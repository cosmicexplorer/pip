"""
The main purpose of this module is to expose LinkCollector.collect_sources().
"""

from __future__ import annotations

import abc
import email.message
import functools
import itertools
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from optparse import Values
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    NamedTuple,
)

from pip._vendor import requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError

from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.filetypes import FileExtensions
from pip._internal.utils.misc import redact_auth_from_url
from pip._internal.utils.urls import ParsedUrl
from pip._internal.vcs import vcs

from .sources import CandidatesFromPage, LinkSource, build_source

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, MutableMapping, Sequence

    from typing_extensions import Self

    ResponseHeaders = MutableMapping[str, str]

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MaybeCachedResponse(abc.ABC):
    etag: str | None
    date: str | None

    @abc.abstractmethod
    def calculate_checksum(self) -> bytes | None: ...


@dataclass(frozen=True, slots=True)
class ServerCachedResponse(MaybeCachedResponse):
    @classmethod
    def parse(cls, resp: Response) -> Self:
        assert resp.status_code == 304
        return cls(
            etag=resp.headers.get("ETag", None),
            date=resp.headers.get("Date", None),
        )

    def calculate_checksum(self) -> bytes | None:
        return None


class ApiSemantics:
    @staticmethod
    @functools.cache
    def _vcs_scheme_regex() -> re.Pattern[str]:
        joined = "|".join(map(re.escape, vcs.schemes))
        return re.compile(
            # Match either "svn+...", or just "svn" itself (from an svn://-prefixed url
            # string).
            f"^({joined})(?:[+]|$)"
            # NB: we could do re.IGNORECASE, but our schemes will all be normalized
            #     anyway, and matching exact casing is faster.
        )

    @staticmethod
    def vcs_scheme_no_web_lookup(url: ParsedUrl) -> str | None:
        """Check for VCS schemes that do not support lookup as web pages.

        Returns the matched VCS scheme, or None if there's no match."""
        if m := ApiSemantics._vcs_scheme_regex().match(url.scheme):
            return m.group(1)
        return None

    class _NotAPIContent(Exception):
        def __init__(self, content_type: str, request_desc: str) -> None:
            super().__init__(content_type, request_desc)
            self.content_type = content_type
            self.request_desc = request_desc

    _api_response_types: ClassVar[tuple[str, ...]] = (
        "text/html",
        "application/vnd.pypi.simple.v1+html",
        "application/vnd.pypi.simple.v1+json",
    )

    @staticmethod
    @functools.cache
    def _api_header_regex() -> re.Pattern[str]:
        joined = "|".join(map(re.escape, ApiSemantics._api_response_types))
        return re.compile(f"^{joined}", flags=re.IGNORECASE)

    @classmethod
    def _ensure_api_header(cls, response: Response) -> None:
        """
        Check the Content-Type header to ensure the response contains a Simple
        API Response.

        :raises:`_NotAPIContent` if the content type is not a valid content-type.
        """
        content_type = response.headers.get("Content-Type", "Unknown")

        if not cls._api_header_regex().match(content_type):
            raise cls._NotAPIContent(content_type, response.request.method)

    class _NotHTTP(Exception):
        pass

    _allowed_simple_api_schemes: ClassVar[frozenset[str]] = frozenset(["http", "https"])

    @classmethod
    def _ensure_api_response(
        cls, url: ParsedUrl, session: PipSession, headers: ResponseHeaders | None = None
    ) -> None:
        """
        Send a HEAD request to the URL, and ensure the response contains a simple
        API Response.

        Raises `_NotHTTP` if the URL is not available for a HEAD request, or
        `_NotAPIContent` if the content type is not a valid content type.
        """
        if url.scheme not in cls._allowed_simple_api_schemes:
            raise cls._NotHTTP()

        resp = session.head(str(url), allow_redirects=True, headers=headers)
        raise_for_status(resp)

        cls._ensure_api_header(resp)

    @staticmethod
    def _format_weighted_content_types(
        weighted_types: Iterable[tuple[str, float | None]],
    ) -> str:
        """
        Specify a set of content types understood for a request and assign them relative
        weights or "quality values". The value for a given type defaults to 1.0 if
        not provided.

        See https://www.rfc-editor.org/rfc/rfc9110.html#name-quality-values.
        """
        # NB: the comma is a stronger delimiter than the semicolon in this context!
        return ", ".join(
            ty if weight is None else f"{ty}; q={weight}"
            for ty, weight in weighted_types
        )

    PREFERRED_INDEX_CONTENT: ClassVar[tuple[tuple[str, float | None], ...]] = (
        # We prefer json over parsing HTML if at all possible.
        ("application/vnd.pypi.simple.v1+json", None),
        ("application/vnd.pypi.simple.v1+html", 0.1),
        ("text/html", 0.01),
    )

    DEFAULT_INDEX_HEADERS: ClassVar[dict[str, str]] = {
        "Accept": _format_weighted_content_types(PREFERRED_INDEX_CONTENT),
        # This workflow can be tested against PyPI with a curl command:
        #
        # > curl --write-out '%{stderr}%{http_code}\n%{stdout}%{header_json}' \
        #        -H 'Accept: application/vnd.pypi.simple.v1+json' \
        #        'https://pypi.org/simple/setuptools/' \
        #        -o pypi-setuptools.json \
        #   | jq
        # 200
        # {
        #   "date": [
        #     "Sat, 30 Aug 2025 00:08:59 GMT"
        #   ],
        #   "cache-control": [
        #     "max-age=600, public"
        #   ],
        #   "etag": [
        #     "\"u2vXpcVCamYifjmRb05NcA\""
        #   ],
        # }
        # > sha256sum pypi-setuptools.json
        # de48e8e6382ebe353ab61550cc627a50a125d5f4964c49ad6992ad820f2bdce8  pypi-setuptools.json # noqa: E501
        # > jq -C <pypi-setuptools.json | less -R
        # {
        #   "alternate-locations": [],
        #   "files": [
        #     {
        #       "core-metadata": false,
        #       "data-dist-info-metadata": false,
        #       "filename": "setuptools-0.6b1-py2.3.egg",
        #       "hashes": {
        #         "sha256": "ae0a6ec6090a92d08fe7f3dbf9f1b2ce889bce2a3d7724b62322a29b92cf93f0" # noqa: E501
        #       },
        #     },
        #   ],
        # }
        # "Cache-Control": "",
        # "Cache-Control": "max-age=0, must-revalidate",
        "Cache-Control": "no-cache",
        # NB: This is very counterintuitive, but "no-cache" in *request* headers means
        #     "always check for updates, but 304 if nothing changed since last time".
        #
        #     (See pypa/pip#5670 for the previous iteration of this.)
        #
        # We record the 'ETag' and 'Date' headers from every index response and write
        # these values (if provided by the server) to file paths within a subdirectory
        # of pip's cache dir (e.g. ~/.cache/pip) generated from hashing the Link object
        # (see pip._internal.cache.Cache.cache_path()).
        #
        # Unlike the default HTTPAdapter in the cachecontrol library, we cache these
        # (very small) metadata strings *separately* from the response itself. This
        # allows us to retrieve the ETag and Date from the most recent successful
        # request to each Link *before* we make the request to that Link, in order to
        # specify the If-None-Match and If-Modified-Since headers for that request. We
        # provide both of these in case a server only supports one and not the other.
        #
        # This workflow can be tested against PyPI with a curl command:
        #
        # > curl --write-out '%{stderr}%{http_code}\n%{stdout}%{header_json}' \
        #        -H 'Accept: application/vnd.pypi.simple.v1+json' \
        #        -H 'Cache-Control: no-cache' \
        #        -H 'If-Modified-Since: Fri, 29 Aug 2025 23:05:25 GMT' \
        #        -H 'If-None-Match: "u2vXpcVCamYifjmRb05NcA"' \
        #        'https://pypi.org/simple/setuptools/' \
        #        -o pypi-setuptools.json \
        #   | jq
        # 304
        # {
        #   "date": [
        #     "Sat, 30 Aug 2025 00:08:59 GMT"
        #   ],
        #   "cache-control": [
        #     "max-age=600, public"
        #   ],
        #   "etag": [
        #     "\"u2vXpcVCamYifjmRb05NcA\""
        #   ],
        #   "x-served-by": [
        #     "cache-iad-kcgs7200038-IAD"
        #   ],
        #   "x-cache": [
        #     "HIT"
        #   ],
        # }
        # > sha256sum pypi-setuptools.json
        # de48e8e6382ebe353ab61550cc627a50a125d5f4964c49ad6992ad820f2bdce8  pypi-setuptools.json # noqa: E501
        #
        # MDN also describes the following as mostly equivalent to no-cache:
        #
        #         Cache-Control: max-age=0, must-revalidate
        #
        # PyPI's behavior is equivalent to no-cache for the above form, but some servers
        # may respond differently.
        "Accept-Encoding": "gzip, deflate",
        # By default, PyPI will not elect to use a compressed transfer encoding, but pip
        # can count on guaranteed built-in support for at least the gzip and deflate
        # encodings from our vendored urllib3. This can be simulated on the command line
        # as well:
        #
        # > curl --write-out '%{stderr}%{http_code}\n%{stdout}%{header_json}' \
        #        -H 'Accept: application/vnd.pypi.simple.v1+json' \
        #        -H 'Accept-Encoding: gzip, deflate' \
        #        'https://pypi.org/simple/setuptools/' \
        #        -o >(gunzip >pypi-setuptools.json) \
        #   | jq
        # 200
        # {
        #   "content-encoding": [
        #     "gzip"
        #   ],
        #   "content-type": [
        #     "application/vnd.pypi.simple.v1+json"
        #   ],
        #   "vary": [
        #     "Accept, Accept-Encoding"
        #   ],
        # }
        # > sha256sum pypi-setuptools.json
        # de48e8e6382ebe353ab61550cc627a50a125d5f4964c49ad6992ad820f2bdce8  pypi-setuptools.json # noqa: E501
        #
        # Note that a 304 response will be 0 bytes in size. This is exactly what we want
        # for pip, but the above command line's shell redirections and output
        # substitution would truncate the output file if we added the If-Modified-Since
        # or If-None-Match headers to the curl request to get a successful cache hit.
        #
        # Instead, curl has a built-in option --compress which will both send the
        # Accept-Encoding header as well as transparently decompress any output:
        #
        # > curl --write-out '%{stderr}%{http_code}\n%{stdout}%{header_json}' \
        #        --compressed \
        #        -H 'Accept: application/vnd.pypi.simple.v1+json' \
        #        -H 'Cache-Control: no-cache' \
        #        -H 'If-Modified-Since: Fri, 29 Aug 2025 23:05:25 GMT' \
        #        'https://pypi.org/simple/setuptools/' \
        #        -o pypi-setuptools.json \
        #  | jq
        # 304
        # {
        #   "date": [
        #     "Sat, 30 Aug 2025 04:02:08 GMT"
        #   ],
        #   "etag": [
        #     "\"wu2HZkJTeVb9z3QVpwD9lQ\""
        #   ],
        # }
        # > sha256sum pypi-setuptools.json
        # de48e8e6382ebe353ab61550cc627a50a125d5f4964c49ad6992ad820f2bdce8  pypi-setuptools.json # noqa: E501
        #
        # Note finally that the ETag header value provided by the server in response to
        # the request for a compressed transfer encoding will almost definitely be
        # different than the ETag it generates for the uncompressed version. So while
        # the cryptographic checksum of the *decompressed* output will remain stable
        # regardless of the selected transfer encoding, the cached ETag and Date headers
        # should probably incorporate the selected transfer encoding into their cache
        # key along with the Link itself.
        #
        # Useful references:
        # - https://developer.mozilla.org/en-US/docs/Web/HTTP/Reference/Headers/Cache-Control#no-cache_2 # noqa: E501
        # - https://httpwg.org/specs/rfc9111.html#validation.sent
        # - https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Caching#etagif-none-match # noqa: E501
        # - https://www.fastly.com/documentation/guides/concepts/shielding/#debugging
        # - https://www.fastly.com/documentation/guides/full-site-delivery/caching/checking-cache/#using-a-fastly-debug-header-with-curl # noqa: E501
    }

    @classmethod
    def _get_simple_response(
        cls, url: ParsedUrl, session: PipSession, headers: ResponseHeaders | None = None
    ) -> Response:
        """Access an Simple API response with GET, and return the response.

        This consists of three parts:

        1. If the URL looks suspiciously like an archive, send a HEAD first to
           check the Content-Type is HTML or Simple API, to avoid downloading a
           large file. Raise `_NotHTTP` if the content type cannot be determined, or
           `_NotAPIContent` if it is not HTML or a Simple API.
        2. Actually perform the request. Raise HTTP exceptions on network failures.
        3. Check the Content-Type header to make sure we got a Simple API response,
           and raise `_NotAPIContent` otherwise.
        """
        if ext := FileExtensions.archive_file_extension(url._filename):
            logger.warning(
                "Prospective URL '%s' has extension '%s' "
                "and may not be a simple API endpoint. "
                "Sending a HEAD response first to confirm.",
                url,
                ext,
            )
            cls._ensure_api_response(url, session=session, headers=headers)

        logger.debug("Getting page %s", url.with_redacted_auth_info())

        if headers:
            logger.debug("(additional) headers: %s", headers)
        resp = session.get_for_index(
            url,
            headers={
                **cls.DEFAULT_INDEX_HEADERS,
                **(headers or {}),
            },
        )
        if resp.status_code == 304:
            return resp
        raise_for_status(resp)

        # The check for archives above only works if the url ends with
        # something that looks like an archive. However that is not a
        # requirement of an url. Unless we issue a HEAD request on every
        # url we cannot know ahead of time for sure if something is a
        # Simple API response or not. However we can check after we've
        # downloaded it.
        cls._ensure_api_header(resp)

        logger.debug(
            "Fetched page %s as %s",
            url.with_redacted_auth_info(),
            resp.headers.get("Content-Type", "Unknown"),
        )

        return resp

    @staticmethod
    def _get_encoding_from_headers(headers: ResponseHeaders) -> str | None:
        """Determine if we have any encoding information in our headers."""
        if headers and "Content-Type" in headers:
            m = email.message.Message()
            m["content-type"] = headers["Content-Type"]
            charset = m.get_param("charset")
            if charset:
                return str(charset)
        return None

    @staticmethod
    def _handle_get_simple_fail(
        link: Link,
        reason: str | Exception,
        meth: Callable[..., None] | None = None,
    ) -> None:
        if meth is None:
            meth = logger.debug
        meth("Could not fetch URL %s: %s - skipping", link, reason)

    @classmethod
    def _make_index_content(
        cls,
        response: Response,
    ) -> MaybeCachedResponse:
        if response.status_code == 304:
            return ServerCachedResponse.parse(response)
        assert response.status_code == 200, response
        encoding = cls._get_encoding_from_headers(response.headers)
        return IndexContent.create(
            content=response.content,
            content_type=response.headers["Content-Type"],
            encoding=encoding,
            content_length=int(response.headers["Content-Length"]),
            url=response.url,
            content_encoding=response.headers.get("Content-Encoding", None),
            etag=response.headers.get("ETag", None),
            date=response.headers.get("Date", None),
        )

    _index_html_url: ClassVar[ParsedUrl] = ParsedUrl.parse("index.html")

    @classmethod
    def _get_index_content(
        cls, link: Link, *, session: PipSession, headers: ResponseHeaders | None = None
    ) -> MaybeCachedResponse | None:
        url = link.url_without_fragment()

        # Check for VCS schemes that do not support lookup as web pages.
        if vcs_scheme := cls.vcs_scheme_no_web_lookup(url):
            logger.warning(
                "Cannot look at %s URL %s "
                "because it does not support lookup as web pages.",
                vcs_scheme,
                link,
            )
            return None

        # Tack index.html onto file:// URLs that point to directories
        if url.scheme == "file" and os.path.isdir(
            urllib.request.url2pathname(url.path)
        ):
            # TODO: In the future, it would be nice if pip supported PEP 691
            #       style responses in the file:// URLs, however there's no
            #       standard file extension for application/vnd.pypi.simple.v1+json
            #       so we'll need to come up with something on our own.
            url = url.join(cls._index_html_url)
            logger.debug(" file: URL is directory, getting %s", url)

        try:
            resp = cls._get_simple_response(url, session=session, headers=headers)
        except cls._NotHTTP:
            logger.warning(
                "Skipping page %s because it looks like an archive, and cannot "
                "be checked by a HTTP HEAD request.",
                link,
            )
        except cls._NotAPIContent as exc:
            logger.warning(
                "Skipping page %s because the %s request got Content-Type: %r. "
                "The only supported Content-Type values are:\n%s",
                link,
                exc.request_desc,
                exc.content_type,
                "\n".join(f"- {ty}" for ty in cls._api_response_types),
            )
        except (NetworkConnectionError, RetryError) as exc:
            cls._handle_get_simple_fail(link, exc)
        except SSLError as exc:
            reason = "There was a problem confirming the ssl certificate: "
            reason += str(exc)
            cls._handle_get_simple_fail(link, reason, meth=logger.info)
        except requests.ConnectionError as exc:
            cls._handle_get_simple_fail(link, f"connection error: {exc}")
        except requests.Timeout:
            cls._handle_get_simple_fail(link, "timed out")
        else:
            return cls._make_index_content(resp)
        return None


@dataclass(frozen=True, slots=True)
class IndexContent(MaybeCachedResponse):
    """Represents one response (or page), along with its URL.

    :param encoding: the encoding to decode the given content.
    :param url: the URL from which the HTML was downloaded.
    :param etag: The ``ETag`` header from an HTTP request against ``url``.
    :param date: The ``Date`` header from an HTTP request against ``url``.
    """

    # FIXME: use stream=True and calculate checksum along with incremental link parsing!
    content: bytes
    content_type: str
    encoding: str | None
    content_length: int
    url: ParsedUrl
    content_encoding: str | None

    @classmethod
    def create(
        cls,
        content: bytes,
        content_type: str,
        encoding: str | None,
        url: str | ParsedUrl,
        content_length: int | None = None,
        content_encoding: str | None = None,
        etag: str | None = None,
        date: str | None = None,
    ) -> Self:
        """
        :param encoding: the encoding to decode the given content.
        :param url: the URL from which the HTML was downloaded.
        """
        if isinstance(url, str):
            url = ParsedUrl.parse(url)
        url = url.with_quoted_path()
        if content_length is None:
            content_length = len(content)
        return cls(
            etag=etag,
            date=date,
            content=content,
            content_type=content_type,
            encoding=encoding,
            content_length=content_length,
            url=url,
            content_encoding=content_encoding,
        )

    def calculate_checksum(self) -> bytes:
        hasher = sha256()
        hasher.update(self.content)
        return hasher.digest()

    def __str__(self) -> str:
        return str(self.url.with_redacted_auth_info())

    _json_content_regex: ClassVar[re.Pattern[str]] = re.compile(
        re.escape("application/vnd.pypi.simple.v1+json"),
        flags=re.IGNORECASE,
    )

    def parse_links(self) -> Iterator[Link]:
        """
        Parse a Simple API's Index Content, and yield its anchor elements as
        Link objects.
        """
        if self.__class__._json_content_regex.match(self.content_type):
            data = json.loads(self.content)
            for file in data.get("files", ()):
                link = Link.from_json(file, self.url, page_content=self)
                if link is None:
                    continue
                yield link
            return

        parser = HTMLLinkParser(self.url)
        encoding = self.encoding or "utf-8"
        parser.feed(self.content.decode(encoding))

        base_url = parser.base_url or self.url
        for anchor in parser.anchors:
            link = Link.from_element(
                anchor, page_url=self.url, base_url=base_url, page_content=self
            )
            if link is None:
                continue
            yield link


class HTMLLinkParser(HTMLParser):
    """
    HTMLParser that keeps the first base HREF and a list of all anchor
    elements' attributes.
    """

    def __init__(self, url: ParsedUrl) -> None:
        super().__init__(convert_charrefs=True)

        self.url: ParsedUrl = url
        self.base_url: ParsedUrl | None = None
        self.anchors: list[dict[str, str | None]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "base" and self.base_url is None:
            href = self.get_href(attrs)
            if href is not None:
                self.base_url = ParsedUrl.parse(href).with_quoted_path()
        elif tag == "a":
            self.anchors.append(dict(attrs))

    def get_href(self, attrs: list[tuple[str, str | None]]) -> str | None:
        for name, value in attrs:
            if name == "href":
                return value
        return None


class CollectedSources(NamedTuple):
    find_links: Sequence[LinkSource | None]
    index_urls: Sequence[LinkSource | None]


class LinkCollector:
    """
    Responsible for collecting Link objects from all configured locations,
    making network requests as needed.

    The class's main method is its collect_sources() method.
    """

    def __init__(
        self,
        session: PipSession,
        search_scope: SearchScope,
    ) -> None:
        self.search_scope = search_scope
        self.session = session

    @classmethod
    def create(
        cls,
        session: PipSession,
        options: Values,
        suppress_no_index: bool = False,
    ) -> Self:
        """
        :param session: The Session to use to make requests.
        :param suppress_no_index: Whether to ignore the --no-index option
            when constructing the SearchScope object.
        """
        index_urls = [options.index_url] + options.extra_index_urls
        if options.no_index and not suppress_no_index:
            logger.debug(
                "Ignoring indexes: %s",
                ",".join(redact_auth_from_url(url) for url in index_urls),
            )
            index_urls = []

        # Make sure find_links is a list before passing to create().
        find_links = options.find_links or []

        search_scope = SearchScope.create(
            find_links=find_links,
            index_urls=index_urls,
            no_index=options.no_index,
        )
        return cls(
            session=session,
            search_scope=search_scope,
        )

    @property
    def find_links(self) -> list[str]:
        return self.search_scope.find_links

    def fetch_response(
        self, location: Link, headers: ResponseHeaders | None = None
    ) -> MaybeCachedResponse | None:
        """
        Fetch an HTML page containing package links.
        """
        return ApiSemantics._get_index_content(
            location, session=self.session, headers=headers
        )

    def collect_sources(
        self,
        project_name: str,
        candidates_from_page: CandidatesFromPage,
    ) -> CollectedSources:
        # The OrderedDict calls deduplicate sources by URL.
        index_url_sources = OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=False,
                project_name=project_name,
            )
            for loc in self.search_scope.get_index_urls_locations(project_name)
        ).values()
        find_links_sources = OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=True,
                project_name=project_name,
            )
            for loc in self.find_links
        ).values()

        if logger.isEnabledFor(logging.DEBUG):
            lines = [
                f"* {s.link}"
                for s in itertools.chain(find_links_sources, index_url_sources)
                if s is not None and s.link is not None
            ]
            lines = [
                f"{len(lines)} location(s) to search "
                f"for versions of {project_name}:"
            ] + lines
            logger.debug("\n".join(lines))

        return CollectedSources(
            find_links=list(find_links_sources),
            index_urls=list(index_url_sources),
        )
