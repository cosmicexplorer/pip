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
from dataclasses import dataclass
from hashlib import sha256
from html.parser import HTMLParser
from optparse import Values
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Callable,
    ClassVar,
    Generic,
    TypeVar,
)

from pip._vendor import requests
from pip._vendor.filelock import FileLock
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError

from pip._internal.cache import (
    FetchResolveCache,
    SerializableEntry,
)
from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link, LinkJsonCodec
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.cache_serde import (
    Bz2TextStream,
    JsonCodec,
    SerializableProcess,
)
from pip._internal.utils.filetypes import FileExtensions
from pip._internal.utils.urls import ParsedUrl
from pip._internal.vcs import vcs

from .caching import _ChecksumSanitizer, _ETagSanitizer, _TimestampSanitizer
from .sources import _LocalFindLinks

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, MutableMapping

    from typing_extensions import Self

    from pip._vendor.packaging.utils import NormalizedName

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
        url: ParsedUrl,
        reason: str | Exception,
        meth: Callable[..., None] | None = None,
    ) -> None:
        if meth is None:
            meth = logger.debug
        meth("Could not fetch URL %s: %s - skipping", url, reason)

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

    @classmethod
    def _get_index_content(
        cls,
        url: ParsedUrl,
        *,
        session: PipSession,
        headers: ResponseHeaders | None = None,
    ) -> MaybeCachedResponse | None:
        # Check for VCS schemes that do not support lookup as web pages.
        if vcs_scheme := cls.vcs_scheme_no_web_lookup(url):
            logger.warning(
                "Cannot look at %s URL %s "
                "because it does not support lookup as web pages.",
                vcs_scheme,
                url,
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
            url = url.join_path("index.html")
            logger.debug(" file: URL is directory, getting %s", url)

        try:
            resp = cls._get_simple_response(url, session=session, headers=headers)
        except cls._NotHTTP:
            logger.warning(
                "Skipping page %s because it looks like an archive, and cannot "
                "be checked by a HTTP HEAD request.",
                url,
            )
        except cls._NotAPIContent as exc:
            logger.warning(
                "Skipping page %s because the %s request got Content-Type: %r. "
                "The only supported Content-Type values are:\n%s",
                url,
                exc.request_desc,
                exc.content_type,
                "\n".join(f"- {ty}" for ty in cls._api_response_types),
            )
        except (NetworkConnectionError, RetryError) as exc:
            cls._handle_get_simple_fail(url, exc)
        except SSLError as exc:
            reason = "There was a problem confirming the ssl certificate: "
            reason += str(exc)
            cls._handle_get_simple_fail(url, reason, meth=logger.info)
        except requests.ConnectionError as exc:
            cls._handle_get_simple_fail(url, f"connection error: {exc}")
        except requests.Timeout:
            cls._handle_get_simple_fail(url, "timed out")
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


PageParsed = TypeVar("PageParsed")
PageResult = TypeVar("PageResult")


class FetchComputeProcess(abc.ABC, Generic[PageParsed, PageResult]):
    @abc.abstractmethod
    def compute(self, parsed: PageParsed) -> PageResult: ...


class ComputeMerger(abc.ABC, Generic[PageResult]):
    @abc.abstractmethod
    def merge(self, results: Iterable[PageResult]) -> PageResult: ...


@dataclass(frozen=True, eq=False)
class LinkCollector:
    """
    Responsible for collecting Link objects from all configured locations,
    making network requests as needed.

    The class's main method is its collect_sources() method.
    """

    session: PipSession
    search_scope: SearchScope
    fetch_resolve_cache: FetchResolveCache | None = None

    @classmethod
    def create(
        cls,
        session: PipSession,
        options: Values,
        *,
        suppress_no_index: bool = False,
    ) -> Self:
        """
        :param session: The Session to use to make requests.
        :param suppress_no_index: Whether to ignore the --no-index option
            when constructing the SearchScope object.
        """
        index_urls = [options.index_url] + options.extra_index_urls
        # Make sure find_links is a list before passing to create().
        find_links = options.find_links or []

        search_scope = SearchScope.create(
            find_links=find_links,
            index_urls=index_urls,
            no_index=options.no_index,
            suppress_no_index=suppress_no_index,
        ).filter_urls(session.is_secure_origin)

        if bool(options.cache_dir) and ("metadata-cache" in options.features_enabled):
            fetch_resolve_cache = FetchResolveCache(options.cache_dir)
        else:
            fetch_resolve_cache = None

        return cls(
            session=session,
            search_scope=search_scope,
            fetch_resolve_cache=fetch_resolve_cache,
        )

    def _uncached_fetch_and_process(
        self,
        location: ParsedUrl,
        compute: FetchComputeProcess[Iterable[Link], PageResult],
    ) -> PageResult:
        logger.debug(
            "(fetch cache disabled) Fetching project page and analyzing links: %s",
            location,
        )

        index_response = self._fetch_response(location)
        if index_response is None:
            return compute.compute([])
        assert isinstance(index_response, IndexContent), index_response

        return compute.compute(index_response.parse_links())

    def fetch_and_process(
        self,
        location: ParsedUrl,
        entry: SerializableEntry,
        process: SerializableProcess[PageResult],
        compute: FetchComputeProcess[Iterable[Link], PageResult],
    ) -> PageResult:
        if self.fetch_resolve_cache is None:
            return self._uncached_fetch_and_process(location, compute)

        cached_path = self.fetch_resolve_cache.fetch_cache_path(location)
        os.makedirs(str(cached_path), exist_ok=True)

        lock = FileLock(str(cached_path / ".fetch.lock"))

        etag_path = cached_path / "etag"
        date_path = cached_path / "modified-since-date"
        checksum_path = cached_path / "checksum"
        parsed_links_path = cached_path / "parsed-links"
        cur_entry_path = cached_path / FetchResolveCache.hashed_entry_name(entry)

        links_codec = JsonCodec(
            codec=LinkJsonCodec(),
            stream=Bz2TextStream(),
        )

        with lock.acquire(timeout=30):
            headers: dict[str, str] = {}
            # NB: mutates headers!
            prev_etag, _prev_date, prev_checksum = self._try_load_http_cache_headers(
                etag_path, date_path, checksum_path, location, headers
            )

            logger.debug("Fetching project page: %s", location)

            index_response = self._fetch_response(location, headers=headers)
            if index_response is None:
                return compute.compute([])

            new_checksum: _ChecksumSanitizer.PageChecksum | None = None
            if (new_checksum_bytes := index_response.calculate_checksum()) is not None:
                new_checksum = _ChecksumSanitizer.PageChecksum(
                    checksum=new_checksum_bytes
                )
                page_unmodified = new_checksum == prev_checksum
            else:
                # The only way we wouldn't be able to calculate a checksum is if we
                # received a 304 (server cached) response.
                page_unmodified = True
                # If that's the case, then we must have received a checksum.
                assert prev_checksum is not None
                # In case the ETag or Date were updated, we still write the HTTP caching
                # info to file before returning.
                new_checksum = prev_checksum

            page_links: tuple[Link, ...] | None = None
            # Only try our persistent link parsing and evaluation caches if we know the
            # page was unmodified via checksum.
            if page_unmodified:
                if (result := process.load_from_path(cur_entry_path)) is not None:
                    return result

                page_links = links_codec.load_from_path(parsed_links_path)
            else:
                # The page data was modified, so we have to clear not only our prior
                # parsing, but any optimistic evaluation cache results.
                FetchResolveCache.clear_hashed_entries(cached_path, type(entry))
                try:
                    parsed_links_path.unlink()
                except OSError:
                    pass

            if page_links is None:
                logger.debug(
                    "extracting new parsed links from index response %s", index_response
                )
                # This is necessary for mypy to allow us to call .parse_links(), but the
                # correctness is more subtle. This will be true when the response is not
                # a 304 Not Modified, but rather a 200 OK. When does a 304 occur? When
                # we have cached ETag and Date parameters which are provided as headers
                # to the index request. Because we call ._write_http_cache_info()
                # strictly *after* we call ._write_parsed_links_cache(), we ensure that
                # even in the presence of parallel pip executions, we never have
                # a cached ETag/Date without also caching the result of link parsing.
                assert not page_unmodified and isinstance(
                    index_response, IndexContent
                ), (
                    page_unmodified,
                    index_response,
                )
                page_links = tuple(index_response.parse_links())
                links_codec.write_to_path(parsed_links_path, page_links)
            assert page_links is not None

            computed_result = compute.compute(page_links)
            process.write_to_path(cur_entry_path, computed_result)

            # NB: We have to write these cache entries in reverse order of calculating
            #     the actual results to avoid an unfortunate race condition where we
            #     don't have any copy of the previous page when we get a 304!
            self.__class__._write_http_cache_info(
                etag_path=etag_path,
                new_etag_str=index_response.etag,
                date_path=date_path,
                date_str=index_response.date,
                project_url=location,
                prev_etag=prev_etag,
                new_checksum=new_checksum,
                checksum_path=checksum_path,
                page_unmodified=page_unmodified,
            )

            return computed_result

    @classmethod
    def _try_load_http_cache_headers(
        cls,
        etag_path: Path,
        date_path: Path,
        checksum_path: Path,
        project_url: ParsedUrl,
        headers: MutableMapping[str, str],
    ) -> tuple[
        _ETagSanitizer.CacheableETag | None,
        _TimestampSanitizer.CacheableTimestamp | None,
        _ChecksumSanitizer.PageChecksum | None,
    ]:
        """Reads ETag and Date values from cache paths to update request headers.

        If either or neither path is a readable file, the headers are unchanged. They
        will be populated with the ETag and Date headers from the server response, if
        the server provides these headers in its response.
        """
        etag: _ETagSanitizer.CacheableETag | None = None
        try:
            etag = _ETagSanitizer.CacheableETag.cache_deserialize(
                etag_path.read_bytes()
            )
            logger.debug(
                "found cached etag for url %s at %s: %s",
                project_url,
                etag_path,
                etag,
            )
            headers["If-None-Match"] = etag.format_http_header()
        except OSError as e:
            logger.debug("no etag found for url %s (%s)", project_url, e)

        date: _TimestampSanitizer.CacheableTimestamp | None = None
        try:
            date = _TimestampSanitizer.CacheableTimestamp.cache_deserialize(
                date_path.read_bytes()
            )
            logger.debug(
                "found cached date for url %s at %s: %s",
                project_url,
                date_path,
                date,
            )
            headers["If-Modified-Since"] = date.format_http_header()
        except OSError as e:
            logger.debug("no date found for url %s (%s)", project_url, e)

        checksum: _ChecksumSanitizer.PageChecksum | None = None
        try:
            checksum = _ChecksumSanitizer.PageChecksum.cache_deserialize(
                checksum_path.read_bytes()
            )
            logger.debug(
                "found checksum for url %s at %s: %s",
                project_url,
                checksum_path,
                checksum,
            )
        except OSError as e:
            logger.debug("no checksum found for url %s (%s)", project_url, e)

        return (etag, date, checksum)

    @classmethod
    def _write_http_cache_info(
        cls,
        etag_path: Path,
        new_etag_str: str | None,
        date_path: Path,
        date_str: str | None,
        project_url: ParsedUrl,
        prev_etag: _ETagSanitizer.CacheableETag | None,
        new_checksum: _ChecksumSanitizer.PageChecksum,
        checksum_path: Path,
        page_unmodified: bool,
    ) -> None:
        new_etag = _ETagSanitizer.CacheableETag.parse_http_header(
            new_etag_str, project_url
        )
        if new_etag is None:
            try:
                etag_path.unlink()
            except OSError:
                pass
        else:
            if new_etag != prev_etag:
                logger.debug(
                    "etag for url %s updated from %s -> %s",
                    project_url,
                    prev_etag,
                    new_etag,
                )
                etag_path.write_bytes(new_etag.cache_serialize())
            else:
                logger.debug(
                    "etag was unmodified for url %s (%s)", project_url, prev_etag
                )
                assert page_unmodified

        new_date = _TimestampSanitizer.CacheableTimestamp.parse_http_header(
            date_str, project_url
        )
        if new_date is None:
            try:
                date_path.unlink()
            except OSError:
                pass
        else:
            date_path.write_bytes(new_date.cache_serialize())
            logger.debug('date "%s" written for url %s', new_date, project_url)

        checksum_path.write_bytes(new_checksum.cache_serialize())
        logger.debug("checksum %s written for url %s", new_checksum, project_url)

    def _fetch_response(
        self, location: ParsedUrl, headers: ResponseHeaders | None = None
    ) -> MaybeCachedResponse | None:
        """
        Fetch an HTML page containing package links.
        """
        return ApiSemantics._get_index_content(
            location, session=self.session, headers=headers
        )

    ### In-memory caching for local and remote calls.
    # NB: Traverses the top-level directory contents when first called.
    @staticmethod
    @functools.cache
    def _path_links(path: Path) -> _LocalFindLinks:
        return _LocalFindLinks(path=path)

    @functools.cached_property
    def _find_links_is_api(self) -> dict[ParsedUrl, bool]:
        return {}

    # NB: Performs a HEAD request when first called.
    def _remote_find_links_is_api(self, url: ParsedUrl) -> bool:
        if (result := self._find_links_is_api.get(url)) is not None:
            return result
        computed: bool
        try:
            ApiSemantics._ensure_api_response(url, self.session)
            computed = True
        except (ApiSemantics._NotHTTP, ApiSemantics._NotAPIContent):
            computed = False
        self._find_links_is_api[url] = computed
        return computed

    ### Collection of direct `Link`s to dists and `ParsedUrl`s of index pages to parse.
    ## [--find-links processing.]
    def _local_find_links(self) -> Iterator[_LocalFindLinks]:
        for u in self.search_scope.find_links:
            if (p := u.path) is not None:
                yield self.__class__._path_links(p)

    ## -> Local dists.
    def _collect_local_find_links_files(self, name: NormalizedName) -> Iterator[Link]:
        for local_links in self._local_find_links():
            for project_url in local_links.urls_for_project(name):
                yield Link(project_url)

    ## -> Local indices.
    def _collect_local_find_links_pages(self) -> Iterator[ParsedUrl]:
        for local_links in self._local_find_links():
            yield from local_links.possible_index_pages()

    def _remote_find_links(self) -> Iterator[ParsedUrl]:
        for u in self.search_scope.find_links:
            if u.path is None:
                yield u.url

    ## -> Remote dists.
    def _collect_remote_find_links_files(self) -> Iterator[Link]:
        for url in self._remote_find_links():
            if not self._remote_find_links_is_api(url):
                yield Link(url)

    ## -> Remote indices.
    def _collect_remote_find_links_pages(self) -> Iterator[ParsedUrl]:
        for url in self._remote_find_links():
            if self._remote_find_links_is_api(url):
                yield url

    ## [--[extra-]index-url[s] processing.]
    def _collect_local_and_remote_index_pages(
        self,
        name: NormalizedName,
    ) -> tuple[tuple[ParsedUrl, ...], tuple[ParsedUrl, ...]]:
        local_index_pages: list[ParsedUrl] = []
        remote_index_pages: list[ParsedUrl] = []

        for u in self.search_scope.index_pages_for_name(name):
            if u.path is not None:
                local_index_pages.append(u.url)
            else:
                remote_index_pages.append(u.url)

        return tuple(local_index_pages), tuple(remote_index_pages)

    ### Generate all valid `Link` values for the given project, then evaluate them
    ### (which includes filtering out invalid ones) to get `InstallationCandidate`s.
    def _collect_files(self, name: NormalizedName) -> Iterator[Link]:
        yield from self._collect_local_find_links_files(name)
        yield from self._collect_remote_find_links_files()

    # TODO: cache this somehow? maybe by checksumming file contents?
    def _collect_and_process_files(
        self,
        name: NormalizedName,
        compute: FetchComputeProcess[Iterable[Link], PageResult],
    ) -> PageResult:
        return compute.compute(self._collect_files(name))

    def _collect_pages(self, name: NormalizedName) -> Iterator[ParsedUrl]:
        yield from self._collect_local_find_links_pages()
        local_index_pages, remote_index_pages = (
            self._collect_local_and_remote_index_pages(name)
        )
        yield from local_index_pages
        yield from self._collect_remote_find_links_pages()
        yield from remote_index_pages

    def _collect_and_process_pages(
        self,
        name: NormalizedName,
        entry: SerializableEntry,
        process: SerializableProcess[PageResult],
        compute: FetchComputeProcess[Iterable[Link], PageResult],
    ) -> Iterator[PageResult]:
        for location in self._collect_pages(name):
            yield self.fetch_and_process(location, entry, process, compute)

    def collect_and_process(
        self,
        name: NormalizedName,
        entry: SerializableEntry,
        process: SerializableProcess[PageResult],
        compute: FetchComputeProcess[Iterable[Link], PageResult],
        merger: ComputeMerger[PageResult],
    ) -> PageResult:
        files_result = self._collect_and_process_files(name, compute)
        pages_results = self._collect_and_process_pages(name, entry, process, compute)
        return merger.merge(itertools.chain([files_result], pages_results))
