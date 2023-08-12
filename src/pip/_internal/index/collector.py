"""
The main purpose of this module is to expose LinkCollector.collect_sources().
"""

import collections
import email.message
import itertools
import json
import logging
import os
import urllib.parse
import urllib.request
from html.parser import HTMLParser
from optparse import Values
from typing import (
    TYPE_CHECKING,
    Callable,
    Dict,
    Iterable,
    List,
    MutableMapping,
    NamedTuple,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from pip._vendor import requests
from pip._vendor.requests import Response
from pip._vendor.requests.exceptions import RetryError, SSLError

from pip._internal.exceptions import NetworkConnectionError
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.network.session import PipSession
from pip._internal.network.utils import raise_for_status
from pip._internal.utils.filetypes import is_archive_file
from pip._internal.utils.misc import redact_auth_from_url
from pip._internal.vcs import vcs

from .sources import CandidatesFromPage, LinkSource, build_source

if TYPE_CHECKING:
    from typing import Protocol
else:
    Protocol = object

logger = logging.getLogger(__name__)

ResponseHeaders = MutableMapping[str, str]


def _match_vcs_scheme(url: str) -> Optional[str]:
    """Look for VCS schemes in the URL.

    Returns the matched VCS scheme, or None if there's no match.
    """
    for scheme in vcs.schemes:
        if url.lower().startswith(scheme) and url[len(scheme)] in "+:":
            return scheme
    return None


class _NotAPIContent(Exception):
    def __init__(self, content_type: str, request_desc: str) -> None:
        super().__init__(content_type, request_desc)
        self.content_type = content_type
        self.request_desc = request_desc


def _ensure_api_header(response: Response) -> None:
    """
    Check the Content-Type header to ensure the response contains a Simple
    API Response.

    Raises `_NotAPIContent` if the content type is not a valid content-type.
    """
    content_type = response.headers.get("Content-Type", "Unknown")

    content_type_l = content_type.lower()
    if content_type_l.startswith(
        (
            "text/html",
            "application/vnd.pypi.simple.v1+html",
            "application/vnd.pypi.simple.v1+json",
        )
    ):
        return

    raise _NotAPIContent(content_type, response.request.method)


class _NotHTTP(Exception):
    pass


def _ensure_api_response(
    url: str, session: PipSession, headers: Optional[Dict[str, str]] = None
) -> None:
    """
    Send a HEAD request to the URL, and ensure the response contains a simple
    API Response.

    Raises `_NotHTTP` if the URL is not available for a HEAD request, or
    `_NotAPIContent` if the content type is not a valid content type.
    """
    scheme, netloc, path, query, fragment = urllib.parse.urlsplit(url)
    if scheme not in {"http", "https"}:
        raise _NotHTTP()

    resp = session.head(url, allow_redirects=True, headers=headers)
    raise_for_status(resp)

    _ensure_api_header(resp)


def _get_simple_response(
    url: str, session: PipSession, headers: Optional[Dict[str, str]] = None
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
    if is_archive_file(Link(url).filename):
        _ensure_api_response(url, session=session, headers=headers)

    logger.debug("Getting page %s", redact_auth_from_url(url))

    logger.debug("headers: %s", str(headers))
    resp = session.get(
        url,
        headers={
            "Accept": ", ".join(
                [
                    "application/vnd.pypi.simple.v1+json",
                    "application/vnd.pypi.simple.v1+html; q=0.1",
                    "text/html; q=0.01",
                ]
            ),
            # We don't want to blindly returned cached data for
            # /simple/, because authors generally expecting that
            # twine upload && pip install will function, but if
            # they've done a pip install in the last ~10 minutes
            # it won't. Thus by setting this to zero we will not
            # blindly use any cached data, however the benefit of
            # using max-age=0 instead of no-cache, is that we will
            # still support conditional requests, so we will still
            # minimize traffic sent in cases where the page hasn't
            # changed at all, we will just always incur the round
            # trip for the conditional GET now instead of only
            # once per 10 minutes.
            # For more information, please see pypa/pip#5670.
            "Cache-Control": "max-age=0",
            **headers,
        },
    )
    raise_for_status(resp)

    # The check for archives above only works if the url ends with
    # something that looks like an archive. However that is not a
    # requirement of an url. Unless we issue a HEAD request on every
    # url we cannot know ahead of time for sure if something is a
    # Simple API response or not. However we can check after we've
    # downloaded it.
    _ensure_api_header(resp)

    logger.debug(
        "Fetched page %s as %s",
        redact_auth_from_url(url),
        resp.headers.get("Content-Type", "Unknown"),
    )

    return resp


def _get_encoding_from_headers(headers: ResponseHeaders) -> Optional[str]:
    """Determine if we have any encoding information in our headers."""
    if headers and "Content-Type" in headers:
        m = email.message.Message()
        m["content-type"] = headers["Content-Type"]
        charset = m.get_param("charset")
        if charset:
            return str(charset)
    return None


def parse_links(page: "IndexContent") -> Iterable[Link]:
    """
    Parse a Simple API's Index Content, and yield its anchor elements as Link objects.
    """

    content_type_l = page.content_type.lower()
    if content_type_l.startswith("application/vnd.pypi.simple.v1+json"):
        data = json.loads(page.content)
        for file in data.get("files", []):
            link = Link.from_json(file, page.url, page_content=page)
            if link is None:
                continue
            yield link
        return

    parser = HTMLLinkParser(page.url)
    encoding = page.encoding or "utf-8"
    parser.feed(page.content.decode(encoding))

    url = page.url
    base_url = parser.base_url or url
    for anchor in parser.anchors:
        link = Link.from_element(
            anchor, page_url=url, base_url=base_url, page_content=page
        )
        if link is None:
            continue
        yield link


class IndexContent:
    """Represents one response (or page), along with its URL"""

    def __init__(
        self,
        content: bytes,
        content_type: str,
        encoding: Optional[str],
        url: str,
        etag: Optional[str] = None,
        date: Optional[str] = None,
    ) -> None:
        """
        :param encoding: the encoding to decode the given content.
        :param url: the URL from which the HTML was downloaded.
        """
        self.content = content
        self.content_type = content_type
        self.encoding = encoding
        self.url = url
        self.etag = etag
        self.date = date

    def __str__(self) -> str:
        return redact_auth_from_url(self.url)


class HTMLLinkParser(HTMLParser):
    """
    HTMLParser that keeps the first base HREF and a list of all anchor
    elements' attributes.
    """

    def __init__(self, url: str) -> None:
        super().__init__(convert_charrefs=True)

        self.url: str = url
        self.base_url: Optional[str] = None
        self.anchors: List[Dict[str, Optional[str]]] = []

    def handle_starttag(self, tag: str, attrs: List[Tuple[str, Optional[str]]]) -> None:
        if tag == "base" and self.base_url is None:
            href = self.get_href(attrs)
            if href is not None:
                self.base_url = href
        elif tag == "a":
            self.anchors.append(dict(attrs))

    def get_href(self, attrs: List[Tuple[str, Optional[str]]]) -> Optional[str]:
        for name, value in attrs:
            if name == "href":
                return value
        return None


def _handle_get_simple_fail(
    link: Link,
    reason: Union[str, Exception],
    meth: Optional[Callable[..., None]] = None,
) -> None:
    if meth is None:
        meth = logger.debug
    meth("Could not fetch URL %s: %s - skipping", link, reason)


def _make_index_content(response: Response) -> IndexContent:
    encoding = _get_encoding_from_headers(response.headers)
    return IndexContent(
        response.content,
        response.headers["Content-Type"],
        encoding=encoding,
        url=response.url,
        etag=response.headers.get("ETag", None),
        date=response.headers.get("Date", None),
    )


def _get_index_content(
    link: Link, *, session: PipSession, headers: Optional[Dict[str, str]] = None
) -> Optional["IndexContent"]:
    url = link.url_without_fragment

    # Check for VCS schemes that do not support lookup as web pages.
    vcs_scheme = _match_vcs_scheme(url)
    if vcs_scheme:
        logger.warning(
            "Cannot look at %s URL %s because it does not support lookup as web pages.",
            vcs_scheme,
            link,
        )
        return None

    # Tack index.html onto file:// URLs that point to directories
    scheme, _, path, _, _, _ = urllib.parse.urlparse(url)
    if scheme == "file" and os.path.isdir(urllib.request.url2pathname(path)):
        # add trailing slash if not present so urljoin doesn't trim
        # final segment
        if not url.endswith("/"):
            url += "/"
        # TODO: In the future, it would be nice if pip supported PEP 691
        #       style responses in the file:// URLs, however there's no
        #       standard file extension for application/vnd.pypi.simple.v1+json
        #       so we'll need to come up with something on our own.
        url = urllib.parse.urljoin(url, "index.html")
        logger.debug(" file: URL is directory, getting %s", url)

    try:
        resp = _get_simple_response(url, session=session, headers=headers)
    except _NotHTTP:
        logger.warning(
            "Skipping page %s because it looks like an archive, and cannot "
            "be checked by a HTTP HEAD request.",
            link,
        )
    except _NotAPIContent as exc:
        logger.warning(
            "Skipping page %s because the %s request got Content-Type: %s. "
            "The only supported Content-Types are application/vnd.pypi.simple.v1+json, "
            "application/vnd.pypi.simple.v1+html, and text/html",
            link,
            exc.request_desc,
            exc.content_type,
        )
    except (NetworkConnectionError, RetryError) as exc:
        _handle_get_simple_fail(link, exc)
    except SSLError as exc:
        reason = "There was a problem confirming the ssl certificate: "
        reason += str(exc)
        _handle_get_simple_fail(link, reason, meth=logger.info)
    except requests.ConnectionError as exc:
        _handle_get_simple_fail(link, f"connection error: {exc}")
    except requests.Timeout:
        _handle_get_simple_fail(link, "timed out")
    else:
        return _make_index_content(resp)
    return None


class CollectedSources(NamedTuple):
    find_links: Sequence[Optional[LinkSource]]
    index_urls: Sequence[Optional[LinkSource]]


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
    ) -> "LinkCollector":
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
        link_collector = LinkCollector(
            session=session,
            search_scope=search_scope,
        )
        return link_collector

    @property
    def find_links(self) -> List[str]:
        return self.search_scope.find_links

    def fetch_response(
        self, location: Link, headers: Optional[Dict[str, str]] = None
    ) -> Optional[IndexContent]:
        """
        Fetch an HTML page containing package links.
        """
        logger.debug("headers: %s", str(headers))
        return _get_index_content(location, session=self.session, headers=headers)

    def collect_sources(
        self,
        project_name: str,
        candidates_from_page: CandidatesFromPage,
    ) -> CollectedSources:
        # The OrderedDict calls deduplicate sources by URL.
        index_url_sources = collections.OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=False,
            )
            for loc in self.search_scope.get_index_urls_locations(project_name)
        ).values()
        find_links_sources = collections.OrderedDict(
            build_source(
                loc,
                candidates_from_page=candidates_from_page,
                page_validator=self.session.is_secure_origin,
                expand_dir=True,
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
