from __future__ import annotations

import dataclasses
import functools
import itertools
import os
import re
import string
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, ClassVar, NewType, Protocol, cast

from pip._internal.utils.misc import pairwise, redact_netloc, split_auth_from_netloc

from .compat import WINDOWS

if TYPE_CHECKING:
    from typing_extensions import Self


@functools.cache
def path_to_url(path: str) -> ParsedUrl:
    """
    Convert a path to a file: URL.  The path will be made absolute and have
    quoted path parts.
    """
    path = os.path.normpath(os.path.abspath(path))
    return ParsedUrl.parse(
        url=urllib.request.pathname2url(path),
        scheme="file",
    )


@functools.cache
def url_to_path(url: ParsedUrl) -> str:
    """
    Convert a file: URL to a path.
    """
    return url.as_filesystem_path


PathComponent = NewType("PathComponent", str)


class PathSegments(Protocol):
    def path_segments(self) -> Iterable[PathComponent]: ...


@dataclasses.dataclass
class UrlPath:
    segments: list[PathComponent]

    def __post_init__(self) -> None:
        assert self.segments

    def is_absolute(self) -> bool:
        return (len(self.segments) > 1) and self.segments[0] == ""

    def pop_filename(self) -> None:
        """
        the last item is not a directory, so will not be taken into account
        in resolving the relative path
        """
        if self.segments[-1] != "":
            self.segments.pop()

    def remove_redundant_inner_segments(self) -> None:
        """
        filter out elements that would cause redundant slashes on re-joining
        the resolved_path
        """
        self.segments[1:-1] = filter(None, self.segments[1:-1])

    def prepend_segments(self, lhs: Self) -> None:
        self.segments = lhs.segments + self.segments

    @classmethod
    def join_right(cls, lhs: PathSegments, rhs: PathSegments) -> str:
        """Perform the very complex join logic of urllib.parse.urljoin()."""
        joined = cls(list(rhs.path_segments()))
        if not joined.is_absolute():
            base = cls(list(lhs.path_segments()))
            base.pop_filename()
            joined.prepend_segments(base)
            joined.remove_redundant_inner_segments()

        resolved_path: list[str] = []

        for seg in joined.segments:
            if seg == "..":
                try:
                    resolved_path.pop()
                except IndexError:
                    # ignore any .. segments that would otherwise cause an IndexError
                    # when popped from resolved_path if resolving for rfc3986
                    pass
            elif seg == ".":
                continue
            else:
                resolved_path.append(seg)

        if joined.segments[-1] in (".", ".."):
            # do some post-processing here. if the last segment was a relative dir,
            # then we need to append the trailing '/'
            resolved_path.append("")

        return "/".join(resolved_path) or "/"


@dataclasses.dataclass(frozen=True)
class ParsedUrl:
    scheme: str
    netloc: str
    path: str
    params: str
    query: str
    fragment: str

    __slots__ = [
        "scheme",
        "netloc",
        "path",
        "params",
        "query",
        "fragment",
        "__dict__",
    ]

    @classmethod
    def parse(cls, url: str, scheme: str = "") -> Self:
        scheme, netloc, path, params, query, fragment = urllib.parse.urlparse(
            url, scheme=scheme
        )

        return cls(
            scheme=scheme,
            netloc=netloc,
            path=path,
            params=params,
            query=query,
            fragment=fragment,
        )

    def _unparse_args(self) -> tuple[str, str, str, str, str, str]:
        return (
            self.scheme,
            self.netloc,
            self.path,
            self.params,
            self.query,
            self.fragment,
        )

    def _unsplit_args(self) -> tuple[str, str, str, str, str]:
        scheme, netloc, url, params, query, fragment = self._unparse_args()
        if params:
            url = f"{url};{params}"
        return scheme, netloc, url, query, fragment

    @functools.cached_property
    def _unparsed_output(self) -> str:
        return ParsedUrl._unsplit(self._unsplit_args())

    @staticmethod
    def _unsplit_components(args: tuple[str, str, str, str, str]) -> Iterator[str]:
        """
        This method is adapted from urllib.parse.urlsplit() in the stdlib in order to
        modify the definition of uses_netloc to include pip's myriad vcs schemes.

        That method is completely uncommented, and it's difficult to infer what it does.
        """
        scheme, netloc, url, query, fragment = args
        if scheme:
            yield scheme
            yield ":"
        # NB: this usage of uses_netloc is very importantly different than in the join()
        #     method! In particular, join() does *not* check for truthiness of `scheme`!
        #     The result of this is that empty schemes do not get another prefix '/'!
        if netloc or (scheme and ParsedUrl.scheme_uses_netloc(scheme)):
            # NB: we are ignoring an additional `or url[:2] == "//"` clauses in the
            # stdlib added in 2024 (e237b25a4fa5626fcd1b1848aa03f725f892e40e) because
            # I don't understand it.
            yield "//"
            yield netloc or ""
            if url and url[:1] != "/":
                yield "/"
            yield url
        if query:
            yield "?"
            yield query
        if fragment:
            yield "#"
            yield fragment

    @staticmethod
    def _unsplit(args: tuple[str, str, str, str, str]) -> str:
        return "".join(ParsedUrl._unsplit_components(args))

    def __str__(self) -> str:
        return self._unparsed_output

    _uses_relative: ClassVar[frozenset[str]] = frozenset(urllib.parse.uses_relative)
    # See docs/html/topics/vcs-support.md.
    _git_schemes: ClassVar[tuple[str, ...]] = (
        "git",
        "git+file",
        "git+https",
        "git+ssh",
        "git+http",
        "git+git",
    )
    _hg_schemes: ClassVar[tuple[str, ...]] = (
        "hg+file",
        "hg+http",
        "hg+https",
        "hg+ssh",
        "hg+static-http",
    )
    _svn_schemes: ClassVar[tuple[str, ...]] = (
        "svn",
        "svn+svn",
        "svn+http",
        "svn+https",
        "svn+ssh",
    )
    _bzr_schemes: ClassVar[tuple[str, ...]] = (
        "bzr+http",
        "bzr+https",
        "bzr+ssh",
        "bzr+sftp",
        "bzr+ftp",
        "bzr+lp",
    )
    _vcs_schemes: ClassVar[tuple[str, ...]] = (
        _git_schemes + _hg_schemes + _svn_schemes + _bzr_schemes
    )

    @functools.cache
    @staticmethod
    def _uses_netloc() -> frozenset[str]:
        from pip._internal.vcs.versioncontrol import vcs

        return frozenset(
            itertools.chain(
                urllib.parse.uses_netloc,
                vcs.all_schemes,
            )
        )

    @classmethod
    def scheme_uses_relative(cls, scheme: str) -> bool:
        return scheme in cls._uses_relative

    @classmethod
    def scheme_uses_netloc(cls, scheme: str) -> bool:
        return scheme in cls._uses_netloc()

    def path_segments(self) -> list[PathComponent]:
        return cast(list[PathComponent], self.path.split("/"))

    def join(self, url: Self) -> Self:
        if not url.scheme:
            url = dataclasses.replace(url, scheme=self.scheme)

        if self.scheme != url.scheme or not type(self).scheme_uses_relative(url.scheme):
            return url
        if type(self).scheme_uses_netloc(url.scheme):
            if url.netloc:
                return url
            url = dataclasses.replace(url, netloc=self.netloc)

        if not url.path and not url.params:
            url = dataclasses.replace(url, path=self.path, params=self.params)
            if not url.query:
                url = dataclasses.replace(url, query=self.query)
            return url

        return dataclasses.replace(
            url,
            path=UrlPath.join_right(self, url),
        )

    # percent-encoded:                   /
    _reserved_chars_re: ClassVar[re.Pattern[str]] = re.compile("(@|%2F)", re.IGNORECASE)

    def _pip_path_parts(self) -> Iterator[tuple[str, str]]:
        """
        Split on the reserved characters prior to cleaning so that
        revision strings in VCS URLs are properly preserved.
        """
        parts = type(self)._reserved_chars_re.split(self.path)
        for to_clean, reserved in pairwise(itertools.chain(parts, ("",))):
            # Normalize %xx escapes (e.g. %2f -> %2F)
            yield to_clean, reserved.upper()

    @staticmethod
    def _clean_file_url_path(path_part: str) -> str:
        """
        Clean the first part of a URL path that corresponds to a local
        filesystem path (i.e. the first part after splitting on "@" characters).
        """
        # We unquote prior to quoting to make sure nothing is double quoted.
        # Also, on Windows the path part might contain a drive letter which
        # should not be quoted. On Linux where drive letters do not
        # exist, the colon should be quoted. We rely on urllib.request
        # to do the right thing here.
        ret = urllib.request.pathname2url(urllib.request.url2pathname(path_part))
        if ret.startswith("///"):
            # Remove any URL authority section, leaving only the URL path.
            ret = ret.removeprefix("//")
        return ret

    @staticmethod
    def _clean_url_path_part(path_part: str) -> str:
        """
        Clean a "part" of a URL path (i.e. after splitting on "@" characters).
        """
        # We unquote prior to quoting to make sure nothing is double quoted.
        return urllib.parse.quote(urllib.parse.unquote(path_part))

    def _clean_path_parts(self) -> Iterator[str]:
        if self.netloc:
            for to_clean, reserved in self._pip_path_parts():
                yield ParsedUrl._clean_url_path_part(to_clean)
                yield reserved
        else:
            # If the netloc is empty, then the URL refers to a local filesystem path.
            for to_clean, reserved in self._pip_path_parts():
                yield ParsedUrl._clean_file_url_path(to_clean)
                yield reserved

    def ensure_quoted_path(self) -> Self:
        """
        Make sure a path is fully quoted.
        For example, if ' ' occurs in the URL, it will be replaced with "%20",
        and without double-quoting other characters.
        """
        return dataclasses.replace(
            self,
            path="".join(self._clean_path_parts()),
        )

    @functools.cached_property
    def with_quoted_path(self) -> Self:
        return self.ensure_quoted_path()

    @functools.cached_property
    def unquoted_path(self) -> str:
        "The .path property is hot, so calculate its value ahead of time."
        return urllib.parse.unquote(self.path)

    def remove_auth_info(self) -> Self:
        netloc, _ = split_auth_from_netloc(self.netloc)
        return dataclasses.replace(
            self,
            netloc=netloc,
        )

    @functools.cached_property
    def with_removed_auth_info(self) -> Self:
        return self.remove_auth_info()

    def redact_auth_info(self) -> Self:
        return dataclasses.replace(
            self,
            netloc=redact_netloc(self.netloc),
        )

    def no_fragment_or_query(self) -> Self:
        return dataclasses.replace(
            self,
            query="",
            fragment="",
        )

    @functools.cached_property
    def with_no_fragment_or_query(self) -> Self:
        return self.no_fragment_or_query()

    @functools.cached_property
    def with_redacted_auth_info(self) -> Self:
        return self.redact_auth_info()

    def to_path(self) -> str:
        """
        Convert a file: URL to a path.
        """
        assert (
            self.scheme == "file"
        ), f"You can only turn file: urls into filenames (not {self!r})"
        netloc = self.netloc
        if not netloc or netloc == "localhost":
            # According to RFC 8089, same as empty authority.
            netloc = ""
        elif WINDOWS:
            # If we have a UNC path, prepend UNC share notation.
            netloc = "\\\\" + netloc
        else:
            raise ValueError(
                f"non-local file URIs are not supported on this platform: {self!r}"
            )

        path = urllib.request.url2pathname(netloc + self.path)

        # On Windows, urlsplit parses the path as something like "/C:/Users/foo".
        # This creates issues for path-related functions like io.open(), so we try
        # to detect and strip the leading slash.
        if (
            WINDOWS
            and not netloc  # Not UNC.
            and len(path) >= 3
            and path[0] == "/"  # Leading slash to strip.
            and path[1] in string.ascii_letters  # Drive letter.
            and path[2:4]
            in (":", ":/")  # Colon + end of string, or colon + absolute path.
        ):
            path = path[1:]

        return path

    @functools.cached_property
    def as_filesystem_path(self) -> str:
        return self.to_path()

    def no_fragment(self) -> Self:
        return dataclasses.replace(self, fragment="")

    @functools.cached_property
    def without_fragment(self) -> Self:
        return self.no_fragment()
