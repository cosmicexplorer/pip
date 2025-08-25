from __future__ import annotations

import abc
import dataclasses
import functools
import itertools
import os
import re
import string
import urllib.parse
import urllib.request
from collections.abc import Iterable, Iterator
from typing import TYPE_CHECKING, Any, ClassVar, NewType, Protocol, cast

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

    @functools.cache
    @staticmethod
    def _uses_netloc() -> frozenset[str]:
        # See docs/html/topics/vcs-support.md.
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

    @functools.cached_property
    def _path_kind(self) -> _PathKind:
        if self.netloc:
            return _UrlPath()
        # If the netloc is empty, then the URL refers to a local filesystem path.
        return _FilePath()

    def ensure_quoted_path(self) -> Self:
        """
        Make sure a path is fully quoted.
        For example, if ' ' occurs in the URL, it will be replaced with "%20",
        and without double-quoting other characters.
        """
        # For some reason, dataclasses.replace() shows up very heavily on a profile
        # output. It's not clear if this improves performance, but it's correct and
        # makes the dataclasses module disappear from the profile.
        return self.__class__(
            self.scheme,
            self.netloc,
            _PathSanitizer.sanitize_path(self.path, self._path_kind),
            self.params,
            self.query,
            self.fragment,
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


class _PathKind(abc.ABC):
    # So we can consider caching.
    @abc.abstractmethod
    def __hash__(self) -> int: ...
    @abc.abstractmethod
    def __eq__(self, rhs: Any) -> bool: ...

    @abc.abstractmethod
    def clean_path_part(self, path_part: str) -> str:
        """Clean this kind of URL path component string (e.g. local or remote)."""
        ...


class _PathSanitizer:
    # TODO: check if IGNORECASE is as performant as [fF]
    # percent-encoded:                   /
    _reserved_chars_re: ClassVar[re.Pattern[str]] = re.compile("(@|%2F)", re.IGNORECASE)

    @staticmethod
    def _pip_path_split(path: str) -> tuple[str, ...]:
        """
        Note that splitting on a regex with groups is very different from splitting on
        a literal string, as each matched group with index > 0 will be replicated in the
        split output:

        >>> re.compile(',').split('a,b')
        ['a', 'b']
        >>> re.compile('(,)').split('a,b')
        ['a', ',', 'b']

        Compare to literal, performed in the reverse order:
        >>> 'a,b'.split(',')
        ['a', 'b']
        """
        return tuple(_PathSanitizer._reserved_chars_re.split(path))

    @staticmethod
    def _pip_path_clean(path: str, kind: _PathKind) -> Iterator[str]:
        """
        Split on the reserved characters prior to cleaning so that
        revision strings in VCS URLs are properly preserved.
        """
        parts = _PathSanitizer._pip_path_split(path)
        for to_clean, reserved in pairwise(parts + ("",)):
            yield kind.clean_path_part(to_clean)
            # Normalize %xx escapes (e.g. %2f -> %2F)
            if reserved:
                yield reserved.upper()

    @staticmethod
    def sanitize_path(path: str, kind: _PathKind) -> str:
        return "".join(_PathSanitizer._pip_path_clean(path, kind))


class _FilePath(_PathKind):
    # This has no instance state.
    def __hash__(self) -> int:
        return hash(id(type(self)))

    def __eq__(self, rhs: Any) -> bool:
        return type(self) is type(rhs)

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

    def clean_path_part(self, path_part: str) -> str:
        return type(self)._clean_file_url_path(path_part)


class _UrlPath(_PathKind):
    # This has no instance state.
    def __hash__(self) -> int:
        return hash(id(type(self)))

    def __eq__(self, rhs: Any) -> bool:
        return type(self) is type(rhs)

    @staticmethod
    def _clean_url_path_part(path_part: str) -> str:
        """
        Clean a "part" of a URL path (i.e. after splitting on "@" characters).
        """
        # We unquote prior to quoting to make sure nothing is double quoted.
        return urllib.parse.quote(urllib.parse.unquote(path_part))

    def clean_path_part(self, path_part: str) -> str:
        return type(self)._clean_url_path_part(path_part)
