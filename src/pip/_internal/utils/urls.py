from __future__ import annotations

import abc
import functools
import itertools
import mimetypes
import os
import posixpath
import re
import string
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, NewType, cast

from .compat import WINDOWS

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import Self


class _PathUrlCoercions:
    @staticmethod
    def path_to_url(path: str, *, normalize: bool = True) -> ParsedUrl:
        """
        Convert a path to a file: URL.  The path will be made absolute and have
        quoted path parts.
        """
        if normalize:
            path = os.path.normpath(os.path.abspath(path))
        return ParsedUrl.parse(
            url=urllib.request.pathname2url(path),
            scheme="file",
        )

    @staticmethod
    def url_to_path(url: ParsedUrl) -> str:
        """
        Convert a file: URL to a path.
        """
        # NB: this remains a private member, because it can be easy to confuse variable
        #     types otherwise.
        return url._path_for_filesystem

    _file_scheme_pattern: ClassVar[re.Pattern[str]] = re.compile(
        r"^file:",
        flags=re.IGNORECASE,
    )

    @classmethod
    def coerce_file_uri_to_path(cls, path_or_uri: str) -> str:
        """
        If a path string begins with file:, parse it as a URL, then extract the intended
        local filesystem path.
        """
        if cls._file_scheme_pattern.match(path_or_uri):
            return cls.url_to_path(ParsedUrl.parse(path_or_uri))
        return path_or_uri


def path_to_url(path: str, *, normalize: bool = True) -> ParsedUrl:
    return _PathUrlCoercions.path_to_url(path, normalize=normalize)


def url_to_path(url: ParsedUrl) -> str:
    return _PathUrlCoercions.url_to_path(url)


def coerce_file_uri_to_path(path_or_uri: str) -> str:
    return _PathUrlCoercions.coerce_file_uri_to_path(path_or_uri)


PathComponent = NewType("PathComponent", str)


class PathSegments(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def path_segments(self) -> Iterable[PathComponent]: ...


@dataclass
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


@dataclass(frozen=True)
class ParsedUrl(PathSegments):
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
    ]

    # NB: Why cache *every single constructed value*? Well, because we want to reuse
    #     every cached_property we possibly can, particularly since some implementations
    #     like quoting/unquoting can be very expensive to compute.
    # TODO: determine if calculating hash values overcomes the perf from above (lol).
    @staticmethod
    @functools.cache
    def _cached_create(
        scheme: str,
        netloc: str,
        path: str,
        params: str,
        query: str,
        fragment: str,
    ) -> ParsedUrl:
        return ParsedUrl(
            scheme=scheme,
            netloc=netloc,
            path=path,
            params=params,
            query=query,
            fragment=fragment,
        )

    @staticmethod
    @functools.cache
    def _cached_parse(
        url: str,
        scheme: str,
    ) -> ParsedUrl:
        scheme, netloc, path, params, query, fragment = urllib.parse.urlparse(
            url, scheme=scheme
        )
        return ParsedUrl._cached_create(
            scheme=scheme,
            netloc=netloc,
            path=path,
            params=params,
            query=query,
            fragment=fragment,
        )

    @classmethod
    def parse(cls, url: str, scheme: str = "") -> ParsedUrl:
        return cls._cached_parse(url=url, scheme=scheme)

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
        return self.__class__._unsplit(self._unsplit_args())

    @classmethod
    def _unsplit_components(cls, args: tuple[str, str, str, str, str]) -> Iterator[str]:
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
        if netloc or (scheme and cls.scheme_uses_netloc(scheme)):
            # NB: we are ignoring an additional `or url[:2] == "//"` clauses in the
            # stdlib added in 2024 (e237b25a4fa5626fcd1b1848aa03f725f892e40e) because
            # I don't understand it.
            yield "//"
            yield netloc or ""
            yield url
        if query:
            yield "?"
            yield query
        if fragment:
            yield "#"
            yield fragment

    @classmethod
    def _unsplit(cls, args: tuple[str, str, str, str, str]) -> str:
        return "".join(cls._unsplit_components(args))

    def __str__(self) -> str:
        return self._unparsed_output

    @functools.cached_property
    def _hash(self) -> int:
        return hash(
            (
                self.scheme,
                self.netloc,
                self.path,
                self.params,
                self.query,
                self.fragment,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            hash(self) == hash(other)
            and self.scheme == other.scheme
            and self.netloc == other.netloc
            and self.path == other.path
            # NB: we have intentionally reordered the fields here for short-circuiting.
            and self.fragment == other.fragment
            and self.query == other.query
            and self.params == other.params
        )

    _uses_relative: ClassVar[frozenset[str]] = frozenset(urllib.parse.uses_relative)

    @staticmethod
    @functools.cache
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

    @functools.cached_property
    def _path_segments(self) -> tuple[str, ...]:
        return tuple(self.path.split("/"))

    def path_segments(self) -> tuple[PathComponent, ...]:
        return cast(tuple[PathComponent, ...], self._path_segments)

    def with_scheme(self, *, scheme: str) -> ParsedUrl:
        if scheme == self.scheme:
            return self
        return self.__class__._cached_create(
            scheme=scheme,
            netloc=self.netloc,
            path=self.path,
            params=self.params,
            query=self.query,
            fragment=self.fragment,
        )

    def with_netloc(self, *, netloc: str) -> ParsedUrl:
        if netloc == self.netloc:
            return self
        return self.__class__._cached_create(
            scheme=self.scheme,
            netloc=netloc,
            path=self.path,
            params=self.params,
            query=self.query,
            fragment=self.fragment,
        )

    def with_path_and_params(self, *, path: str, params: str) -> ParsedUrl:
        if self.path == path and self.params == params:
            return self
        return self.__class__._cached_create(
            scheme=self.scheme,
            netloc=self.netloc,
            path=path,
            params=params,
            query=self.query,
            fragment=self.fragment,
        )

    def with_query(self, *, query: str) -> ParsedUrl:
        if query == self.query:
            return self
        return self.__class__._cached_create(
            scheme=self.scheme,
            netloc=self.netloc,
            path=self.path,
            params=self.params,
            query=query,
            fragment=self.fragment,
        )

    def with_path(self, *, path: str) -> ParsedUrl:
        if path == self.path:
            return self
        return self.__class__._cached_create(
            scheme=self.scheme,
            netloc=self.netloc,
            path=path,
            params=self.params,
            query=self.query,
            fragment=self.fragment,
        )

    def with_fragment_and_query(self, *, fragment: str, query: str) -> ParsedUrl:
        if fragment == self.fragment and query == self.query:
            return self
        return self.__class__._cached_create(
            scheme=self.scheme,
            netloc=self.netloc,
            path=self.path,
            params=self.params,
            query=query,
            fragment=fragment,
        )

    def with_fragment(self, *, fragment: str) -> ParsedUrl:
        if fragment == self.fragment:
            return self
        return self.__class__._cached_create(
            scheme=self.scheme,
            netloc=self.netloc,
            path=self.path,
            params=self.params,
            query=self.query,
            fragment=fragment,
        )

    def join(self, url: ParsedUrl) -> ParsedUrl:
        if not url.scheme:
            url = url.with_scheme(scheme=self.scheme)

        if self.scheme != url.scheme or not self.__class__.scheme_uses_relative(
            url.scheme
        ):
            return url
        if self.__class__.scheme_uses_netloc(url.scheme):
            if url.netloc:
                return url
            url = url.with_netloc(netloc=self.netloc)

        if not url.path and not url.params:
            url = url.with_path_and_params(path=self.path, params=self.params)
            if not url.query:
                url = url.with_query(query=self.query)
            return url

        return url.with_path(path=UrlPath.join_right(self, url))

    @functools.cached_property
    def _path_kind(self) -> _PathKind:
        if self.netloc:
            return _UrlPath()
        # If the netloc is empty, then the URL refers to a local filesystem path.
        return _FilePath()

    @functools.cached_property
    def _quoted_path(self) -> str:
        return _PathSanitizer.sanitize_path(self.path, self._path_kind)

    @functools.cached_property
    def _as_quoted_path(self) -> ParsedUrl:
        return self.with_path(path=self._quoted_path)

    def with_quoted_path(self) -> ParsedUrl:
        """
        Make sure a path is fully quoted.
        For example, if ' ' occurs in the URL, it will be replaced with "%20",
        and without double-quoting other characters.
        """
        # For some reason, dataclasses.replace() shows up very heavily on a profile
        # output. It's not clear if this improves performance, but it's correct and
        # makes the dataclasses module disappear from the profile.
        return self._as_quoted_path

    @functools.cached_property
    def _netloc_without_auth_info(self) -> str:
        from pip._internal.utils.misc import split_auth_from_netloc

        netloc, _ = split_auth_from_netloc(self.netloc)
        return netloc

    @functools.cached_property
    def _as_removed_auth_info(self) -> ParsedUrl:
        return self.with_netloc(netloc=self._netloc_without_auth_info)

    def with_removed_auth_info(self) -> ParsedUrl:
        return self._as_removed_auth_info

    @functools.cached_property
    def _netloc_with_redacted_auth_info(self) -> str:
        from pip._internal.utils.misc import redact_netloc

        return redact_netloc(self.netloc)

    @functools.cached_property
    def _as_redacted_auth_info(self) -> ParsedUrl:
        return self.with_netloc(netloc=self._netloc_with_redacted_auth_info)

    def with_redacted_auth_info(self) -> ParsedUrl:
        return self._as_redacted_auth_info

    @functools.cached_property
    def _as_no_fragment_or_query(self) -> ParsedUrl:
        return self.with_fragment_and_query(fragment="", query="")

    def with_no_fragment_or_query(self) -> ParsedUrl:
        return self._as_no_fragment_or_query

    @functools.cached_property
    def _path_for_filesystem(self) -> str:
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
    def _base_name(self) -> str:
        return posixpath.basename(self._unquoted_path.rstrip("/"))

    @functools.cached_property
    def _filename(self) -> str:
        if name := self._base_name:
            return name
        # Make sure we don't leak auth information if the netloc
        # includes a username and password.
        return self._netloc_without_auth_info

    @functools.cached_property
    def _splitext(self) -> tuple[str, str]:
        from pip._internal.utils.misc import splitext

        return splitext(self._base_name)

    def splitext(self) -> tuple[str, str]:
        return self._splitext

    @functools.cached_property
    def _as_no_fragment(self) -> ParsedUrl:
        return self.with_fragment(fragment="")

    def without_fragment(self) -> ParsedUrl:
        return self._as_no_fragment

    @staticmethod
    # @functools.cache
    def _unquote_path_str(path: str) -> str:
        return _UrlPath._do_unquote(path)

    @functools.cached_property
    def _unquoted_path(self) -> str:
        "The .path property is hot, so calculate its value ahead of time."
        return self.__class__._unquote_path_str(self.path)

    def unquoted_path(self) -> str:
        return self._unquoted_path

    @staticmethod
    # @functools.cache
    def _split_fragment(fragment: str) -> dict[str, list[str]]:
        return urllib.parse.parse_qs(fragment, keep_blank_values=True)

    @functools.cached_property
    def _fragments(self) -> dict[str, list[str]]:
        return self.__class__._split_fragment(self.fragment)

    def _path_only(self) -> bool:
        return (
            not self.scheme
            and not self.netloc
            and not self.params
            and not self.query
            and not self.fragment
        )

    def join_path(self, rightwards_path: str) -> ParsedUrl:
        right = self.__class__.parse(rightwards_path)
        # We don't want to figure out how to define path joining in more complex cases.
        assert right._path_only(), right
        assert right.path, right
        # NB: If we don't add a trailing slash, our "joining" mechanic drops the last
        #     path component. This *would* be indicative of severe issues with our
        #     abstraction, but fortunately we only use url joining in exactly one place,
        #     when joining hrefs from a simple html index (where it's correct behavior).
        return self.with_trailing_slash_path().join(right)

    @functools.cached_property
    def _as_trailing_slash_path(self) -> ParsedUrl:
        if self.path.endswith("/"):
            return self
        return self.__class__._cached_create(
            scheme=self.scheme,
            netloc=self.netloc,
            path=f"{self.path}/",
            params=self.params,
            query=self.query,
            fragment=self.fragment,
        )

    def with_trailing_slash_path(self) -> ParsedUrl:
        return self._as_trailing_slash_path

    @functools.cached_property
    def _guess_type_result(self) -> tuple[str | None, str | None]:
        return mimetypes.guess_type(str(self), strict=False)

    def guess_type(self) -> tuple[str | None, str | None]:
        return self._guess_type_result


@dataclass(frozen=True)
class _PreQuotedUrl(ParsedUrl):
    _quoted_path: str
    _fragments: dict[str, list[str]]

    @property
    def _as_quoted_path(self) -> Self:
        return self

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ParsedUrl):
            return NotImplemented
        return ParsedUrl.__eq__(self, other)


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
    _reserved_chars_re: ClassVar[re.Pattern[str]] = re.compile("(@|%2[fF])")

    @staticmethod
    def sanitize_path(path: str, kind: _PathKind) -> str:
        if not path.rsplit("@%"):
            return kind.clean_path_part(path)
        parts = _PathSanitizer._reserved_chars_re.split(path)
        for i in range(0, len(parts)):
            if i % 2 == 0:
                parts[i] = kind.clean_path_part(parts[i])
            else:
                # Normalize %xx escapes (e.g. %2f -> %2F).
                parts[i] = parts[i].upper()
        return "".join(parts)


class _FilePath(_PathKind):
    # This has no instance state.
    def __hash__(self) -> int:
        return hash(id(self.__class__))

    def __eq__(self, rhs: Any) -> bool:
        return type(rhs) is self.__class__

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


QuoteSafe = NewType("QuoteSafe", str)
NeedsQuoting = NewType("NeedsQuoting", str)


class _Quoter(dict[bytes | str, str]):
    """See urllib.parse._Quoter."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        assert b"" not in self, (self, args, kwargs)
        self[b""] = ""

    def __setitem__(self, key: bytes | str, value: str) -> None:
        assert type(key) is bytes, (key, value, self)
        super().__setitem__(key, value)
        super().__setitem__(key.decode("utf-8"), value)

    @staticmethod
    @functools.cache
    def with_interned_ascii_codepoints() -> _Quoter:
        return _Quoter()._intern_codepoints(
            code for code in range(0, 2**7) if code not in _Quoter._always_safe()
        )

    @staticmethod
    def _char_range(start: str, fin: str) -> Iterable[int]:
        beg = ord(start)
        end = ord(fin)
        assert beg < end
        return range(beg, end + 1)

    @staticmethod
    @functools.cache
    def _always_safe() -> frozenset[int]:
        return frozenset(
            itertools.chain(
                _Quoter._char_range("A", "Z"),
                _Quoter._char_range("a", "z"),
                _Quoter._char_range("0", "9"),
                b"_.-~",
                b"/",
            )
        )

    @staticmethod
    @functools.cache
    def _always_safe_str() -> str:
        return "".join(
            sorted(
                map(chr, _Quoter._always_safe()),
                reverse=True,
            )
        )

    @staticmethod
    @functools.cache
    def _format_int(k: int) -> str:
        assert k not in _Quoter._always_safe(), (k, _Quoter._always_safe())
        return f"%{k:02X}"

    def _intern_codepoints(self, codepoints: Iterable[int]) -> Self:
        for k in frozenset(codepoints):
            # Ensure the byte string of length 1 is interned.
            # NB: _format_int() ensures the integer itself is interned.
            self[bytes([k])] = self.__class__._format_int(k)
        return self

    _split_chunks_length: ClassVar[int] = 4

    def __missing__(self, k: bytes | str) -> str:
        assert len(k) <= self._split_chunks_length, k
        if isinstance(k, str):
            k = k.encode("utf-8")
        assert isinstance(k, bytes), k

        ret = "".join(map(self.__class__._format_int, k))
        self[k] = ret
        return ret

    _unsafe_regions: ClassVar[re.Pattern[str]] = re.compile(
        # "/" is mentioned in RFC 3986, but ignored here (see urllib.parse.quote).
        r"([^a-zA-Z0-9_.~/-]+)"
    )

    def _basic_quote(self, path: str) -> QuoteSafe:
        split = self._unsafe_regions.split(path)
        # Get only the odd-indexed entries, which are the matched regions.
        for i in range(1, len(split), 2):
            prev = split[i]
            if len(prev) > self._split_chunks_length:
                # NB: list comprehension is faster than generator here.
                split[i] = "".join(
                    [
                        self[prev[j : j + self._split_chunks_length]]
                        for j in range(0, len(prev), self._split_chunks_length)
                    ]
                )
            else:
                split[i] = self[prev]
        return cast(QuoteSafe, "".join(split))

    def do_quote(self, path: str) -> QuoteSafe:
        if not (lhs_to_quote := path.rstrip(self.__class__._always_safe_str())):
            return cast(QuoteSafe, path)
        clean_rhs = cast(QuoteSafe, path[len(lhs_to_quote) :])
        return cast(QuoteSafe, self._basic_quote(lhs_to_quote) + clean_rhs)


class _Unquoter(dict[str, str]):
    _code_length: ClassVar[int] = 3

    def __setitem__(self, key: str, value: str) -> None:
        assert key.startswith("%") and len(key) % self._code_length == 0, key
        if len(key) > self._code_length:
            super().__setitem__(key, value)
            return
        codes = key[1:]
        all_upper = f"%{codes.upper()}"
        all_lower = f"%{codes.lower()}"
        first_up = f"%{codes[0].upper()}{codes[1].lower()}"
        second_up = f"%{codes[0].lower()}{codes[1].upper()}"
        for modified_key in (all_upper, all_lower, first_up, second_up):
            super().__setitem__(modified_key, value)

    def _intern_codepoints(self, codepoints: Iterable[int]) -> Self:
        for k in frozenset(codepoints):
            self[_Quoter._format_int(k)] = chr(k)
        return self

    @staticmethod
    @functools.cache
    def with_interned_ascii_codepoints() -> _Unquoter:
        return _Unquoter()._intern_codepoints(
            code for code in range(0, 2**7) if code not in _Quoter._always_safe()
        )

    @staticmethod
    def _format_code(k: str) -> str:
        assert k.startswith("%") and len(k) == _Unquoter._code_length, k
        code = int(k[1:], base=16)
        return chr(code)

    _split_chunks_length: ClassVar[int] = 4 * _code_length

    def __missing__(self, k: str) -> str:
        assert len(k) <= self._split_chunks_length, k
        assert len(k) % self._code_length == 0, k

        if len(k) > self._code_length:
            # NB: list comprehensions are faster than generators here.
            ret = "".join(
                [
                    # NB: This is a recursive call to self.
                    self[k[i : i + self._code_length]]
                    for i in range(0, len(k), self._code_length)
                ]
            )
        else:
            ret = self.__class__._format_code(k)

        self[k] = ret
        return ret

    _escaped_regions: ClassVar[re.Pattern[str]] = re.compile(r"((?:%[0-9a-fA-F]{2})+)")

    def _basic_unquote(self, path: str) -> str:
        split = self._escaped_regions.split(path)
        # Get only the odd-indexed entries, which are the matched regions.
        for i in range(1, len(split), 2):
            prev = split[i]
            if len(prev) > self._split_chunks_length:
                # NB: list comprehension is faster than generator here.
                split[i] = "".join(
                    [
                        self[prev[j : j + self._split_chunks_length]]
                        for j in range(0, len(prev), self._split_chunks_length)
                    ]
                )
            else:
                split[i] = self[prev]
        return "".join(split)

    def do_unquote(self, path: str) -> str:
        if not path.rstrip("%"):
            return path
        return self._basic_unquote(path)


class _UrlPath(_PathKind):
    _unquoter: ClassVar[_Unquoter] = _Unquoter.with_interned_ascii_codepoints()

    @classmethod
    def _do_unquote(cls, path: str) -> str:
        return cls._unquoter.do_unquote(path)

    _quoter: ClassVar[_Quoter] = _Quoter.with_interned_ascii_codepoints()

    @classmethod
    def _do_quote(cls, path: str) -> str:
        return cls._quoter.do_quote(path)

    # This has no instance state.
    def __hash__(self) -> int:
        return hash(id(self.__class__))

    def __eq__(self, rhs: Any) -> bool:
        return type(rhs) is self.__class__

    @staticmethod
    def _clean_url_path_part(path_part: str) -> str:
        """
        Clean a "part" of a URL path (i.e. after splitting on "@" characters).
        """
        # We unquote prior to quoting to make sure nothing is double quoted.
        return _UrlPath._do_quote(_UrlPath._do_unquote(path_part))

    def clean_path_part(self, path_part: str) -> str:
        return type(self)._clean_url_path_part(path_part)
