from __future__ import annotations

import functools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from pip._vendor.packaging._parser import parse_requirement as _parse_requirement
from pip._vendor.packaging._tokenizer import ParserSyntaxError
from pip._vendor.packaging.requirements import InvalidRequirement
from pip._vendor.packaging.utils import canonicalize_name

# FIXME: this means url parsing would need to be ported to packaging as well, but that's
#        not actually a bad idea since:
# (1) it would lead to correct url comparisons for Requirement
# (2) the approach used for url parsing is quite similar to that used for other parsing
from pip._internal.utils.urls import ParsedUrl

from .markers import Marker
from .specifiers import SpecifierSet

if TYPE_CHECKING:
    from collections.abc import Iterator

    from typing_extensions import Self


@dataclass(frozen=True)
class Requirement:
    name: str
    url: ParsedUrl | None
    extras: frozenset[str]
    specifier: SpecifierSet
    marker: Marker | None

    __slots__ = ["name", "url", "extras", "specifier", "marker", "__dict__"]

    @staticmethod
    @functools.cache
    def _cached_parse(requirement_string: str) -> Requirement:
        try:
            parsed = _parse_requirement(requirement_string)
        except ParserSyntaxError as e:
            raise InvalidRequirement(str(e)) from e

        url: ParsedUrl | None = None
        if u := parsed.url:
            url = ParsedUrl.parse(u)
        marker: Marker | None = None
        if (mk := parsed.marker) is not None:
            marker = Marker.normalize_markers(mk)
        return Requirement(
            name=parsed.name,
            url=url,
            extras=frozenset(parsed.extras or ()),
            specifier=SpecifierSet.parse(parsed.specifier),
            marker=marker,
        )

    @classmethod
    def parse(cls, requirement_string: str) -> Requirement:
        """
        Parse a given requirement string into its parts, such as name, specifier,
        URL, and extras.

        :raises: InvalidRequirement: on a badly-formed requirement string.
        """
        return cls._cached_parse(requirement_string)

    def with_redacted_auth_info(self) -> Self:
        if self.url is None:
            return self
        redacted = self.url.with_redacted_auth_info()
        if redacted is self.url:
            return self
        return self.__class__(
            name=self.name,
            url=redacted,
            extras=self.extras,
            specifier=self.specifier,
            marker=self.marker,
        )

    def _iter_unnamed_parts(self) -> Iterator[str]:
        if self.extras:
            formatted_extras = ",".join(sorted(self.extras))
            yield f"[{formatted_extras}]"

        if self.specifier:
            yield str(self.specifier)

        if self.url:
            yield f"@ {self.url}"
            if self.marker:
                yield " "

        if self.marker:
            yield f"; {self.marker}"

    @functools.cached_property
    def _unnamed_parts(self) -> tuple[str, ...]:
        return tuple(self._iter_unnamed_parts())

    @functools.cached_property
    def _str(self) -> str:
        return self.name + "".join(self._unnamed_parts)

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.parse('{self}')"

    @functools.cached_property
    def _canonical_name(self) -> str:
        return canonicalize_name(self.name)

    @functools.cached_property
    def _hash(self) -> int:
        return hash(
            (
                self.__class__.__name__,
                self._canonical_name,
                self._unnamed_parts,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if type(other) is not self.__class__:
            return NotImplemented

        return (
            hash(self) == hash(other)
            and self._canonical_name == other._canonical_name
            and self.specifier == other.specifier
            and self.extras == other.extras
            and self.url == other.url
            and self.marker == other.marker
        )
