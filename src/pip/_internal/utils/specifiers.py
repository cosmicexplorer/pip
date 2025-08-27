from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Protocol

from pip._vendor.packaging.specifiers import InvalidSpecifier

from .version import InvalidVersion, ParsedVersion

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self


class BaseSpecifier(Protocol):
    def __contains__(self, item: ParsedVersion) -> bool:
        """
        Return whether or not the item is contained in this specifier.
        """
        ...

    def contains(self, item: ParsedVersion, prereleases: bool | None = None) -> bool:
        """
        Determines if the given item is contained within this specifier.
        """
        ...

    def filter(
        self, iterable: Iterable[ParsedVersion], prereleases: bool | None = None
    ) -> Iterator[ParsedVersion]:
        """
        Takes an iterable of items and filters them so that only items which
        are contained within this specifier are allowed in it.
        """
        ...


class Operator(Enum):
    COMPATIBLE = "~="
    EQUAL = "=="
    NOT_EQUAL = "!="
    LESS_THAN_EQUAL = "<="
    GREATER_THAN_EQUAL = ">="
    LESS_THAN = "<"
    GREATER_THAN = ">"
    ARBITRARY = "==="

    @classmethod
    def match_pattern(cls) -> str:
        return "|".join(re.escape(o.value) for o in cls)

    def is_inclusive(self) -> bool:
        return self != type(self).NOT_EQUAL


@dataclass(frozen=True)
class Specifier:
    operator: Operator
    _version: str
    _trailing_dot_star: bool
    _prereleases: bool | None

    __slots__ = [
        "operator",
        "_version",
        "_trailing_dot_star",
        "_prereleases",
        "__dict__",
    ]

    _operator_regex_str: ClassVar[
        str
    ] = f"""
        (?P<operator>(?:{Operator.match_pattern()}))
        """
    _version_regex_str: ClassVar[
        str
    ] = r"""
        (?P<version>
            (?:
                # The identity operators allow for an escape hatch that will
                # do an exact string match of the version you wish to install.
                # This will not be parsed by PEP 440 and we cannot determine
                # any semantic meaning from it. This operator is discouraged
                # but included entirely as an escape hatch.
                (?<====)  # Only match for the identity operator
                \s*
                [^\s;)]*  # The arbitrary version can be just about anything,
                          # we match everything except for whitespace, a
                          # semi-colon for marker support, and a closing paren
                          # since versions can be enclosed in them.
            )
            |
            (?:
                # The (non)equality operators allow for wild card and local
                # versions to be specified so we have to define these two
                # operators separately to enable that.
                (?<===|!=)            # Only match for equals and not equals

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release

                # You cannot use a wild card and a pre-release, post-release, a dev or
                # local version together so group them with a | and make them optional.
                (?:
                    \.\*  # Wild card syntax of .*
                    |
                    (?:                                  # pre release
                        [-_\.]?
                        (?:alpha|beta|preview|pre|a|b|c|rc)
                        [-_\.]?
                        [0-9]*
                    )?
                    (?:                                  # post release
                        (?:-[0-9]+)|(?:[-_\.]?(?:post|rev|r)[-_\.]?[0-9]*)
                    )?
                    (?:[-_\.]?dev[-_\.]?[0-9]*)?         # dev release
                    (?:\+[a-z0-9]+(?:[-_\.][a-z0-9]+)*)? # local
                )?
            )
            |
            (?:
                # The compatible operator requires at least two digits in the
                # release segment.
                (?<=~=)               # Only match for the compatible operator

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)+   # release  (We have a + instead of a *)
                (?:                   # pre release
                    [-_\.]?
                    (?:alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(?:post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
            |
            (?:
                # All other operators only allow a sub set of what the
                # (non)equality operators do. Specifically they do not allow
                # local versions to be specified nor do they allow the prefix
                # matching wild cards.
                (?<!==|!=|~=)         # We have special cases for these
                                      # operators so we want to make sure they
                                      # don't match here.

                \s*
                v?
                (?:[0-9]+!)?          # epoch
                [0-9]+(?:\.[0-9]+)*   # release
                (?:                   # pre release
                    [-_\.]?
                    (?:alpha|beta|preview|pre|a|b|c|rc)
                    [-_\.]?
                    [0-9]*
                )?
                (?:                                   # post release
                    (?:-[0-9]+)|(?:[-_\.]?(?:post|rev|r)[-_\.]?[0-9]*)
                )?
                (?:[-_\.]?dev[-_\.]?[0-9]*)?          # dev release
            )
        )
        """

    _regex: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*" + _operator_regex_str + _version_regex_str + r"\s*$",
        flags=re.VERBOSE | re.IGNORECASE,
    )

    @classmethod
    def parse(cls, spec: str = "", prereleases: bool | None = None) -> Self:
        """
        :param spec:
            The string representation of a specifier which will be parsed and
            normalized before use.
        :param prereleases:
            This tells the specifier if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.
        :raises InvalidSpecifier:
            If the given specifier is invalid (i.e. bad syntax).
        """
        m = cls._regex.match(spec)
        if not m:
            raise InvalidSpecifier(f"Invalid specifier: {spec!r}")
        g = m.groupdict()

        operator = Operator(g["operator"])

        version = g["version"].strip()
        if operator == Operator.EQUAL and version.endswith(".*"):
            trailing_dot_star = True
            version = version[:-2]
        else:
            trailing_dot_star = False
        return cls(
            operator=operator,
            _version=version,
            _trailing_dot_star=trailing_dot_star,
            _prereleases=prereleases,
        )

    @functools.cached_property
    def parsed_version(self) -> ParsedVersion | None:
        try:
            return ParsedVersion.parse(self._version)
        except InvalidVersion:
            return None

    @functools.cached_property
    def prereleases(self) -> bool:
        if self._prereleases is not None:
            return self._prereleases

        if not self.operator.is_inclusive():
            return False

        if v := self.parsed_version:
            return v.is_prerelease
        return False

    @functools.cached_property
    def _str(self) -> str:
        version = self._version
        if self._trailing_dot_star:
            version += ".*"
        return f"{self.operator.value}{version}"

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        pre = (
            f", prereleases={self._prereleases!r}"
            if self._prereleases is not None
            else ""
        )
        return f"{self.__class__.__name__}.parse('{self}'{pre})"

    @functools.cached_property
    def _canonical_spec(self) -> tuple[Operator, str]:
        if v := self.parsed_version:
            if self.operator != Operator.COMPATIBLE:
                v = v.with_trimmed_release()
            return self.operator, str(v)
        # Legacy versions cannot be normalized.
        return self.operator, self._version

    def __hash__(self) -> int:
        return hash(self._canonical_spec)

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._canonical_spec == other._canonical_spec

    def __contains__(self, item: ParsedVersion) -> bool:
        return self.contains(item)

    def contains(self, item: ParsedVersion, prereleases: bool | None = None) -> bool:
        if prereleases is None:
            prereleases = self.prereleases

        if item.is_prerelease and not prereleases:
            return False

        raise NotImplementedError
