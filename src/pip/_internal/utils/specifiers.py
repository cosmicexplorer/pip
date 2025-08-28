from __future__ import annotations

import abc
import functools
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from pip._vendor.packaging.specifiers import InvalidSpecifier

from .containment import (
    ArbitraryEqual,
    Compatible,
    ContainsPredicate,
    EqualBasic,
    EqualPrefix,
    GreaterThan,
    GreaterThanEqual,
    LessThan,
    LessThanEqual,
    NotEqualBasic,
)
from .version import InvalidVersion, ParsedVersion

if TYPE_CHECKING:
    from typing import Any, ClassVar

    from typing_extensions import Self


class BaseSpecifier(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def __contains__(self, item: ParsedVersion) -> bool:
        """
        Return whether or not the item is contained in this specifier.
        """
        ...

    @abc.abstractmethod
    def contains(self, item: ParsedVersion, prereleases: bool | None = None) -> bool:
        """
        Determines if the given item is contained within this specifier.
        """
        ...

    @abc.abstractmethod
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
class Specifier(BaseSpecifier):
    operator: Operator
    _version: str
    _trailing_dot_star: bool
    _prereleases: bool | None

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

    @functools.cached_property
    def _predicate(self) -> ContainsPredicate:
        if self.operator == Operator.ARBITRARY:
            return ArbitraryEqual(self._version)
        if self.operator == Operator.LESS_THAN:
            assert self.parsed_version is not None
            return LessThan(self.parsed_version)
        if self.operator == Operator.GREATER_THAN:
            assert self.parsed_version is not None
            return GreaterThan(self.parsed_version)
        if self.operator == Operator.LESS_THAN_EQUAL:
            assert self.parsed_version is not None
            return LessThanEqual(self.parsed_version)
        if self.operator == Operator.GREATER_THAN_EQUAL:
            assert self.parsed_version is not None
            return GreaterThanEqual(self.parsed_version)
        if self.operator == Operator.NOT_EQUAL:
            assert self.parsed_version is not None
            return NotEqualBasic(self.parsed_version)
        if self.operator == Operator.EQUAL:
            if self._trailing_dot_star:
                return EqualPrefix(self._version)
            assert self.parsed_version is not None
            return EqualBasic(self.parsed_version)
        assert self.operator == Operator.COMPATIBLE, self.operator
        return Compatible(self._version)

    def __contains__(self, item: ParsedVersion) -> bool:
        return self.contains(item)

    def contains(self, item: ParsedVersion, prereleases: bool | None = None) -> bool:
        if prereleases is None:
            prereleases = self.prereleases

        if item.is_prerelease and not prereleases:
            return False

        return self._predicate.evaluate(item)

    def filter(
        self,
        iterable: Iterable[ParsedVersion],
        prereleases: bool | None = None,
    ) -> Iterator[ParsedVersion]:
        yielded = False
        found_prereleases = []

        kw = {"prereleases": prereleases if prereleases is not None else True}

        # Attempt to iterate over all the values in the iterable and if any of
        # them match, yield them.
        for version in iterable:
            if self.contains(version, **kw):
                # If our version is a prerelease, and we were not set to allow
                # prereleases, then we'll store it for later in case nothing
                # else matches this specifier.
                if version.is_prerelease and not (prereleases or self.prereleases):
                    found_prereleases.append(version)
                # Either this is not a prerelease, or we should have been
                # accepting prereleases from the beginning.
                else:
                    yielded = True
                    yield version

        # Now that we've iterated over everything, determine if we've yielded
        # any values, and if we have not and we have any prereleases stored up
        # then we will go ahead and yield the prereleases.
        if not yielded and found_prereleases:
            for version in found_prereleases:
                yield version


@dataclass(frozen=True)
class SpecifierSet(BaseSpecifier):
    _specs: frozenset[Specifier]
    _prereleases: bool | None

    @classmethod
    def parse(
        cls,
        specifiers: str | Iterable[Specifier] = "",
        prereleases: bool | None = None,
    ) -> Self:
        """
        :param specifiers:
            The string representation of a specifier or a comma-separated list of
            specifiers which will be parsed and normalized before use.
            May also be an iterable of ``Specifier`` instances, which will be used
            as is.
        :param prereleases:
            This tells the SpecifierSet if it should accept prerelease versions if
            applicable or not. The default of ``None`` will autodetect it from the
            given specifiers.

        :raises InvalidSpecifier:
            If the given ``specifiers`` are not parseable than this exception will be
            raised.
        """
        if isinstance(specifiers, str):
            # Split on `,` to break each individual specifier into its own item, and
            # strip each item to remove leading/trailing whitespace.
            split_specifiers = [s.strip() for s in specifiers.split(",") if s.strip()]

            # Make each individual specifier a Specifier and save in a frozen set
            # for later.
            specs = frozenset(map(Specifier.parse, split_specifiers))
        else:
            # Save the supplied specifiers in a frozen set.
            specs = frozenset(specifiers)

        return cls(
            _specs=specs,
            _prereleases=prereleases,
        )

    @functools.cached_property
    def prereleases(self) -> bool | None:
        # If we have been given an explicit prerelease modifier, then we'll
        # pass that through here.
        if self._prereleases is not None:
            return self._prereleases

        # If we don't have any specifiers, and we don't have a forced value,
        # then we'll just return None since we don't know if this should have
        # pre-releases or not.
        if not self._specs:
            return None

        # Otherwise we'll see if any of the given specifiers accept
        # prereleases, if any of them do we'll return True, otherwise False.
        return any(s.prereleases for s in self._specs)

    @functools.cached_property
    def _str(self) -> str:
        return ",".join(sorted(map(str, self._specs)))

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        pre = (
            f", prereleases={self._prereleases!r}"
            if self._prereleases is not None
            else ""
        )
        return f"SpecifierSet.parse('{self}'{pre})"

    @functools.cached_property
    def _hash(self) -> int:
        return hash(self._specs)

    def __hash__(self) -> int:
        return self._hash

    def __and__(self, other: Self) -> Self:
        if not isinstance(other, self.__class__):
            return NotImplemented

        specs = self._specs | other._specs

        prereleases: bool | None
        if self._prereleases is None and other._prereleases is not None:
            prereleases = other._prereleases
        elif self._prereleases is not None and other._prereleases is None:
            prereleases = self._prereleases
        elif self._prereleases == other._prereleases:
            prereleases = self._prereleases
        else:
            raise ValueError(
                "Cannot combine SpecifierSets with True and False prerelease overrides."
            )

        return self.__class__(
            _specs=specs,
            _prereleases=prereleases,
        )

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._specs == other._specs

    def __len__(self) -> int:
        return len(self._specs)

    def __iter__(self) -> Iterator[Specifier]:
        return iter(self._specs)

    def __contains__(self, item: ParsedVersion) -> bool:
        return self.contains(item)

    def contains(self, item: ParsedVersion, prereleases: bool | None = None) -> bool:
        if prereleases is None:
            prereleases = self.prereleases
        if not prereleases and item.is_prerelease:
            return False
        return all(s.contains(item, prereleases=prereleases) for s in self._specs)

    def filter(
        self,
        iterable: Iterable[ParsedVersion],
        prereleases: bool | None = None,
    ) -> Iterator[ParsedVersion]:
        if prereleases is None:
            prereleases = self.prereleases
        assert prereleases is not None

        # If we have any specifiers, then we want to wrap our iterable in the
        # filter method for each one, this will act as a logical AND amongst
        # each specifier.
        if self._specs:
            for spec in self._specs:
                iterable = spec.filter(iterable, prereleases=prereleases)
            return iter(iterable)

        # If we do not have any specifiers, then we need to have a rough filter
        # which will filter out any pre-releases, unless there are no final
        # releases.
        filtered: list[ParsedVersion] = []
        found_prereleases: list[ParsedVersion] = []

        for item in iterable:
            # Store any item which is a pre-release for later unless we've
            # already found a final version or we are accepting prereleases
            if item.is_prerelease and not prereleases:
                if not filtered:
                    found_prereleases.append(item)
            else:
                filtered.append(item)

        # If we've found no items except for pre-releases, then we'll go
        # ahead and use the pre-releases
        if not filtered and found_prereleases and prereleases is None:
            return iter(found_prereleases)

        return iter(filtered)
