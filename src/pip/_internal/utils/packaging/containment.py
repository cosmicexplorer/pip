from __future__ import annotations

import abc
import functools
import itertools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from pip._vendor.packaging.utils import canonicalize_version

from .version import ParsedVersion

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import Any, ClassVar


class ContainsPredicate(abc.ABC):
    @abc.abstractmethod
    def evaluate(self, prospective: ParsedVersion) -> bool: ...

    @functools.cached_property
    def _hash(self) -> int:
        return hash(id(self))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        return other is self


@dataclass(frozen=True)
class _BasicPredicate(ContainsPredicate):
    spec: ParsedVersion


@dataclass(frozen=True)
class EqualBasic(_BasicPredicate):
    def evaluate(self, prospective: ParsedVersion) -> bool:
        # If the specifier does not have a local segment, then we want to
        # act as if the prospective version also does not have a local
        # segment.
        if not self.spec.local:
            return prospective.public == self.spec
        return prospective == self.spec


@dataclass(frozen=True)
class NotEqualBasic(_BasicPredicate):
    @functools.cached_property
    def eq_pred(self) -> EqualBasic:
        return EqualBasic(self.spec)

    def evaluate(self, prospective: ParsedVersion) -> bool:
        return not self.eq_pred.evaluate(prospective)


@dataclass(frozen=True)
class LessThanEqual(_BasicPredicate):
    def evaluate(self, prospective: ParsedVersion) -> bool:
        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return prospective.public <= self.spec


@dataclass(frozen=True)
class GreaterThanEqual(_BasicPredicate):
    def evaluate(self, prospective: ParsedVersion) -> bool:
        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return prospective.public >= self.spec


@dataclass(frozen=True)
class LessThan(_BasicPredicate):
    def evaluate(self, prospective: ParsedVersion) -> bool:
        if not prospective < self.spec:
            return False
        # This special case is here so that, unless the specifier itself
        # includes is a pre-release version, that we do not accept pre-release
        # versions for the version mentioned in the specifier (e.g. <3.1 should
        # not match 3.1.dev0, but should match 3.0.dev0).
        if not self.spec.is_prerelease and prospective.is_prerelease:
            if prospective.base_version == self.spec.base_version:
                return False
        # If we've gotten to here, it means that prospective version is both
        # less than the spec version *and* it's not a pre-release of the same
        # version in the spec.
        return True


@dataclass(frozen=True)
class GreaterThan(_BasicPredicate):
    def evaluate(self, prospective: ParsedVersion) -> bool:
        if not prospective > self.spec:
            return False
        # This special case is here so that, unless the specifier itself
        # includes is a post-release version, that we do not accept
        # post-release versions for the version mentioned in the specifier
        # (e.g. >3.1 should not match 3.0.post0, but should match 3.2.post0).
        if not self.spec.is_postrelease and prospective.is_postrelease:
            if prospective.base_version == self.spec.base_version:
                return False
        # Ensure that we do not allow a local version of the version mentioned
        # in the specifier, which is technically greater than, to match.
        if prospective.local is not None:
            if prospective.base_version == self.spec.base_version:
                return False
        # If we've gotten to here, it means that prospective version is both
        # greater than the spec version *and* it's not a pre-release of the
        # same version in the spec.
        return True


@dataclass(frozen=True)
class _StrPredicate(ContainsPredicate):
    spec_str: str


@dataclass(frozen=True)
class ArbitraryEqual(_StrPredicate):
    def evaluate(self, prospective: ParsedVersion) -> bool:
        return str(prospective).lower() == self.spec_str.lower()


@dataclass(frozen=True)
class EqualPrefix(_StrPredicate):
    @functools.cached_property
    def split_spec(self) -> tuple[str, ...]:
        # Split the spec out by bangs and dots, and pretend that there is
        # an implicit dot in between a release segment and a pre-release segment.
        return _VersionSplitter.split(
            canonicalize_version(self.spec_str, strip_trailing_zero=False)
        )

    def evaluate(self, prospective: ParsedVersion) -> bool:
        # In the case of prefix matching we want to ignore local segment.
        split_prospective = _VersionSplitter.split(str(prospective.public))

        # 0-pad the prospective version before shortening it to get the correct
        # shortened version.
        padded_prospective, _ = _VersionSplitter.pad_version(
            split_prospective, self.split_spec
        )

        # Shorten the prospective version to be the same length as the spec
        # so that we can determine if the specifier is a prefix of the
        # prospective version or not.
        return padded_prospective[: len(self.split_spec)] == self.split_spec


@dataclass(frozen=True)
class NotEqualPrefix(_StrPredicate):
    @functools.cached_property
    def eq_prefix_pred(self) -> EqualPrefix:
        return EqualPrefix(self.spec_str)

    def evaluate(self, prospective: ParsedVersion) -> bool:
        return not self.eq_prefix_pred.evaluate(prospective)


@dataclass(frozen=True)
class Compatible(_StrPredicate):
    @functools.cached_property
    def prefix_form(self) -> str:
        return _VersionSplitter.prefix_form(self.spec_str)

    @functools.cached_property
    def ge_pred(self) -> GreaterThanEqual:
        return GreaterThanEqual(ParsedVersion.parse(self.spec_str))

    @functools.cached_property
    def eq_prefix_pred(self) -> EqualPrefix:
        return EqualPrefix(self.prefix_form)

    def evaluate(self, prospective: ParsedVersion) -> bool:
        # Compatible releases have an equivalent combination of >= and ==. That
        # is that ~=2.2 is equivalent to >=2.2,==2.*. This allows us to
        # implement this in terms of the other specifiers instead of
        # implementing it ourselves. The only thing we need to do is construct
        # the other specifiers.
        return self.ge_pred.evaluate(prospective) and self.eq_prefix_pred.evaluate(
            prospective
        )


@dataclass(frozen=True)
class _PrefixMatcher:
    prefixes: tuple[str, ...]

    def __post_init__(self) -> None:
        assert len(self.prefixes) > 0

    __slots__ = ["prefixes", "__dict__"]

    @staticmethod
    def _prefix_pattern(prefixes: Iterable[str]) -> str:
        joined = "|".join(re.escape(p) for p in prefixes)
        return f"^{joined}"

    @functools.cached_property
    def prefix_regex(self) -> re.Pattern[str]:
        return re.compile(type(self)._prefix_pattern(self.prefixes))

    def does_not_match_prefix(self, s: str) -> bool:
        return self.prefix_regex.match(s) is None


class _VersionSplitter:
    _prefix_regex: ClassVar[re.Pattern[str]] = re.compile(
        r"^([0-9]+)((?:a|b|c|rc)[0-9]+)$"
    )

    @staticmethod
    def split(version: str) -> tuple[str, ...]:
        """Split version into components.

        The split components are intended for version comparison. The logic does
        not attempt to retain the original version string, so joining the
        components back with :func:`_VersionSplitter.join` may not produce the original
        version string.
        """
        result: list[str] = []

        epoch, _, rest = version.rpartition("!")
        result.append(epoch or "0")

        for item in rest.split("."):
            if m := _VersionSplitter._prefix_regex.match(item):
                result.extend(m.groups())
            else:
                result.append(item)
        return tuple(result)

    @staticmethod
    def join(components: tuple[str, ...]) -> str:
        """Join split version components into a version string.

        This function assumes the input came from :func:`_VersionSplitter.split`, where
        the first component must be the epoch (either empty or numeric), and all other
        components numeric.
        """
        assert len(components) >= 2, components
        epoch, *rest = components
        joined = ".".join(rest)
        return f"{epoch}!{joined}"

    _suffix_matcher: ClassVar[_PrefixMatcher] = _PrefixMatcher(
        ("dev", "a", "b", "rc", "post")
    )

    @staticmethod
    def _is_not_suffix(segment: str) -> bool:
        return _VersionSplitter._suffix_matcher.does_not_match_prefix(segment)

    @staticmethod
    def prefix_form(version: str) -> str:
        # We want everything but the last item in the version, but we want to
        # ignore suffix segments.
        no_suffix = tuple(
            itertools.takewhile(
                _VersionSplitter._is_not_suffix,
                _VersionSplitter.split(version),
            )
        )
        return _VersionSplitter.join(no_suffix[:-1])

    @staticmethod
    def pad_version(
        left: tuple[str, ...], right: tuple[str, ...]
    ) -> tuple[tuple[str, ...], tuple[str, ...]]:
        left_split: list[list[str]] = []
        right_split: list[list[str]] = []

        # Get the release segment of our versions
        left_split.append(list(itertools.takewhile(lambda x: x.isdigit(), left)))
        right_split.append(list(itertools.takewhile(lambda x: x.isdigit(), right)))

        # Get the rest of our versions
        left_split.append(list(left[len(left_split[0]) :]))
        right_split.append(list(right[len(right_split[0]) :]))

        # Insert our padding
        left_split.insert(1, ["0"] * max(0, len(right_split[0]) - len(left_split[0])))
        right_split.insert(1, ["0"] * max(0, len(left_split[0]) - len(right_split[0])))

        return (
            tuple(itertools.chain.from_iterable(left_split)),
            tuple(itertools.chain.from_iterable(right_split)),
        )
