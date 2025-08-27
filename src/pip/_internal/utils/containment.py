from __future__ import annotations

import functools
import itertools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .version import ParsedVersion

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import ClassVar


class ContainsPredicate:
    @staticmethod
    def compare_compatible(
        prospective: ParsedVersion,
        spec: ParsedVersion,
    ) -> bool:
        # Compatible releases have an equivalent combination of >= and ==. That
        # is that ~=2.2 is equivalent to >=2.2,==2.*. This allows us to
        # implement this in terms of the other specifiers instead of
        # implementing it ourselves. The only thing we need to do is construct
        # the other specifiers.
        prefix = _VersionSplitter.prefix_form(str(spec))

        return ContainsPredicate.compare_greater_than_equal(
            prospective, spec
        ) and ContainsPredicate._compare_equal_prefix(prospective, prefix)

    @staticmethod
    def compare_equal_full(
        prospective: ParsedVersion,
        spec: ParsedVersion,
        trailing_dot_star: bool,
    ) -> bool:
        if trailing_dot_star:
            return ContainsPredicate._compare_equal_prefix(prospective, str(spec))
        return ContainsPredicate.compare_equal_basic(prospective, spec)

    @staticmethod
    def _compare_equal_prefix(prospective: ParsedVersion, spec: str) -> bool:
        # Split the spec out by bangs and dots, and pretend that there is
        # an implicit dot in between a release segment and a pre-release segment.
        split_spec = _VersionSplitter.split(spec)
        # In the case of prefix matching we want to ignore local segment.
        split_prospective = _VersionSplitter.split(str(prospective.public))

        # 0-pad the prospective version before shortening it to get the correct
        # shortened version.
        padded_prospective, _ = _VersionSplitter.pad_version(
            split_prospective, split_spec
        )

        # Shorten the prospective version to be the same length as the spec
        # so that we can determine if the specifier is a prefix of the
        # prospective version or not.
        return padded_prospective[: len(split_spec)] == split_spec

    @staticmethod
    def compare_equal_basic(prospective: ParsedVersion, spec: ParsedVersion) -> bool:
        # If the specifier does not have a local segment, then we want to
        # act as if the prospective version also does not have a local
        # segment.
        if not spec.local:
            return prospective.public == spec
        return prospective == spec

    @staticmethod
    def compare_less_than_equal(
        prospective: ParsedVersion,
        spec: ParsedVersion,
    ) -> bool:
        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return prospective.public <= spec

    @staticmethod
    def compare_greater_than_equal(
        prospective: ParsedVersion,
        spec: ParsedVersion,
    ) -> bool:
        # NB: Local version identifiers are NOT permitted in the version
        # specifier, so local version labels can be universally removed from
        # the prospective version.
        return prospective.public >= spec

    @staticmethod
    def compare_less_than(prospective: ParsedVersion, spec: ParsedVersion) -> bool:
        if not prospective < spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a pre-release version, that we do not accept pre-release
        # versions for the version mentioned in the specifier (e.g. <3.1 should
        # not match 3.1.dev0, but should match 3.0.dev0).
        if not spec.is_prerelease and prospective.is_prerelease:
            if prospective.base_version == spec.base_version:
                return False

        # If we've gotten to here, it means that prospective version is both
        # less than the spec version *and* it's not a pre-release of the same
        # version in the spec.
        return True

    @staticmethod
    def compare_greater_than(prospective: ParsedVersion, spec: ParsedVersion) -> bool:
        if not prospective > spec:
            return False

        # This special case is here so that, unless the specifier itself
        # includes is a post-release version, that we do not accept
        # post-release versions for the version mentioned in the specifier
        # (e.g. >3.1 should not match 3.0.post0, but should match 3.2.post0).
        if not spec.is_postrelease and prospective.is_postrelease:
            if prospective.base_version == spec.base_version:
                return False

        # Ensure that we do not allow a local version of the version mentioned
        # in the specifier, which is technically greater than, to match.
        if prospective.local is not None:
            if prospective.base_version == spec.base_version:
                return False

        # If we've gotten to here, it means that prospective version is both
        # greater than the spec version *and* it's not a pre-release of the
        # same version in the spec.
        return True

    @staticmethod
    def compare_arbitrary(prospective: ParsedVersion, spec: ParsedVersion) -> bool:
        return str(prospective).lower() == str(spec).lower()


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
