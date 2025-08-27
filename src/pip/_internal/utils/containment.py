from __future__ import annotations

import functools
import itertools
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable
    from typing import ClassVar


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

    def matches_prefix(self, s: str) -> bool:
        return self.prefix_regex.match(s) is not None


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
    def is_suffix(segment: str) -> bool:
        return _VersionSplitter._suffix_matcher.matches_prefix(segment)

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
