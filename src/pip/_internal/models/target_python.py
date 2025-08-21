from __future__ import annotations

import functools
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

from pip._vendor.packaging.tags import Tag

from pip._internal.utils.compatibility_tags import get_supported, version_info_to_nodot
from pip._internal.utils.misc import normalize_version_info
from pip._internal.utils.packaging.version import ParsedVersion


@dataclass(frozen=True)
class TargetPython:
    """
    Encapsulates the properties of a Python interpreter one is targeting
    for a package install, download, etc.
    """

    __slots__ = [
        "_given_py_version_info",
        "abis",
        "implementation",
        "platforms",
        "__dict__",
    ]

    _given_py_version_info: tuple[int, ...] | None
    abis: tuple[str, ...] | None
    implementation: str | None
    platforms: tuple[str, ...] | None

    def __post_init__(self) -> None:
        if self._given_py_version_info is not None:
            assert self._given_py_version_info
        if self.abis is not None:
            assert self.abis
        if self.implementation is not None:
            assert self.implementation
        if self.platforms is not None:
            assert self.platforms

    @functools.cached_property
    def _hash(self) -> int:
        return hash(
            (
                self._given_py_version_info,
                self.abis,
                self.implementation,
                self.platforms,
            )
        )

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return (
            hash(self) == hash(other)
            and self._given_py_version_info == other._given_py_version_info
            and self.abis == other.abis
            and self.implementation == other.implementation
            and self.platforms == other.platforms
        )

    @staticmethod
    @functools.cache
    def _cached_create(
        py_version_info: tuple[int, ...] | None,
        abis: tuple[str, ...] | None,
        implementation: str | None,
        platforms: tuple[str, ...] | None,
    ) -> TargetPython:
        return TargetPython(
            _given_py_version_info=py_version_info,
            abis=abis,
            implementation=implementation,
            platforms=platforms,
        )

    @classmethod
    def create(
        cls,
        platforms: Iterable[str] | None = None,
        py_version_info: tuple[int, ...] | None = None,
        abis: Iterable[str] | None = None,
        implementation: str | None = None,
    ) -> TargetPython:
        """
        :param platforms: An iterable of strings or None. If None, matches
            packages that are supported by the current system. Otherwise, will
            match packages that can be built on the platforms passed in. The packages
            matched will only ever be downloaded for distribution: they will
            not be built locally.
        :param py_version_info: An optional tuple of ints representing the
            Python version information to use (e.g. `sys.version_info[:3]`).
            This can have length 1, 2, or 3 when provided.
        :param abis: An iterable of strings or None. This is passed to
            compatibility_tags.py's get_supported() function after converting a non-None
            iterable to tuple.
        :param implementation: A string or None. This is passed to
            compatibility_tags.py's get_supported() function as is.
        """
        if abis is not None:
            abis = tuple(sorted(frozenset(abis)))
        if platforms is not None:
            platforms = tuple(sorted(frozenset(platforms)))
        return cls._cached_create(
            py_version_info=py_version_info or None,
            abis=abis or None,
            implementation=implementation or None,
            platforms=platforms or None,
        )

    @functools.cached_property
    def py_version_info(self) -> tuple[int, int, int]:
        if self._given_py_version_info is None:
            return sys.version_info[:3]
        return normalize_version_info(self._given_py_version_info)

    @functools.cached_property
    def full_py_version(self) -> ParsedVersion:
        return ParsedVersion.parse(".".join(map(str, self.py_version_info)))

    @functools.cached_property
    def py_version(self) -> ParsedVersion:
        return ParsedVersion.parse(".".join(map(str, self.py_version_info[:2])))

    @functools.cached_property
    def format_given(self) -> str:
        """
        Format the given, non-None attributes for display.
        """
        display_version = None
        if self._given_py_version_info is not None:
            display_version = ".".join(map(str, self._given_py_version_info))

        key_values = [
            ("platforms", self.platforms),
            ("version_info", display_version),
            ("abis", self.abis),
            ("implementation", self.implementation),
        ]
        return " ".join(
            f"{key}={value!r}" for key, value in key_values if value is not None
        )

    @functools.cached_property
    def sorted_tags(self) -> tuple[Tag, ...]:
        """
        Return the supported PEP 425 tags to check wheel candidates against.

        The tags are returned in order of preference (most preferred first).
        """
        # Pass version=None if no py_version_info was given since
        # version=None uses special default logic.
        version = None
        if self._given_py_version_info is not None:
            version = version_info_to_nodot(self._given_py_version_info)

        return get_supported(
            version=version,
            platforms=self.platforms,
            abis=self.abis,
            impl=self.implementation,
        )

    @functools.cached_property
    def tag_preferences(self) -> dict[Tag, int]:
        """
        Since the index of the tag in the _supported_tags list is used
        as a priority, precompute a map from tag to index/priority to be
        used in wheel.find_most_preferred_tag.
        """
        return {tag: idx for idx, tag in enumerate(self.sorted_tags)}

    @functools.cached_property
    def unsorted_tags(self) -> frozenset[Tag]:
        """Exactly the same as get_sorted_tags, but returns a set.

        This is important for performance.
        """
        return frozenset(self.sorted_tags)
