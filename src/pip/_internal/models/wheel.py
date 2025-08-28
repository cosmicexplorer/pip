"""Represents a wheel file and provides access to the various parts of the
name that have meaning.
"""

from __future__ import annotations

import functools
import re
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar, Protocol

from pip._vendor.packaging.tags import Tag
from pip._vendor.packaging.utils import (
    BuildTag,
    NormalizedName,
    canonicalize_name,
)
from pip._vendor.packaging.utils import (
    InvalidWheelFilename as _PackagingInvalidWheelFilename,
)

from pip._internal.exceptions import InvalidWheelFilename
from pip._internal.utils.deprecation import deprecated
from pip._internal.utils.filename_parsing import parse_wheel_filename
from pip._internal.utils.version import ParsedVersion

if TYPE_CHECKING:
    from typing_extensions import Self


# TODO upstream fixes:
# (1) fix find_most_preferred_tag() intersection:
# > NB: This method has been completely broken
# >     since d2c280be64e7e6413a481e1b2548d8908fc9c3ae. The `tags` parameter was
# >     never checked. It appears this simply misses out on optimizations as opposed
# >     to producing correctness issues. The intended behavior appears to be
# >     an intersection.
# (2) Fix build_tag():
# > NB: On main, we fail to address the case of no build tag in Wheel.build_tag().
# >     That is a valid result and indeed the common case.


class _ParsedWheelInfo(Protocol):
    @property
    def name(self) -> NormalizedName: ...
    @property
    def version(self) -> str: ...
    @property
    def build_tag(self) -> BuildTag: ...
    @property
    def tag_set(self) -> frozenset[Tag]: ...


@dataclass(frozen=True)
class WheelInfo:
    name: NormalizedName
    version: str
    build_tag: BuildTag
    tag_set: frozenset[Tag]

    @functools.cached_property
    def sorted_tag_strings(self) -> tuple[str, ...]:
        """Return the wheel's tags as a sorted tuple of strings."""
        return tuple(sorted(map(str, self.tag_set)))

    def support_index_min(self, tags: Iterable[Tag]) -> int:
        """Return the lowest index that one of the wheel's file_tag combinations
        achieves in the given list of supported tags.

        For example, if there are 8 supported tags and one of the file tags
        is first in the list, then return 0.

        :param tags: the PEP 425 tags to check the wheel against, in order
            with most preferred first.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
        try:
            return next(i for i, t in enumerate(tags) if t in self.tag_set)
        except StopIteration:
            raise ValueError()

    def find_most_preferred_tag(
        self, tags: Iterable[Tag], tag_to_priority: dict[Tag, int]
    ) -> int:
        """Return the priority of the most preferred tag that one of the wheel's file
        tag combinations achieves in the given list of supported tags using the given
        tag_to_priority mapping, where lower priorities are more-preferred.

        This is used in place of support_index_min in some cases in order to avoid
        an expensive linear scan of a large list of tags.

        :param tags: the PEP 425 tags to check the wheel against.
        :param tag_to_priority: a mapping from tag to priority of that tag, where
            lower is more preferred.

        :raises ValueError: If none of the wheel's file tags match one of
            the supported tags.
        """
        # NB: This method has been completely broken
        # since d2c280be64e7e6413a481e1b2548d8908fc9c3ae. The `tags` parameter was
        # never checked. It appears this simply misses out on optimizations as opposed
        # to producing correctness issues. The intended behavior appears to be
        # an intersection.
        return min(tag_to_priority[tag] for tag in self.tag_set.intersection(tags))

    def supported(self, tags: Iterable[Tag]) -> bool:
        """Return whether the wheel is compatible with one of the given tags.

        :param tags: the PEP 425 tags to check the wheel against.
        """
        return not self.tag_set.isdisjoint(tags)

    @classmethod
    def _from_parsed(cls, info: _ParsedWheelInfo) -> Self:
        return cls(
            name=info.name,
            version=info.version,
            build_tag=info.build_tag,
            tag_set=info.tag_set,
        )

    @classmethod
    def parse_filename(cls, filename: str) -> Self:
        try:
            normal_info = _NormalizedWheelInfo.parse_pep_508_wheel_filename(filename)
            return cls._from_parsed(normal_info)
        except _PackagingInvalidWheelFilename as e:
            legacy_info = _LegacyWheelInfo.parse_legacy_wheel_filename(filename)

            deprecated(
                reason=(
                    f"Wheel filename {filename!r} is not correctly normalised. "
                    "Future versions of pip will raise the following error:\n"
                    f"{e.args[0]}\n\n"
                ),
                replacement=(
                    "to rename the wheel to use a correctly normalised "
                    "name (this may require updating the version in "
                    "the project metadata)"
                ),
                gone_in="25.3",
                issue=12938,
            )

            return cls._from_parsed(legacy_info)


@dataclass(frozen=True)
class _NormalizedWheelInfo:
    name: NormalizedName
    _version: ParsedVersion
    build_tag: BuildTag
    tag_set: frozenset[Tag]

    @functools.cached_property
    def version(self) -> str:
        return str(self._version)

    @classmethod
    def parse_pep_508_wheel_filename(cls, filename: str) -> Self:
        """Parse a normalized (PEP 508) wheel filename.

        The current specification of valid wheel filenames is at
        https://packaging.python.org/en/latest/specifications/dependency-specifiers.

        :raises: packaging.utils.InvalidWheelFilename: if not correctly normalized."""
        name, version, build_tag, tag_set = parse_wheel_filename(filename)
        return cls(name=name, _version=version, build_tag=build_tag, tag_set=tag_set)


@dataclass(frozen=True)
class _LegacyWheelInfo:
    _name: str
    _version: str
    _build_tag: str | None
    pyversions: tuple[str, ...]
    abis: tuple[str, ...]
    plats: tuple[str, ...]

    _legacy_wheel_file_re: ClassVar[re.Pattern[str]] = re.compile(
        r"""^(?P<namever>(?P<name>[^\s-]+?)-(?P<ver>[^\s-]*?))
        ((-(?P<build>\d[^-]*?))?-(?P<pyver>[^\s-]+?)-(?P<abi>[^\s-]+?)-(?P<plat>[^\s-]+?)
        \.whl|\.dist-info)$""",
        re.VERBOSE,
    )

    _build_tag_re: ClassVar[re.Pattern[str]] = re.compile(r"(\d+)(.*)")

    @functools.cached_property
    def name(self) -> NormalizedName:
        return canonicalize_name(self._name)

    @functools.cached_property
    def version(self) -> str:
        return self._version.replace("_", "-")

    @functools.cached_property
    def build_tag(self) -> BuildTag:
        # NB: On main, we fail to address the case of no build tag in Wheel.build_tag().
        #     That is a valid result and indeed the common case.
        if self._build_tag is None:
            return ()
        tag_info = self._build_tag_re.match(self._build_tag)
        assert tag_info is not None
        (numeric, suffix) = tag_info.groups()
        return int(numeric), suffix

    def iter_tags(self) -> Iterator[Tag]:
        for py in self.pyversions:
            for abi in self.abis:
                for plat in self.plats:
                    yield Tag(
                        interpreter=py,
                        abi=abi,
                        platform=plat,
                    )

    @functools.cached_property
    def tag_set(self) -> frozenset[Tag]:
        return frozenset(self.iter_tags())

    @classmethod
    def parse_legacy_wheel_filename(cls, filename: str) -> Self:
        """Parse a legacy (PEP 305+440) wheel filename.

        The name must conform to PEP 305, and the version must conform to PEP 440.
        The build tag must conform to PEP 427.

        :raises: pip._internal.exceptions.InvalidWheelFilename: if not conformant to
                                                                PEP 305 and PEP 440."""
        legacy_wheel_info = cls._legacy_wheel_file_re.match(filename)
        if not legacy_wheel_info:
            raise InvalidWheelFilename(
                f"Invalid wheel filename (PEP 305 name, PEP 440 version): {filename!r}"
                f"\nFilename must match this regexp: {cls._legacy_wheel_file_re!r}"
            )
        info = legacy_wheel_info.groupdict()

        return cls(
            _name=info["name"],
            _version=info["ver"],
            _build_tag=info["build"],
            pyversions=tuple(info["pyver"].split(".")),
            abis=tuple(info["abi"].split(".")),
            plats=tuple(info["plat"].split(".")),
        )
