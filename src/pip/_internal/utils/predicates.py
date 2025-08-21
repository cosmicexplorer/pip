from __future__ import annotations

import functools
import re
from collections import defaultdict
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pip._internal.models.target_python import TargetPython
from pip._internal.utils.packaging.specifiers import SpecifierSet
from pip._internal.utils.packaging.version import ParsedVersion

if TYPE_CHECKING:
    from typing_extensions import Self

    from pip._vendor.packaging.utils import NormalizedName


@dataclass(frozen=True, slots=True)
class RequiresPython:
    specifier: SpecifierSet

    @functools.cache
    @staticmethod
    def _version_sort_key(v: str) -> tuple[int, ...]:
        # TODO: why is this construction necessary over the one in SpecifierSet.__str__?
        return tuple(int(s) for s in v.split(".") if s.isdigit())

    @functools.cache
    @staticmethod
    def _sorted_string(specifier: SpecifierSet) -> str:
        return ",".join(
            sorted(
                map(str, specifier),
                key=RequiresPython._version_sort_key,
            )
        )

    def to_sorted_string(self) -> str:
        return RequiresPython._sorted_string(self.specifier)

    _compat_by_specifier: ClassVar[
        defaultdict[SpecifierSet, dict[ParsedVersion, bool]]
    ] = defaultdict(dict)

    def is_compatible(self, py: TargetPython) -> bool:
        cache = type(self)._compat_by_specifier[self.specifier]
        if py.full_py_version in cache:
            return cache[py.full_py_version]
        new_result = py.full_py_version in self.specifier
        cache[py.full_py_version] = new_result
        return new_result

    @classmethod
    def parse(cls, requires_python: str) -> Self:
        """
        :raises: specifiers.InvalidSpecifier
        """
        return cls(SpecifierSet.parse(requires_python))


@dataclass(frozen=True)
class FragmentMatcher:
    canonical_name: NormalizedName

    __slots__ = ["canonical_name", "__dict__"]

    @functools.cache
    @staticmethod
    def _decanonicalized_regex(canonical_name: NormalizedName) -> re.Pattern[str]:
        return re.compile(
            r"^"
            + r"(?:[-_.]+)".join(
                f"(?:{re.escape(component)})" for component in canonical_name.split("-")
            )
        )

    @functools.cached_property
    def _pattern(self) -> re.Pattern[str]:
        return FragmentMatcher._decanonicalized_regex(self.canonical_name)

    def find_name_version_sep(self, fragment: str) -> int | None:
        """Find the separator's index based on the package's canonical name.

        :param fragment: A <package>+<version> filename "fragment" (stem) or
            egg fragment.

        This function is needed since the canonicalized name does not necessarily
        have the same length as the egg info's name part. An example::

        >>> fragment = 'foo__bar-1.0'
        >>> m = FragmentMatcher('foo-bar')
        >>> m.find_name_version_sep(fragment)
        8
        """
        # Project name and version must be separated by one single dash. Find all
        # occurrences of dashes; if the string in front of it matches the canonical
        # name, this is the one separating the name and version parts.
        if m := self._pattern.search(fragment.lower()):
            if (len(fragment) > m.end()) and (fragment[m.end()] == "-"):
                return m.end()
        return None

    def extract_version_from_fragment(self, fragment: str) -> str | None:
        """Parse the version string from a <package>+<version> filename
        "fragment" (stem) or egg fragment.

        :param fragment: The string to parse. E.g. foo-2.1
        """
        if ind := self.find_name_version_sep(fragment):
            if len(fragment) > ind + 1:
                return fragment[ind + 1 :]
        return None
