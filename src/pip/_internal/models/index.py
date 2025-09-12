from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from pip._internal.utils.urls import ParsedUrl

if TYPE_CHECKING:
    from typing_extensions import Self


@dataclass(frozen=True, slots=True)
class PackageIndex:
    """Represents a Package Index and provides easier access to endpoints"""

    base_url: ParsedUrl
    simple_url: ParsedUrl
    search_url: ParsedUrl

    @classmethod
    def create(
        cls,
        url: str,
        *,
        simple_subdir: str = "simple",
        search_subdir: str = "pypi",
    ) -> Self:
        base_url = ParsedUrl.parse(url)
        simple_url = base_url.join_path(simple_subdir)
        search_url = base_url.join_path(search_subdir)

        return cls(
            base_url=base_url,
            simple_url=simple_url,
            search_url=search_url,
        )


PyPI = PackageIndex.create("https://pypi.org/")
TestPyPI = PackageIndex.create("https://test.pypi.org/")
