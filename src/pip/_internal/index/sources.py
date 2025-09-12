from __future__ import annotations

import functools
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from pip._vendor.packaging.utils import (
    InvalidSdistFilename,
    InvalidWheelFilename,
    NormalizedName,
)

from pip._internal.models.link import Link
from pip._internal.utils.packaging.filename_parsing import (
    parse_sdist_filename,
    parse_wheel_filename,
)
from pip._internal.utils.urls import ParsedUrl, path_to_url

if TYPE_CHECKING:
    from collections.abc import Iterable

    from pip._internal.models.candidate import InstallationCandidate

    FoundCandidates = Iterable[InstallationCandidate]
    FoundLinks = Iterable[Link]
    CandidatesFromPage = Callable[[ParsedUrl], Iterable[InstallationCandidate]]

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class _LocalFindLinks:
    """Scans file or directory and caches results.

    It may seem confusing to treat a directory or file the same way, but there are good
    reasons to do so:
    (1) --find-links accepts either a file or directory path.
    (2) --find-links performs highly nontrivial parsing/fs traversal which is shared by
        both types of argument values.

    The biggest difference between the two is whether the path expands to more than
    one entry, so it's easier to handle that case when expanding entries and otherwise
    retain the rest of the shared logic.
    """

    path: Path

    __slots__ = ["path", "__dict__"]

    def __post_init__(self) -> None:
        assert self.path.is_absolute(), self
        assert self.path.exists(), self

    @functools.cached_property
    def _entries(self) -> tuple[tuple[str, ParsedUrl], ...]:
        # TODO: could optimize url parsing here.
        if self.path.is_file():
            return ((self.path.name, path_to_url(str(self.path), normalize=False)),)
        assert self.path.is_dir(), self
        return tuple(
            # Path auto-normalizes, and we checked that the parent was absolute in
            # __post_init__().
            (entry.name, path_to_url(str(entry), normalize=False))
            for entry in self.path.iterdir()
        )

    @staticmethod
    def _is_html_file(file_url: ParsedUrl) -> bool:
        return file_url.guess_type()[0] == "text/html"

    @functools.cached_property
    def _html_files(self) -> tuple[ParsedUrl, ...]:
        return tuple(
            url for _, url in self._entries if self.__class__._is_html_file(url)
        )

    @staticmethod
    def _try_parse_dist_filename(name: str) -> NormalizedName | None:
        project_filename: NormalizedName | None = None
        try:
            project_filename, _, _, _ = parse_wheel_filename(name)
        except InvalidWheelFilename:
            try:
                project_filename, _ = parse_sdist_filename(name)
            except InvalidSdistFilename:
                pass
        return project_filename

    @functools.cached_property
    def _project_name_to_urls(self) -> dict[NormalizedName, tuple[ParsedUrl, ...]]:
        ret: defaultdict[NormalizedName, list[ParsedUrl]] = defaultdict(list)

        for name, url in self._entries:
            if self.__class__._is_html_file(url):
                continue
            if project_name := self.__class__._try_parse_dist_filename(name):
                ret[project_name].append(url)

        return {n: tuple(us) for n, us in ret.items()}

    def possible_index_pages(self) -> Iterable[ParsedUrl]:
        return self._html_files

    def urls_for_project(self, project_name: NormalizedName) -> Iterable[ParsedUrl]:
        return self._project_name_to_urls.get(project_name, ())
