from __future__ import annotations

import dataclasses
import itertools
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Callable, ClassVar

from pip._internal.models.index import PyPI
from pip._internal.utils.compat import has_tls
from pip._internal.utils.misc import normalize_path
from pip._internal.utils.urls import ParsedUrl, _UrlPath, path_to_url, url_to_path
from pip._internal.vcs.versioncontrol import try_parse_url

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import Self

    from pip._vendor.packaging.utils import NormalizedName

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _PrintableUrlCollection:
    """Class to lazily generate url strings and ensure auth info is redacted."""

    urls: tuple[ParsedUrl, ...]

    @classmethod
    def create(cls, urls: Iterable[ParsedUrl]) -> Self:
        return cls(urls=tuple(urls))

    def _redacted_auth_urls(self) -> Iterator[ParsedUrl]:
        for u in self.urls:
            yield u.with_redacted_auth_info()

    def __str__(self) -> str:
        return ", ".join(map(str, self._redacted_auth_urls()))

    def __bool__(self) -> bool:
        return bool(self.urls)

    _default_pypi_index_only: ClassVar[tuple[ParsedUrl, ...]] = (PyPI.simple_url,)

    def is_default_pypi_index_only(self) -> bool:
        return self.urls == self.__class__._default_pypi_index_only


@dataclass(frozen=True, slots=True)
class SourceLocation:
    url: ParsedUrl
    path: Path | None

    @staticmethod
    def _normalize_path_link(link_or_path: str) -> str:
        """Build find_links.

        If an argument starts with ~, it may be
        a local file relative to a home directory. So try normalizing
        it and if it exists, use the normalized version.

        NB: This is deliberately conservative - it might be fine just to
            blindly normalize anything starting with a ~...
        """
        if link_or_path.startswith("~"):
            normalized = normalize_path(link_or_path)
            if os.path.exists(normalized):
                return normalized
        return link_or_path

    @classmethod
    def parse(cls, location: str) -> Self | None:
        location = cls._normalize_path_link(location)

        path: Path | None = None
        url: ParsedUrl | None = None

        if os.path.exists(location):  # Is a local path.
            url = path_to_url(location)
            path = Path(location)
        elif location.startswith("file:"):  # A file: URL.
            url = ParsedUrl.parse(location)
            path = Path(url_to_path(url))
            if not path.exists():
                logger.warning(
                    "Location '%s' denotes a file:// url, "
                    "but the path '%s' does not exist. This entry will be ignored.",
                    location,
                    path,
                )
                return None
            path = path.absolute()
        elif url := try_parse_url(location):
            pass

        if url is None:
            logger.warning(
                "Location '%s' is ignored: "
                "it is either a non-existing path or lacks a specific scheme.",
                location,
            )
            return None
        return cls(url=url, path=path)

    def join(self, project_name: NormalizedName) -> Self | None:
        path: Path | None = self.path
        if path:
            path = path / project_name
            if not path.exists():
                logger.debug(
                    "Project name '%s' not found for local index source at %s; "
                    "skipping.",
                    project_name,
                    self.path,
                )
                return None
        project_subdir = _UrlPath._quoter.do_quote(project_name)
        # For maximum compatibility with easy_install, ensure the path
        # ends in a trailing slash.  Although this isn't in the spec
        # (and PyPI can handle it without the slash) some other index
        # implementations might break if they relied on easy_install's
        # behavior.
        # NB: SearchScope#get_index_urls_locations() on main would just concatenate the
        #     entire url with the project name instead of just the path, which seems
        #     likely to break much more than easy_install compat could ever hope to.
        url = self.url.join_path(project_subdir).with_trailing_slash_path()
        return self.__class__(url=url, path=path)


@dataclass(frozen=True, slots=True)
class SearchScope:
    """
    Encapsulates the locations that pip is configured to search.
    """

    find_links: tuple[SourceLocation, ...]
    index_urls: tuple[SourceLocation, ...]
    no_index: bool

    def _all_urls(self) -> Iterable[ParsedUrl]:
        return itertools.chain(
            (loc.url for loc in self.index_urls),
            (loc.url for loc in self.find_links),
        )

    def _ssl_urls(self) -> _PrintableUrlCollection:
        """Identify URLs requiring SSL support."""
        return _PrintableUrlCollection.create(
            link for link in self._all_urls() if link.scheme == "https"
        )

    def __post_init__(self) -> None:
        assert isinstance(self.find_links, tuple), self
        assert isinstance(self.index_urls, tuple), self

        # If we don't have TLS enabled, then WARN if anyplace we're looking
        # relies on TLS.
        if not has_tls():
            ssl_urls = self._ssl_urls()
            if ssl_urls:
                logger.warning(
                    "pip is configured with locations that require "
                    "TLS/SSL, but the ssl module in Python is not "
                    "available: %s",
                    ssl_urls,
                )

    @classmethod
    def empty(cls) -> Self:
        return cls(find_links=(), index_urls=(), no_index=False)

    @classmethod
    def create(
        cls,
        find_links: Iterable[str],
        index_urls: Iterable[str],
        *,
        no_index: bool,
        suppress_no_index: bool = False,
    ) -> Self:
        """
        Create a SearchScope object after normalizing the `find_links`.

        It might seem strange to have both no_index as well as suppress_no_index. It
        might also seem strange to have no_index as a field at all. The reason for both
        of these is that the global value of no_index is used to construct the
        SearchScope used to parse a requirements file, whereas suppress_no_index is used
        in self_outdated_check.py.
        """
        parsed_find_links = tuple(filter(None, map(SourceLocation.parse, find_links)))
        parsed_index_urls = tuple(filter(None, map(SourceLocation.parse, index_urls)))

        if no_index and not suppress_no_index:
            ignored_indices = _PrintableUrlCollection.create(
                link.url for link in parsed_index_urls
            )
            if ignored_indices:
                logger.debug("Ignoring indices: %s", ignored_indices)
            parsed_index_urls = ()

        return cls(
            find_links=parsed_find_links,
            index_urls=parsed_index_urls,
            no_index=no_index,
        )

    def get_formatted_locations(self) -> str:
        lines = []

        redacted_index_urls = _PrintableUrlCollection.create(
            loc.url for loc in self.index_urls
        )
        if redacted_index_urls and not redacted_index_urls.is_default_pypi_index_only():
            lines.append(f"Looking in indexes: {redacted_index_urls}")

        redacted_find_links = _PrintableUrlCollection.create(
            loc.url for loc in self.find_links
        )
        if redacted_find_links:
            lines.append(f"Looking in links: {redacted_find_links}")

        return "\n".join(lines)

    def filter_urls(
        self,
        f: Callable[[ParsedUrl], bool],
    ) -> Self:
        find_links: list[SourceLocation] = []
        for u in self.find_links:
            if u.path is not None:
                find_links.append(u)
            elif f(u.url):
                find_links.append(u)

        index_urls: list[SourceLocation] = []
        for u in self.index_urls:
            if u.path is not None:
                index_urls.append(u)
            elif f(u.url):
                index_urls.append(u)

        return dataclasses.replace(
            self,
            find_links=tuple(find_links),
            index_urls=tuple(index_urls),
        )

    def index_pages_for_name(
        self, project_name: NormalizedName
    ) -> Iterator[SourceLocation]:
        for u in self.index_urls:
            if (joined := u.join(project_name)) is not None:
                yield joined
