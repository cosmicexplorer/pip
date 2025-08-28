import functools
from dataclasses import dataclass

from pip._internal.models.link import Link
from pip._internal.utils.packaging.version import ParsedVersion


@dataclass(frozen=True)
class InstallationCandidate:
    """Represents a potential "candidate" for installation."""

    __slots__ = ["name", "version", "link", "__dict__"]

    name: str
    version: ParsedVersion
    link: Link

    @functools.cached_property
    def _description(self) -> str:
        return f"{self.name!r} candidate (version {self.version} at {self.link})"

    def __str__(self) -> str:
        return self._description
