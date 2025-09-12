import functools
from dataclasses import dataclass
from typing import Any

from pip._internal.models.link import Link, PersistentLinkCacheArgs
from pip._internal.utils.cache_serde import SerializationCodec
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


class CandidateJsonCodec(SerializationCodec[InstallationCandidate, Any]):
    def serialize(self, val: InstallationCandidate) -> Any:
        return {
            "name": val.name,
            "version": str(val.version),
            "link": val.link.cache_args().to_json(),
        }

    def deserialize(self, data: Any) -> InstallationCandidate:
        link_cache_args = PersistentLinkCacheArgs.from_json(data["link"])
        link = Link.from_cache_args(link_cache_args)
        return InstallationCandidate(
            data["name"],
            ParsedVersion.parse(data["version"]),
            link,
        )
