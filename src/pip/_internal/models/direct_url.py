"""PEP 610"""

from __future__ import annotations

import abc
import functools
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, TypeVar

from pip._internal.utils.misc import hash_file
from pip._internal.utils.urls import ParsedUrl

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from typing_extensions import Self

__all__ = [
    "DirectUrl",
    "DirectUrlValidationError",
    "DirInfo",
    "ArchiveInfo",
    "VcsInfo",
]

T = TypeVar("T")

DIRECT_URL_METADATA_NAME = "direct_url.json"


class DirectUrlValidationError(Exception):
    pass


def _get(
    d: Mapping[str, Any], expected_type: type[T], key: str, default: T | None = None
) -> T | None:
    """Get value from dictionary and verify expected type."""
    if key not in d:
        return default
    value = d[key]
    if not isinstance(value, expected_type):
        raise DirectUrlValidationError(
            f"{value!r} has unexpected type for {key} (expected {expected_type})"
        )
    return value


def _get_required(
    d: Mapping[str, Any], expected_type: type[T], key: str, default: T | None = None
) -> T:
    value = _get(d, expected_type, key, default)
    if value is None:
        raise DirectUrlValidationError(f"{key} must have a value")
    return value


def _exactly_one_of(parsed_infos: Iterator[InfoType | None]) -> InfoType:
    infos = tuple(filter(None, parsed_infos))
    if not infos:
        raise DirectUrlValidationError(
            "missing one of archive_info, dir_info, vcs_info"
        )
    if len(infos) > 1:
        raise DirectUrlValidationError(
            "more than one of archive_info, dir_info, vcs_info"
        )
    return infos[0]


def _filter_none(**kwargs: Any) -> dict[str, Any]:
    """Make dict excluding None values."""
    return {k: v for k, v in kwargs.items() if v is not None}


class InfoType(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def name(self) -> str: ...

    @classmethod
    @abc.abstractmethod
    def _from_dict(cls, d: Mapping[str, Any] | None) -> Self | None: ...

    @abc.abstractmethod
    def _to_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True)
class VcsInfo(InfoType):
    name: ClassVar[str] = "vcs_info"

    vcs: str
    commit_id: str
    requested_revision: str | None = None

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any] | None) -> Self | None:
        if d is None:
            return None
        return cls(
            vcs=_get_required(d, str, "vcs"),
            commit_id=_get_required(d, str, "commit_id"),
            requested_revision=_get(d, str, "requested_revision"),
        )

    def _to_dict(self) -> dict[str, Any]:
        return _filter_none(
            vcs=self.vcs,
            requested_revision=self.requested_revision,
            commit_id=self.commit_id,
        )


@dataclass(frozen=True)
class ArchiveInfo(InfoType):
    name: ClassVar[str] = "archive_info"

    hash: str | None
    hashes: dict[str, str] | None

    def __post_init__(self) -> None:
        if self.hash is not None:
            assert self.hash
        if self.hashes is not None:
            assert self.hashes

    def try_compute_hash_from_local_file(self, local_file_path: str | None) -> None:
        """
        Make sure we have a hash in download_info. If we got it as part of the
        URL, it will have been verified and we can rely on it. Otherwise we
        compute it from the downloaded file.

        FIXME: https://github.com/pypa/pip/issues/11943
        """
        if self.hashes:
            return
        if local_file_path is None:
            return
        hash = hash_file(local_file_path)[0].hexdigest()
        # We populate info.hash for backward compatibility.
        object.__setattr__(self, "hash", f"sha256={hash}")
        # We're working around the immutable parse() method, so we have to copy its
        # logic again here.
        object.__setattr__(self, "hashes", {"sha256": hash})

    @classmethod
    def parse(
        cls,
        hash: str | None = None,
        hashes: Mapping[str, str] | None = None,
    ) -> Self:
        # Auto-populate the hashes key to upgrade to the new format automatically.
        # We don't back-populate the legacy hash key from hashes.
        if hash is not None:
            try:
                hash_name, hash_value = hash.split("=", 1)
            except ValueError:
                raise DirectUrlValidationError(
                    f"invalid archive_info.hash format: {hash!r}"
                )
            if hashes is None:
                hashes = {hash_name: hash_value}
            elif hash_name not in hashes:
                hashes = {**hashes, hash_name: hash_value}
        return cls(hash=hash, hashes=dict(hashes) if hashes else None)

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any] | None) -> ArchiveInfo | None:
        if d is None:
            return None
        return cls.parse(hash=_get(d, str, "hash"), hashes=_get(d, dict, "hashes"))

    def _to_dict(self) -> dict[str, Any]:
        return _filter_none(hash=self.hash, hashes=self.hashes)


@dataclass(frozen=True)
class DirInfo(InfoType):
    name: ClassVar[str] = "dir_info"

    editable: bool = False

    @classmethod
    def _from_dict(cls, d: Mapping[str, Any] | None) -> Self | None:
        if d is None:
            return None
        return cls(editable=_get_required(d, bool, "editable", default=False))

    def _to_dict(self) -> dict[str, Any]:
        return _filter_none(editable=self.editable or None)


@dataclass(frozen=True)
class DirectUrl:
    url: ParsedUrl
    info: InfoType
    subdirectory: str | None

    def __post_init__(self) -> None:
        if self.subdirectory is not None:
            assert self.subdirectory

    @classmethod
    def create(
        cls,
        url: str | ParsedUrl,
        info: InfoType,
        subdirectory: str | None = None,
    ) -> Self:
        if isinstance(url, str):
            url = ParsedUrl.parse(url)
        return cls(
            url=url,
            info=info,
            subdirectory=subdirectory or None,
        )

    ENV_VAR_RE: ClassVar[re.Pattern[str]] = re.compile(
        r"^\$\{[A-Za-z0-9-_]+\}(:\$\{[A-Za-z0-9-_]+\})?$"
    )

    def _remove_auth_from_netloc(self, netloc: str) -> str:
        if "@" not in netloc:
            return netloc
        user_pass, netloc_no_user_pass = netloc.split("@", 1)
        if (
            isinstance(self.info, VcsInfo)
            and self.info.vcs == "git"
            and user_pass == "git"
        ):
            return netloc
        if self.__class__.ENV_VAR_RE.match(user_pass):
            return netloc
        return netloc_no_user_pass

    @functools.cached_property
    def redacted_url(self) -> ParsedUrl:
        """url with user:password part removed unless it is formed with
        environment variables as specified in PEP 610, or it is ``git``
        in the case of a git URL.
        """
        netloc = self._remove_auth_from_netloc(self.url.netloc)
        return self.url.with_netloc(netloc=netloc)

    def validate(self) -> None:
        self.from_dict(self.to_dict())

    @classmethod
    def _try_parse_infos(cls, d: Mapping[str, Any]) -> Iterator[InfoType | None]:
        yield ArchiveInfo._from_dict(_get(d, dict, "archive_info"))
        yield DirInfo._from_dict(_get(d, dict, "dir_info"))
        yield VcsInfo._from_dict(_get(d, dict, "vcs_info"))

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> Self:
        return cls.create(
            url=_get_required(d, str, "url"),
            subdirectory=_get(d, str, "subdirectory"),
            info=_exactly_one_of(cls._try_parse_infos(d)),
        )

    def to_dict(self) -> dict[str, Any]:
        res = _filter_none(
            url=str(self.redacted_url),
            subdirectory=self.subdirectory,
        )
        res[self.info.name] = self.info._to_dict()
        return res

    @classmethod
    def from_json(cls, s: str) -> Self:
        return cls.from_dict(json.loads(s))

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def is_local_editable(self) -> bool:
        return isinstance(self.info, DirInfo) and self.info.editable
