from __future__ import annotations

import functools
import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, NoReturn, Protocol, cast

from pip._internal.exceptions import HashMismatch, HashMissing, InstallationError

if TYPE_CHECKING:
    from hashlib import _Hash

    from typing_extensions import Self


# The recommended hash algo of the moment. Change this whenever the state of
# the art changes; it won't hurt backward compatibility.
FAVORITE_HASH = "sha256"


# Names of hashlib algorithms allowed by the --hash option and ``pip hash``
# Currently, those are the ones at least as collision-resistant as sha256.
STRONG_HASHES = ["sha256", "sha384", "sha512"]


class FileHasher(Protocol):
    def update(self, buf: bytes | bytearray | memoryview) -> None: ...


@dataclass(frozen=True, slots=True)
class MultiHasher:
    hashers: dict[str, FileHasher]

    def __post_init__(self) -> None:
        assert self.hashers

    def update(self, buf: bytes | bytearray | memoryview) -> None:
        for h in self.hashers.values():
            h.update(buf)

    @classmethod
    def for_hash_names(cls, names: Iterable[str]) -> Self:
        hashers = {}
        for hash_name in names:
            try:
                hashers[hash_name] = cast(FileHasher, hashlib.new(hash_name))
            except (ValueError, TypeError):
                raise InstallationError(f"Unknown hash name: {hash_name}")
        return cls(hashers=hashers)


@dataclass
class Hashes:
    """A wrapper that builds multiple hashes at once and checks them against
    known-good values

    """

    _allowed: dict[str, frozenset[str]]

    __slots__ = ["_allowed", "__dict__"]

    @staticmethod
    def _normalize_hashes(
        hashes: Iterable[tuple[str, Iterable[str]]],
    ) -> dict[str, frozenset[str]]:
        return {
            # Make sure values are always sorted (to ease equality checks)
            alg: frozenset(k.lower() for k in keys)
            for alg, keys in hashes
        }

    def __init__(self, hashes: Mapping[str, Iterable[str]] | None = None) -> None:
        """
        :param hashes: A dict of algorithm names pointing to lists of allowed
            hex digests
        """
        self._allowed = Hashes._normalize_hashes((hashes or {}).items())

    def __and__(self, other: Hashes) -> Hashes:
        if not isinstance(other, Hashes):
            return NotImplemented

        # If either of the Hashes object is entirely empty (i.e. no hash
        # specified at all), all hashes from the other object are allowed.
        if not other:
            return self
        if not self:
            return other

        # Otherwise only hashes that present in both objects are allowed.
        return Hashes(
            {
                alg: (self._allowed[alg] & other._allowed[alg])
                for alg in (self._allowed.keys() & other._allowed.keys())
            }
        )

    @functools.cached_property
    def digest_count(self) -> int:
        return sum(len(digests) for digests in self._allowed.values())

    def is_hash_allowed(self, hash_name: str, hex_digest: str) -> bool:
        """Return whether the given hex digest is allowed."""
        if allowed := self._allowed.get(hash_name):
            return hex_digest in allowed
        return False

    def _create_multi_hasher(self) -> MultiHasher:
        return MultiHasher.for_hash_names(self._allowed.keys())

    def _raise(self, gots: Mapping[str, _Hash]) -> NoReturn:
        raise HashMismatch(self._allowed, gots)

    # NB: There is no git blame regarding the decision to avoid using the same type for
    #     the digest fileobj used for every other file operation.
    @staticmethod
    def _file_digest(
        file: Any, hash_factory: Callable[[], MultiHasher]
    ) -> Mapping[str, _Hash]:
        result = cast(MultiHasher, hashlib.file_digest(file, hash_factory))  # type: ignore[arg-type]
        return result.hashers  # type: ignore[return-value]

    def check_against_file(self, file: Any) -> None:
        """Check good hashes against a file-like object

        Raise HashMismatch if none match.

        """
        # FIXME: file_digest() is 3.11+
        gots = type(self)._file_digest(file, self._create_multi_hasher)
        for hash_name, got in gots.items():
            if got.hexdigest() in self._allowed[hash_name]:
                return
        self._raise(gots)

    def check_against_path(self, path: str) -> None:
        with open(path, "rb") as file:
            return self.check_against_file(file)

    def has_one_of(self, hashes: dict[str, str]) -> bool:
        """Return whether any of the given hashes are allowed."""
        for hash_name, hex_digest in hashes.items():
            if self.is_hash_allowed(hash_name, hex_digest):
                return True
        return False

    def __bool__(self) -> bool:
        """Return whether I know any known-good hashes."""
        return bool(self._allowed)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Hashes):
            return NotImplemented
        return self._allowed == other._allowed

    @functools.cached_property
    def _cmp_key(self) -> tuple[tuple[str, frozenset[str]], ...]:
        return tuple(
            sorted(
                self._allowed.items(),
                key=lambda it: it[0],
            )
        )

    def __hash__(self) -> int:
        return hash(self._cmp_key)


class MissingHashes(Hashes):
    """A workalike for Hashes used when we're missing a hash for a requirement

    It computes the actual hash of the requirement and raises a HashMissing
    exception showing it to the user.

    """

    def __init__(self) -> None:
        """Don't offer the ``hashes`` kwarg."""
        # Pass our favorite hash in to generate a "gotten hash". With the
        # empty list, it will never match, so an error will always raise.
        super().__init__(hashes={FAVORITE_HASH: frozenset()})

    def _raise(self, gots: Mapping[str, _Hash]) -> NoReturn:
        raise HashMissing(gots[FAVORITE_HASH].hexdigest())
