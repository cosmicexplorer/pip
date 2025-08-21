from __future__ import annotations

import functools
import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar

from pip._vendor.packaging.version import VERSION_PATTERN, InvalidVersion

from .comparisons import Infinity, InfinityType, NegativeInfinity, NegativeInfinityType

if TYPE_CHECKING:
    from typing import SupportsInt


@dataclass(frozen=True)
class ParsedVersion:
    epoch: int
    release: tuple[int, ...]
    _dev: tuple[str, int] | None
    pre: tuple[str, int] | None
    _post: tuple[str, int] | None
    _local: tuple[int | str, ...] | None

    __slots__ = ["epoch", "release", "_dev", "pre", "_post", "_local", "__dict__"]

    def __post_init__(self) -> None:
        assert len(self.release) > 0

    _regex: ClassVar[re.Pattern[str]] = re.compile(
        r"^\s*" + VERSION_PATTERN + r"\s*$",
        flags=re.VERBOSE | re.IGNORECASE,
    )

    @staticmethod
    @functools.cache
    def _cached_create(
        epoch: int,
        release: tuple[int, ...],
        _dev: tuple[str, int] | None,
        pre: tuple[str, int] | None,
        _post: tuple[str, int] | None,
        _local: tuple[int | str, ...] | None,
    ) -> ParsedVersion:
        return ParsedVersion(
            epoch=epoch,
            release=release,
            _dev=_dev,
            pre=pre,
            _post=_post,
            _local=_local,
        )

    @staticmethod
    @functools.cache
    def _cached_parse(version: str) -> ParsedVersion:
        m = ParsedVersion._regex.match(version)
        if not m:
            raise InvalidVersion(f"Invalid version: {version!r}")
        g = m.groupdict()

        return ParsedVersion._cached_create(
            epoch=int(g["epoch"] or 0),
            release=tuple(map(int, g["release"].split("."))),
            pre=ParsedVersion._parse_letter_version(g["pre_l"], g["pre_n"]),
            _post=ParsedVersion._parse_letter_version(
                g["post_l"], g["post_n1"] or g["post_n2"]
            ),
            _dev=ParsedVersion._parse_letter_version(g["dev_l"], g["dev_n"]),
            _local=ParsedVersion._parse_local_version(g["local"]),
        )

    @classmethod
    def parse(cls, version: str) -> ParsedVersion:
        """Parse a version string into a normalized representation.

        :param version:
            The string representation of a version which will be parsed and normalized
            before use.
        :raises InvalidVersion:
            If the ``version`` does not conform to PEP 440 in any way then this
            exception will be raised.
        """
        return cls._cached_parse(version)

    @staticmethod
    def _generate_alternates(
        alternate_pairs: Iterable[tuple[str, Iterable[str]]],
    ) -> dict[str, str]:
        """Generate a reverse dict mapping alternate values to the normalized string."""
        ret = {}
        for normalized, alternates in alternate_pairs:
            for alt in alternates:
                assert alt not in ret
                ret[alt] = normalized
        return ret

    _letter_alternates: ClassVar[dict[str, str]] = _generate_alternates(
        {
            "a": ["alpha"],
            "b": ["beta"],
            "rc": ["c", "pre", "preview"],
            "post": ["rev", "r"],
        }.items()
    )

    @classmethod
    def _normalize_letter(cls, letter: str) -> str:
        # We normalize any letters to their lower case form.
        letter = letter.lower()

        # We consider some words to be alternate spellings of other words and
        # in those cases we want to normalize the spellings to our preferred
        # spelling.
        return cls._letter_alternates.get(letter, letter)

    @classmethod
    def _given_letter_version(
        cls,
        letter: str,
        number: str | bytes | SupportsInt | None,
    ) -> tuple[str, int]:
        assert letter
        # We consider there to be an implicit 0 in a pre-release if there is
        # not a numeral associated with it.
        if number is None:
            number = 0
        return cls._normalize_letter(letter), int(number)

    @staticmethod
    def _number_only_version(
        number: str | bytes | SupportsInt | None,
    ) -> tuple[str, int] | None:
        if not number:
            return None
        # We assume if we are given a number, but we are not given a letter
        # then this is using the implicit post release syntax (e.g. 1.0-1)
        return "post", int(number)

    @classmethod
    def _parse_letter_version(
        cls,
        letter: str | None,
        number: str | bytes | SupportsInt | None,
    ) -> tuple[str, int] | None:
        if letter:
            return cls._given_letter_version(letter, number)
        return cls._number_only_version(number)

    _local_version_separators: ClassVar[re.Pattern[str]] = re.compile(r"[\._-]")

    @staticmethod
    def _normalize_local_component(part: str) -> int | str:
        if part.isdigit():
            return int(part)
        return part.lower()

    @classmethod
    def _parse_local_version(cls, local: str | None) -> tuple[int | str, ...] | None:
        """
        Takes a string like abc.1.twelve and turns it into ("abc", 1, "twelve").
        """
        if local is None:
            return None
        return tuple(
            map(
                cls._normalize_local_component,
                cls._local_version_separators.split(local),
            )
        )

    # Normalize components for the purpose of comparison.
    @staticmethod
    def _trim_trailing_zeros(release: tuple[int, ...]) -> tuple[int, ...]:
        """
        When we compare a release version, we want to compare it with all of the
        trailing zeros removed.
        """
        nonzeros = (index for index, val in enumerate(release) if val)
        last_nonzero = max(nonzeros, default=0)
        return release[: last_nonzero + 1]

    @staticmethod
    def _cmp_local(
        local: tuple[int | str, ...] | None,
    ) -> NegativeInfinityType | tuple[tuple[int | NegativeInfinityType, str], ...]:
        if local is None:
            # Versions without a local segment should sort before those with one.
            return NegativeInfinity
        # Versions with a local segment need that segment parsed to implement
        # the sorting rules in PEP440.
        # - Alpha numeric segments sort before numeric segments
        # - Alpha numeric segments sort lexicographically
        # - Numeric segments sort numerically
        # - Shorter versions sort before longer versions when the prefixes
        #   match exactly
        return tuple(
            (i, "") if isinstance(i, int) else (NegativeInfinity, i) for i in local
        )

    def _cmp_pre(self) -> InfinityType | NegativeInfinityType | tuple[str, int]:
        # We need to "trick" the sorting algorithm to put 1.0.dev0 before 1.0a0.
        # We'll do this by abusing the pre segment, but we _only_ want to do this
        # if there is not a pre or a post segment. If we have one of those then
        # the normal sorting rules will handle this case correctly.
        if self.pre is None and self._post is None and self._dev is not None:
            return NegativeInfinity
        # Versions without a pre-release (except as noted above) should sort after
        # those with one.
        if self.pre is None:
            return Infinity
        return self.pre

    def _cmp_post(self) -> NegativeInfinityType | tuple[str, int]:
        # Versions without a post segment should sort before those with one.
        if self._post is None:
            return NegativeInfinity
        return self._post

    def _cmp_dev(self) -> InfinityType | tuple[str, int]:
        # Versions without a development segment should sort after those with one.
        if self._dev is None:
            return Infinity
        return self._dev

    @functools.cached_property
    def _key(
        self,
    ) -> tuple[
        int,
        tuple[int, ...],
        InfinityType | NegativeInfinityType | tuple[str, int],
        NegativeInfinityType | tuple[str, int],
        InfinityType | tuple[str, int],
        NegativeInfinityType | tuple[tuple[int | NegativeInfinityType, str], ...],
    ]:
        _local = self.__class__._cmp_local(self._local)

        return (
            self.epoch,
            self.trimmed_release(),
            self._cmp_pre(),
            self._cmp_post(),
            self._cmp_dev(),
            _local,
        )

    @property
    def post(self) -> int | None:
        return self._post[1] if self._post else None

    @property
    def dev(self) -> int | None:
        return self._dev[1] if self._dev else None

    @functools.cached_property
    def local(self) -> str | None:
        if not self._local:
            return None
        return ".".join(map(str, self._local))

    def trimmed_release(self) -> tuple[int, ...]:
        return self.__class__._trim_trailing_zeros(self.release)

    def with_trimmed_release(self) -> ParsedVersion:
        trimmed = self.trimmed_release()
        if len(trimmed) == len(self.release):
            return self
        return self.__class__._cached_create(
            self.epoch,
            trimmed,
            self._dev,
            self.pre,
            self._post,
            self._local,
        )

    @functools.cached_property
    def _release_dot(self) -> str:
        return ".".join(map(str, self.release))

    @functools.cached_property
    def _base_str(self) -> str:
        if self.epoch != 0:
            return f"{self.epoch}!{self._release_dot}"
        return self._release_dot

    @functools.cached_property
    def base_version(self) -> ParsedVersion:
        return self.__class__.parse(self._base_str)

    @property
    def is_prerelease(self) -> bool:
        return self._dev is not None or self.pre is not None

    @property
    def is_postrelease(self) -> bool:
        return self._post is not None

    @property
    def is_devrelease(self) -> bool:
        return self._dev is not None

    @property
    def major(self) -> int:
        return self.release[0]

    @property
    def minor(self) -> int:
        return self.release[1] if len(self.release) >= 2 else 0

    @property
    def micro(self) -> int:
        return self.release[2] if len(self.release) >= 3 else 0

    @functools.cached_property
    def _pre_str(self) -> str:
        if self.pre is None:
            return ""
        return "".join(map(str, self.pre))

    @functools.cached_property
    def _post_str(self) -> str:
        if self._post is None:
            return ""
        return f".post{self._post[1]}"

    @functools.cached_property
    def _dev_str(self) -> str:
        if self._dev is None:
            return ""
        return f".dev{self._dev[1]}"

    @functools.cached_property
    def _public_str(self) -> str:
        return "".join(
            (
                self._base_str,
                self._pre_str,
                self._post_str,
                self._dev_str,
            )
        )

    @functools.cached_property
    def _str(self) -> str:
        if (local := self.local) is not None:
            return f"{self._public_str}+{local}"
        return self._public_str

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.parse('{self}')"

    @functools.cached_property
    def public(self) -> ParsedVersion:
        if self.local is None:
            return self
        return self.__class__.parse(self._public_str)

    def __hash__(self) -> int:
        return hash(self._key)

    def __lt__(self, other: Any) -> bool:
        if type(other) is not self.__class__:
            return NotImplemented

        return self._key < other._key

    def __le__(self, other: Any) -> bool:
        if type(other) is not self.__class__:
            return NotImplemented

        return self._key <= other._key

    def __eq__(self, other: Any) -> bool:
        if type(other) is not self.__class__:
            return NotImplemented

        return hash(self) == hash(other) and self._key == other._key

    def __ge__(self, other: Any) -> bool:
        if type(other) is not self.__class__:
            return NotImplemented

        return self._key >= other._key

    def __gt__(self, other: Any) -> bool:
        if type(other) is not self.__class__:
            return NotImplemented

        return self._key > other._key

    def __ne__(self, other: Any) -> bool:
        if type(other) is not self.__class__:
            return NotImplemented

        return self._key != other._key
