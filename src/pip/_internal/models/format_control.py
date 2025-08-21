from __future__ import annotations

import enum
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

from pip._vendor.packaging.utils import (
    NormalizedName,
    canonicalize_name,
    is_normalized_name,
)

from pip._internal.exceptions import CommandError

if TYPE_CHECKING:
    pass


class FormatControlBuilder:
    """Helper for managing formats from which a package can be installed."""

    __slots__ = ["no_binary", "only_binary"]

    def __init__(
        self,
        no_binary: set[str] | None = None,
        only_binary: set[str] | None = None,
    ) -> None:
        if no_binary is None:
            no_binary = set()
        if only_binary is None:
            only_binary = set()

        self.no_binary = no_binary
        self.only_binary = only_binary

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented

        if self.__slots__ != other.__slots__:
            return False

        return all(getattr(self, k) == getattr(other, k) for k in self.__slots__)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.no_binary}, {self.only_binary})"

    @staticmethod
    def handle_mutual_excludes(value: str, target: set[str], other: set[str]) -> None:
        if value.startswith("-"):
            raise CommandError(
                "--no-binary / --only-binary option requires 1 argument."
            )
        new = value.split(",")
        while ":all:" in new:
            other.clear()
            target.clear()
            target.add(":all:")
            del new[: new.index(":all:") + 1]
            # Without a none, we want to discard everything as :all: covers it
            if ":none:" not in new:
                return
        for name in new:
            if name == ":none:":
                target.clear()
                continue
            name = canonicalize_name(name)
            other.discard(name)
            target.add(name)

    def disallow_binaries(self) -> None:
        self.handle_mutual_excludes(
            ":all:",
            self.no_binary,
            self.only_binary,
        )

    @staticmethod
    def _canonical_names_only(names: Iterable[str]) -> Iterator[NormalizedName]:
        for name in names:
            if name == ":all:":
                continue
            assert is_normalized_name(
                name
            ), "all project names should already have been normalized"
            yield cast(NormalizedName, name)

    def build(self) -> FormatControl:
        default_binary = ":all:" in self.only_binary
        default_source = ":all:" in self.no_binary
        no_binary = frozenset(self._canonical_names_only(self.no_binary))
        only_binary = frozenset(self._canonical_names_only(self.only_binary))
        return FormatControl(
            no_binary=no_binary,
            only_binary=only_binary,
            default_binary=default_binary,
            default_source=default_source,
        )


class AllowedFormats(enum.Enum):
    SourceOnly = enum.auto()
    BinaryOnly = enum.auto()
    AnyFormat = enum.auto()

    def allows_binary(self) -> bool:
        return self != type(self).SourceOnly

    def allows_source(self) -> bool:
        return self != type(self).BinaryOnly


@dataclass(frozen=True, slots=True)
class FormatControl:
    no_binary: frozenset[NormalizedName]
    only_binary: frozenset[NormalizedName]
    default_binary: bool
    default_source: bool

    def get_allowed_formats(self, project_name: str) -> AllowedFormats:
        canonical_name = canonicalize_name(project_name)
        if canonical_name in self.only_binary:
            return AllowedFormats.BinaryOnly
        if canonical_name in self.no_binary:
            return AllowedFormats.SourceOnly
        if self.default_binary:
            return AllowedFormats.BinaryOnly
        if self.default_source:
            return AllowedFormats.SourceOnly
        return AllowedFormats.AnyFormat
