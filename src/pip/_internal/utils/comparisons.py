from __future__ import annotations

from typing import Any


class InfinityType:
    def __repr__(self) -> str:
        return "Infinity"

    def __hash__(self) -> int:
        # FIXME: packaging.version hashes repr(self)--is there a reason for that?
        return hash(id(self.__class__))

    def __lt__(self, other: Any) -> bool:
        return False

    def __le__(self, other: Any) -> bool:
        # FIXME: packaging.version just returns False here?
        return self.__eq__(other)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def __gt__(self, other: Any) -> bool:
        return True

    def __ge__(self, other: Any) -> bool:
        return True

    def __neg__(self) -> NegativeInfinityType:
        return NegativeInfinity


Infinity = InfinityType()


class NegativeInfinityType:
    def __repr__(self) -> str:
        return "-Infinity"

    def __hash__(self) -> int:
        # FIXME: packaging.version hashes repr(self)--is there a reason for that?
        return hash(id(self.__class__))

    def __lt__(self, other: Any) -> bool:
        return True

    def __le__(self, other: Any) -> bool:
        return True

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, self.__class__)

    def __gt__(self, other: Any) -> bool:
        return False

    def __ge__(self, other: Any) -> bool:
        # FIXME: packaging.version just returns False here?
        return self.__eq__(other)

    def __neg__(self) -> InfinityType:
        return Infinity


NegativeInfinity = NegativeInfinityType()
