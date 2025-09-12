from __future__ import annotations

import abc
import bz2
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import (
    TYPE_CHECKING,
    Any,
    BinaryIO,
    Generic,
    TextIO,
    TypeVar,
)

if TYPE_CHECKING:

    pass

logger = logging.getLogger(__name__)


_T = TypeVar("_T")


class SerializableProcess(abc.ABC, Generic[_T]):
    @abc.abstractmethod
    def load_from_path(self, path: Path) -> _T | None: ...

    @abc.abstractmethod
    def write_to_path(self, path: Path, value: _T) -> None: ...


_Data = TypeVar("_Data")


class SerializationCodec(abc.ABC, Generic[_T, _Data]):
    @abc.abstractmethod
    def serialize(self, val: _T) -> _Data: ...

    @abc.abstractmethod
    def deserialize(self, data: _Data) -> _T: ...


_IO = TypeVar("_IO", TextIO, BinaryIO)


class FileStreamProtocol(abc.ABC, Generic[_IO]):
    @abc.abstractmethod
    def open_for_read(self, path: Path) -> _IO: ...

    @abc.abstractmethod
    def open_for_write(self, path: Path) -> _IO: ...


class Bz2TextStream(FileStreamProtocol[TextIO]):
    def open_for_read(self, path: Path) -> TextIO:
        return bz2.open(str(path), mode="rt", encoding="utf-8")

    def open_for_write(self, path: Path) -> TextIO:
        return bz2.open(str(path), mode="wt", encoding="utf-8")


@dataclass(frozen=True, slots=True)
class JsonCodec(SerializableProcess[tuple[_T, ...]]):
    codec: SerializationCodec[_T, Any]
    stream: FileStreamProtocol[TextIO]

    def load_from_path(self, path: Path) -> tuple[_T, ...] | None:
        try:
            with self.stream.open_for_read(path) as f:
                serialized = json.load(f)
            logger.debug("read json-serialized input from %s", path)

            return tuple(self.codec.deserialize(val) for val in serialized)
        except (OSError, json.decoder.JSONDecodeError, KeyError) as e:
            logger.debug(
                "could not read cached values at %s %s(%s)",
                path,
                e.__class__.__name__,
                e,
            )
        return None

    def write_to_path(self, path: Path, value: tuple[_T, ...]) -> None:
        serialized = [self.codec.serialize(val) for val in value]

        with self.stream.open_for_write(path) as f:
            logger.debug("writing serialized values to %s", path)
            json.dump(serialized, f)
