from __future__ import annotations

import datetime
import enum
import functools
import logging
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pip._internal.utils.urls import ParsedUrl

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class _TimestampSanitizer:
    # Buckle up ************. Read the following:
    # - https://docs.python.org/3/library/time.html#time.strptime
    # - https://docs.python.org/3/library/time.html#time.tzname
    #
    # time.strptime()'s %Z directive can only parse UTC and exactly two local timezones.
    # This makes error handling incredibly aggravating, but luckily we only concern
    # ourselves with two cases:
    # (1) UTC time zone (every server should be doing this).
    # (2) the same time zone as the local machine.
    #
    # We really *should* be able to parse arbitrary timestamps and convert them to UTC,
    # but the python stdlib can't do this for us.
    _local_names: ClassVar[frozenset[str]] = frozenset(time.tzname)

    @dataclass(frozen=True, slots=True)
    class LocalTimezone:
        tz: datetime.tzinfo
        name: str

        @classmethod
        def create(cls, timestamp: datetime.datetime) -> Self:
            with_local_tz = timestamp.astimezone()
            tz = with_local_tz.tzinfo
            assert tz is not None, (timestamp, with_local_tz)
            name = tz.tzname(with_local_tz)
            assert name in _TimestampSanitizer._local_names, (
                timestamp,
                with_local_tz,
                tz,
                name,
            )
            return cls(tz=tz, name=name)

        @classmethod
        def now(cls) -> Self:
            return cls.create(datetime.datetime.now())

    @staticmethod
    @functools.cache
    def _now_local() -> LocalTimezone:
        return _TimestampSanitizer.LocalTimezone.now()

    _utc_zones: ClassVar[frozenset[str]] = frozenset(["UTC", "GMT"])

    class TzKind(enum.Enum):
        utc = enum.auto()
        local_matching = enum.auto()
        local_non_matching = enum.auto()

        @classmethod
        def parse(cls, parsed_time: time.struct_time) -> _TimestampSanitizer.TzKind:
            if parsed_time.tm_zone in _TimestampSanitizer._utc_zones:
                return cls.utc
            # NB: time.tzname is only defined for the current host.
            #     This method is not useful for general time string parsing.
            assert parsed_time.tm_zone in _TimestampSanitizer._local_names, parsed_time
            # The local time zone provided by the server response header matched the one
            # used on the local machine.
            if parsed_time.tm_zone == _TimestampSanitizer._now_local().name:
                return cls.local_matching
            # The server provided a local time zone which did not match the one used on
            # the local machine. Because time.strptime() only understands utc and
            # exactly two local timezones, this implies that the server timezone is the
            # same except for the DST setting. This is worth warning the user about.
            return cls.local_non_matching

    _HTTP_DATE_FORMAT: ClassVar[str] = "%a, %d %b %Y %H:%M:%S %Z"

    @dataclass(frozen=True, slots=True)
    class ParsedDateHeader:
        parsed_time: time.struct_time
        parsed_date: datetime.datetime

        class InvalidTimestamp(Exception):
            pass

        @classmethod
        def parse(cls, date_str: str) -> Self:
            """Parse a 'Date' HTTP response header, without handling timezones."""
            date_str = date_str.strip()
            try:
                parsed_time = time.strptime(
                    date_str, _TimestampSanitizer._HTTP_DATE_FORMAT
                )
                parsed_date = datetime.datetime.strptime(
                    date_str, _TimestampSanitizer._HTTP_DATE_FORMAT
                )
            except ValueError as e:
                raise cls.InvalidTimestamp(
                    f"could not parse {date_str!r} as timestamp "
                    f"(expected {_TimestampSanitizer._HTTP_DATE_FORMAT!r})"
                ) from e
            return cls(parsed_time=parsed_time, parsed_date=parsed_date)

        @property
        def _tz_string(self) -> str:
            return self.parsed_time.tm_zone

        def tz_kind(self) -> _TimestampSanitizer.TzKind:
            return _TimestampSanitizer.TzKind.parse(self.parsed_time)

    @dataclass(frozen=True, slots=True)
    class CacheableTimestamp:
        timestamp: datetime.datetime

        def __post_init__(self) -> None:
            assert self.timestamp.tzname() is not None, (self.timestamp, self)

        @dataclass(frozen=True, slots=True)
        class SerializableTimestamp:
            epoch: int

            def __post_init__(self) -> None:
                assert isinstance(self.epoch, int), self
                assert self.epoch >= 0, self

            @classmethod
            def from_timestamp(cls, timestamp: datetime.datetime) -> Self:
                epoch = timestamp.timestamp()
                # The timestamp will only have second resolution according to the parse
                # format string _HTTP_DATE_FORMAT.
                assert not (epoch % 1), (epoch, timestamp)
                return cls(epoch=int(epoch))

            def to_timestamp(self) -> datetime.datetime:
                return datetime.datetime.fromtimestamp(
                    self.epoch, tz=datetime.timezone.utc
                )

            def serialize(self) -> bytes:
                return self.epoch.to_bytes(length=4, byteorder="big", signed=False)

            @classmethod
            def deserialize(cls, data: bytes) -> Self:
                return cls(epoch=int.from_bytes(data, byteorder="big", signed=False))

        def cache_serialize(self) -> bytes:
            epoch = self.__class__.SerializableTimestamp.from_timestamp(self.timestamp)
            return epoch.serialize()

        @classmethod
        def cache_deserialize(cls, data: bytes) -> Self:
            epoch = cls.SerializableTimestamp.deserialize(data)
            return cls(timestamp=epoch.to_timestamp())

        def format_http_header(self) -> str:
            return self.timestamp.strftime(_TimestampSanitizer._HTTP_DATE_FORMAT)

        @classmethod
        def parse_http_header(cls, date_str: str | None, url: ParsedUrl) -> Self | None:
            if parsed_date := cls._parse_http_date_header(
                date_str, url.with_redacted_auth_info()
            ):
                return cls(timestamp=parsed_date)
            return None

        @staticmethod
        def _parse_http_date_header(
            date_str: str | None,
            url: ParsedUrl,
        ) -> datetime.datetime | None:
            if date_str is None:
                logger.debug(
                    "No 'date' header was provided in response for url %s",
                    url,
                )
                return None

            try:
                parsed_timestamp = _TimestampSanitizer.ParsedDateHeader.parse(date_str)
            except _TimestampSanitizer.ParsedDateHeader.InvalidTimestamp:
                logger.exception(
                    "Received an unparseable 'date' header from the server: %r. "
                    "However, this may be a valid date format that pip should handle. "
                    "Please feel free to file a bug.",
                    date_str,
                )
                return None

            # strptime() doesn't set the timezone according to the parsed %Z arg, which
            # may be any of "UTC", "GMT", or any element of `time.tzname`.
            kind = parsed_timestamp.tz_kind()
            if kind == _TimestampSanitizer.TzKind.utc:
                logger.debug(
                    "A UTC timezone %r was provided "
                    "in the 'date' response header for url %s",
                    parsed_timestamp._tz_string,
                    url,
                )
                return parsed_timestamp.parsed_date.replace(
                    tzinfo=datetime.timezone.utc
                )
            if kind == _TimestampSanitizer.TzKind.local_matching:
                logger.debug(
                    "A local timezone %r was provided "
                    "in the 'date' response header for url %s",
                    parsed_timestamp._tz_string,
                    url,
                )
                return parsed_timestamp.parsed_date.replace(
                    tzinfo=_TimestampSanitizer._now_local().tz
                )
            assert kind == _TimestampSanitizer.TzKind.local_non_matching, kind
            logger.warning(
                "A local timezone %r was provided "
                "in the 'date' response header for url %s -- however, "
                "it differed from the timezone on the local machine %r by DST setting. "
                "This may be some sort of bug, "
                "but it is also a limitation of the current caching approach. "
                "Please file a bug.",
                parsed_timestamp._tz_string,
                url,
                _TimestampSanitizer._now_local().name,
            )
            return None
