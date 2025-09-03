from __future__ import annotations

import abc
import datetime
import enum
import functools
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from pip._internal.utils.urls import ParsedUrl

if TYPE_CHECKING:
    from typing_extensions import Self

logger = logging.getLogger(__name__)


class _CacheableHTTPHeader(metaclass=abc.ABCMeta):
    @classmethod
    @abc.abstractmethod
    def response_header_name(cls) -> str: ...

    @abc.abstractmethod
    def request_caching_header_name(self) -> str: ...

    @abc.abstractmethod
    def cache_serialize(self) -> bytes: ...

    @classmethod
    @abc.abstractmethod
    def cache_deserialize(cls, data: bytes) -> Self: ...

    @abc.abstractmethod
    def format_http_header(self) -> str: ...

    @classmethod
    @abc.abstractmethod
    def parse_http_header(
        cls, header_str: str | None, url: ParsedUrl
    ) -> Self | None: ...


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
    class Epoch:
        epoch: int

        def __post_init__(self) -> None:
            assert self.epoch >= 0, self

        def __str__(self) -> str:
            return f"epoch({self.epoch})"

        @classmethod
        def from_timestamp(cls, timestamp: datetime.datetime) -> Self:
            assert timestamp.tzname() is not None, timestamp
            epoch = timestamp.timestamp()
            # The timestamp will only have second resolution according to the parse
            # format string _HTTP_DATE_FORMAT.
            assert not (epoch % 1), (epoch, timestamp)
            return cls(epoch=int(epoch))

        def to_timestamp(self) -> datetime.datetime:
            return datetime.datetime.fromtimestamp(self.epoch, tz=datetime.timezone.utc)

        def serialize(self) -> bytes:
            return self.epoch.to_bytes(length=4, byteorder="big", signed=False)

        @classmethod
        def deserialize(cls, data: bytes) -> Self:
            return cls(epoch=int.from_bytes(data, byteorder="big", signed=False))

    @dataclass(frozen=True, slots=True)
    class CacheableTimestamp(_CacheableHTTPHeader):
        epoch: _TimestampSanitizer.Epoch

        @classmethod
        def response_header_name(cls) -> str:
            return "Date"

        @classmethod
        def request_caching_header_name(cls) -> str:
            return "If-Modified-Since"

        def cache_serialize(self) -> bytes:
            return self.epoch.serialize()

        @classmethod
        def cache_deserialize(cls, data: bytes) -> Self:
            epoch = _TimestampSanitizer.Epoch.deserialize(data)
            return cls(epoch=epoch)

        @functools.cached_property
        def _as_datetime(self) -> datetime.datetime:
            return self.epoch.to_timestamp()

        @functools.cached_property
        def _as_http_formatted(self) -> str:
            return self._as_datetime.strftime(_TimestampSanitizer._HTTP_DATE_FORMAT)

        def __str__(self) -> str:
            return f"<{self.epoch}, synthesized({self._as_http_formatted!r})>"

        def format_http_header(self) -> str:
            return self._as_http_formatted

        @classmethod
        def parse_http_header(cls, date_str: str | None, url: ParsedUrl) -> Self | None:
            if parsed_date := cls._parse_http_date_header(date_str, url):
                epoch = _TimestampSanitizer.Epoch.from_timestamp(parsed_date)
                return cls(epoch=epoch)
            return None

        @staticmethod
        def _parse_http_date_header(
            date_str: str | None,
            url: ParsedUrl,
        ) -> datetime.datetime | None:
            # FIXME: make this type-safe!!!!!
            url = url.with_redacted_auth_info()
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


class _ETagSanitizer:
    @dataclass(frozen=True, slots=True)
    class CacheableETag(_CacheableHTTPHeader):
        etag: str

        # https://httpwg.org/specs/rfc9110.html#field.etag
        _strict_regex: ClassVar[re.Pattern[str]] = re.compile(
            "[\x21\x23-\x7e\x80-\xff]*"
        )

        def __post_init__(self) -> None:
            assert self.__class__._strict_regex.fullmatch(self.etag), self

        def __str__(self) -> str:
            return f"<etag({self.etag!r})>"

        @classmethod
        def response_header_name(cls) -> str:
            return "ETag"

        @classmethod
        def request_caching_header_name(cls) -> str:
            return "If-None-Match"

        def cache_serialize(self) -> bytes:
            return self.etag.encode("utf-8")

        @classmethod
        def cache_deserialize(cls, data: bytes) -> Self:
            return cls(etag=data.decode("utf-8"))

        def format_http_header(self) -> str:
            return f'"{self.etag}"'

        _etag_regex: ClassVar[re.Pattern[str]] = re.compile(
            r'(?P<weak>W/)?"(?P<tag>.*)"',
            flags=re.DOTALL,
        )

        _http_spec_url: ClassVar[ParsedUrl] = ParsedUrl.parse(
            "https://httpwg.org/specs/rfc9110.html#field.etag"
        )

        @classmethod
        def parse_http_header(cls, etag_str: str | None, url: ParsedUrl) -> Self | None:
            # FIXME: make this type-safe!!!!!
            url = url.with_redacted_auth_info()
            if etag_str is None:
                logger.debug(
                    "No 'etag' header was provided in response for url %s",
                    url,
                )
                return None

            if m := cls._etag_regex.fullmatch(etag_str):
                g = m.groupdict()
                if g["weak"]:
                    logger.debug(
                        "Received a weak etag value %r in response for url %s",
                        etag_str,
                        url,
                    )
                etag = g["tag"]
                if cls._strict_regex.fullmatch(etag):
                    return cls(etag=etag)
                logger.error(
                    "'etag' value %r (from header string %r) in response for url %s "
                    "contained invalid or deprecated characters "
                    "-- should match pattern %s. "
                    "See specification: %s",
                    etag,
                    etag_str,
                    url,
                    cls._strict_regex,
                    cls._http_spec_url,
                )
                return None

            logger.error(
                "Failed to parse 'etag' header %r in response for url %s "
                "-- should match pattern %s, "
                "with characters additionally restricted to %s. "
                "See specification: %s",
                etag_str,
                url,
                cls._etag_regex,
                cls._strict_regex,
                cls._http_spec_url,
            )
            return None
