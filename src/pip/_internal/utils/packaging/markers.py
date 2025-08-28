from __future__ import annotations

import functools
import operator
from dataclasses import dataclass
from enum import Enum
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    TypedDict,
    cast,
)

from pip._vendor.packaging._parser import MarkerAtom, MarkerList, Op, Value, Variable
from pip._vendor.packaging._parser import parse_marker as _parse_marker
from pip._vendor.packaging._tokenizer import ParserSyntaxError
from pip._vendor.packaging.markers import (
    InvalidMarker,
    UndefinedComparison,
    format_full_version,
)
from pip._vendor.packaging.specifiers import InvalidSpecifier
from pip._vendor.packaging.utils import canonicalize_name

from .specifiers import Specifier
from .version import ParsedVersion

if TYPE_CHECKING:
    from collections.abc import Set
    from types import ModuleType

    from typing_extensions import Self

__all__ = [
    "Marker",
    "Environment",
    "EvaluateContext",
]


class EvaluateContext(Enum):
    METADATA = "metadata"
    LOCK_FILE = "lock_file"
    REQUIREMENT = "requirement"


class Environment(TypedDict):
    implementation_name: str
    """The implementation's identifier, e.g. ``'cpython'``."""

    implementation_version: ParsedVersion
    """
    The implementation's version, e.g. ``'3.13.0a2'`` for CPython 3.13.0a2, or
    ``'7.3.13'`` for PyPy3.10 v7.3.13.
    """

    os_name: str
    """
    The value of :py:data:`os.name`. The name of the operating system dependent module
    imported, e.g. ``'posix'``.
    """

    platform_machine: str
    """
    Returns the machine type, e.g. ``'i386'``.

    An empty string if the value cannot be determined.
    """

    platform_release: str
    """
    The system's release, e.g. ``'2.2.0'`` or ``'NT'``.

    An empty string if the value cannot be determined.
    """

    platform_system: str
    """
    The system/OS name, e.g. ``'Linux'``, ``'Windows'`` or ``'Java'``.

    An empty string if the value cannot be determined.
    """

    platform_version: str
    """
    The system's release version, e.g. ``'#3 on degas'``.

    An empty string if the value cannot be determined.
    """

    python_full_version: ParsedVersion
    """
    The Python version as string ``'major.minor.patchlevel'``.

    Note that unlike the Python :py:data:`sys.version`, this value will always include
    the patchlevel (it defaults to 0).
    """

    platform_python_implementation: str
    """
    A string identifying the Python implementation, e.g. ``'CPython'``.
    """

    python_version: ParsedVersion
    """The Python version as string ``'major.minor'``."""

    sys_platform: str
    """
    This string contains a platform identifier that can be used to append
    platform-specific components to :py:data:`sys.path`, for instance.

    For Unix systems, except on Linux and AIX, this is the lowercased OS name as
    returned by ``uname -s`` with the first part of the version as returned by
    ``uname -r`` appended, e.g. ``'sunos5'`` or ``'freebsd8'``, at the time when Python
    was built.
    """


class _EnvConfigure:
    @staticmethod
    def _full_version(ver: str) -> ParsedVersion:
        """
        Work around platform.python_version() returning something that is not PEP 440
        compliant for non-tagged Python builds.
        """
        if ver.endswith("+"):
            ver += "local"
        return ParsedVersion.parse(ver)

    @staticmethod
    def configure(os: ModuleType, platform: ModuleType, sys: ModuleType) -> Environment:
        iver = format_full_version(sys.implementation.version)
        return {
            "implementation_name": sys.implementation.name,
            "implementation_version": ParsedVersion.parse(iver),
            "os_name": os.name,
            "platform_machine": platform.machine(),
            "platform_release": platform.release(),
            "platform_system": platform.system(),
            "platform_version": platform.version(),
            "python_full_version": _EnvConfigure._full_version(
                platform.python_version()
            ),
            "platform_python_implementation": platform.python_implementation(),
            "python_version": ParsedVersion.parse(
                ".".join(platform.python_version_tuple()[:2])
            ),
            "sys_platform": sys.platform,
        }

    @staticmethod
    @functools.cache
    def default() -> Environment:
        import os
        import platform
        import sys

        return _EnvConfigure.configure(os, platform, sys)


class _OpEvaluator:
    _version_match_operators: ClassVar[
        dict[str, Callable[[ParsedVersion, ParsedVersion], bool]]
    ] = {
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }

    _operators: ClassVar[dict[str, Callable[[str, str | Set[str]], bool]]] = {
        "in": lambda lhs, rhs: lhs in rhs,
        "not in": lambda lhs, rhs: lhs not in rhs,
        "<": operator.lt,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
        ">=": operator.ge,
        ">": operator.gt,
    }

    @classmethod
    def _eval_op(
        cls,
        lhs: str | ParsedVersion,
        op: Op,
        rhs: str | ParsedVersion | Set[str],
    ) -> bool:
        if isinstance(lhs, ParsedVersion):
            assert isinstance(rhs, (str, ParsedVersion)), rhs
            try:
                spec = Specifier.parse("".join([op.serialize(), str(rhs)]))
            except InvalidSpecifier:
                raise UndefinedComparison(
                    f"Undefined version matching op {op!r} on {lhs!r} and {rhs!r}."
                )
            else:
                return spec.contains(lhs, prereleases=True)

        if isinstance(rhs, ParsedVersion):
            if isinstance(lhs, str):
                lhs = ParsedVersion.parse(lhs)
            if v_oper := cls._version_match_operators.get(op.serialize()):
                return v_oper(lhs, rhs)
            raise UndefinedComparison(
                f"Undefined version comparison op {op!r} on {lhs!r} and {rhs!r}."
            )

        if oper := cls._operators.get(op.serialize()):
            return oper(lhs, rhs)
        raise UndefinedComparison(f"Undefined {op!r} on {lhs!r} and {rhs!r}.")

    _markers_allowing_set: ClassVar[frozenset[str]] = frozenset(
        ["extras", "dependency_groups"]
    )

    @classmethod
    def _normalize(
        cls,
        lhs: str | ParsedVersion,
        rhs: str | ParsedVersion | Set[str],
        key: str,
    ) -> tuple[str | ParsedVersion, str | ParsedVersion | Set[str]]:
        # PEP 685 – Comparison of extra names for optional distribution dependencies
        # https://peps.python.org/pep-0685/
        # > When comparing extra names, tools MUST normalize the names being
        # > compared using the semantics outlined in PEP 503 for names
        if key == "extra":
            assert isinstance(lhs, str), lhs
            assert isinstance(rhs, str), "extra value must be a string"
            return (canonicalize_name(lhs), canonicalize_name(rhs))
        if key in cls._markers_allowing_set:
            assert isinstance(lhs, str), lhs
            if isinstance(rhs, str):  # pragma: no cover
                return (canonicalize_name(lhs), canonicalize_name(rhs))
            assert not isinstance(rhs, ParsedVersion), rhs
            return (canonicalize_name(lhs), {canonicalize_name(v) for v in rhs})
        # other environment markers don't have such standards
        return lhs, rhs

    @classmethod
    def evaluate_markers(
        cls,
        markers: MarkerList,
        environment: dict[str, str | ParsedVersion | Set[str]],
    ) -> bool:
        groups: list[list[bool]] = [[]]

        for marker in markers:
            assert isinstance(marker, (list, tuple, str))

            if isinstance(marker, list):
                groups[-1].append(cls.evaluate_markers(marker, environment))
            elif isinstance(marker, tuple):
                lhs, op, rhs = marker

                environment_key: str
                if isinstance(lhs, Variable):
                    environment_key = lhs.value
                    lhs_value = environment[environment_key]
                    rhs_value = rhs.value
                else:
                    lhs_value = lhs.value
                    environment_key = rhs.value
                    rhs_value = environment[environment_key]
                assert isinstance(lhs_value, (str, ParsedVersion)), lhs_value

                lhs_value, rhs_value = cls._normalize(
                    lhs_value, rhs_value, key=environment_key
                )

                groups[-1].append(cls._eval_op(lhs_value, op, rhs_value))
            else:
                assert marker in ["and", "or"]
                if marker == "or":
                    groups.append([])

        return any(all(item) for item in groups)


@dataclass(frozen=True)
class Marker:
    _markers: MarkerList

    __slots__ = ["_markers", "__dict__"]

    @classmethod
    def parse(cls, marker: str) -> Self:
        # Note: We create a Marker object without calling this constructor in
        #       packaging.requirements.Requirement. If any additional logic is
        #       added here, make sure to mirror/adapt Requirement.
        try:
            markers = cls._normalize_extra_values(_parse_marker(marker))
        except ParserSyntaxError as e:
            raise InvalidMarker(str(e)) from e
        return cls(_markers=markers)

    def evaluate(
        self,
        environment: dict[str, str | ParsedVersion | Set[str]] | None = None,
        context: EvaluateContext = EvaluateContext.METADATA,
    ) -> bool:
        """Evaluate a marker.

        Return the boolean from evaluating the given marker against the
        environment. environment is an optional argument to override all or
        part of the determined environment. The *context* parameter specifies what
        context the markers are being evaluated for, which influences what markers
        are considered valid. Acceptable values are "metadata" (for core metadata;
        default), "lock_file", and "requirement" (i.e. all other situations).

        The environment is determined from the current Python process.
        """
        current_environment = cast(
            "dict[str, str | ParsedVersion | Set[str]]", _EnvConfigure.default()
        )
        if context == EvaluateContext.LOCK_FILE:
            current_environment.update(
                extras=frozenset(), dependency_groups=frozenset()
            )
        elif context == EvaluateContext.METADATA:
            current_environment["extra"] = ""
        if environment is not None:
            current_environment.update(environment)
            # The API used to allow setting extra to None. We need to handle this
            # case for backwards compatibility.
            if "extra" in current_environment and current_environment["extra"] is None:
                current_environment["extra"] = ""

        return _OpEvaluator.evaluate_markers(self._markers, current_environment)

    @functools.cached_property
    def _str(self) -> str:
        return self.__class__._format_marker(self._markers)

    def __str__(self) -> str:
        return self._str

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}.parse('{self}')"

    @functools.cached_property
    def _hash(self) -> int:
        return hash((self.__class__.__name__, str(self)))

    def __hash__(self) -> int:
        return self._hash

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return hash(self) == hash(other) and str(self) == str(other)

    @staticmethod
    def _normalize_extra_values(results: MarkerList) -> MarkerList:
        """
        Normalize extra values.
        """
        if isinstance(results[0], tuple):
            lhs, op, rhs = results[0]
            if isinstance(lhs, Variable) and lhs.value == "extra":
                normalized_extra = canonicalize_name(rhs.value)
                rhs = Value(normalized_extra)
            elif isinstance(rhs, Variable) and rhs.value == "extra":
                normalized_extra = canonicalize_name(lhs.value)
                lhs = Value(normalized_extra)
            results[0] = lhs, op, rhs  # type: ignore[index]
        return results

    @classmethod
    def _format_marker(
        cls, marker: list[str] | MarkerAtom | str, first: bool | None = True
    ) -> str:
        assert isinstance(marker, (list, tuple, str))

        # Sometimes we have a structure like [[...]] which is a single item list
        # where the single item is itself it's own list. In that case we want skip
        # the rest of this function so that we don't get extraneous () on the
        # outside.
        if (
            isinstance(marker, list)
            and len(marker) == 1
            and isinstance(marker[0], (list, tuple))
        ):
            return cls._format_marker(marker[0])

        if isinstance(marker, list):
            inner = (cls._format_marker(m, first=False) for m in marker)
            if first:
                return " ".join(inner)
            else:
                return "(" + " ".join(inner) + ")"
        elif isinstance(marker, tuple):
            return " ".join([m.serialize() for m in marker])
        else:
            return marker
