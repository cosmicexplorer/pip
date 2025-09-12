from __future__ import annotations

import abc
import enum
import functools
import re
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
)

from pip._vendor.packaging.specifiers import InvalidSpecifier
from pip._vendor.packaging.tags import parse_tag
from pip._vendor.packaging.utils import NormalizedName, canonicalize_name

from pip._internal.cache import SerializableEntry
from pip._internal.exceptions import InvalidWheelFilename
from pip._internal.models.format_control import AllowedFormats
from pip._internal.models.link import Link
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import WheelInfo
from pip._internal.utils._log import getLogger
from pip._internal.utils.filetypes import FileExtensions
from pip._internal.utils.predicates import FragmentMatcher, RequiresPython
from pip._internal.utils.unpacking import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from typing_extensions import Self


logger = getLogger(__name__)


### Classification for `Link`s to dists (which may be local or remote).
###
### This classification method *could* be done via exceptions, but pip processes quite
### a few `Link`s over a run, so instead we return a result type and explicitly
### propagate the early return.
## Union tag to identify the result.
class LinkType(enum.Enum):
    candidate = enum.auto()
    different_project = enum.auto()
    yanked = enum.auto()
    format_unsupported = enum.auto()
    format_invalid = enum.auto()
    platform_mismatch = enum.auto()
    requires_python_mismatch = enum.auto()


## Basic result wrapper. Currently there is only failure and success.
class EvaluationResult(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def kind(self) -> LinkType: ...


## Successful result!
@dataclass(frozen=True)
class FoundCandidate(EvaluationResult):
    version: str

    kind: ClassVar[LinkType] = LinkType.candidate


## Basic error result, with a method detailing the error.
##
## The __str__() method is used to delay generating string output for logging until
## necessary, or not at all, depending on the log level.
##
## This allows for complex logic for useful debugging, without e.g. performing URL
## parsing or sorting large lists of output until needed.
class EvaluationFailure(EvaluationResult):
    @abc.abstractmethod
    def detail(self) -> str: ...

    def __str__(self) -> str:
        return self.detail()


## This avoids sorting or even evaluating the list of errors until __str__() is called.
@dataclass(frozen=True)
class EvalFailureSet:
    failures: tuple[EvaluationFailure, ...]

    def __bool__(self) -> bool:
        return len(self.failures) > 0

    def __str__(self) -> str:
        if not self.failures:
            return "none"
        return "; ".join(sorted(map(str, self.failures)))


class _InvalidWheelFilename(EvaluationFailure):
    kind: ClassVar[LinkType] = LinkType.format_invalid

    def detail(self) -> str:
        return "invalid wheel filename"


@dataclass(frozen=True)
class _WrongName(EvaluationFailure):
    project_name: str

    kind: ClassVar[LinkType] = LinkType.different_project

    def detail(self) -> str:
        return f"wrong project name (not {self.project_name})"


@dataclass(frozen=True)
class _UnsupportedTags(EvaluationFailure):
    info: WheelInfo

    kind: ClassVar[LinkType] = LinkType.platform_mismatch

    def detail(self) -> str:
        # Include the wheel's tags in the reason string to
        # simplify troubleshooting compatibility issues.
        file_tags = ", ".join(self.info.sorted_tag_strings)
        return (
            f"none of the wheel's tags ({file_tags}) are compatible "
            "(run pip debug --verbose to show compatible tags)"
        )


class _PyVerMismatch(EvaluationFailure):
    kind: ClassVar[LinkType] = LinkType.platform_mismatch

    def detail(self) -> str:
        return "Python version is incorrect"


@dataclass(frozen=True)
class _PyTagParseError(EvaluationFailure):
    py_tag_string: str

    kind: ClassVar[LinkType] = LinkType.platform_mismatch

    def detail(self) -> str:
        return f"Could not parse python tags: {self.py_tag_string!r}"


class _NotAFile(EvaluationFailure):
    kind: ClassVar[LinkType] = LinkType.format_unsupported

    def detail(self) -> str:
        return "not a file"


@dataclass(frozen=True)
class _UnsupportedArchive(EvaluationFailure):
    ext: str

    kind: ClassVar[LinkType] = LinkType.format_unsupported

    def detail(self) -> str:
        return f"unsupported archive format: {self.ext}"


@dataclass(frozen=True)
class _NoBinariesAllowed(EvaluationFailure):
    project_name: str

    kind: ClassVar[LinkType] = LinkType.format_unsupported

    def detail(self) -> str:
        return f"No binaries permitted for {self.project_name}"


class _MacOSX10(EvaluationFailure):
    kind: ClassVar[LinkType] = LinkType.format_unsupported

    def detail(self) -> str:
        return "macosx10 one"


@dataclass(frozen=True)
class _NoSourcesAllowed(EvaluationFailure):
    project_name: str

    kind: ClassVar[LinkType] = LinkType.format_unsupported

    def detail(self) -> str:
        return f"No sources permitted for {self.project_name}"


@dataclass(frozen=True)
class _MissingVersion(EvaluationFailure):
    project_name: str

    kind: ClassVar[LinkType] = LinkType.format_invalid

    def detail(self) -> str:
        return f"Missing project version for {self.project_name}"


@dataclass(frozen=True)
class _YankedReason(EvaluationFailure):
    yanked_reason: str | None

    kind: ClassVar[LinkType] = LinkType.yanked

    def detail(self) -> str:
        reason = self.yanked_reason or "<none given>"
        return f"yanked for reason: {reason}"


@dataclass(frozen=True)
class _IncompatibleRequiresPython(EvaluationFailure):
    version: str
    predicate: RequiresPython

    kind: ClassVar[LinkType] = LinkType.requires_python_mismatch

    def detail(self) -> str:
        descriptor = self.predicate.to_sorted_string()
        return f"{self.version} Requires-Python {descriptor}"


@dataclass(frozen=True)
class LinkEvaluator(SerializableEntry):
    """
    Responsible for evaluating links for a particular project.
    """

    project_name: str
    formats: AllowedFormats
    target_python: TargetPython
    allow_yanked: bool
    ignore_requires_python: bool

    __slots__ = [
        "project_name",
        "formats",
        "target_python",
        "allow_yanked",
        "ignore_requires_python",
    ]

    @classmethod
    def suffix(cls) -> str:
        return ".evaluation"

    def serialize(self) -> dict[str, Any]:
        return {
            # "project_name": self.project_name,
            "canonical_name": self._canonical_name,
            "formats": self.formats.value,
            "target_python": tuple(map(str, self.target_python.sorted_tags)),
            "allow_yanked": self.allow_yanked,
            "ignore_requires_python": self.ignore_requires_python,
        }

    # Don't include an allow_yanked default value to make sure each call
    # site considers whether yanked releases are allowed. This also causes
    # that decision to be made explicit in the calling code, which helps
    # people when reading the code.
    @classmethod
    def create(
        cls,
        project_name: str,
        formats: AllowedFormats,
        target_python: TargetPython,
        allow_yanked: bool,
        ignore_requires_python: bool | None = None,
    ) -> Self:
        """
        :param project_name: The user supplied package name.
        :param formats: The formats allowed for this package.
        :param target_python: The target Python interpreter to use when
            evaluating link compatibility. This is used, for example, to
            check wheel compatibility, as well as when checking the Python
            version, e.g. the Python version embedded in a link filename
            (or egg fragment) and against an HTML link's optional PEP 503
            "data-requires-python" attribute.
        :param allow_yanked: Whether files marked as yanked (in the sense
            of PEP 592) are permitted to be candidates for install.
        :param ignore_requires_python: Whether to ignore incompatible
            PEP 503 "data-requires-python" values in HTML links. Defaults
            to False.
        """
        return cls(
            project_name=project_name,
            formats=formats,
            target_python=target_python,
            allow_yanked=allow_yanked,
            ignore_requires_python=(
                False if ignore_requires_python is None else ignore_requires_python
            ),
        )

    _py_version_re: ClassVar[re.Pattern[str]] = re.compile(r"-py(?:.+)$")
    _single_py_version: ClassVar[re.Pattern[str]] = re.compile(r"py([1-3](?:\.[0-9])?)")

    @functools.cached_property
    def _canonical_name(self) -> NormalizedName:
        return canonicalize_name(self.project_name)

    def _parse_wheel(
        self, filename: str
    ) -> tuple[str | None, EvaluationFailure | None]:
        try:
            wheel = WheelInfo.parse_filename(filename)
        except InvalidWheelFilename:
            return None, _InvalidWheelFilename()
        if wheel.name != self._canonical_name:
            return None, _WrongName(self.project_name)

        # FIXME: optimize this!
        if not wheel.supported(self.target_python):
            return None, _UnsupportedTags(wheel)
        return wheel.version, None

    def _split_version_and_py(
        self, version: str
    ) -> tuple[str | None, EvaluationFailure | None]:
        py_ver_match = self._py_version_re.search(version)
        if not py_ver_match:
            return version, None
        version = version[: py_ver_match.start()]
        py_tag_string = py_ver_match.group(0)[1:]

        if single_version := self._single_py_version.match(py_tag_string):
            if single_version.group(1) != self.target_python.py_version:
                return None, _PyVerMismatch()
            return version, None

        try:
            py_tag = parse_tag(py_tag_string)
        except ValueError:
            return None, _PyTagParseError(py_tag_string)
        raise ValueError(f"unexpected tag string: {py_tag!r} from version {version!r}")

    @functools.cached_property
    def _fragment_matcher(self) -> FragmentMatcher:
        return FragmentMatcher(self._canonical_name)

    def _parse_version(self, link: Link) -> tuple[str | None, EvaluationFailure | None]:
        version: str | None = None

        if link.egg_fragment:
            egg_info = link.egg_fragment
            ext = link.ext
        else:
            egg_info, ext = link.splitext()
            if not ext:
                return None, _NotAFile()
            if ext not in SUPPORTED_EXTENSIONS:
                return None, _UnsupportedArchive(ext)
            if (
                not self.formats.allows_binary()
                and ext == FileExtensions.WHEEL_EXTENSION
            ):
                return None, _NoBinariesAllowed(self.project_name)
            if "macosx10" in link.path and ext == ".zip":
                return None, _MacOSX10()
            if ext == FileExtensions.WHEEL_EXTENSION:
                version, result = self._parse_wheel(link.filename)
                if result is not None:
                    return None, result

        # This should be up by the self.ok_binary check, but see issue 2700.
        if not self.formats.allows_source() and ext != FileExtensions.WHEEL_EXTENSION:
            return None, _NoSourcesAllowed(self.project_name)

        if version is None:
            version = self._fragment_matcher.extract_version_from_fragment(
                egg_info,
            )
        if version is None:
            return None, _MissingVersion(self.project_name)

        version, result = self._split_version_and_py(version)
        if result is not None:
            return None, result
        assert version is not None
        # FIXME: optimize the version canonicalization--do it only once!
        return version, None

    @staticmethod
    def _requires_python_is_compatible(
        target_python: TargetPython, ignore_requires_python: bool, link: Link
    ) -> tuple[bool, RequiresPython | None]:
        if link.requires_python is None:
            return True, None
        try:
            predicate = RequiresPython.parse(link.requires_python)
        except InvalidSpecifier:
            logger.debug(
                "Ignoring invalid Requires-Python (%r) for link: %s",
                link.requires_python,
                link,
            )
            return True, None
        if predicate.is_compatible(target_python):
            return True, None

        if ignore_requires_python:
            logger.debug(
                "Ignoring failed Requires-Python check (%s not in: %r) for link: %s",
                target_python.full_py_version,
                link.requires_python,
                link,
            )
            return True, None

        logger.verbose(
            "Link requires a different Python (%s not in: %r): %s",
            target_python.full_py_version,
            link.requires_python,
            link,
        )
        return False, predicate

    def evaluate_link(self, link: Link) -> EvaluationResult:
        """
        Determine whether a link is a candidate for installation.

        :return: A tuple (result, detail), where *result* is an enum
            representing whether the evaluation found a candidate, or the reason
            why one is not found. If a candidate is found, *detail* will be the
            candidate's version string; if one is not found, it contains the
            reason the link fails to qualify.
        """
        version: str | None = None
        if link.is_yanked and not self.allow_yanked:
            return _YankedReason(link.yanked_reason or None)

        version, result = self._parse_version(link)
        if result is not None:
            return result
        assert version is not None

        is_compatible, predicate = LinkEvaluator._requires_python_is_compatible(
            self.target_python, self.ignore_requires_python, link
        )
        if not is_compatible:
            assert predicate is not None
            return _IncompatibleRequiresPython(version, predicate)

        logger.debug("Found link %s, version: %s", link, version)

        return FoundCandidate(version)
