"""Routines related to PyPI, indexes"""

from __future__ import annotations

import abc
import enum
import functools
import itertools
import logging
import re
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from typing import (
    TYPE_CHECKING,
    ClassVar,
)

from pip._vendor.packaging.specifiers import InvalidSpecifier
from pip._vendor.packaging.tags import parse_tag
from pip._vendor.packaging.utils import BuildTag, NormalizedName, canonicalize_name
from pip._vendor.packaging.version import InvalidVersion

from pip._internal.exceptions import (
    BestVersionAlreadyInstalled,
    DistributionNotFound,
    InvalidWheelFilename,
)
from pip._internal.index.collector import LinkCollector, parse_links
from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.format_control import (
    AllowedFormats,
    FormatControlBuilder,
)
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import WheelInfo
from pip._internal.req import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.filetypes import FileExtensions
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import build_netloc
from pip._internal.utils.packaging.specifiers import BaseSpecifier, SpecifierSet
from pip._internal.utils.packaging.version import ParsedVersion
from pip._internal.utils.predicates import FragmentMatcher, RequiresPython
from pip._internal.utils.unpacking import SUPPORTED_EXTENSIONS

if TYPE_CHECKING:
    from typing_extensions import Self, TypeGuard

    CandidateSortingKey = tuple[int, int, int, ParsedVersion, int | None, BuildTag, int]

__all__ = ["BestCandidateResult", "PackageFinder"]


logger = getLogger(__name__)


class LinkType(enum.Enum):
    candidate = enum.auto()
    different_project = enum.auto()
    yanked = enum.auto()
    format_unsupported = enum.auto()
    format_invalid = enum.auto()
    platform_mismatch = enum.auto()
    requires_python_mismatch = enum.auto()


class EvaluationResult(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def kind(self) -> LinkType: ...


@dataclass(frozen=True)
class _FoundCandidate(EvaluationResult):
    version: str

    kind: ClassVar[LinkType] = LinkType.candidate


class EvaluationFailure(EvaluationResult):
    @abc.abstractmethod
    def detail(self) -> str: ...

    def __str__(self) -> str:
        return self.detail()


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
class LinkEvaluator:
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
        "__dict__",
    ]

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

    @functools.cached_property
    def _evaluations(self) -> dict[Link, EvaluationResult]:
        return {}

    def evaluate_link(self, link: Link) -> EvaluationResult:
        if (result := self._evaluations.get(link)) is not None:
            return result
        new_result = self._do_evaluate_link(link)
        self._evaluations[link] = new_result
        return new_result

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

    def _do_evaluate_link(self, link: Link) -> EvaluationResult:
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

        return _FoundCandidate(version)


@dataclass(frozen=True, slots=True)
class _CandidateHashFilter:
    project_name: str
    hashes: Hashes | None

    @dataclass(frozen=True)
    class HashMatches:
        match_count: int
        matches_or_no_digest: tuple[InstallationCandidate, ...]
        non_matches: tuple[InstallationCandidate, ...]
        candidates: tuple[InstallationCandidate, ...]

        __slots__ = [
            "match_count",
            "matches_or_no_digest",
            "non_matches",
            "candidates",
            "__dict__",
        ]

        @functools.cached_property
        def filtered(self) -> tuple[InstallationCandidate, ...]:
            if self.match_count:
                return self.matches_or_no_digest
            return self.candidates

        @dataclass(frozen=True)
        class NonMatches:
            non_matches: tuple[InstallationCandidate, ...]

            def __str__(self) -> str:
                n = len(self.non_matches)
                if not n:
                    return "discarding no candidates"
                joined = "\n  ".join(str(c.link) for c in self.non_matches)
                return f"discarding {n} non-matches:\n  {joined}"

        def discarded_candidates(self) -> NonMatches:
            if len(self.filtered) == len(self.candidates):
                return self.__class__.NonMatches(())
            return self.__class__.NonMatches(self.non_matches)

        @classmethod
        def calculate(
            cls, hashes: Hashes, candidates: tuple[InstallationCandidate, ...]
        ) -> Self:
            matches_or_no_digest = []
            # Collect the non-matches for logging purposes.
            non_matches = []
            match_count = 0
            for candidate in candidates:
                link = candidate.link
                if not link.has_hash:
                    pass
                elif link.is_hash_allowed(hashes=hashes):
                    match_count += 1
                else:
                    non_matches.append(candidate)
                    continue
                matches_or_no_digest.append(candidate)
            return cls(
                match_count=match_count,
                matches_or_no_digest=tuple(matches_or_no_digest),
                non_matches=tuple(non_matches),
                candidates=candidates,
            )

    def filter_unallowed_hashes(
        self,
        candidates: Iterable[InstallationCandidate],
    ) -> tuple[InstallationCandidate, ...]:
        """
        Filter out candidates whose hashes aren't allowed, and return a new
        list of candidates.

        If at least one candidate has an allowed hash, then all candidates with
        either an allowed hash or no hash specified are returned.  Otherwise,
        the given candidates are returned.

        Including the candidates with no hash specified when there is a match
        allows a warning to be logged if there is a more preferred candidate
        with no hash specified.  Returning all candidates in the case of no
        matches lets pip report the hash of the candidate that would otherwise
        have been installed (e.g. permitting the user to more easily update
        their requirements file with the desired hash).
        """
        # Make sure we're not returning back the given value.
        candidates = tuple(candidates)
        if not self.hashes:
            logger.debug(
                "Given no hashes to check %s links for project %r: "
                "discarding no candidates",
                len(candidates),
                self.project_name,
            )
            return candidates
        assert self.hashes is not None

        hash_matches = self.__class__.HashMatches.calculate(self.hashes, candidates)

        logger.debug(
            "Checked %s links for project %r against %s hashes "
            "(%s matches, %s no digest): %s",
            len(hash_matches.candidates),
            self.project_name,
            self.hashes.digest_count,
            hash_matches.match_count,
            len(hash_matches.matches_or_no_digest) - hash_matches.match_count,
            hash_matches.discarded_candidates(),
        )

        return hash_matches.filtered


@dataclass
class CandidatePreferences:
    """
    Encapsulates some of the preferences for filtering and sorting
    InstallationCandidate objects.
    """

    prefer_binary: bool = False
    allow_all_prereleases: bool = False


@dataclass(frozen=True)
class BestCandidateResult:
    """A collection of candidates, returned by `PackageFinder.find_best_candidate`.

    This class is only intended to be instantiated by CandidateEvaluator's
    `compute_best_candidate()` method.

    :param all_candidates: A sequence of all available candidates found.
    :param applicable_candidates: The applicable candidates.
    :param best_candidate: The most preferred candidate found, or None
        if no applicable candidates were found.
    """

    all_candidates: tuple[InstallationCandidate, ...]
    applicable_candidates: tuple[InstallationCandidate, ...]
    best_candidate: InstallationCandidate | None


@dataclass(frozen=True)
class CandidateEvaluator:
    """
    Responsible for filtering and sorting candidates for installation based
    on what tags are valid.

    :param _supported_tags: The PEP 425 tags supported by the target
        Python in order of preference (most preferred first).
    """

    _project_name: str
    _target_python: TargetPython
    _specifier: BaseSpecifier
    _prefer_binary: bool = False
    _allow_all_prereleases: bool = False
    _hashes: Hashes | None = None

    @classmethod
    def create(
        cls,
        project_name: str,
        target_python: TargetPython | None = None,
        prefer_binary: bool = False,
        allow_all_prereleases: bool = False,
        specifier: BaseSpecifier | None = None,
        hashes: Hashes | None = None,
    ) -> CandidateEvaluator:
        """Create a CandidateEvaluator object.

        :param target_python: The target Python interpreter to use when
            checking compatibility. If None (the default), a TargetPython
            object will be constructed from the running Python.
        :param specifier: An optional object implementing `filter`
            (e.g. `packaging.specifiers.SpecifierSet`) to filter applicable
            versions.
        :param hashes: An optional collection of allowed hashes.
        """
        if target_python is None:
            target_python = TargetPython.create()
        if specifier is None:
            specifier = SpecifierSet.empty()

        return cls(
            _project_name=project_name,
            _target_python=target_python,
            _specifier=specifier,
            _prefer_binary=prefer_binary,
            _allow_all_prereleases=allow_all_prereleases,
            _hashes=hashes,
        )

    @functools.cached_property
    def _candidate_hash_filter(self) -> _CandidateHashFilter:
        return _CandidateHashFilter(self._project_name, self._hashes)

    def get_applicable_candidates(
        self,
        candidates: Iterable[InstallationCandidate],
    ) -> tuple[InstallationCandidate, ...]:
        """
        Return the applicable candidates from a list of candidates.
        """
        # Using None infers from the specifier instead.
        allow_prereleases = self._allow_all_prereleases or None

        # We turn the version object into a str here because otherwise
        # when we're debundled but setuptools isn't, Python will see
        # packaging.version.Version and
        # pkg_resources._vendor.packaging.version.Version as different
        # types. This way we'll use a str as a common data interchange
        # format. If we stop using the pkg_resources provided specifier
        # and start using our own, we can drop the cast to str().
        applicable_candidates = self._specifier.filter_arg(
            candidates,
            key=lambda c: c.version,
            prereleases=allow_prereleases,
        )
        filtered_applicable_candidates = (
            self._candidate_hash_filter.filter_unallowed_hashes(applicable_candidates)
        )

        return tuple(
            sorted(
                filtered_applicable_candidates,
                key=self._sort_key,
            )
        )

    @functools.cached_property
    def _cur_key_cache(self) -> dict[InstallationCandidate, CandidateSortingKey]:
        return {}

    def _sort_key(self, candidate: InstallationCandidate) -> CandidateSortingKey:
        if (result := self._cur_key_cache.get(candidate)) is not None:
            return result
        new_result = self._do_sort_key(candidate)
        self._cur_key_cache[candidate] = new_result
        return new_result

    def _do_sort_key(self, candidate: InstallationCandidate) -> CandidateSortingKey:
        """
        Function to pass as the `key` argument to a call to sorted() to sort
        InstallationCandidates by preference.

        Returns a tuple such that tuples sorting as greater using Python's
        default comparison operator are more preferred.

        The preference is as follows:

        First and foremost, candidates with allowed (matching) hashes are
        always preferred over candidates without matching hashes. This is
        because e.g. if the only candidate with an allowed hash is yanked,
        we still want to use that candidate.

        Second, excepting hash considerations, candidates that have been
        yanked (in the sense of PEP 592) are always less preferred than
        candidates that haven't been yanked. Then:

        If not finding wheels, they are sorted by version only.
        If finding wheels, then the sort order is by version, then:
          1. existing installs
          2. wheels ordered via Wheel.support_index_min(self._supported_tags)
          3. source archives
        If prefer_binary was set, then all wheels are sorted above sources.
        Egg fragments are sorted below links without them.

        Note: it was considered to embed this logic into the Link
              comparison operators, but then different sdist links
              with the same version, would have to be considered equal
        """
        build_tag: BuildTag = ()
        binary_preference = 0
        link = candidate.link
        if link.is_wheel:
            # can raise InvalidWheelFilename
            wheel = WheelInfo.parse_filename(link.filename)
            pri = -wheel.support_index_min(self._target_python)
            if self._prefer_binary:
                binary_preference = 1
            build_tag = wheel.build_tag
        else:  # sdist
            pri = -len(self._target_python.sorted_tags)
        has_allowed_hash = int(link.is_hash_allowed(self._hashes))
        yank_value = -1 * int(link.is_yanked)  # -1 for yanked.
        egg_value = -1 * int(link.egg_fragment is not None)
        return (
            has_allowed_hash,
            yank_value,
            binary_preference,
            candidate.version,
            pri,
            build_tag,
            egg_value,
        )

    def compute_best_candidate(
        self,
        candidates: Iterable[InstallationCandidate],
    ) -> BestCandidateResult:
        """
        Compute and return a `BestCandidateResult` instance.
        """
        candidates = tuple(candidates)
        applicable_candidates = self.get_applicable_candidates(candidates)

        if applicable_candidates:
            best_candidate = applicable_candidates[-1]
        else:
            best_candidate = None

        return BestCandidateResult(
            candidates,
            applicable_candidates=applicable_candidates,
            best_candidate=best_candidate,
        )


class PackageFinder:
    """This finds packages.

    This is meant to match easy_install's technique for looking for
    packages, by reading pages and looking for appropriate links.
    """

    def __init__(
        self,
        link_collector: LinkCollector,
        target_python: TargetPython,
        allow_yanked: bool,
        format_control: FormatControlBuilder | None = None,
        candidate_prefs: CandidatePreferences | None = None,
        ignore_requires_python: bool | None = None,
    ) -> None:
        """
        This constructor is primarily meant to be used by the create() class
        method and from tests.

        :param format_control: A FormatControl object, used to control
            the selection of source packages / binary packages when consulting
            the index and links.
        :param candidate_prefs: Options to use when creating a
            CandidateEvaluator object.
        """
        if candidate_prefs is None:
            candidate_prefs = CandidatePreferences()

        format_control = format_control or FormatControlBuilder(set(), set())

        self._allow_yanked = allow_yanked
        self._candidate_prefs = candidate_prefs
        self._ignore_requires_python = ignore_requires_python
        self._link_collector = link_collector
        self._target_python = target_python

        self.format_control = format_control.build()

        # These are boring links that have already been logged somehow.
        self._logged_links: set[Link] = set()
        self._requires_python_skipped_reasons: list[EvaluationFailure] = []

    # Don't include an allow_yanked default value to make sure each call
    # site considers whether yanked releases are allowed. This also causes
    # that decision to be made explicit in the calling code, which helps
    # people when reading the code.
    @classmethod
    def create(
        cls,
        link_collector: LinkCollector,
        selection_prefs: SelectionPreferences,
        target_python: TargetPython | None = None,
    ) -> PackageFinder:
        """Create a PackageFinder.

        :param selection_prefs: The candidate selection preferences, as a
            SelectionPreferences object.
        :param target_python: The target Python interpreter to use when
            checking compatibility. If None (the default), a TargetPython
            object will be constructed from the running Python.
        """
        if target_python is None:
            target_python = TargetPython.create()

        candidate_prefs = CandidatePreferences(
            prefer_binary=selection_prefs.prefer_binary,
            allow_all_prereleases=selection_prefs.allow_all_prereleases,
        )

        return cls(
            candidate_prefs=candidate_prefs,
            link_collector=link_collector,
            target_python=target_python,
            allow_yanked=selection_prefs.allow_yanked,
            format_control=selection_prefs.format_control,
            ignore_requires_python=selection_prefs.ignore_requires_python,
        )

    @property
    def target_python(self) -> TargetPython:
        return self._target_python

    @property
    def search_scope(self) -> SearchScope:
        return self._link_collector.search_scope

    @search_scope.setter
    def search_scope(self, search_scope: SearchScope) -> None:
        self._link_collector.search_scope = search_scope

    @property
    def find_links(self) -> list[str]:
        return self._link_collector.find_links

    @property
    def index_urls(self) -> list[str]:
        return self.search_scope.index_urls

    @property
    def proxy(self) -> str | None:
        return self._link_collector.session.pip_proxy

    @property
    def trusted_hosts(self) -> Iterable[str]:
        for host_port in self._link_collector.session.pip_trusted_origins:
            yield build_netloc(*host_port)

    @property
    def custom_cert(self) -> str | None:
        # session.verify is either a boolean (use default bundle/no SSL
        # verification) or a string path to a custom CA bundle to use. We only
        # care about the latter.
        verify = self._link_collector.session.verify
        return verify if isinstance(verify, str) else None

    @property
    def client_cert(self) -> str | None:
        cert = self._link_collector.session.cert
        assert not isinstance(cert, tuple), "pip only supports PEM client certs"
        return cert

    @property
    def allow_all_prereleases(self) -> bool:
        return self._candidate_prefs.allow_all_prereleases

    def set_allow_all_prereleases(self) -> None:
        self._candidate_prefs.allow_all_prereleases = True

    @property
    def prefer_binary(self) -> bool:
        return self._candidate_prefs.prefer_binary

    def set_prefer_binary(self) -> None:
        self._candidate_prefs.prefer_binary = True

    def requires_python_skipped_reasons(self) -> EvalFailureSet:
        return EvalFailureSet(tuple(self._requires_python_skipped_reasons))

    def make_link_evaluator(self, project_name: str) -> LinkEvaluator:
        return LinkEvaluator.create(
            project_name=project_name,
            formats=self.format_control.get_allowed_formats(project_name),
            target_python=self._target_python,
            allow_yanked=self._allow_yanked,
            ignore_requires_python=self._ignore_requires_python,
        )

    def _sort_links(self, links: Iterable[Link]) -> tuple[Link, ...]:
        """
        Returns elements of links in order, non-egg links first, egg links
        second, while eliminating duplicates
        """
        eggs, no_eggs = [], []
        seen: set[Link] = set()
        for link in links:
            if link not in seen:
                seen.add(link)
                if link.egg_fragment:
                    eggs.append(link)
                else:
                    no_eggs.append(link)
        return tuple(no_eggs) + tuple(eggs)

    def _log_skipped_link(self, link: Link, result: EvaluationFailure) -> None:
        if link not in self._logged_links:
            # Put the link at the end so the reason is more visible and because
            # the link string is usually very long.
            logger.debug("Skipping link: %s: %s", result, link)
            self._logged_links.add(link)
        if result.kind == LinkType.requires_python_mismatch:
            self._requires_python_skipped_reasons.append(result)

    def get_install_candidate(
        self, link_evaluator: LinkEvaluator, link: Link
    ) -> InstallationCandidate | None:
        """
        If the link is a candidate for install, convert it to an
        InstallationCandidate and return it. Otherwise, return None.
        """
        result = link_evaluator.evaluate_link(link)
        if result.kind != LinkType.candidate:
            assert isinstance(result, EvaluationFailure), result
            self._log_skipped_link(link, result)
            return None
        assert isinstance(result, _FoundCandidate), result
        version = result.version

        try:
            return InstallationCandidate(
                name=link_evaluator.project_name,
                version=ParsedVersion.parse(version),
                link=link,
            )
        except InvalidVersion:
            return None

    def _evaluate_links(
        self, link_evaluator: LinkEvaluator, links: Iterable[Link]
    ) -> list[InstallationCandidate]:
        """
        Convert links that are candidates to InstallationCandidate objects.
        """
        candidates = []
        for link in self._sort_links(links):
            candidate = self.get_install_candidate(link_evaluator, link)
            if candidate is not None:
                candidates.append(candidate)

        return candidates

    def process_project_url(
        self, project_url: Link, link_evaluator: LinkEvaluator
    ) -> list[InstallationCandidate]:
        logger.debug(
            "Fetching project page and analyzing links: %s",
            project_url,
        )
        index_response = self._link_collector.fetch_response(project_url)
        if index_response is None:
            return []

        page_links = list(parse_links(index_response))

        with indent_log():
            package_links = self._evaluate_links(
                link_evaluator,
                links=page_links,
            )

        return package_links

    @functools.cached_property
    def _all_candidates(self) -> dict[str, list[InstallationCandidate]]:
        return {}

    def find_all_candidates(self, project_name: str) -> list[InstallationCandidate]:
        if (result := self._all_candidates.get(project_name)) is not None:
            return result
        new_result = self._do_find_all_candidates(project_name)
        self._all_candidates[project_name] = new_result
        return new_result

    def _do_find_all_candidates(self, project_name: str) -> list[InstallationCandidate]:
        """Find all available InstallationCandidate for project_name

        This checks index_urls and find_links.
        All versions found are returned as an InstallationCandidate list.

        See LinkEvaluator.evaluate_link() for details on which files
        are accepted.
        """
        link_evaluator = self.make_link_evaluator(project_name)

        collected_sources = self._link_collector.collect_sources(
            project_name=project_name,
            candidates_from_page=functools.partial(
                self.process_project_url,
                link_evaluator=link_evaluator,
            ),
        )

        page_candidates_it = itertools.chain.from_iterable(
            source.page_candidates()
            for sources in collected_sources
            for source in sources
            if source is not None
        )
        page_candidates = list(page_candidates_it)

        file_links_it = itertools.chain.from_iterable(
            source.file_links()
            for sources in collected_sources
            for source in sources
            if source is not None
        )
        file_candidates = self._evaluate_links(
            link_evaluator,
            sorted(file_links_it, reverse=True),
        )

        if logger.isEnabledFor(logging.DEBUG) and file_candidates:
            paths = []
            for candidate in file_candidates:
                assert candidate.link.url  # we need to have a URL
                try:
                    paths.append(candidate.link.file_path)
                except Exception:
                    paths.append(candidate.link.url)  # it's not a local file

            logger.debug("Local files found: %s", ", ".join(paths))

        # This is an intentional priority ordering
        return file_candidates + page_candidates

    def make_candidate_evaluator(
        self,
        project_name: str,
        specifier: BaseSpecifier | None = None,
        hashes: Hashes | None = None,
    ) -> CandidateEvaluator:
        """Create a CandidateEvaluator object to use."""
        candidate_prefs = self._candidate_prefs
        return CandidateEvaluator.create(
            project_name=project_name,
            target_python=self._target_python,
            prefer_binary=candidate_prefs.prefer_binary,
            allow_all_prereleases=candidate_prefs.allow_all_prereleases,
            specifier=specifier,
            hashes=hashes,
        )

    @functools.cached_property
    def _best_candidates(
        self,
    ) -> defaultdict[
        str,
        dict[tuple[BaseSpecifier | None, Hashes | None], BestCandidateResult],
    ]:
        return defaultdict(dict)

    def find_best_candidate(
        self,
        project_name: str,
        specifier: BaseSpecifier | None = None,
        hashes: Hashes | None = None,
    ) -> BestCandidateResult:
        sub_cache = self._best_candidates[project_name]
        if (result := sub_cache.get((specifier, hashes))) is not None:
            return result
        new_result = self._do_find_best_candidate(project_name, specifier, hashes)
        sub_cache[(specifier, hashes)] = new_result
        return new_result

    def _do_find_best_candidate(
        self,
        project_name: str,
        specifier: BaseSpecifier | None = None,
        hashes: Hashes | None = None,
    ) -> BestCandidateResult:
        """Find matches for the given project and specifier.

        :param specifier: An optional object implementing `filter`
            (e.g. `packaging.specifiers.SpecifierSet`) to filter applicable
            versions.

        :return: A `BestCandidateResult` instance.
        """
        candidates = self.find_all_candidates(project_name)
        candidate_evaluator = self.make_candidate_evaluator(
            project_name=project_name,
            specifier=specifier,
            hashes=hashes,
        )
        return candidate_evaluator.compute_best_candidate(candidates)

    @staticmethod
    def _format_versions(cand_iter: Iterable[InstallationCandidate]) -> str:
        # This repeated parse_version and str() conversion is needed to
        # handle different vendoring sources from pip and pkg_resources.
        # If we stop using the pkg_resources provided specifier and start
        # using our own, we can drop the cast to str().
        versions = frozenset(c.version for c in cand_iter)
        if not versions:
            return "none"
        return ", ".join(map(str, sorted(versions)))

    def find_requirement(
        self, req: InstallRequirement, upgrade: bool
    ) -> InstallationCandidate | None:
        """Try to find a Link matching req

        Expects req, an InstallRequirement and upgrade, a boolean
        Returns a InstallationCandidate if found,
        Raises DistributionNotFound or BestVersionAlreadyInstalled otherwise
        """
        name = req.name
        assert name is not None, "find_requirement() called with no name"

        hashes = req.hashes(trust_internet=False)
        best_candidate_result = self.find_best_candidate(
            name,
            specifier=req.specifier,
            hashes=hashes,
        )
        best_candidate = best_candidate_result.best_candidate

        installed_version: ParsedVersion | None = None
        if req.satisfied_by is not None:
            installed_version = req.satisfied_by.version

        if installed_version is None and best_candidate is None:
            logger.critical(
                "Could not find a version that satisfies the requirement %s "
                "(from versions: %s)",
                req,
                self.__class__._format_versions(best_candidate_result.all_candidates),
            )

            raise DistributionNotFound(f"No matching distribution found for {req}")

        def _should_install_candidate(
            candidate: InstallationCandidate | None,
        ) -> TypeGuard[InstallationCandidate]:
            if installed_version is None:
                return True
            if best_candidate is None:
                return False
            return best_candidate.version > installed_version

        if not upgrade and installed_version is not None:
            if _should_install_candidate(best_candidate):
                logger.debug(
                    "Existing installed version (%s) satisfies requirement "
                    "(most up-to-date version is %s)",
                    installed_version,
                    best_candidate.version,
                )
            else:
                logger.debug(
                    "Existing installed version (%s) is most up-to-date and "
                    "satisfies requirement",
                    installed_version,
                )
            return None

        if _should_install_candidate(best_candidate):
            logger.debug(
                "Using version %s (newest of versions: %s)",
                best_candidate.version,
                self.__class__._format_versions(
                    best_candidate_result.applicable_candidates
                ),
            )
            return best_candidate

        # We have an existing version, and its the best version
        logger.debug(
            "Installed version (%s) is most up-to-date (past versions: %s)",
            installed_version,
            self.__class__._format_versions(
                best_candidate_result.applicable_candidates
            ),
        )
        raise BestVersionAlreadyInstalled
