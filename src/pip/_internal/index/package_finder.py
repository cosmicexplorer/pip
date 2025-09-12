"""Routines related to PyPI, indexes"""

from __future__ import annotations

import dataclasses
import functools
from collections import defaultdict
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
)

from pip._vendor.packaging.utils import BuildTag, NormalizedName, canonicalize_name
from pip._vendor.packaging.version import InvalidVersion

from pip._internal.cache import FetchResolveCache
from pip._internal.exceptions import (
    BestVersionAlreadyInstalled,
    DistributionNotFound,
)
from pip._internal.index.collector import (
    ComputeMerger,
    FetchComputeProcess,
    LinkCollector,
)
from pip._internal.models.candidate import CandidateJsonCodec, InstallationCandidate
from pip._internal.models.format_control import (
    FormatControl,
    FormatControlBuilder,
)
from pip._internal.models.link import Link
from pip._internal.models.search_scope import SearchScope
from pip._internal.models.selection_prefs import SelectionPreferences
from pip._internal.models.target_python import TargetPython
from pip._internal.models.wheel import WheelInfo
from pip._internal.req import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.cache_serde import Bz2TextStream, JsonCodec
from pip._internal.utils.hashes import Hashes
from pip._internal.utils.misc import build_netloc
from pip._internal.utils.packaging.specifiers import BaseSpecifier, SpecifierSet
from pip._internal.utils.packaging.version import ParsedVersion

from .hash_filter import CandidateHashFilter
from .link_evaluator import (
    EvalFailureSet,
    EvaluationFailure,
    FoundCandidate,
    LinkEvaluator,
    LinkType,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from typing_extensions import TypeGuard

    CandidateSortingKey = tuple[int, int, int, ParsedVersion, int | None, BuildTag, int]

__all__ = ["FormatControl", "BestCandidateResult", "PackageFinder"]


logger = getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ComputeLinkEvaluation(
    FetchComputeProcess["Iterable[Link]", tuple[InstallationCandidate, ...]]
):
    link_evaluator: LinkEvaluator
    _requires_python_skipped_reasons: list[EvaluationFailure] = field(
        default_factory=list
    )

    def _log_skipped_link(self, link: Link, result: EvaluationFailure) -> None:
        # Put the link at the end so the reason is more visible and because
        # the link string is usually very long.
        logger.debug("Skipping link: %s: %s", result, link)
        if result.kind == LinkType.requires_python_mismatch:
            self._requires_python_skipped_reasons.append(result)

    def requires_python_skipped_reasons(self) -> EvalFailureSet:
        return EvalFailureSet(tuple(self._requires_python_skipped_reasons))

    def get_install_candidate(self, link: Link) -> InstallationCandidate | None:
        """
        If the link is a candidate for install, convert it to an
        InstallationCandidate and return it. Otherwise, return None.
        """
        result = self.link_evaluator.evaluate_link(link)
        if result.kind != LinkType.candidate:
            assert isinstance(result, EvaluationFailure), result
            self._log_skipped_link(link, result)
            return None
        assert isinstance(result, FoundCandidate), result
        version = result.version

        try:
            return InstallationCandidate(
                name=self.link_evaluator.project_name,
                version=ParsedVersion.parse(version),
                link=link,
            )
        except InvalidVersion:
            return None

    def _evaluate_links(self, links: Iterable[Link]) -> Iterator[InstallationCandidate]:
        """
        Convert links that are candidates to InstallationCandidate objects.
        """
        seen: set[Link] = set()
        for link in links:
            if link in seen:
                continue
            seen.add(link)
            if (candidate := self.get_install_candidate(link)) is not None:
                yield candidate

    def compute(self, links: Iterable[Link]) -> tuple[InstallationCandidate, ...]:
        return tuple(self._evaluate_links(links))


class CandidateMerger(ComputeMerger[tuple[InstallationCandidate, ...]]):
    def merge(
        self, results: Iterable[tuple[InstallationCandidate, ...]]
    ) -> tuple[InstallationCandidate, ...]:
        return tuple(cand for res in results for cand in res)


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
    def _candidate_hash_filter(self) -> CandidateHashFilter:
        return CandidateHashFilter(self._project_name, self._hashes)

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
        fetch_resolve_cache: FetchResolveCache | None = None,
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

        self._fetch_resolve_cache = fetch_resolve_cache

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
        fetch_resolve_cache: FetchResolveCache | None = None,
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
            fetch_resolve_cache=fetch_resolve_cache,
        )

    @property
    def target_python(self) -> TargetPython:
        return self._target_python

    @property
    def search_scope(self) -> SearchScope:
        return self._link_collector.search_scope

    @search_scope.setter
    def search_scope(self, search_scope: SearchScope) -> None:
        self._link_collector = dataclasses.replace(
            self._link_collector, search_scope=search_scope
        )

    @property
    def find_links(self) -> tuple[str, ...]:
        return tuple(str(u.url) for u in self._link_collector.search_scope.find_links)

    @property
    def index_urls(self) -> tuple[str, ...]:
        return tuple(str(u.url) for u in self.search_scope.index_urls)

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

    def make_link_evaluator(self, project_name: str) -> LinkEvaluator:
        return LinkEvaluator.create(
            project_name=project_name,
            formats=self.format_control.get_allowed_formats(project_name),
            target_python=self._target_python,
            allow_yanked=self._allow_yanked,
            ignore_requires_python=self._ignore_requires_python,
        )

    @functools.cached_property
    def _all_candidates(
        self,
    ) -> dict[NormalizedName, tuple[tuple[InstallationCandidate, ...], EvalFailureSet]]:
        return {}

    def find_all_candidates(
        self, name: str
    ) -> tuple[tuple[InstallationCandidate, ...], EvalFailureSet]:
        project_name = canonicalize_name(name)
        if (result := self._all_candidates.get(project_name)) is not None:
            return result
        new_result = self._do_find_all_candidates(project_name)
        self._all_candidates[project_name] = new_result
        return new_result

    def _do_find_all_candidates(
        self, project_name: NormalizedName
    ) -> tuple[tuple[InstallationCandidate, ...], EvalFailureSet]:
        """Find all available InstallationCandidate for project_name

        This checks index_urls and find_links.
        All versions found are returned as an InstallationCandidate list.

        See LinkEvaluator.evaluate_link() for details on which files
        are accepted.
        """
        link_evaluator = self.make_link_evaluator(project_name)
        process = JsonCodec(codec=CandidateJsonCodec(), stream=Bz2TextStream())
        compute = ComputeLinkEvaluation(link_evaluator=link_evaluator)
        merger = CandidateMerger()

        results = self._link_collector.collect_and_process(
            project_name,
            link_evaluator,
            process,
            compute,
            merger,
        )
        reasons = compute.requires_python_skipped_reasons()
        return results, reasons

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
        candidates, _ = self.find_all_candidates(project_name)
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
