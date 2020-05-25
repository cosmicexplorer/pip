import json
import logging

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.specifiers import SpecifierSet
from pip._vendor.packaging.version import Version
from pip._vendor.pkg_resources import Distribution

from pip._internal.models.link import Link
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.typing import MYPY_CHECK_RUNNING

from .base import Candidate
from .base import Requirement as BaseRequirement
from .candidates import (
    AlreadyInstalledCandidate,
    EditableCandidate,
    ExtrasCandidate,
    LinkCandidate,
    RequiresPythonCandidate,
)
from .requirements import (
    ExplicitRequirement,
    RequiresPythonRequirement,
    SpecifierRequirement,
)

if MYPY_CHECK_RUNNING:
    from typing import Any, Dict, Optional, Type

    from .factory import Factory

try:
    from json import JSONDecodeError
except ImportError:
    # PY2
    JSONDecodeError = ValueError  # type: ignore


logger = logging.getLogger(__name__)


# Persistent dependency caching across pip invocations:
class PersistentRequirementDependencyCache(object):
    def __init__(self, file_path, factory):
        # type: (str, Factory) -> None
        self._file_path = file_path
        self._factory = factory
        self._cache = None      # type: Optional[RequirementDependencyCache]

    def __enter__(self):
        # type: () -> RequirementDependencyCache
        try:
            with open(self._file_path, 'r') as f:
                cache_from_config = RequirementDependencyCache.deserialize(
                    f.read(),
                    factory=self._factory,
                )
        except (OSError, JSONDecodeError, EOFError) as e:
            # If the file does not exist, or the cache was not readable for any
            # reason, just start anew.
            logger.debug('error reading dependency cache: {}.'.format(e))
            cache_from_config = RequirementDependencyCache({})

        self._cache = cache_from_config
        return self._cache

    def __exit__(self, exc_type, exc_value, exc_tb):
        # type: (Type[Any], Any, Any) -> None
        try:
            with open(self._file_path, 'w') as f:
                assert self._cache is not None
                f.write(self._cache.serialize())
        finally:
            self._cache = None


if MYPY_CHECK_RUNNING:
    from typing import List, Union

    RequirementsCacheDict = Dict[str, List[BaseRequirement]]

    SerdeCand = Union[
        AlreadyInstalledCandidate,
        EditableCandidate,
        ExtrasCandidate,
        LinkCandidate,
        RequiresPythonCandidate,
    ]


class RequirementDependencyCache(object):
    def __init__(self, cache_from_config):
        # type: (RequirementsCacheDict) -> None
        self._cache = cache_from_config

    def add_dependency_links(self, candidate, dependencies):
        # type: (Candidate, List[BaseRequirement]) -> None
        key = json.dumps(self._serialize_candidate(candidate))
        prev_deps = self._cache.get(key, None)

        if prev_deps is not None:
            assert dependencies == prev_deps, (
                'dependencies: {}, prev_deps: {}'
                .format(dependencies, prev_deps))
        self._cache[key] = dependencies

    def get(self, candidate):
        # type: (Candidate) -> Optional[List[BaseRequirement]]
        key = json.dumps(self._serialize_candidate(candidate))
        return self._cache.get(key, None)

    def serialize(self):
        # type: () -> str
        return json.dumps({
            cand: [
                self._serialize_requirement(dep_req)
                for dep_req in deps
            ]
            for cand, deps in self._cache.items()
        })

    def _serialize_candidate(self, cand):
        # type: (Candidate) -> Dict[str, Any]
        if isinstance(cand, AlreadyInstalledCandidate):
            tag = 'already-installed'
            payload = {
                'dist_location': cand.dist.location,
                'requirement': str(cand._ireq.req),
            }
        elif isinstance(cand, EditableCandidate):
            tag = 'editable'
            payload = {
                'link': cand.link.url,
                'requirement': str(cand._ireq.req),
                'name': cand.name,
                'version': str(cand.version),
            }
        elif isinstance(cand, ExtrasCandidate):
            tag = 'extras'
            payload = {
                'base': self._serialize_candidate(cand.base),
                'extras': list(cand.extras),
            }
        elif isinstance(cand, LinkCandidate):
            tag = 'link'
            payload = {
                'link': cand.link.url,
                'requirement': str(cand._ireq.req),
                'name': cand.name,
                'version': str(cand.version),
            }
        else:
            assert isinstance(cand, RequiresPythonCandidate), cand
            tag = 'requires-python'
            payload = {
                'version': str(cand.version),
            }

        return dict(tag=tag, payload=payload)

    @classmethod
    def _deserialize_candidate(cls, info, factory):
        # type: (Dict[str, Any], Factory) -> SerdeCand
        tag = info['tag']
        payload = info['payload']

        if tag == 'already-installed':
            dist_location = payload['dist_location']
            requirement = Requirement(payload['requirement'])
            dist = Distribution(
                location=dist_location,
                project_name=requirement.name,
                # TODO: make a better error message (or default value) if there
                # was no specifier!
                version=list(requirement.specifier)[0].version,
            )
            return AlreadyInstalledCandidate(
                dist=dist,
                # FIXME: this is using the same requirement as the `parent`
                # arg!
                parent=InstallRequirement(requirement, None),
                factory=factory,
            )
        elif tag == 'editable':
            link = payload['link']
            requirement = Requirement(payload['requirement'])
            name = payload['name']
            version = (Version(payload['version'])
                       if payload['version']
                       else None)
            return EditableCandidate(
                link=Link(link),
                parent=InstallRequirement(requirement, None),
                factory=factory,
                name=name,
                version=version,
            )
        elif tag == 'extras':
            base = cls._deserialize_candidate(payload['base'], factory)
            # NB: This removes any ordering that may be present in the
            # serialized version!
            extras = set(payload['extras'])
            return ExtrasCandidate(base=base, extras=extras)  # type: ignore
        elif tag == 'link':
            link = payload['link']
            requirement = Requirement(payload['requirement'])
            name = payload['name']
            version = (Version(payload['version'])
                       if payload['version']
                       else None)
            return LinkCandidate(
                link=Link(link),
                parent=InstallRequirement(requirement, None),
                factory=factory,
                name=name,
                version=version,
            )
        else:
            assert tag == 'requires-python', tag
            version = Version(payload['version'])
            return RequiresPythonCandidate(version.release)

    def _serialize_requirement(self, req):
        # type: (BaseRequirement) -> Dict[str, Any]
        if isinstance(req, ExplicitRequirement):
            tag = 'explicit'
            payload = self._serialize_candidate(req.candidate)
        elif isinstance(req, RequiresPythonRequirement):
            tag = 'requires-python'
            payload = {
                'specifier': str(req.specifier),
                'candidate': self._serialize_candidate(req._candidate),
            }
        else:
            assert isinstance(req, SpecifierRequirement), req
            tag = 'specifier'
            payload = {
                'requirement': str(req._ireq.req)
            }
        return dict(tag=tag, payload=payload)

    @classmethod
    def _deserialize_requirement(cls, info, factory):
        # type: (Dict[str, Any], Factory) -> BaseRequirement
        tag = info['tag']
        payload = info['payload']

        if tag == 'explicit':
            candidate = cls._deserialize_candidate(payload, factory)
            return ExplicitRequirement(candidate)
        elif tag == 'requires-python':
            specifier = SpecifierSet(payload['specifier'])
            candidate = cls._deserialize_candidate(payload['candidate'],
                                                   factory)
            return RequiresPythonRequirement(specifier, candidate)
        else:
            assert tag == 'specifier', tag
            req = Requirement(payload['requirement'])
            return SpecifierRequirement(InstallRequirement(req, None), factory)

    @classmethod
    def deserialize(cls, s, factory):
        # type: (str, Factory) -> RequirementDependencyCache
        return cls({
            cand: [
                cls._deserialize_requirement(dep_req, factory)
                for dep_req in deps
            ]
            for cand, deps in json.loads(s).items()
        })
