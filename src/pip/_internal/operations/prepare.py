"""Prepares a distribution for installation
"""

# The following comment should be removed at some point in the future.
# mypy: strict-optional=False

import email.errors
import gzip
import json
import mimetypes
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from pip._vendor.packaging.utils import canonicalize_name
from pip._vendor.requests.exceptions import InvalidSchema

from pip._internal.cache import LinkMetadataCache, should_cache
from pip._internal.distributions import make_distribution_for_install_requirement
from pip._internal.exceptions import (
    DirectoryUrlHashUnsupported,
    HashMismatch,
    HashUnpinned,
    InstallationError,
    MetadataInconsistent,
    NetworkConnectionError,
    VcsHashUnsupported,
)
from pip._internal.index.package_finder import PackageFinder
from pip._internal.metadata import (
    BaseDistribution,
    get_metadata_distribution,
)
from pip._internal.models.direct_url import ArchiveInfo
from pip._internal.models.link import Link
from pip._internal.models.wheel import Wheel
from pip._internal.network.download import BatchDownloader, Downloader
from pip._internal.network.lazy_wheel import (
    HTTPRangeRequestUnsupported,
    dist_from_wheel_url,
)
from pip._internal.network.session import PipSession
from pip._internal.operations.build.build_tracker import BuildTracker
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils._log import getLogger
from pip._internal.utils.direct_url_helpers import (
    direct_url_for_editable,
    direct_url_from_link,
)
from pip._internal.utils.hashes import Hashes, MissingHashes
from pip._internal.utils.logging import indent_log
from pip._internal.utils.misc import (
    display_path,
    hash_file,
    hide_url,
)
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.unpacking import unpack_file
from pip._internal.vcs import vcs

logger = getLogger(__name__)


def _get_prepared_distribution(
    req: InstallRequirement,
    build_tracker: BuildTracker,
    finder: PackageFinder,
    build_isolation: bool,
    check_build_deps: bool,
) -> Tuple[bool, BaseDistribution]:
    """Prepare a distribution for installation."""
    abstract_dist = make_distribution_for_install_requirement(req)
    tracker_id = abstract_dist.build_tracker_id
    builds_metadata = tracker_id is not None
    if builds_metadata:
        with build_tracker.track(req, tracker_id):
            abstract_dist.prepare_distribution_metadata(
                finder, build_isolation, check_build_deps
            )
    return (builds_metadata, abstract_dist.get_metadata_distribution())


def unpack_vcs_link(link: Link, location: str, verbosity: int) -> None:
    vcs_backend = vcs.get_backend_for_scheme(link.scheme)
    assert vcs_backend is not None
    vcs_backend.unpack(location, url=hide_url(link.url), verbosity=verbosity)


class File:
    def __init__(self, path: str, content_type: Optional[str]) -> None:
        self.path = path
        if content_type is None:
            self.content_type = mimetypes.guess_type(path)[0]
        else:
            self.content_type = content_type


def get_http_url(
    link: Link,
    download: Downloader,
    download_dir: Optional[str] = None,
    hashes: Optional[Hashes] = None,
) -> File:
    temp_dir = TempDirectory(kind="unpack", globally_managed=True)
    # If a download dir is specified, is the file already downloaded there?
    already_downloaded_path = None
    if download_dir:
        already_downloaded_path = _check_download_dir(link, download_dir, hashes)

    if already_downloaded_path:
        from_path = already_downloaded_path
        content_type = None
    else:
        # let's download to a tmp dir
        from_path, content_type = download(link, temp_dir.path)
        if hashes:
            hashes.check_against_path(from_path)

    return File(from_path, content_type)


def get_file_url(
    link: Link, download_dir: Optional[str] = None, hashes: Optional[Hashes] = None
) -> File:
    """Get file and optionally check its hash."""
    # If a download dir is specified, is the file already there and valid?
    already_downloaded_path = None
    if download_dir:
        already_downloaded_path = _check_download_dir(link, download_dir, hashes)

    if already_downloaded_path:
        from_path = already_downloaded_path
    else:
        from_path = link.file_path

    # If --require-hashes is off, `hashes` is either empty, the
    # link's embedded hash, or MissingHashes; it is required to
    # match. If --require-hashes is on, we are satisfied by any
    # hash in `hashes` matching: a URL-based or an option-based
    # one; no internet-sourced hash will be in `hashes`.
    if hashes:
        hashes.check_against_path(from_path)
    return File(from_path, None)


def unpack_url(
    link: Link,
    location: str,
    download: Downloader,
    verbosity: int,
    download_dir: Optional[str] = None,
    hashes: Optional[Hashes] = None,
) -> Optional[File]:
    """Unpack link into location, downloading if required.

    :param hashes: A Hashes object, one of whose embedded hashes must match,
        or HashMismatch will be raised. If the Hashes is empty, no matches are
        required, and unhashable types of requirements (like VCS ones, which
        would ordinarily raise HashUnsupported) are allowed.
    """
    # non-editable vcs urls
    if link.is_vcs:
        unpack_vcs_link(link, location, verbosity=verbosity)
        return None

    assert not link.is_existing_dir()

    # file urls
    if link.is_file:
        file = get_file_url(link, download_dir, hashes=hashes)

    # http urls
    else:
        file = get_http_url(
            link,
            download,
            download_dir,
            hashes=hashes,
        )

    # unpack the archive to the build dir location. even when only downloading
    # archives, they have to be unpacked to parse dependencies, except wheels
    if not link.is_wheel:
        unpack_file(file.path, location, file.content_type)

    return file


def _check_download_dir(
    link: Link,
    download_dir: str,
    hashes: Optional[Hashes],
    warn_on_hash_mismatch: bool = True,
) -> Optional[str]:
    """Check download_dir for previously downloaded file with correct hash
    If a correct file is found return its path else None

    If a file is found at the given path, but with an invalid hash, the file is deleted.
    """
    download_path = os.path.join(download_dir, link.filename)

    if not os.path.exists(download_path):
        return None

    # If already downloaded, does its hash match?
    logger.info("File was already downloaded %s", download_path)
    if hashes:
        try:
            hashes.check_against_path(download_path)
        except HashMismatch:
            if warn_on_hash_mismatch:
                logger.warning(
                    "Previously-downloaded file %s has bad hash. Re-downloading.",
                    download_path,
                )
            os.unlink(download_path)
            return None
    return download_path


@dataclass(frozen=True)
class CacheableDist:
    metadata: str
    filename: Path
    canonical_name: str

    @classmethod
    def from_dist(cls, link: Link, dist: BaseDistribution) -> "CacheableDist":
        """Extract the serializable data necessary to generate a metadata-only dist."""
        return cls(
            metadata=str(dist.metadata),
            filename=Path(link.filename),
            canonical_name=dist.canonical_name,
        )

    def to_dist(self) -> BaseDistribution:
        """Return a metadata-only dist from the deserialized cache entry."""
        return get_metadata_distribution(
            metadata_contents=self.metadata.encode("utf-8"),
            filename=str(self.filename),
            canonical_name=self.canonical_name,
        )

    def to_json(self) -> Dict[str, Any]:
        return {
            "metadata": self.metadata,
            "filename": str(self.filename),
            "canonical_name": self.canonical_name,
        }

    @classmethod
    def from_json(cls, args: Dict[str, Any]) -> "CacheableDist":
        return cls(
            metadata=args["metadata"],
            filename=Path(args["filename"]),
            canonical_name=args["canonical_name"],
        )


class RequirementPreparer:
    """Prepares a Requirement"""

    def __init__(  # noqa: PLR0913
        self,
        build_dir: str,
        download_dir: Optional[str],
        src_dir: str,
        build_isolation: bool,
        check_build_deps: bool,
        build_tracker: BuildTracker,
        session: PipSession,
        progress_bar: str,
        finder: PackageFinder,
        require_hashes: bool,
        use_user_site: bool,
        lazy_wheel: bool,
        verbosity: int,
        legacy_resolver: bool,
        metadata_cache: Optional[LinkMetadataCache] = None,
    ) -> None:
        super().__init__()

        self.src_dir = src_dir
        self.build_dir = build_dir
        self.build_tracker = build_tracker
        self._session = session
        self._download = Downloader(session, progress_bar)
        self._batch_download = BatchDownloader(session, progress_bar)
        self.finder = finder

        # Where still-packed archives should be written to. If None, they are
        # not saved, and are deleted immediately after unpacking.
        self.download_dir = download_dir

        # Is build isolation allowed?
        self.build_isolation = build_isolation

        # Should check build dependencies?
        self.check_build_deps = check_build_deps

        # Should hash-checking be required?
        self.require_hashes = require_hashes

        # Should install in user site-packages?
        self.use_user_site = use_user_site

        # Should wheels be downloaded lazily?
        self.use_lazy_wheel = lazy_wheel

        # How verbose should underlying tooling be?
        self.verbosity = verbosity

        # Are we using the legacy resolver?
        self.legacy_resolver = legacy_resolver

        # Memoized downloaded files, as mapping of url: path.
        self._downloaded: Dict[str, str] = {}

        # Previous "header" printed for a link-based InstallRequirement
        self._previous_requirement_header = ("", "")

        self._metadata_cache = metadata_cache

    def _log_preparing_link(self, req: InstallRequirement) -> None:
        """Provide context for the requirement being prepared."""
        if req.link.is_file and not req.is_wheel_from_cache:
            message = "Processing %s"
            information = str(display_path(req.link.file_path))
        else:
            message = "Collecting %s"
            information = str(req.req or req)

        # If we used req.req, inject requirement source if available (this
        # would already be included if we used req directly)
        if req.req and req.comes_from:
            if isinstance(req.comes_from, str):
                comes_from: Optional[str] = req.comes_from
            else:
                comes_from = req.comes_from.from_path()
            if comes_from:
                information += f" (from {comes_from})"

        if (message, information) != self._previous_requirement_header:
            self._previous_requirement_header = (message, information)
            logger.info(message, information)

        if req.is_wheel_from_cache:
            with indent_log():
                logger.info("Using cached %s", req.link.filename)

    def _ensure_link_req_src_dir(
        self, req: InstallRequirement, parallel_builds: bool
    ) -> None:
        """Ensure source_dir of a linked InstallRequirement."""
        # Since source_dir is only set for editable requirements.
        if req.link.is_wheel:
            # We don't need to unpack wheels, so no need for a source
            # directory.
            return
        assert req.source_dir is None
        if req.link.is_existing_dir():
            # build local directories in-tree
            req.source_dir = req.link.file_path
            return

        # We always delete unpacked sdists after pip runs.
        req.ensure_has_source_dir(
            self.build_dir,
            autodelete=True,
            parallel_builds=parallel_builds,
        )
        req.ensure_pristine_source_checkout()

    def _get_linked_req_hashes(self, req: InstallRequirement) -> Hashes:
        # By the time this is called, the requirement's link should have
        # been checked so we can tell what kind of requirements req is
        # and raise some more informative errors than otherwise.
        # (For example, we can raise VcsHashUnsupported for a VCS URL
        # rather than HashMissing.)
        if not self.require_hashes:
            return req.hashes(trust_internet=True)

        # We could check these first 2 conditions inside unpack_url
        # and save repetition of conditions, but then we would
        # report less-useful error messages for unhashable
        # requirements, complaining that there's no hash provided.
        if req.link.is_vcs:
            raise VcsHashUnsupported()
        if req.link.is_existing_dir():
            raise DirectoryUrlHashUnsupported()

        # Unpinned packages are asking for trouble when a new version
        # is uploaded.  This isn't a security check, but it saves users
        # a surprising hash mismatch in the future.
        # file:/// URLs aren't pinnable, so don't complain about them
        # not being pinned.
        if not req.is_direct and not req.is_pinned:
            raise HashUnpinned()

        # If known-good hashes are missing for this requirement,
        # shim it with a facade object that will provoke hash
        # computation and then raise a HashMissing exception
        # showing the user what the hash should be.
        return req.hashes(trust_internet=False) or MissingHashes()

    def _fetch_metadata_only(
        self,
        req: InstallRequirement,
    ) -> Optional[BaseDistribution]:
        if self.legacy_resolver:
            logger.debug(
                "Metadata-only fetching is not used in the legacy resolver",
            )
            return None
        if self.require_hashes:
            # Hash checking also means hashes are provided for all reqs, so no resolve
            # is necessary and metadata-only fetching provides no speedup.
            logger.debug(
                "Metadata-only fetching is not used as hash checking is required",
            )
            return None

        return (
            self._fetch_cached_metadata(req)
            or self._fetch_metadata_using_link_data_attr(req)
            or self._fetch_metadata_using_lazy_wheel(req)
        )

    def _locate_metadata_cache_entry(self, link: Link) -> Optional[Path]:
        """If the metadata cache is active, generate a filesystem path from the hash of
        the given Link."""
        if self._metadata_cache is None:
            return None

        return self._metadata_cache.cache_path(link)

    def _fetch_cached_metadata(
        self, req: InstallRequirement
    ) -> Optional[BaseDistribution]:
        cached_path = self._locate_metadata_cache_entry(req.link)
        if cached_path is None:
            return None

        # Quietly continue if the cache entry does not exist.
        if not os.path.isfile(cached_path):
            logger.debug(
                "no cached metadata for link %s at %s",
                req.link,
                cached_path,
            )
            return None

        try:
            with gzip.open(cached_path, mode="rt", encoding="utf-8") as f:
                logger.debug(
                    "found cached metadata for link %s at %s", req.link, f.name
                )
                args = json.load(f)
                cached_dist = CacheableDist.from_json(args)
                return cached_dist.to_dist()
        except (OSError, json.JSONDecodeError, KeyError) as e:
            logger.exception(
                "error reading cached metadata for link %s at %s %s(%s)",
                req.link,
                cached_path,
                e.__class__.__name__,
                str(e),
            )
            raise

    def _cache_metadata(
        self,
        req: InstallRequirement,
        metadata_dist: BaseDistribution,
    ) -> None:
        cached_path = self._locate_metadata_cache_entry(req.link)
        if cached_path is None:
            return

        # The cache file exists already, so we have nothing to do.
        if os.path.isfile(cached_path):
            logger.debug(
                "metadata for link %s is already cached at %s", req.link, cached_path
            )
            return

        # The metadata cache is split across several subdirectories, so ensure the
        # containing directory for the cache file exists before writing.
        os.makedirs(str(cached_path.parent), exist_ok=True)
        try:
            with gzip.open(cached_path, mode="wt", encoding="utf-8") as f:
                cacheable_dist = CacheableDist.from_dist(req.link, metadata_dist)
                args = cacheable_dist.to_json()
                logger.debug("caching metadata for link %s at %s", req.link, f.name)
                json.dump(args, f)
        except (OSError, email.errors.HeaderParseError) as e:
            # TODO: Some dists raise email.errors.HeaderParseError when calling str() or
            # bytes() on the metadata, which is an email.Message. This is probably a bug
            # in email parsing.
            logger.exception(
                "error caching metadata for dist %s from %s: %s(%s)",
                metadata_dist,
                req.link,
                e.__class__.__name__,
                str(e),
            )
            raise

    def _fetch_metadata_using_link_data_attr(
        self,
        req: InstallRequirement,
    ) -> Optional[BaseDistribution]:
        """Fetch metadata from the data-dist-info-metadata attribute, if possible."""
        # (1) Get the link to the metadata file, if provided by the backend.
        metadata_link = req.link.metadata_link()
        if metadata_link is None:
            return None
        assert req.req is not None
        logger.verbose(
            "Obtaining dependency information for %s from %s",
            req.req,
            metadata_link,
        )
        # (2) Download the contents of the METADATA file, separate from the dist itself.
        #     NB: this request will hit the CacheControl HTTP cache, which will be very
        #     quick since the METADATA file is very small. Therefore, we can rely on
        #     HTTP caching instead of LinkMetadataCache.
        metadata_file = get_http_url(
            metadata_link,
            self._download,
            hashes=metadata_link.as_hashes(),
        )
        with open(metadata_file.path, "rb") as f:
            metadata_contents = f.read()
        # (3) Generate a dist just from those file contents.
        metadata_dist = get_metadata_distribution(
            metadata_contents,
            req.link.filename,
            req.req.name,
        )
        # (4) Ensure the Name: field from the METADATA file matches the name from the
        #     install requirement.
        #
        #     NB: raw_name will fall back to the name from the install requirement if
        #     the Name: field is not present, but it's noted in the raw_name docstring
        #     that that should NEVER happen anyway.
        if canonicalize_name(metadata_dist.raw_name) != canonicalize_name(req.req.name):
            raise MetadataInconsistent(
                req, "Name", req.req.name, metadata_dist.raw_name
            )
        return metadata_dist

    def _fetch_metadata_using_lazy_wheel(
        self,
        req: InstallRequirement,
    ) -> Optional[BaseDistribution]:
        """Fetch metadata using lazy wheel, if possible."""
        # --use-feature=fast-deps must be provided.
        if not self.use_lazy_wheel:
            return None
        if req.link.is_file or not req.link.is_wheel:
            logger.debug(
                "Lazy wheel is not used as %r does not point to a remote wheel",
                req.link,
            )
            return None

        wheel = Wheel(req.link.filename)
        name = canonicalize_name(wheel.name)
        logger.info(
            "Obtaining dependency information from %s %s",
            name,
            wheel.version,
        )

        try:
            lazy_wheel_dist = dist_from_wheel_url(
                name, req.link.url_without_fragment, self._session
            )
        except HTTPRangeRequestUnsupported:
            logger.debug("%s does not support range requests", req.link)
            return None

        # If we've used the lazy wheel approach, then PEP 658 metadata is not available.
        # If the wheel is very large (>1GB), then retrieving it from the CacheControl
        # HTTP cache may take multiple seconds, even on a fast computer, and the
        # preparer will unnecessarily copy the cached response to disk before deleting
        # it at the end of the run. Caching the dist metadata in LinkMetadataCache means
        # later pip executions can retrieve metadata within milliseconds and avoid
        # thrashing the disk.
        self._cache_metadata(req, lazy_wheel_dist)
        return lazy_wheel_dist

    def _complete_partial_requirements(
        self,
        partially_downloaded_reqs: Iterable[InstallRequirement],
        parallel_builds: bool = False,
    ) -> None:
        """Download any requirements which were only fetched by metadata."""
        # Download to a temporary directory. These will be copied over as
        # needed for downstream 'download', 'wheel', and 'install' commands.
        temp_dir = TempDirectory(kind="unpack", globally_managed=True).path

        # Map each link to the requirement that owns it. This allows us to set
        # `req.local_file_path` on the appropriate requirement after passing
        # all the links at once into BatchDownloader.
        links_to_fully_download: Dict[Link, InstallRequirement] = {}
        for req in partially_downloaded_reqs:
            assert req.link

            # (1) File URLs don't need to be downloaded, so skip them.
            if req.link.scheme == "file":
                continue
            # (2) If this is e.g. a git url, we don't know how to handle that in the
            #     BatchDownloader, so leave it for self._prepare_linked_requirement() at
            #     the end of this method, which knows how to handle any URL.
            can_simply_download = True
            try:
                # This will raise InvalidSchema if our Session can't download it.
                self._session.get_adapter(req.link.url)
            except InvalidSchema:
                can_simply_download = False
            if can_simply_download:
                links_to_fully_download[req.link] = req

        batch_download = self._batch_download(
            links_to_fully_download.keys(),
            temp_dir,
        )
        for link, (filepath, _) in batch_download:
            logger.debug("Downloading link %s to %s", link, filepath)
            req = links_to_fully_download[link]
            # Record the downloaded file path so wheel reqs can extract a Distribution
            # in .get_dist().
            req.local_file_path = filepath
            # Record that the file is downloaded so we don't do it again in
            # _prepare_linked_requirement().
            self._downloaded[req.link.url] = filepath

            # If this is an sdist, we need to unpack it after downloading, but the
            # .source_dir won't be set up until we are in _prepare_linked_requirement().
            # Add the downloaded archive to the install requirement to unpack after
            # preparing the source dir.
            if not req.is_wheel:
                req.needs_unpacked_archive(Path(filepath))

        # This step is necessary to ensure all lazy wheels are processed
        # successfully by the 'download', 'wheel', and 'install' commands.
        for req in partially_downloaded_reqs:
            self._prepare_linked_requirement(req, parallel_builds)

    def prepare_linked_requirement(
        self, req: InstallRequirement, parallel_builds: bool = False
    ) -> BaseDistribution:
        """Prepare a requirement to be obtained from req.link."""
        assert req.link
        self._log_preparing_link(req)
        with indent_log():
            # Check if the relevant file is already available
            # in the download directory
            file_path = None
            if self.download_dir is not None and req.link.is_wheel:
                hashes = self._get_linked_req_hashes(req)
                file_path = _check_download_dir(
                    req.link,
                    self.download_dir,
                    hashes,
                    # When a locally built wheel has been found in cache, we don't warn
                    # about re-downloading when the already downloaded wheel hash does
                    # not match. This is because the hash must be checked against the
                    # original link, not the cached link. It that case the already
                    # downloaded file will be removed and re-fetched from cache (which
                    # implies a hash check against the cache entry's origin.json).
                    warn_on_hash_mismatch=not req.is_wheel_from_cache,
                )

            if file_path is not None:
                # The file is already available, so mark it as downloaded
                self._downloaded[req.link.url] = file_path
            else:
                # The file is not available, attempt to fetch only metadata
                metadata_dist = self._fetch_metadata_only(req)
                if metadata_dist is not None:
                    # These reqs now have the dependency information from the downloaded
                    # metadata, without having downloaded the actual dist at all.
                    req.cache_virtual_metadata_only_dist(metadata_dist)
                    return metadata_dist

            # None of the optimizations worked, fully prepare the requirement
            return self._prepare_linked_requirement(req, parallel_builds)

    def _ensure_download_info(self, reqs: Iterable[InstallRequirement]) -> None:
        """
        `pip install --report` extracts the download info from each requirement for its
        JSON output, so we need to make sure every requirement has this before finishing
        the resolve. But .download_info will only be populated by the point this method
        is called for requirements already found in the wheel cache, so we need to
        synthesize it for uncached results. Luckily, a DirectUrl can be parsed directly
        from a url without any other context. However, this also means the download info
        will only contain a hash if the link itself declares the hash.
        """
        for req in reqs:
            self._populate_download_info(req)

    def _force_fully_prepared(
        self, reqs: Iterable[InstallRequirement], require_concrete: bool
    ) -> None:
        """
        The legacy resolver seems to prepare requirements differently that can leave
        them half-done in certain code paths. I'm not quite sure how it's doing things,
        but at least we can do this to make sure they do things right.
        """
        for req in reqs:
            req.prepared = True
            if require_concrete:
                assert req.is_concrete

    def finalize_linked_requirements(
        self,
        reqs: Iterable[InstallRequirement],
        hydrate_virtual_reqs: bool,
        parallel_builds: bool = False,
    ) -> None:
        """Prepare linked requirements more, if needed.

        Neighboring .metadata files as per PEP 658 or lazy wheels via fast-deps will be
        preferred to extract metadata from any concrete requirement (one that has been
        mapped to a Link) without downloading the underlying wheel or sdist. When ``pip
        install --dry-run`` is called, we want to avoid ever downloading the underlying
        dist, but we still need to provide all of the results that pip commands expect
        from the typical resolve process.

        Those expectations vary, but one distinction lies in whether the command needs
        an actual physical dist somewhere on the filesystem, or just the metadata about
        it from the resolver (as in ``pip install --report``). If the command requires
        actual physical filesystem locations for the resolved dists, it must call this
        method with ``hydrate_virtual_reqs=True`` to fully download anything
        that remains.
        """
        if not hydrate_virtual_reqs:
            self._ensure_download_info(reqs)
            self._force_fully_prepared(reqs, require_concrete=False)
            return

        partially_downloaded_reqs: List[InstallRequirement] = []
        for req in reqs:
            if req.is_concrete:
                continue
            # Determine if any of these requirements were already downloaded.
            if self.download_dir is not None and req.link.is_wheel:
                hashes = self._get_linked_req_hashes(req)
                # If the file is there, but doesn't match the hash, delete it and print
                # a warning. We will be downloading it again via
                # partially_downloaded_reqs.
                file_path = _check_download_dir(
                    req.link, self.download_dir, hashes, warn_on_hash_mismatch=True
                )
                if file_path is not None:
                    # If the hash does match, then we still need to generate a concrete
                    # dist, but we don't have to download the wheel again.
                    self._downloaded[req.link.url] = file_path
            partially_downloaded_reqs.append(req)

        self._complete_partial_requirements(
            partially_downloaded_reqs,
            parallel_builds=parallel_builds,
        )
        # NB: Must call this method before returning!
        self._force_fully_prepared(reqs, require_concrete=True)

    def _prepare_linked_requirement(
        self, req: InstallRequirement, parallel_builds: bool
    ) -> BaseDistribution:
        assert req.link
        link = req.link

        hashes = self._get_linked_req_hashes(req)

        if hashes and req.is_wheel_from_cache:
            assert req.download_info is not None
            assert link.is_wheel
            assert link.is_file
            # We need to verify hashes, and we have found the requirement in the cache
            # of locally built wheels.
            if (
                isinstance(req.download_info.info, ArchiveInfo)
                and req.download_info.info.hashes
                and hashes.has_one_of(req.download_info.info.hashes)
            ):
                # At this point we know the requirement was built from a hashable source
                # artifact, and we verified that the cache entry's hash of the original
                # artifact matches one of the hashes we expect. We don't verify hashes
                # against the cached wheel, because the wheel is not the original.
                hashes = None
            else:
                logger.warning(
                    "The hashes of the source archive found in cache entry "
                    "don't match, ignoring cached built wheel "
                    "and re-downloading source."
                )
                req.link = req.cached_wheel_source_link
                link = req.link

        self._ensure_link_req_src_dir(req, parallel_builds)

        if link.is_existing_dir():
            local_file = None
        elif link.url not in self._downloaded:
            try:
                local_file = unpack_url(
                    link,
                    req.source_dir,
                    self._download,
                    self.verbosity,
                    self.download_dir,
                    hashes,
                )
            except NetworkConnectionError as exc:
                raise InstallationError(
                    "Could not install requirement {} because of HTTP "
                    "error {} for URL {}".format(req, exc, link)
                )
        else:
            file_path = self._downloaded[link.url]
            if hashes:
                hashes.check_against_path(file_path)
            local_file = File(file_path, content_type=None)

        # For use in later processing,
        # preserve the file path on the requirement.
        if local_file:
            req.local_file_path = local_file.path

        self._populate_download_info(req)

        (builds_metadata, dist) = _get_prepared_distribution(
            req,
            self.build_tracker,
            self.finder,
            self.build_isolation,
            self.check_build_deps,
        )
        if builds_metadata and should_cache(req):
            self._cache_metadata(req, dist)
        return dist

    def _populate_download_info(self, req: InstallRequirement) -> None:
        # If download_info is set, we got it from the wheel cache.
        if req.download_info is None:
            # Editables don't go through this function (see
            # prepare_editable_requirement).
            assert not req.editable
            req.download_info = direct_url_from_link(req.link, req.source_dir)
            # Make sure we have a hash in download_info. If we got it as part of the
            # URL, it will have been verified and we can rely on it. Otherwise we
            # compute it from the downloaded file.
            # FIXME: https://github.com/pypa/pip/issues/11943
            if (
                isinstance(req.download_info.info, ArchiveInfo)
                and not req.download_info.info.hashes
                and req.local_file_path
            ):
                hash = hash_file(req.local_file_path)[0].hexdigest()
                # We populate info.hash for backward compatibility.
                # This will automatically populate info.hashes.
                req.download_info.info.hash = f"sha256={hash}"

    def save_linked_requirement(self, req: InstallRequirement) -> None:
        assert self.download_dir is not None
        assert req.link is not None
        assert req.is_concrete
        link = req.link
        if link.is_vcs or (link.is_existing_dir() and req.editable):
            # Make a .zip of the source_dir we already created.
            req.archive(self.download_dir)
            return

        if link.is_existing_dir():
            logger.debug(
                "Not copying link to destination directory "
                "since it is a directory: %s",
                link,
            )
            return
        if req.local_file_path is None:
            # No distribution was downloaded for this requirement.
            return

        download_location = os.path.join(self.download_dir, link.filename)
        if not os.path.exists(download_location):
            shutil.copy(req.local_file_path, download_location)
            download_path = display_path(download_location)
            logger.info("Saved %s", download_path)

    def prepare_editable_requirement(
        self,
        req: InstallRequirement,
    ) -> BaseDistribution:
        """Prepare an editable requirement."""
        assert req.editable, "cannot prepare a non-editable req as editable"

        logger.info("Obtaining %s", req)

        with indent_log():
            if self.require_hashes:
                raise InstallationError(
                    "The editable requirement {} cannot be installed when "
                    "requiring hashes, because there is no single file to "
                    "hash.".format(req)
                )
            req.ensure_has_source_dir(self.src_dir)
            req.update_editable()
            assert req.source_dir
            req.download_info = direct_url_for_editable(req.unpacked_source_directory)

            (_, dist) = _get_prepared_distribution(
                req,
                self.build_tracker,
                self.finder,
                self.build_isolation,
                self.check_build_deps,
            )

            req.check_if_exists(self.use_user_site)

        # This should already have been populated by the preparation of the source dist.
        assert req.is_concrete
        return dist

    def prepare_installed_requirement(
        self,
        req: InstallRequirement,
        skip_reason: str,
    ) -> BaseDistribution:
        """Prepare an already-installed requirement."""
        assert req.satisfied_by, "req should have been satisfied but isn't"
        assert skip_reason is not None, (
            "did not get skip reason skipped but req.satisfied_by "
            "is set to {}".format(req.satisfied_by)
        )
        logger.info(
            "Requirement %s: %s (%s)", skip_reason, req, req.satisfied_by.version
        )
        with indent_log():
            if self.require_hashes:
                logger.debug(
                    "Since it is already installed, we are trusting this "
                    "package without checking its hash. To ensure a "
                    "completely repeatable environment, install into an "
                    "empty virtualenv."
                )
            (_, dist) = _get_prepared_distribution(
                req,
                self.build_tracker,
                self.finder,
                self.build_isolation,
                self.check_build_deps,
            )

        assert req.is_concrete
        return dist
