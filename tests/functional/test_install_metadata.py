import json
import re
import shutil
import uuid
from dataclasses import dataclass
from enum import Enum
from hashlib import sha256
from pathlib import Path
from textwrap import dedent
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

import pytest
from pip._vendor.packaging.requirements import Requirement

from pip._internal.models.direct_url import DirectUrl
from pip._internal.utils.urls import path_to_url
from tests.lib import (
    PipTestEnvironment,
    TestData,
    TestPipResult,
)


class MetadataKind(Enum):
    """All the types of values we might be provided for the data-dist-info-metadata
    attribute from PEP 658."""

    # Valid: will read metadata from the dist instead.
    No = "none"
    # Valid: will read the .metadata file, but won't check its hash.
    Unhashed = "unhashed"
    # Valid: will read the .metadata file and check its hash matches.
    Sha256 = "sha256"
    # Invalid: will error out after checking the hash.
    WrongHash = "wrong-hash"
    # Invalid: will error out after failing to fetch the .metadata file.
    NoFile = "no-file"


@dataclass(frozen=True)
class FakePackage:
    """Mock package structure used to generate a PyPI repository.

    FakePackage name and version should correspond to sdists (.tar.gz files) in our test
    data."""

    name: str
    version: str
    filename: str
    metadata: MetadataKind
    # This will override any dependencies specified in the actual dist's METADATA.
    requires_dist: Tuple[str, ...] = ()
    # This will override the Name specified in the actual dist's METADATA.
    metadata_name: Optional[str] = None

    def metadata_filename(self) -> str:
        """This is specified by PEP 658."""
        return f"{self.filename}.metadata"

    def generate_additional_tag(self) -> str:
        """This gets injected into the <a> tag in the generated PyPI index page for this
        package."""
        if self.metadata == MetadataKind.No:
            return ""
        if self.metadata in [MetadataKind.Unhashed, MetadataKind.NoFile]:
            return 'data-dist-info-metadata="true"'
        if self.metadata == MetadataKind.WrongHash:
            return 'data-dist-info-metadata="sha256=WRONG-HASH"'
        assert self.metadata == MetadataKind.Sha256
        checksum = sha256(self.generate_metadata()).hexdigest()
        return f'data-dist-info-metadata="sha256={checksum}"'

    def requires_str(self) -> str:
        if not self.requires_dist:
            return ""
        joined = " and ".join(self.requires_dist)
        return f"Requires-Dist: {joined}"

    def generate_metadata(self) -> bytes:
        """This is written to `self.metadata_filename()` and will override the actual
        dist's METADATA, unless `self.metadata == MetadataKind.NoFile`."""
        return dedent(
            f"""\
        Metadata-Version: 2.1
        Name: {self.metadata_name or self.name}
        Version: {self.version}
        {self.requires_str()}
        """
        ).encode("utf-8")


@pytest.fixture(scope="function")
def write_index_html_content(tmpdir: Path) -> Callable[[str], Path]:
    """Generate a PyPI package index.html within a temporary local directory."""
    html_dir = tmpdir / "index_html_content"
    html_dir.mkdir()

    def generate_index_html_subdir(index_html: str) -> Path:
        """Create a new subdirectory after a UUID and write an index.html."""
        new_subdir = html_dir / uuid.uuid4().hex
        new_subdir.mkdir()

        (new_subdir / "index.html").write_text(index_html)

        return new_subdir

    return generate_index_html_subdir


@pytest.fixture(scope="function")
def html_index_for_packages(
    shared_data: TestData,
    write_index_html_content: Callable[[str], Path],
) -> Callable[..., Path]:
    """Generate a PyPI HTML package index within a local directory pointing to
    blank data."""

    def generate_html_index_for_packages(
        packages: Dict[str, List[FakePackage]]
    ) -> Path:
        """
        Produce a PyPI directory structure pointing to the specified packages.
        """
        # (1) Generate the content for a PyPI index.html.
        pkg_links = "\n".join(
            f'    <a href="{pkg}/index.html">{pkg}</a>' for pkg in packages.keys()
        )
        index_html = f"""\
<!DOCTYPE html>
<html>
  <head>
    <meta name="pypi:repository-version" content="1.0">
    <title>Simple index</title>
  </head>
  <body>
{pkg_links}
  </body>
</html>"""
        # (2) Generate the index.html in a new subdirectory of the temp directory.
        index_html_subdir = write_index_html_content(index_html)

        # (3) Generate subdirectories for individual packages, each with their own
        # index.html.
        for pkg, links in packages.items():
            pkg_subdir = index_html_subdir / pkg
            pkg_subdir.mkdir()

            download_links: List[str] = []
            for package_link in links:
                # (3.1) Generate the <a> tag which pip can crawl pointing to this
                # specific package version.
                download_links.append(
                    f'    <a href="{package_link.filename}" {package_link.generate_additional_tag()}>{package_link.filename}</a><br/>'  # noqa: E501
                )
                # (3.2) Copy over the corresponding file in `shared_data.packages`.
                cached_file = shared_data.packages / package_link.filename
                new_file = pkg_subdir / package_link.filename
                # (3.3) Write a metadata file, if applicable.
                if package_link.metadata != MetadataKind.NoFile:
                    (pkg_subdir / package_link.metadata_filename()).write_bytes(
                        package_link.generate_metadata()
                    )

            # (3.4) After collating all the download links and copying over the files,
            # write an index.html with the generated download links for each
            # copied file for this specific package name.
            download_links_str = "\n".join(download_links)
            pkg_index_content = f"""\
<!DOCTYPE html>
<html>
  <head>
    <meta name="pypi:repository-version" content="1.0">
    <title>Links for {pkg}</title>
  </head>
  <body>
    <h1>Links for {pkg}</h1>
{download_links_str}
  </body>
</html>"""
            (pkg_subdir / "index.html").write_text(pkg_index_content)

        return index_html_subdir

    return generate_html_index_for_packages


@pytest.fixture(scope="function")
def install_with_generated_html_index(
    script: PipTestEnvironment,
    html_index_for_packages: Callable[[Dict[str, List[FakePackage]]], Path],
    tmpdir: Path,
) -> Callable[..., Tuple[TestPipResult, Dict[str, Any]]]:
    """Execute `pip download` against a generated PyPI index."""
    output_file = tmpdir / "output_file.json"

    def run_for_generated_index(
        packages: Dict[str, List[FakePackage]],
        args: List[str],
        *,
        dry_run: bool = True,
        allow_error: bool = False,
    ) -> Tuple[TestPipResult, Dict[str, Any]]:
        """
        Produce a PyPI directory structure pointing to the specified packages, then
        execute `pip install --report ... -i ...` pointing to our generated index.
        """
        index_dir = html_index_for_packages(packages)
        pip_args = [
            "install",
            *(("--dry-run",) if dry_run else ()),
            "--ignore-installed",
            "--report",
            str(output_file),
            "-i",
            path_to_url(str(index_dir)),
            *args,
        ]
        result = script.pip(*pip_args, allow_error=allow_error)
        try:
            with open(output_file, "rb") as f:
                report = json.load(f)
        except FileNotFoundError:
            if allow_error:
                report = {}
            else:
                raise
        return (result, report)

    return run_for_generated_index


def iter_dists(report: Dict[str, Any]) -> Iterator[Tuple[Requirement, DirectUrl]]:
    """Parse a (req,url) tuple from each installed dist in the --report json."""
    for inst in report["install"]:
        metadata = inst["metadata"]
        name = metadata["name"]
        version = metadata["version"]
        req = Requirement(f"{name}=={version}")
        direct_url = DirectUrl.from_dict(inst["download_info"])
        yield (req, direct_url)


# The package database we generate for testing PEP 658 support.
_simple_packages: Dict[str, List[FakePackage]] = {
    "simple": [
        FakePackage("simple", "1.0", "simple-1.0.tar.gz", MetadataKind.Sha256),
        FakePackage("simple", "2.0", "simple-2.0.tar.gz", MetadataKind.No),
        # This will raise a hashing error.
        FakePackage("simple", "3.0", "simple-3.0.tar.gz", MetadataKind.WrongHash),
    ],
    "simple2": [
        # Override the dependencies here in order to force pip to download
        # simple-1.0.tar.gz as well.
        FakePackage(
            "simple2",
            "1.0",
            "simple2-1.0.tar.gz",
            MetadataKind.Unhashed,
            ("simple==1.0",),
        ),
        # This will raise an error when pip attempts to fetch the metadata file.
        FakePackage("simple2", "2.0", "simple2-2.0.tar.gz", MetadataKind.NoFile),
        # This has a METADATA file with a mismatched name.
        FakePackage(
            "simple2",
            "3.0",
            "simple2-3.0.tar.gz",
            MetadataKind.Sha256,
            metadata_name="not-simple2",
        ),
    ],
    "colander": [
        # Ensure we can read the dependencies from a metadata file within a wheel
        # *without* PEP 658 metadata.
        FakePackage(
            "colander", "0.9.9", "colander-0.9.9-py2.py3-none-any.whl", MetadataKind.No
        ),
    ],
    "compilewheel": [
        # Ensure we can override the dependencies of a wheel file by injecting PEP
        # 658 metadata.
        FakePackage(
            "compilewheel",
            "1.0",
            "compilewheel-1.0-py2.py3-none-any.whl",
            MetadataKind.Unhashed,
            ("simple==1.0",),
        ),
    ],
    "has-script": [
        # Ensure we check PEP 658 metadata hashing errors for wheel files.
        FakePackage(
            "has-script",
            "1.0",
            "has.script-1.0-py2.py3-none-any.whl",
            MetadataKind.WrongHash,
        ),
    ],
    "translationstring": [
        FakePackage(
            "translationstring", "1.1", "translationstring-1.1.tar.gz", MetadataKind.No
        ),
    ],
    "priority": [
        # Ensure we check for a missing metadata file for wheels.
        FakePackage(
            "priority", "1.0", "priority-1.0-py2.py3-none-any.whl", MetadataKind.NoFile
        ),
    ],
    "requires-simple-extra": [
        # Metadata name is not canonicalized.
        FakePackage(
            "requires-simple-extra",
            "0.1",
            "requires_simple_extra-0.1-py2.py3-none-any.whl",
            MetadataKind.Sha256,
            metadata_name="Requires_Simple.Extra",
        ),
    ],
}


@pytest.mark.parametrize(
    "requirement_to_install, expected_outputs",
    [
        ("simple2==1.0", ["simple2==1.0", "simple==1.0"]),
        ("simple==2.0", ["simple==2.0"]),
        (
            "colander",
            ["colander==0.9.9", "translationstring==1.1"],
        ),
        (
            "compilewheel",
            ["compilewheel==1.0", "simple==1.0"],
        ),
    ],
)
def test_install_with_metadata(
    install_with_generated_html_index: Callable[
        ..., Tuple[TestPipResult, Dict[str, Any]]
    ],
    requirement_to_install: str,
    expected_outputs: List[str],
) -> None:
    """Verify that if a data-dist-info-metadata attribute is present, then it is used
    instead of the actual dist's METADATA."""
    _, report = install_with_generated_html_index(
        _simple_packages,
        [requirement_to_install],
    )
    installed = sorted(str(r) for r, _ in iter_dists(report))
    assert installed == expected_outputs


@pytest.mark.parametrize(
    "requirement_to_install, real_hash",
    [
        (
            "simple==3.0",
            "95e0f200b6302989bcf2cead9465cf229168295ea330ca30d1ffeab5c0fed996",
        ),
        (
            "has-script",
            "16ba92d7f6f992f6de5ecb7d58c914675cf21f57f8e674fb29dcb4f4c9507e5b",
        ),
    ],
)
def test_incorrect_metadata_hash(
    install_with_generated_html_index: Callable[
        ..., Tuple[TestPipResult, Dict[str, Any]]
    ],
    requirement_to_install: str,
    real_hash: str,
) -> None:
    """Verify that if a hash for data-dist-info-metadata is provided, it must match the
    actual hash of the metadata file."""
    result, _ = install_with_generated_html_index(
        _simple_packages,
        [requirement_to_install],
        allow_error=True,
    )
    assert result.returncode != 0
    expected_msg = f"""\
        Expected sha256 WRONG-HASH
             Got        {real_hash}"""
    assert expected_msg in result.stderr


@pytest.mark.parametrize(
    "requirement_to_install, expected_url",
    [
        ("simple2==2.0", "simple2-2.0.tar.gz.metadata"),
        ("priority", "priority-1.0-py2.py3-none-any.whl.metadata"),
    ],
)
def test_metadata_not_found(
    install_with_generated_html_index: Callable[
        ..., Tuple[TestPipResult, Dict[str, Any]]
    ],
    requirement_to_install: str,
    expected_url: str,
) -> None:
    """Verify that if a data-dist-info-metadata attribute is provided, that pip will
    fetch the .metadata file at the location specified by PEP 658, and error
    if unavailable."""
    result, _ = install_with_generated_html_index(
        _simple_packages,
        [requirement_to_install],
        allow_error=True,
    )
    assert result.returncode != 0
    expected_re = re.escape(expected_url)
    pattern = re.compile(
        f"ERROR: 404 Client Error: FileNotFoundError for url:.*{expected_re}"
    )
    assert pattern.search(result.stderr), (pattern, result.stderr)


def test_produces_error_for_mismatched_package_name_in_metadata(
    install_with_generated_html_index: Callable[
        ..., Tuple[TestPipResult, Dict[str, Any]]
    ],
) -> None:
    """Verify that the package name from the metadata matches the requested package."""
    result, _ = install_with_generated_html_index(
        _simple_packages,
        ["simple2==3.0"],
        allow_error=True,
    )
    assert result.returncode != 0
    assert (
        "simple2-3.0.tar.gz has inconsistent Name: expected 'simple2', but metadata "
        "has 'not-simple2'"
    ) in result.stdout


@pytest.mark.parametrize(
    "requirement",
    (
        "requires-simple-extra==0.1",
        "REQUIRES_SIMPLE-EXTRA==0.1",
        "REQUIRES....simple-_-EXTRA==0.1",
    ),
)
def test_canonicalizes_package_name_before_verifying_metadata(
    install_with_generated_html_index: Callable[
        ..., Tuple[TestPipResult, Dict[str, Any]]
    ],
    requirement: str,
) -> None:
    """Verify that the package name from the command line and the package's
    METADATA are both canonicalized before comparison, while the name from the METADATA
    is always used verbatim to represent the installed candidate in --report.

    Regression test for https://github.com/pypa/pip/issues/12038
    """
    _, report = install_with_generated_html_index(
        _simple_packages,
        [requirement],
    )
    reqs = [str(r) for r, _ in iter_dists(report)]
    assert reqs == ["Requires_Simple.Extra==0.1"]
