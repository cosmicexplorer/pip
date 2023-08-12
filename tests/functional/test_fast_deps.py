import fnmatch
import json
import os
import pathlib
from os.path import basename
from pathlib import Path
from typing import Iterable, List

from pip._vendor.packaging.utils import canonicalize_name
from pytest import mark

from tests.conftest import HTMLIndexWithRangeServer, RangeHandler
from tests.lib import PipTestEnvironment, TestData, TestPipResult


def pip(script: PipTestEnvironment, command: str, requirement: str) -> TestPipResult:
    return script.pip(
        command,
        "--prefer-binary",
        "--no-cache-dir",
        "--use-feature=fast-deps",
        requirement,
        # TODO: remove this when fast-deps is on by default.
        allow_stderr_warning=True,
    )


def assert_installed(script: PipTestEnvironment, names: str) -> None:
    list_output = json.loads(script.pip("list", "--format=json").stdout)
    installed = {canonicalize_name(item["name"]) for item in list_output}
    assert installed.issuperset(map(canonicalize_name, names))


@mark.network
@mark.parametrize(
    ("requirement", "expected"),
    (
        ("Paste==3.4.2", ("Paste", "six")),
        ("Paste[flup]==3.4.2", ("Paste", "six", "flup")),
    ),
)
def test_install_from_pypi(
    requirement: str, expected: str, script: PipTestEnvironment
) -> None:
    pip(script, "install", requirement)
    assert_installed(script, expected)


@mark.network
@mark.parametrize(
    ("requirement", "expected"),
    (
        ("Paste==3.4.2", ("Paste-3.4.2-*.whl", "six-*.whl")),
        ("Paste[flup]==3.4.2", ("Paste-3.4.2-*.whl", "six-*.whl", "flup-*")),
    ),
)
def test_download_from_pypi(
    requirement: str, expected: Iterable[str], script: PipTestEnvironment
) -> None:
    result = pip(script, "download", requirement)
    created = [basename(f) for f in result.files_created]
    assert all(fnmatch.filter(created, f) for f in expected)


@mark.network
def test_build_wheel_with_deps(data: TestData, script: PipTestEnvironment) -> None:
    result = pip(script, "wheel", os.fspath(data.packages / "requiresPaste"))
    created = [basename(f) for f in result.files_created]
    assert fnmatch.filter(created, "requirespaste-3.1.4-*.whl")
    assert fnmatch.filter(created, "Paste-3.4.2-*.whl")
    assert fnmatch.filter(created, "six-*.whl")


@mark.network
def test_require_hash(script: PipTestEnvironment, tmp_path: pathlib.Path) -> None:
    reqs = tmp_path / "requirements.txt"
    reqs.write_text(
        "idna==2.10"
        " --hash=sha256:"
        "b97d804b1e9b523befed77c48dacec60e6dcb0b5391d57af6a65a312a90648c0"
        " --hash=sha256:"
        "b307872f855b18632ce0c21c5e45be78c0ea7ae4c15c828c20788b26921eb3f6"
    )
    result = script.pip(
        "download",
        "--use-feature=fast-deps",
        "-r",
        str(reqs),
        allow_stderr_warning=True,
    )
    created = [basename(f) for f in result.files_created]
    assert fnmatch.filter(created, "idna-2.10*")


@mark.network
def test_hash_mismatch(script: PipTestEnvironment, tmp_path: pathlib.Path) -> None:
    reqs = tmp_path / "requirements.txt"
    reqs.write_text("idna==2.10 --hash=sha256:irna")
    result = script.pip(
        "download",
        "--use-feature=fast-deps",
        "-r",
        str(reqs),
        expect_error=True,
    )
    assert "DO NOT MATCH THE HASHES" in result.stderr


@mark.parametrize("range_handler", list(RangeHandler))
def test_download_range(
    script: PipTestEnvironment,
    tmpdir: Path,
    html_index_with_range_server: HTMLIndexWithRangeServer,
    range_handler: RangeHandler,
) -> None:
    """Execute `pip download` against a generated PyPI index."""
    download_dir = tmpdir / "download_dir"

    def run_for_generated_index(args: List[str]) -> TestPipResult:
        """
        Produce a PyPI directory structure pointing to the specified packages, then
        execute `pip download -i ...` pointing to our generated index.
        """
        pip_args = [
            "download",
            "--use-feature=fast-deps",
            "-d",
            str(download_dir),
            "-i",
            "http://localhost:8000",
            *args,
        ]
        return script.pip(*pip_args, allow_stderr_warning=True)

    with html_index_with_range_server(range_handler) as handler:
        run_for_generated_index(["colander", "compilewheel==2.0", "has-script"])
        generated_files = os.listdir(download_dir)
        assert fnmatch.filter(generated_files, "colander*.whl")
        assert fnmatch.filter(generated_files, "compilewheel*.whl")
        assert fnmatch.filter(generated_files, "has.script*.whl")

        colander_wheel_path = "/colander/colander-0.9.9-py2.py3-none-any.whl"
        compile_wheel_path = "/compilewheel/compilewheel-2.0-py2.py3-none-any.whl"
        has_script_path = "/has-script/has.script-1.0-py2.py3-none-any.whl"

        if range_handler == RangeHandler.Always200OK:
            assert not handler.head_request_paths
            assert not handler.positive_range_request_paths
            assert {colander_wheel_path} == handler.negative_range_request_paths
            # Tries a range request, finds it's unsupported, so doesn't try it again.
            assert handler.get_request_counts[colander_wheel_path] == 2
            assert handler.get_request_counts[compile_wheel_path] == 1
            assert handler.get_request_counts[has_script_path] == 1
        elif range_handler == RangeHandler.NoNegativeRange:
            assert {
                colander_wheel_path,
                compile_wheel_path,
                has_script_path,
            } == handler.head_request_paths
            assert {
                colander_wheel_path,
                compile_wheel_path,
                has_script_path,
            } == handler.positive_range_request_paths
            # Tries this first, finds that negative offsets are unsupported, so doesn't
            # try it again.
            assert {colander_wheel_path} == handler.negative_range_request_paths
            # One more for the first wheel, because it has the failing negative
            # byte request.
            assert handler.get_request_counts[colander_wheel_path] == 3
            # The entire .dist-info dir should have been pulled in with a single
            # ranged GET. The second GET is for the end of the download, pulling down
            # the entire file contents.
            assert handler.get_request_counts[compile_wheel_path] == 2
            # The entire file should have been pulled in with a single ranged GET.
            assert handler.get_request_counts[has_script_path] == 2
        else:
            assert range_handler == RangeHandler.SupportsNegativeRange
            # The negative byte index worked, so no head requests.
            assert not handler.head_request_paths
            # The negative range request was in bounds and pulled in the entire
            # .dist-info directory for compile-wheel==2.0, so we didn't need another
            # range request.
            assert {
                colander_wheel_path,
                has_script_path,
            } == handler.positive_range_request_paths
            assert {
                colander_wheel_path,
                compile_wheel_path,
                has_script_path,
            } == handler.negative_range_request_paths
            assert handler.get_request_counts[colander_wheel_path] == 3
            assert handler.get_request_counts[compile_wheel_path] == 2
            # One more than last time, because the negative byte index failed.
            assert handler.get_request_counts[has_script_path] == 3
