"""Tests for various classes in pip._internal.models"""

from pip._internal.models import candidate, index
from pip._internal.models.link import Link
from pip._internal.utils.packaging.version import ParsedVersion


class TestPackageIndex:
    """Tests for pip._internal.models.index.PackageIndex"""

    def test_gives_right_urls(self) -> None:
        url = "https://mypypi.internal/path/"
        pack_index = index.PackageIndex.create(url)

        assert str(pack_index.base_url) == url

        assert str(pack_index.simple_url) == url + "simple"
        assert str(pack_index.search_url) == url + "pypi"

    def test_PyPI_urls_are_correct(self) -> None:
        pack_index = index.PyPI

        assert str(pack_index.base_url) == "https://pypi.org/"
        assert str(pack_index.simple_url) == "https://pypi.org/simple"
        assert str(pack_index.search_url) == "https://pypi.org/pypi"

    def test_TestPyPI_urls_are_correct(self) -> None:
        pack_index = index.TestPyPI

        assert str(pack_index.base_url) == "https://test.pypi.org/"
        assert str(pack_index.simple_url) == "https://test.pypi.org/simple"
        assert str(pack_index.search_url) == "https://test.pypi.org/pypi"


class TestInstallationCandidate:
    def test_sets_correct_variables(self) -> None:
        obj = candidate.InstallationCandidate(
            "A",
            ParsedVersion.parse("1.0.0"),
            Link("https://somewhere.com/path/A-1.0.0.tar.gz"),
        )
        assert obj.name == "A"
        assert obj.version == ParsedVersion.parse("1.0.0")
        assert obj.link.url == "https://somewhere.com/path/A-1.0.0.tar.gz"
