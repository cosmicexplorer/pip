from __future__ import annotations

from pip._internal.models.candidate import InstallationCandidate
from pip._internal.models.link import Link
from pip._internal.utils.packaging.version import ParsedVersion


def make_mock_candidate(
    version: str, yanked_reason: str | None = None, hex_digest: str | None = None
) -> InstallationCandidate:
    url = f"https://example.com/pkg-{version}.tar.gz"
    if hex_digest is not None:
        assert len(hex_digest) == 64
        url += f"#sha256={hex_digest}"

    link = Link(url, yanked_reason=yanked_reason)
    candidate = InstallationCandidate("mypackage", ParsedVersion.parse(version), link)

    return candidate
