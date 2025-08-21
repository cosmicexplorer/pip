"""Filetype information."""

from __future__ import annotations

import re
from typing import ClassVar


class FileExtensions:

    WHEEL_EXTENSION: ClassVar[str] = ".whl"
    BZ2_EXTENSIONS: ClassVar[tuple[str, ...]] = (".tar.bz2", ".tbz")
    XZ_EXTENSIONS: ClassVar[tuple[str, ...]] = (
        ".tar.xz",
        ".txz",
        ".tlz",
        ".tar.lz",
        ".tar.lzma",
    )
    ZIP_EXTENSIONS: ClassVar[tuple[str, ...]] = (".zip", WHEEL_EXTENSION)
    TAR_EXTENSIONS: ClassVar[tuple[str, ...]] = (".tar.gz", ".tgz", ".tar")
    ARCHIVE_EXTENSIONS: ClassVar[tuple[str, ...]] = (
        ZIP_EXTENSIONS + BZ2_EXTENSIONS + TAR_EXTENSIONS + XZ_EXTENSIONS
    )

    _all_archive_exts_joiner: ClassVar[str] = "|".join(
        map(re.escape, ARCHIVE_EXTENSIONS)
    )
    _all_archive_exts_regex: ClassVar[re.Pattern[str]] = re.compile(
        f"(?:{_all_archive_exts_joiner})$",
        flags=re.IGNORECASE,
    )

    @staticmethod
    def archive_file_extension(name: str) -> str | None:
        """Return True if `name` has a conventional archive filename extension."""
        # NB: os.path.splitext() is unreliable for this purposes as it will only ever
        # contain "at most one period": https://docs.python.org/3/library/os.path.html#os.path.splitext
        if m := FileExtensions._all_archive_exts_regex.search(name):
            return m.group(0)
        return None
