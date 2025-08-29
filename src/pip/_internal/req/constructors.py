"""Backing implementation for InstallRequirement's various constructors

The idea here is that these formed a major chunk of InstallRequirement's size
so, moving them and support code dedicated to them outside of that class
helps creates for better understandability for the rest of the code.

These are meant to be used elsewhere within pip to create instances of
InstallRequirement.
"""

from __future__ import annotations

import copy
import logging
import os
import re
from collections.abc import Collection
from dataclasses import dataclass
from typing import ClassVar

from pip._vendor.packaging.requirements import InvalidRequirement

from pip._internal.exceptions import InstallationError
from pip._internal.models.index import PyPI, TestPyPI
from pip._internal.models.link import Link
from pip._internal.models.wheel import WheelInfo
from pip._internal.req.req_file import ParsedRequirement
from pip._internal.req.req_install import InstallRequirement
from pip._internal.utils.filetypes import FileExtensions
from pip._internal.utils.misc import is_installable_dir
from pip._internal.utils.packaging.markers import Marker
from pip._internal.utils.packaging.requirements import Requirement
from pip._internal.utils.packaging.specifiers import Operator
from pip._internal.utils.packaging_utils import get_requirement
from pip._internal.utils.urls import ParsedUrl, coerce_file_uri_to_path, path_to_url
from pip._internal.vcs import has_vcs_url_scheme, vcs

__all__ = [
    "install_req_from_editable",
    "install_req_from_line",
    "parse_editable",
]

logger = logging.getLogger(__name__)


def _strip_extras(path: str) -> tuple[str, str | None]:
    m = re.match(r"^(.+)(\[[^\]]+\])$", path)
    extras = None
    if m:
        path_no_extras = m.group(1)
        extras = m.group(2)
    else:
        path_no_extras = path

    return path_no_extras, extras


# TODO: what is "placeholder" and why does it work?
def convert_extras(extras: str | None) -> frozenset[str]:
    if not extras:
        return frozenset()
    return frozenset(get_requirement("placeholder" + extras.lower()).extras)


def _set_requirement_extras(req: Requirement, new_extras: set[str]) -> Requirement:
    """
    Returns a new requirement based on the given one, with the supplied extras. If the
    given requirement already has extras those are replaced (or dropped if no new extras
    are given).
    """
    match: re.Match[str] | None = re.fullmatch(
        # see https://peps.python.org/pep-0508/#complete-grammar
        r"([\w\t .-]+)(\[[^\]]*\])?(.*)",
        str(req),
        flags=re.ASCII,
    )
    # ireq.req is a valid requirement so the regex should always match
    assert (
        match is not None
    ), f"regex match on requirement {req} failed, this should never happen"
    pre: str | None = match.group(1)
    post: str | None = match.group(3)
    assert (
        pre is not None and post is not None
    ), f"regex group selection for requirement {req} failed, this should never happen"
    extras: str = "[{}]".format(",".join(sorted(new_extras)) if new_extras else "")
    return get_requirement(f"{pre}{extras}{post}")


def parse_editable(editable_req: str) -> tuple[Link, frozenset[str]]:
    """Parses an editable requirement into:
        - a requirement name
        - an URL
        - extras
        - editable options
    Accepted requirements:
        svn+http://blahblah@rev#egg=Foobar[baz]&subdirectory=version_subdir
        .[some_extra]
    """

    # If a file path is specified with extras, strip off the extras.
    url_no_extras, extras = _strip_extras(editable_req)

    parsed_url = None
    url_no_extras = coerce_file_uri_to_path(url_no_extras)
    # TODO: consider _looks_like_path()????
    if os.path.isdir(url_no_extras):
        # Treating it as code that has already been checked out
        parsed_url = path_to_url(url_no_extras)
        return Link(parsed_url), convert_extras(extras)
    if _ResourcePathMatcher.looks_like_path(url_no_extras):
        logger.debug(
            "editable requirement '%s' was converted to file-like URL '%s', "
            "but it did not point to an existing directory path",
            editable_req,
            url_no_extras,
        )

    if not has_vcs_url_scheme(url_no_extras):
        backends = ", ".join(vcs.all_schemes)
        raise InstallationError(
            f"{editable_req} is not a valid editable requirement. "
            f"It should either be a path to a local project or a VCS URL "
            f"(beginning with {backends})."
        )
    # TODO: why no extras here? why is the egg fragment correct?
    return Link(editable_req), frozenset()


def check_first_requirement_in_file(filename: str) -> None:
    """Check if file is parsable as a requirements file.

    This is heavily based on ``pkg_resources.parse_requirements``, but
    simplified to just check the first meaningful line.

    :raises InvalidRequirement: If the first meaningful line cannot be parsed
        as an requirement.
    """
    with open(filename, encoding="utf-8", errors="ignore") as f:
        # Create a steppable iterator, so we can handle \-continuations.
        lines = (
            line
            for line in (line.strip() for line in f)
            if line and not line.startswith("#")  # Skip blank lines/comments.
        )

        for line in lines:
            # Drop comments -- a hash without a space may be in a URL.
            if " #" in line:
                line = line[: line.find(" #")]
            # If there is a line continuation, drop it, and append the next line.
            if line.endswith("\\"):
                line = line[:-2].strip() + next(lines, "")
            get_requirement(line)
            return


def deduce_helpful_msg(req: str) -> str:
    """Returns helpful msg in case requirements file does not exist,
    or cannot be parsed.

    :params req: Requirements file path
    """
    if not os.path.exists(req):
        return f" File '{req}' does not exist."
    msg = " The path does exist. "
    # Try to parse and check if it is a requirements file.
    try:
        check_first_requirement_in_file(req)
    except InvalidRequirement:
        logger.debug("Cannot parse '%s' as requirements file", req)
    else:
        msg += (
            f"The argument you provided "
            f"({req}) appears to be a"
            f" requirements file. If that is the"
            f" case, use the '-r' flag to install"
            f" the packages specified within it."
        )
    return msg


@dataclass(frozen=True)
class RequirementParts:
    requirement: Requirement | None
    link: Link | None
    markers: Marker | None
    extras: frozenset[str]


def parse_req_from_editable(editable_req: str) -> RequirementParts:
    link, extras_override = parse_editable(editable_req)

    name = link.egg_fragment
    if name is not None:
        try:
            req: Requirement | None = get_requirement(name)
        except InvalidRequirement as exc:
            raise InstallationError(f"Invalid requirement: {name!r}: {exc}")
    else:
        req = None

    return RequirementParts(req, link, None, extras_override)


# ---- The actual constructors follow ----


def install_req_from_editable(
    editable_req: str,
    comes_from: InstallRequirement | str | None = None,
    *,
    use_pep517: bool | None = None,
    isolated: bool = False,
    global_options: list[str] | None = None,
    hash_options: dict[str, list[str]] | None = None,
    constraint: bool = False,
    user_supplied: bool = False,
    permit_editable_wheels: bool = False,
    config_settings: dict[str, str | list[str]] | None = None,
) -> InstallRequirement:
    parts = parse_req_from_editable(editable_req)

    return InstallRequirement(
        parts.requirement,
        comes_from=comes_from,
        user_supplied=user_supplied,
        editable=True,
        permit_editable_wheels=permit_editable_wheels,
        link=parts.link,
        constraint=constraint,
        use_pep517=use_pep517,
        isolated=isolated,
        global_options=global_options,
        hash_options=hash_options,
        config_settings=config_settings,
        extras=parts.extras,
    )


class _ResourcePathMatcher:
    _path_like_regex: ClassVar[re.Pattern[str]] = re.compile(
        "|".join(
            [
                r"^\.",
                re.escape(os.path.sep),
                *([re.escape(os.path.altsep)] if os.path.altsep is not None else []),
            ]
        )
    )

    @staticmethod
    def looks_like_path(name: str) -> bool:
        """Checks whether the string "looks like" a path on the filesystem.

        This does not check whether the target actually exists, only judge from the
        appearance.

        Returns true if any of the following conditions is true:
        * a path separator is found (either os.path.sep or os.path.altsep);
        * a dot is found (which represents the current directory).
        """
        return _ResourcePathMatcher._path_like_regex.search(name) is not None

    _pep_440_url_heuristic_regex: ClassVar[re.Pattern[str]] = re.compile(
        r"^.*(?=@)",
        flags=re.DOTALL,
    )

    # NB: This commit 16af35c61345866d582b43abfc649a0d79b16c9c inverted all of the logic
    #     in this method and and made it extremely difficult to follow.
    #     See 5b93c0919912efc548caa3e71df5e8bdf9706d2d which introduces the original
    #     version of this logic which the below implementation attempts to revert to.
    @staticmethod
    def get_path_or_url(path: str, name: str) -> ParsedUrl | None:
        """
        First, it checks whether a provided path is an installable directory. If it
        is, returns the path.

        If false, check if the path is an archive file (such as a .whl).
        The function checks if the path is a file. If false, if the path has
        an @, it will treat it as a PEP 440 URL requirement and return the path.
        """
        if _ResourcePathMatcher.looks_like_path(name) and os.path.isdir(path):
            if not is_installable_dir(path):
                raise InstallationError(
                    f"Requirement string {name!r} points to a readable directory "
                    f"at {path!r}, but that directory is not installable. "
                    "For example, neither 'setup.py' nor 'pyproject.toml' were found."
                )
            return path_to_url(path)

        if FileExtensions.archive_file_extension(path):
            # The file ends in an archive-like file extension.

            if os.path.isfile(path):
                # It was a real file!
                return path_to_url(path)

            if pre_at := _ResourcePathMatcher._pep_440_url_heuristic_regex.match(name):
                if _ResourcePathMatcher.looks_like_path(pre_at.group(0)):
                    url_path = path_to_url(path)
                    logger.warning(
                        "Requirement %r looks like an archive filename "
                        "(before the @: %r), but the file at path %r does not exist. "
                        "Interpreting it as a url: %r.",
                        name,
                        pre_at,
                        path,
                        url_path,
                    )
                    return url_path
            else:
                url_path = path_to_url(path)
                logger.warning(
                    "Requirement %r looks like an archive filename, "
                    "but the file at path %r does not exist. "
                    "Interpreting it as a url: %r.",
                    name,
                    path,
                    url_path,
                )
                return url_path

        # No such luck!
        return None


def parse_req_from_line(name: str, line_source: str | None) -> RequirementParts:
    if has_vcs_url_scheme(name):
        marker_sep = "; "
    else:
        marker_sep = ";"
    if marker_sep in name:
        name, markers_as_string = name.split(marker_sep, 1)
        markers_as_string = markers_as_string.strip()
        if not markers_as_string:
            markers = None
        else:
            markers = Marker.parse(markers_as_string)
    else:
        markers = None
    name = name.strip()
    req_as_string = None
    link = None
    extras_as_string = None

    if has_vcs_url_scheme(name):
        name, extras_as_string = _strip_extras(name)
        link = Link(name)
    else:
        p, extras_as_string = _strip_extras(name)
        url = _ResourcePathMatcher.get_path_or_url(p, name)
        if url is not None:
            link = Link(url)

    # it's a local file, dir, or url
    if link:
        # Handle relative file URLs
        # FIXME: optimize the reparsing here too!
        if link.scheme == "file" and re.search(r"\.\./", link.url):
            link = Link(path_to_url(link.path))
        # wheel file
        if link.is_wheel:
            wheel = WheelInfo.parse_filename(
                link.filename
            )  # can raise InvalidWheelFilename
            req_as_string = f"{wheel.name}=={wheel.version}"
        else:
            # set the req to the egg fragment.  when it's not there, this
            # will become an 'unnamed' requirement
            req_as_string = link.egg_fragment

    # a requirement specifier
    else:
        req_as_string = name

    extras = convert_extras(extras_as_string)

    def with_source(text: str) -> str:
        if not line_source:
            return text
        return f"{text} (from {line_source})"

    def _parse_req_string(req_as_string: str) -> Requirement:
        try:
            return get_requirement(req_as_string)
        except InvalidRequirement as exc:
            if os.path.sep in req_as_string:
                add_msg = "It looks like a path."
                add_msg += deduce_helpful_msg(req_as_string)
            elif "=" in req_as_string and not any(
                op.value in req_as_string for op in Operator
            ):
                add_msg = "= is not a valid operator. Did you mean == ?"
            else:
                add_msg = ""
            msg = with_source(f"Invalid requirement: {req_as_string!r}: {exc}")
            if add_msg:
                msg += f"\nHint: {add_msg}"
            raise InstallationError(msg)

    if req_as_string is not None:
        req: Requirement | None = _parse_req_string(req_as_string)
    else:
        req = None

    return RequirementParts(req, link, markers, extras)


def install_req_from_line(
    name: str,
    comes_from: str | InstallRequirement | None = None,
    *,
    use_pep517: bool | None = None,
    isolated: bool = False,
    global_options: list[str] | None = None,
    hash_options: dict[str, list[str]] | None = None,
    constraint: bool = False,
    line_source: str | None = None,
    user_supplied: bool = False,
    config_settings: dict[str, str | list[str]] | None = None,
) -> InstallRequirement:
    """Creates an InstallRequirement from a name, which might be a
    requirement, directory containing 'setup.py', filename, or URL.

    :param line_source: An optional string describing where the line is from,
        for logging purposes in case of an error.
    """
    parts = parse_req_from_line(name, line_source)

    return InstallRequirement(
        parts.requirement,
        comes_from,
        link=parts.link,
        markers=parts.markers,
        use_pep517=use_pep517,
        isolated=isolated,
        global_options=global_options,
        hash_options=hash_options,
        config_settings=config_settings,
        constraint=constraint,
        extras=parts.extras,
        user_supplied=user_supplied,
    )


def install_req_from_req_string(
    req_string: str,
    comes_from: InstallRequirement | None = None,
    isolated: bool = False,
    use_pep517: bool | None = None,
    user_supplied: bool = False,
) -> InstallRequirement:
    try:
        req = get_requirement(req_string)
    except InvalidRequirement as exc:
        raise InstallationError(f"Invalid requirement: {req_string!r}: {exc}")

    domains_not_allowed = [
        PyPI.file_storage_domain,
        TestPyPI.file_storage_domain,
    ]
    if (
        req.url
        and comes_from
        and comes_from.link
        and comes_from.link.netloc in domains_not_allowed
    ):
        # Explicitly disallow pypi packages that depend on external urls
        raise InstallationError(
            "Packages installed from PyPI cannot depend on packages "
            "which are not also hosted on PyPI.\n"
            f"{comes_from.name} depends on {req} "
        )

    return InstallRequirement(
        req,
        comes_from,
        isolated=isolated,
        use_pep517=use_pep517,
        user_supplied=user_supplied,
    )


def install_req_from_parsed_requirement(
    parsed_req: ParsedRequirement,
    isolated: bool = False,
    use_pep517: bool | None = None,
    user_supplied: bool = False,
    config_settings: dict[str, str | list[str]] | None = None,
) -> InstallRequirement:
    if parsed_req.is_editable:
        req = install_req_from_editable(
            parsed_req.requirement,
            comes_from=parsed_req.comes_from,
            use_pep517=use_pep517,
            constraint=parsed_req.constraint,
            isolated=isolated,
            user_supplied=user_supplied,
            config_settings=config_settings,
        )

    else:
        req = install_req_from_line(
            parsed_req.requirement,
            comes_from=parsed_req.comes_from,
            use_pep517=use_pep517,
            isolated=isolated,
            global_options=(
                parsed_req.options.get("global_options", [])
                if parsed_req.options
                else []
            ),
            hash_options=(
                parsed_req.options.get("hashes", {}) if parsed_req.options else {}
            ),
            constraint=parsed_req.constraint,
            line_source=parsed_req.line_source,
            user_supplied=user_supplied,
            config_settings=config_settings,
        )
    return req


def install_req_from_link_and_ireq(
    link: Link, ireq: InstallRequirement
) -> InstallRequirement:
    return InstallRequirement(
        req=ireq.req,
        comes_from=ireq.comes_from,
        editable=ireq.editable,
        link=link,
        markers=ireq.markers,
        use_pep517=ireq.use_pep517,
        isolated=ireq.isolated,
        global_options=ireq.global_options,
        hash_options=ireq.hash_options,
        config_settings=ireq.config_settings,
        user_supplied=ireq.user_supplied,
    )


def install_req_drop_extras(ireq: InstallRequirement) -> InstallRequirement:
    """
    Creates a new InstallationRequirement using the given template but without
    any extras. Sets the original requirement as the new one's parent
    (comes_from).
    """
    return InstallRequirement(
        req=(
            _set_requirement_extras(ireq.req, set()) if ireq.req is not None else None
        ),
        comes_from=ireq,
        editable=ireq.editable,
        link=ireq.link,
        markers=ireq.markers,
        use_pep517=ireq.use_pep517,
        isolated=ireq.isolated,
        global_options=ireq.global_options,
        hash_options=ireq.hash_options,
        constraint=ireq.constraint,
        extras=[],
        config_settings=ireq.config_settings,
        user_supplied=ireq.user_supplied,
        permit_editable_wheels=ireq.permit_editable_wheels,
    )


def install_req_extend_extras(
    ireq: InstallRequirement,
    extras: Collection[str],
) -> InstallRequirement:
    """
    Returns a copy of an installation requirement with some additional extras.
    Makes a shallow copy of the ireq object.
    """
    result = copy.copy(ireq)
    result.extras = {*ireq.extras, *extras}
    result.req = (
        _set_requirement_extras(ireq.req, result.extras)
        if ireq.req is not None
        else None
    )
    return result
