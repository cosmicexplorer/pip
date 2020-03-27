# The following comment should be removed at some point in the future.
# mypy: disallow-untyped-defs=False

from __future__ import absolute_import

import logging
import os
import sys

from pip._vendor.packaging.requirements import Requirement
from pip._vendor.packaging.version import Version

from pip._internal.cli import cmdoptions
from pip._internal.cli.cmdoptions import make_target_python
from pip._internal.cli.req_command import (
    RequirementCommand,
    SessionCommandMixin,
)
from pip._internal.models.link import Link
from pip._internal.req import RequirementSet
from pip._internal.req.req_tracker import get_requirement_tracker
from pip._internal.utils.filesystem import check_path_owner
from pip._internal.utils.misc import ensure_dir, normalize_path, write_output
from pip._internal.utils.temp_dir import TempDirectory
from pip._internal.utils.typing import MYPY_CHECK_RUNNING


if MYPY_CHECK_RUNNING:
    from typing import List, Tuple
    from pip._internal.index.package_finder import PackageFinder
    from pip._internal.req.req_install import InstallRequirement


logger = logging.getLogger(__name__)


class ResolveCommand(RequirementCommand, SessionCommandMixin):
    """
    EXPERIMENTAL

    Translate the list of input requirements to a list of transitive ==
    requirements. The effect of resolving the output list of requirements
    should be the exact same as the input list, given some specific state of
    --index and --find-links urls at the time this command was run.

    This supports all of the same inputs as the `download` command.
    """

    usage = """
      %prog [options] <requirement specifier> [package-index-options] ...
      %prog [options] -r <requirements file> [package-index-options] ...
      %prog [options] <vcs project url> ...
      %prog [options] <local project path> ...
      %prog [options] <archive url/path> ..."""

    def __init__(self, *args, **kw):
        super(ResolveCommand, self).__init__(*args, **kw)

        cmd_opts = self.cmd_opts

        cmd_opts.add_option(cmdoptions.constraints())
        cmd_opts.add_option(cmdoptions.requirements())
        cmd_opts.add_option(cmdoptions.build_dir())
        cmd_opts.add_option(cmdoptions.no_deps())
        cmd_opts.add_option(cmdoptions.global_options())
        cmd_opts.add_option(cmdoptions.no_binary())
        cmd_opts.add_option(cmdoptions.only_binary())
        cmd_opts.add_option(cmdoptions.prefer_binary())
        cmd_opts.add_option(cmdoptions.src())
        cmd_opts.add_option(cmdoptions.pre())
        cmd_opts.add_option(cmdoptions.require_hashes())
        cmd_opts.add_option(cmdoptions.progress_bar())
        cmd_opts.add_option(cmdoptions.no_build_isolation())
        cmd_opts.add_option(cmdoptions.use_pep517())
        cmd_opts.add_option(cmdoptions.no_use_pep517())

        cmd_opts.add_option(
            '-d', '--dest', '--destination-dir', '--destination-directory',
            dest='download_dir',
            metavar='dir',
            default=os.curdir,
            help=("Download packages into <dir>."),
        )

        cmdoptions.add_target_python_options(cmd_opts)

        index_opts = cmdoptions.make_option_group(
            cmdoptions.index_group,
            self.parser,
        )

        self.parser.insert_option_group(0, index_opts)
        self.parser.insert_option_group(0, cmd_opts)

    class InvalidPackageUrlError(Exception):
        pass

    def _infer_package_name_and_version_from_url(self, req, finder):
        # type: (InstallRequirement, PackageFinder) -> Tuple[str, Version, str]
        name = req.req.name
        assert req.link
        url = req.link.url
        assert url
        link_evaluator = finder.make_link_evaluator(name)
        is_candidate, result = link_evaluator.evaluate_link(Link(url))
        if not is_candidate:
            raise self.InvalidPackageUrlError(result)

        return (name, Version(result), url)

    def run(self, options, args):
        options.ignore_installed = True
        # editable doesn't really make sense for `pip resolve`, but the bowels
        # of the RequirementSet code require that property.
        options.editables = []

        cmdoptions.check_dist_restriction(options)

        options.download_dir = normalize_path(options.download_dir)

        ensure_dir(options.download_dir)

        session = self.get_default_session(options)

        target_python = make_target_python(options)
        finder = self._build_package_finder(
            options=options,
            session=session,
            target_python=target_python,
        )
        build_delete = (not options.build_dir)

        with get_requirement_tracker() as req_tracker, TempDirectory(
            options.build_dir, delete=build_delete, kind="download"
        ) as directory:

            requirement_set = RequirementSet()
            self.populate_requirement_set(
                requirement_set,
                args,
                options,
                finder,
                session,
                None
            )

            preparer = self.make_requirement_preparer(
                temp_build_dir=directory,
                options=options,
                req_tracker=req_tracker,
                session=session,
                finder=finder,
                download_dir=options.download_dir,
                use_user_site=False,
                quickly_parse_sub_requirements=(
                    options.quickly_parse_sub_requirements),
            )

            resolver = self.make_resolver(
                preparer=preparer,
                finder=finder,
                options=options,
                py_version_info=options.python_version,
                quickly_parse_sub_requirements=(
                    options.quickly_parse_sub_requirements),
                session=self._session,
            )

            self.trace_basic_info(finder)

            completed_requirement_set = resolver.resolve(
                requirement_set.requirements.values(),
                check_supported_wheels=True)

        requirement_resolve_output_entries = []  # type: List[str]
        for req in completed_requirement_set.requirements.values():
            name, version, url = self._infer_package_name_and_version_from_url(
                req, finder)
            entry = '{}=={} ({})'.format(name, version, url)
            requirement_resolve_output_entries.append(entry)

        sys.stdout.write(
            'Resolve output:\n{}\n'
            .format('\n'.join(requirement_resolve_output_entries)))

        return requirement_set
