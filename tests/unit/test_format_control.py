from optparse import Values

import pytest

from pip._internal.cli import cmdoptions
from pip._internal.cli.base_command import Command
from pip._internal.cli.status_codes import SUCCESS
from pip._internal.models.format_control import AllowedFormats, FormatControlBuilder


class SimpleCommand(Command):
    def __init__(self) -> None:
        super().__init__("fake", "fake summary")

    def add_options(self) -> None:
        self.cmd_opts.add_option(cmdoptions.no_binary())
        self.cmd_opts.add_option(cmdoptions.only_binary())

    def run(self, options: Values, args: list[str]) -> int:
        self.options = options
        return SUCCESS


def test_no_binary_overrides() -> None:
    cmd = SimpleCommand()
    cmd.main(["fake", "--only-binary=:all:", "--no-binary=fred"])
    format_control = FormatControlBuilder({"fred"}, {":all:"})
    assert cmd.options.format_control == format_control


def test_only_binary_overrides() -> None:
    cmd = SimpleCommand()
    cmd.main(["fake", "--no-binary=:all:", "--only-binary=fred"])
    format_control = FormatControlBuilder({":all:"}, {"fred"})
    assert cmd.options.format_control == format_control


def test_none_resets() -> None:
    cmd = SimpleCommand()
    cmd.main(["fake", "--no-binary=:all:", "--no-binary=:none:"])
    format_control = FormatControlBuilder(set(), set())
    assert cmd.options.format_control == format_control


def test_none_preserves_other_side() -> None:
    cmd = SimpleCommand()
    cmd.main(["fake", "--no-binary=:all:", "--only-binary=fred", "--no-binary=:none:"])
    format_control = FormatControlBuilder(set(), {"fred"})
    assert cmd.options.format_control == format_control


def test_comma_separated_values() -> None:
    cmd = SimpleCommand()
    cmd.main(["fake", "--no-binary=1,2,3"])
    format_control = FormatControlBuilder({"1", "2", "3"}, set())
    assert cmd.options.format_control == format_control


@pytest.mark.parametrize(
    "no_binary,only_binary,argument,expected",
    [
        ({"fred"}, set(), "fred", AllowedFormats.SourceOnly),
        ({"fred"}, {":all:"}, "fred", AllowedFormats.SourceOnly),
        (set(), {"fred"}, "fred", AllowedFormats.BinaryOnly),
        ({":all:"}, {"fred"}, "fred", AllowedFormats.BinaryOnly),
    ],
)
def test_fmt_ctl_matches(
    no_binary: set[str], only_binary: set[str], argument: str, expected: AllowedFormats
) -> None:
    fmt = FormatControlBuilder(no_binary, only_binary).build()
    assert fmt.get_allowed_formats(argument) == expected
