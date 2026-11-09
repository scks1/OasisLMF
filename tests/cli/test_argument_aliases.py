"""Commands accept the `ods_tools check` names for the reinsurance OED files (#2228)."""
import argparse

import pytest

import oasislmf.cli.root  # noqa: F401 - imports every command so the subclasses below are registered
from oasislmf.cli.command import ARGUMENT_ALIASES, OasisComputationCommand
from oasislmf.cli.exposure import RunCmd
from oasislmf.manager import OasisManager


def _all_subclasses(cls):
    for sub in cls.__subclasses__():
        yield sub
        yield from _all_subclasses(sub)


def _commands_with_cli_param(name):
    commands = set()
    for cmd_cls in _all_subclasses(OasisComputationCommand):
        params = OasisManager.computations_params.get(getattr(cmd_cls, 'computation_name', None), [])
        if any(p['name'] == name and p.get('help') for p in params):
            commands.add(cmd_cls)
    return sorted(commands, key=lambda c: f'{c.__module__}.{c.__name__}')


def _parser(cmd_cls):
    parser = argparse.ArgumentParser()
    cmd_cls(argv=[]).add_args(parser)
    return parser


ALIASED_COMMANDS = _commands_with_cli_param('oed_info_csv')


def test_aliased_commands_found():
    assert RunCmd in ALIASED_COMMANDS
    assert len(ALIASED_COMMANDS) > 1


@pytest.mark.parametrize('cmd_cls', ALIASED_COMMANDS, ids=lambda c: c.__name__)
def test_ri_info_and_ri_scope_aliases(cmd_cls):
    args = _parser(cmd_cls).parse_args(['--ri-info', 'info.csv', '--ri-scope', 'scope.csv'])
    assert args.oed_info_csv == 'info.csv'
    assert args.oed_scope_csv == 'scope.csv'


def test_existing_names_still_work():
    args = _parser(RunCmd).parse_args(['-i', 'info.csv', '--oed-scope-csv', 'scope.csv'])
    assert args.oed_info_csv == 'info.csv'
    assert args.oed_scope_csv == 'scope.csv'


def test_aliases_listed_in_help():
    help_text = _parser(RunCmd).format_help()
    for aliases in ARGUMENT_ALIASES.values():
        for alias in aliases:
            assert alias in help_text


def test_location_and_account_still_filter_the_exposure():
    """`--location`/`--account` keep their existing meaning; only the RI file names are aliased."""
    args = _parser(RunCmd).parse_args(['--location', '1', '2', '--account', 'A'])
    assert args.location == ['1', '2']
    assert args.account == ['A']
