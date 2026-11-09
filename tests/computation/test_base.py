import json
import logging
import os

import pytest

from oasislmf.computation.base import ComputationStep
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.log_config import OasisLogConfig


class DummyComputationStep(ComputationStep):
    """Minimal concrete ComputationStep used to exercise _apply_log_config()."""

    def run(self):
        pass


@pytest.fixture
def oasis_loggers():
    """Give each test a clean 'oasislmf'/'ods_tools' logger state to assert against,
    and restore whatever was there before once the test is done so other tests /
    modules relying on global logger state aren't affected.
    """
    logger = logging.getLogger('oasislmf')
    ods_logger = logging.getLogger('ods_tools')

    orig_level = logger.level
    orig_ods_level = ods_logger.level
    orig_handlers = list(logger.handlers)

    for h in orig_handlers:
        logger.removeHandler(h)

    handler = logging.StreamHandler()
    handler.name = 'oasislmf'
    handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(handler)
    logger.setLevel(logging.WARNING)
    ods_logger.setLevel(logging.WARNING)

    yield logger, ods_logger, handler

    logger.removeHandler(handler)
    for h in orig_handlers:
        logger.addHandler(h)
    logger.setLevel(orig_level)
    ods_logger.setLevel(orig_ods_level)


def test_no_log_kwargs__logger_untouched(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep()

    assert logger.level == logging.WARNING
    assert ods_logger.level == logging.WARNING
    assert handler.formatter._fmt == '%(message)s'


def test_verbose_only__logger_untouched(oasis_loggers):
    """verbose must never influence the logger here - that's setup_logger()'s job,
    and it has context (the nested MDK 'logging' config block) this class doesn't."""
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(verbose=True)

    assert logger.level == logging.WARNING
    assert ods_logger.level == logging.WARNING
    assert handler.formatter._fmt == '%(message)s'


def test_log_level_set__applied_to_logger_and_ods_tools(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(log_level='DEBUG')

    assert logger.level == logging.DEBUG
    # DEBUG on the main logger also raises ods_tools to DEBUG (see get_ods_tools_level)
    assert ods_logger.level == logging.DEBUG


def test_log_level_set_non_debug__ods_tools_stays_warning(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(log_level='ERROR')

    assert logger.level == logging.ERROR
    assert ods_logger.level == logging.WARNING


def test_log_format_set__formatter_applied_to_all_handlers(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(log_format='compact')

    expected_fmt = OasisLogConfig.FORMAT_TEMPLATES['compact']
    assert handler.formatter._fmt == expected_fmt
    # log_format alone must not touch the level
    assert logger.level == logging.WARNING


def test_log_level_and_log_format_set_together(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(log_level='DEBUG', log_format='simple')

    assert logger.level == logging.DEBUG
    assert ods_logger.level == logging.DEBUG
    assert handler.formatter._fmt == OasisLogConfig.FORMAT_TEMPLATES['simple']


def test_verbose_does_not_override_explicit_log_level(oasis_loggers):
    """Regression test: verbose must not be folded into the level computation
    alongside an explicit log_level - the explicit value always wins outright."""
    logger, ods_logger, handler = oasis_loggers

    DummyComputationStep(verbose=True, log_level='ERROR')

    assert logger.level == logging.ERROR


def test_global_params_choices_and_no_cli_flag():
    """log_level/log_format/verbose should be collectable step params with the right
    choices, so they show up correctly in the generated computation settings schema,
    but must not be exposed as their own CLI flag (no 'help' key) since those are
    already registered directly by OasisBaseCommand."""
    params = {p['name']: p for p in DummyComputationStep.get_params()}

    assert params['log_level']['choices'] == OasisLogConfig.STANDARD_LEVELS
    assert params['log_format']['choices'] == list(OasisLogConfig.FORMAT_TEMPLATES.keys())
    assert params['verbose']['default'] is False

    for name in ('verbose', 'log_level', 'log_format'):
        assert 'help' not in params[name]


def test_mdk_config_logging_block_honoured_when_log_level_reapplied(oasis_loggers, tmp_path):
    """ods_tools_level from the MDK config file's "logging" block must still be applied
    when the level is re-applied from computation settings."""
    logger, ods_logger, handler = oasis_loggers
    config_fp = tmp_path / 'oasislmf.json'
    config_fp.write_text(json.dumps({'logging': {'ods_tools_level': 'ERROR'}}))

    DummyComputationStep(log_level='DEBUG', config=str(config_fp))

    assert logger.level == logging.DEBUG
    assert ods_logger.level == logging.ERROR


@pytest.mark.parametrize('config_content', [None, '{not valid json'])
def test_unreadable_mdk_config__warns_and_falls_back_to_defaults(oasis_loggers, tmp_path, caplog, config_content):
    logger, ods_logger, handler = oasis_loggers
    config_fp = tmp_path / 'oasislmf.json'
    if config_content is not None:
        config_fp.write_text(config_content)

    with caplog.at_level(logging.WARNING, logger='oasislmf.computation.base'):
        DummyComputationStep(log_level='DEBUG', config=str(config_fp))

    assert f"Could not re-load MDK config file for logging: {config_fp}" in caplog.text
    assert logger.level == logging.DEBUG
    assert ods_logger.level == logging.DEBUG


def test_log_format_set__other_handlers_untouched(oasis_loggers):
    logger, ods_logger, handler = oasis_loggers
    other_handler = logging.StreamHandler()
    other_handler.name = 'other'
    other_handler.setFormatter(logging.Formatter('%(message)s'))
    logger.addHandler(other_handler)
    try:
        DummyComputationStep(log_format='compact')
    finally:
        logger.removeHandler(other_handler)

    assert handler.formatter._fmt == OasisLogConfig.FORMAT_TEMPLATES['compact']
    assert other_handler.formatter._fmt == '%(message)s'


def test_get_default_run_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    DummyComputationStep.run_dir_key = 'dummy'
    try:
        run_dir = DummyComputationStep.get_default_run_dir()
    finally:
        del DummyComputationStep.run_dir_key

    assert os.path.dirname(run_dir) == os.path.join(str(tmp_path), 'runs')
    assert os.path.basename(run_dir).startswith('dummy-')


class DummySchemaStep(DummyComputationStep):
    step_params = [
        {'name': 'some_flag', 'default': False, 'help': 'A boolean flag'},
        {'name': 'some_choice', 'choices': ['a', 'b']},
        {'name': 'some_count', 'type': int},
        {'name': 'some_ratio', 'type': float},
        {'name': 'some_settings_json', 'is_path': True},
    ]
    settings_params = [{'name': 'some_settings_json', 'loader': json.load}]


def test_computation_settings_json_schema():
    """Global logging params appear in the schema, but 'config' (exclude_from_schema)
    and settings files (settings_params) must not."""
    properties = DummySchemaStep.get_computation_settings_json_schema()['properties']

    assert 'config' not in properties
    assert 'some_settings_json' not in properties
    assert properties['log_level']['enum'] == OasisLogConfig.STANDARD_LEVELS
    assert properties['log_format']['enum'] == list(OasisLogConfig.FORMAT_TEMPLATES.keys())
    assert properties['verbose'] == {'type': 'boolean'}
    assert properties['some_flag'] == {'type': 'boolean', 'description': 'A boolean flag'}
    assert properties['some_choice'] == {'type': 'string', 'enum': ['a', 'b']}
    assert properties['some_count'] == {'type': 'integer'}
    assert properties['some_ratio'] == {'type': 'number'}


def _load_json(fp):
    with open(fp) as f:
        return json.load(f)


class DummySettingsStep(DummyComputationStep):
    step_params = [
        {'name': 'num_processes', 'type': int, 'default': 1},
        {'name': 'run_dir', 'default': None},
        {'name': 'settings_json', 'is_path': True},
    ]
    settings_params = [{'name': 'settings_json', 'loader': _load_json}]


@pytest.fixture
def settings_file(tmp_path):
    def write(computation_settings):
        path = tmp_path / 'analysis_settings.json'
        path.write_text(json.dumps({'computation_settings': computation_settings}))
        return str(path)
    return write


def test_get_arguments_admin_computation_settings_override_everything(settings_file):
    """The default role keeps the old behaviour: the settings file wins over kwargs."""
    fp = settings_file({'num_processes': 8, 'run_dir': '/elsewhere'})
    args = DummySettingsStep.get_arguments(settings_json=fp, run_dir='/runs/1')
    assert args['num_processes'] == 8
    assert args['run_dir'] == '/elsewhere'


def test_get_arguments_user_computation_settings_only_change_defaults(settings_file, caplog):
    """With the 'user' role, values the caller passed are kept; defaults can still be changed (#1819)."""
    fp = settings_file({'num_processes': 8, 'run_dir': '/elsewhere'})
    with caplog.at_level(logging.WARNING, logger='oasislmf.computation.base'):
        args = DummySettingsStep.get_arguments(settings_json=fp, run_dir='/runs/1', computation_settings_user_role='user')
    assert args['num_processes'] == 8
    assert args['run_dir'] == '/runs/1'
    assert "ignoring ['run_dir']" in caplog.text
    assert "by the 'user' role" in caplog.text


def test_get_arguments_user_role_treats_none_kwargs_as_unset(settings_file):
    """OasisManager passes every parameter, unset ones as None: those stay changeable."""
    fp = settings_file({'run_dir': '/elsewhere'})
    args = DummySettingsStep.get_arguments(settings_json=fp, run_dir=None, computation_settings_user_role='user')
    assert args['run_dir'] == '/elsewhere'


def test_get_arguments_user_role_keeps_env_override(settings_file, monkeypatch):
    monkeypatch.setenv('OASIS_ENV_OVERRIDE', 'True')
    monkeypatch.setenv('OASIS_NUM_PROCESSES', '4')
    fp = settings_file({'num_processes': 8})
    args = DummySettingsStep.get_arguments(settings_json=fp, computation_settings_user_role='user')
    assert args['num_processes'] == 4


@pytest.mark.parametrize('role', ['admin', 'user'])
def test_get_arguments_settings_file_cannot_set_its_own_role_or_config(settings_file, role, caplog):
    """'computation_settings_user_role' and 'config' (exclude_from_schema) can't come from the file."""
    fp = settings_file({'computation_settings_user_role': 'admin', 'config': 'other.json', 'run_dir': '/elsewhere'})
    with caplog.at_level(logging.WARNING, logger='oasislmf.computation.base'):
        args = DummySettingsStep.get_arguments(settings_json=fp, run_dir='/runs/1', computation_settings_user_role=role)
    assert args['computation_settings_user_role'] == role
    assert args['config'] is None
    assert args['run_dir'] == ('/elsewhere' if role == 'admin' else '/runs/1')
    assert "'computation_settings_user_role'" in caplog.text and "'config'" in caplog.text


def test_get_arguments_rejects_unknown_role(settings_file):
    with pytest.raises(OasisException, match="computation_settings_user_role must be one of"):
        DummySettingsStep.get_arguments(settings_json=settings_file({}), computation_settings_user_role='staff')


def test_computation_settings_user_role_not_in_schema():
    properties = DummySettingsStep.get_computation_settings_json_schema()['properties']
    assert 'computation_settings_user_role' not in properties
    assert 'num_processes' in properties
