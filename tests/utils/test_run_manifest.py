import hashlib
import json
import os
from unittest import mock

import pytest

from oasislmf.computation.generate.files import GenerateFiles
from oasislmf.computation.generate.losses import GenerateLosses
from oasislmf.computation.run.manifest import VerifyRun, WriteRunManifest
from oasislmf.computation.run.model import RunModel
from oasislmf.manager import OasisManager
from oasislmf.utils.exceptions import OasisException
from oasislmf.utils.run_manifest import (MANIFEST_NAME, build_manifest, compare_manifest, read_manifest, verify_run,
                                         write_manifest)
from tests.computation.data.common import MIN_MODEL_SETTINGS, MIN_RUN_SETTINGS


def _write(path, text):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write(text)


@pytest.fixture
def run_dir(tmp_path):
    model_data = tmp_path / 'model_data'
    _write(str(model_data / 'footprint.csv'), 'event_id,areaperil_id\n1,1\n')
    run = tmp_path / 'run'
    _write(str(run / 'input' / 'items.csv'), 'item_id\n1\n')
    _write(str(run / 'input' / 'analysis_settings.json'), '{}')
    _write(str(run / 'output' / 'gul_S1_palt.csv'), 'SummaryId,MeanLoss\n1,10.0\n')
    _write(str(run / 'work' / 'scratch.bin'), 'not hashed')
    os.symlink(str(model_data), str(run / 'static'))
    _write(str(tmp_path / 'analysis_settings.json'), '{"model_name_id": "M"}')
    return run


def _settings(run_dir):
    return {'analysis_settings_json': str(run_dir.parent / 'analysis_settings.json')}


def test_build_manifest_hashes_inputs_and_outputs(run_dir):
    manifest = build_manifest(str(run_dir), settings_files=_settings(run_dir))

    assert list(manifest['files']) == ['input/analysis_settings.json', 'input/items.csv', 'output/gul_S1_palt.csv']
    assert manifest['files']['input/items.csv'] == {'sha256': hashlib.sha256(b'item_id\n1\n').hexdigest(), 'size': 10}
    assert manifest['model_data'] == {'path': str((run_dir.parent / 'model_data').resolve()), 'files': None}
    assert manifest['settings']['analysis_settings_json']['sha256'] == hashlib.sha256(b'{"model_name_id": "M"}').hexdigest()
    env = manifest['environment']
    assert env['packages']['oasislmf'] and env['packages']['numpy'] and env['python']


def test_build_manifest_records_oasis_environment(run_dir, monkeypatch):
    monkeypatch.setenv('OASIS_AREAPERIL_TYPE', 'u8')
    monkeypatch.setenv('NOT_OASIS', 'x')

    env = build_manifest(str(run_dir))['environment']['oasis_env']

    assert env['OASIS_AREAPERIL_TYPE'] == 'u8'
    assert 'NOT_OASIS' not in env


def test_build_manifest_can_hash_model_data(run_dir):
    manifest = build_manifest(str(run_dir), hash_model_data=True)

    assert list(manifest['model_data']['files']) == ['footprint.csv']


def test_model_data_in_a_static_dir_of_links_is_hashed_through_the_links(run_dir):
    os.remove(str(run_dir / 'static'))
    (run_dir / 'static').mkdir()
    os.symlink(str(run_dir.parent / 'model_data' / 'footprint.csv'), str(run_dir / 'static' / 'footprint.csv'))
    write_manifest(str(run_dir), hash_model_data=True)
    assert read_manifest(str(run_dir))['model_data']['path'] == str(run_dir / 'static')

    _write(str(run_dir.parent / 'model_data' / 'footprint.csv'), 'changed\n')

    assert verify_run(str(run_dir))['model_data'] == ['footprint.csv']


def test_build_manifest_needs_a_run_dir(tmp_path):
    with pytest.raises(OasisException, match='not found'):
        build_manifest(str(tmp_path / 'nope'))


def test_manifest_does_not_hash_itself(run_dir):
    write_manifest(str(run_dir))
    _write(str(run_dir / 'output' / MANIFEST_NAME), '{}')

    assert 'output/run_manifest.json' not in build_manifest(str(run_dir))['files']


def test_unchanged_run_verifies(run_dir):
    write_manifest(str(run_dir), settings_files=_settings(run_dir), hash_model_data=True)

    result = verify_run(str(run_dir))

    assert result['match']
    assert result['files']['matched'] == 3
    assert result['environment'] == {}


def test_changed_missing_and_added_files_are_reported(run_dir):
    write_manifest(str(run_dir))
    _write(str(run_dir / 'output' / 'gul_S1_palt.csv'), 'SummaryId,MeanLoss\n1,10.5\n')
    os.remove(str(run_dir / 'input' / 'items.csv'))
    _write(str(run_dir / 'output' / 'extra.csv'), 'x\n')

    result = verify_run(str(run_dir))

    assert not result['match']
    assert result['files']['changed'] == ['output/gul_S1_palt.csv']
    assert result['files']['missing'] == ['input/items.csv']
    assert result['files']['added'] == ['output/extra.csv']


def test_changed_settings_file_is_reported(run_dir):
    write_manifest(str(run_dir), settings_files=_settings(run_dir))
    _write(str(run_dir.parent / 'analysis_settings.json'), '{"model_name_id": "N"}')

    result = verify_run(str(run_dir))

    assert not result['match'] and result['settings'] == ['analysis_settings_json']


def test_changed_model_data_is_reported_when_hashed(run_dir):
    write_manifest(str(run_dir), hash_model_data=True)
    _write(str(run_dir.parent / 'model_data' / 'footprint.csv'), 'event_id,areaperil_id\n1,2\n')

    assert verify_run(str(run_dir))['model_data'] == ['footprint.csv']


def test_model_data_is_not_compared_when_not_hashed(run_dir):
    write_manifest(str(run_dir))
    _write(str(run_dir.parent / 'model_data' / 'footprint.csv'), 'changed\n')

    assert verify_run(str(run_dir))['match']


def test_environment_differences_are_reported_but_do_not_fail(run_dir, monkeypatch):
    write_manifest(str(run_dir))
    monkeypatch.setenv('OASIS_AREAPERIL_TYPE', 'u8')

    result = verify_run(str(run_dir))

    assert result['match']
    assert result['environment']['oasis_env.OASIS_AREAPERIL_TYPE'] == {'expected': None, 'actual': 'u8'}


def _copy_run(run_dir, name):
    other = run_dir.parent / name
    for sub in ('input', 'output'):
        for f in os.listdir(str(run_dir / sub)):
            with open(str(run_dir / sub / f)) as src:
                _write(str(other / sub / f), src.read())
    return other


def test_run_verifies_against_an_identical_run(run_dir):
    write_manifest(str(run_dir), settings_files=_settings(run_dir))
    other = _copy_run(run_dir, 'other')

    result = verify_run(str(other), against=str(run_dir))

    assert result['match'], result
    assert result['settings'] == []


def test_run_against_a_different_run_lists_the_differences(run_dir):
    write_manifest(str(run_dir))
    other = _copy_run(run_dir, 'other')
    _write(str(other / 'output' / 'gul_S1_palt.csv'), 'SummaryId,MeanLoss\n1,10.000001\n')

    result = verify_run(str(other), against=str(run_dir / MANIFEST_NAME))

    assert result['files']['changed'] == ['output/gul_S1_palt.csv']


def test_run_against_another_uses_its_own_settings_and_environment(run_dir):
    write_manifest(str(run_dir), settings_files=_settings(run_dir))
    other = _copy_run(run_dir, 'other')
    _write(str(run_dir.parent / 'other_settings.json'), '{"model_name_id": "Other"}')
    write_manifest(str(other), settings_files={'analysis_settings_json': str(run_dir.parent / 'other_settings.json')})

    result = verify_run(str(other), against=str(run_dir))

    assert result['settings'] == ['analysis_settings_json']
    assert not result['match']


def test_compare_manifest_checks_model_data_only_when_both_hashed(run_dir):
    hashed = build_manifest(str(run_dir), hash_model_data=True)
    unhashed = build_manifest(str(run_dir))
    hashed['model_data']['files']['footprint.csv']['sha256'] = '0'

    assert compare_manifest(hashed, unhashed)['model_data'] == []
    assert compare_manifest(hashed, build_manifest(str(run_dir), hash_model_data=True))['model_data'] == ['footprint.csv']


def test_read_manifest_errors(run_dir):
    with pytest.raises(OasisException, match="oasislmf model manifest"):
        read_manifest(str(run_dir))
    _write(str(run_dir / MANIFEST_NAME), json.dumps({'manifest_version': 99}))
    with pytest.raises(OasisException, match='manifest version 99'):
        read_manifest(str(run_dir))


def test_steps_write_and_verify(run_dir, tmp_path):
    settings = _settings(run_dir)['analysis_settings_json']
    manifest = WriteRunManifest(model_run_dir=str(run_dir), analysis_settings_json=settings).run()
    assert manifest['settings']['analysis_settings_json']['path'] == settings
    assert os.path.isfile(str(run_dir / MANIFEST_NAME))

    report = str(tmp_path / 'report.json')
    assert VerifyRun(model_run_dir=str(run_dir), verify_report_json=report).run()['match']
    with open(report) as f:
        assert json.load(f)['match']

    _write(str(run_dir / 'output' / 'gul_S1_palt.csv'), 'changed\n')
    with pytest.raises(OasisException, match='does not match its manifest'):
        VerifyRun(model_run_dir=str(run_dir)).run()


def test_manager_exposes_the_steps():
    assert hasattr(OasisManager(), 'write_run_manifest') and hasattr(OasisManager(), 'verify_run')


@pytest.mark.parametrize('enabled', [True, False])
def test_run_model_writes_the_manifest_when_asked(tmp_path, monkeypatch, enabled):
    calls = []

    def fake_run(name):
        def run(self):
            calls.append(name)
        return run

    monkeypatch.setattr(GenerateFiles, 'run', fake_run('GenerateFiles'))
    monkeypatch.setattr(GenerateLosses, 'run', fake_run('GenerateLosses'))
    run = str(tmp_path / 'run')
    (tmp_path / 'model_data').mkdir()
    analysis_fp, model_fp = str(tmp_path / 'analysis.json'), str(tmp_path / 'model.json')
    _write(analysis_fp, json.dumps(MIN_RUN_SETTINGS))
    _write(model_fp, json.dumps(MIN_MODEL_SETTINGS))
    with mock.patch('oasislmf.computation.run.model.get_exposure_data'), \
            mock.patch('oasislmf.computation.run.model.write_run_manifest') as writer:
        step = RunModel(model_run_dir=run, model_data_dir=str(tmp_path / 'model_data'), analysis_settings_json=analysis_fp,
                        model_settings_json=model_fp, run_manifest=enabled)
        step.run()

    assert calls == ['GenerateFiles', 'GenerateLosses']
    assert writer.called == enabled
    if enabled:
        writer.assert_called_once_with(step, run)
