__all__ = [
    'WriteRunManifest',
    'VerifyRun',
]

import json

from ..base import ComputationStep
from ...utils.exceptions import OasisException
from ...utils.inputs import str2bool
from ...utils.run_manifest import verify_run, write_manifest

SETTINGS_FILE_PARAMS = ('analysis_settings_json', 'model_settings_json')
MAX_LISTED = 20


def write_run_manifest(step, run_dir):
    """Write the manifest of a run from a computation step's settings parameters.

    Args:
        step (ComputationStep): Step with ``analysis_settings_json``, ``model_settings_json``, ``model_data_dir`` and
            ``run_manifest_model_data`` attributes (any may be missing or None).
        run_dir (str): Model run directory.

    Returns:
        dict: The manifest written.
    """
    return write_manifest(
        run_dir,
        settings_files={name: getattr(step, name, None) for name in SETTINGS_FILE_PARAMS},
        model_data_dir=getattr(step, 'model_data_dir', None),
        hash_model_data=bool(getattr(step, 'run_manifest_model_data', False)),
    )


class WriteRunManifest(ComputationStep):
    """Write ``run_manifest.json``: the hashes of a finished run's inputs, outputs and settings, and the versions it ran with."""

    step_params = [
        {'name': 'model_run_dir', 'flag': '-r', 'is_path': True, 'pre_exist': True, 'required': True,
         'help': 'Model run directory'},
        {'name': 'analysis_settings_json', 'flag': '-a', 'is_path': True, 'pre_exist': True,
         'help': 'Analysis settings JSON file path'},
        {'name': 'model_settings_json', 'flag': '-M', 'is_path': True, 'pre_exist': False,
         'help': 'Model settings JSON file path'},
        {'name': 'model_data_dir', 'flag': '-d', 'is_path': True, 'pre_exist': True,
         'help': "Model data directory path (default: the run's 'static' directory)"},
        {'name': 'run_manifest_model_data', 'type': str2bool, 'const': True, 'nargs': '?', 'default': False,
         'help': 'Also hash every model data file (slow for large models)'},
    ]

    def run(self):
        manifest = write_run_manifest(self, self.model_run_dir)
        self.logger.info(f"Wrote the manifest of {len(manifest['files'])} files to '{self.model_run_dir}'")
        return manifest


def _listed(paths):
    shown = ', '.join(paths[:MAX_LISTED])
    return shown + (f' and {len(paths) - MAX_LISTED} more' if len(paths) > MAX_LISTED else '')


class VerifyRun(ComputationStep):
    """Check a run's inputs, outputs and settings against its own manifest, or against another run's."""

    step_params = [
        {'name': 'model_run_dir', 'flag': '-r', 'is_path': True, 'pre_exist': True, 'required': True,
         'help': 'Model run directory to check'},
        {'name': 'verify_against', 'is_path': True, 'pre_exist': True,
         'help': "Another run's directory or run_manifest.json to compare with (default: the run's own manifest)"},
        {'name': 'verify_report_json', 'is_path': True, 'pre_exist': False,
         'help': 'Write the comparison to this JSON file'},
    ]

    def run(self):
        result = verify_run(self.model_run_dir, self.verify_against)
        if self.verify_report_json:
            with open(self.verify_report_json, 'w') as f:
                json.dump(result, f, indent=2)

        reference = self.verify_against or 'its manifest'
        for key, what in (('changed', 'differ'), ('missing', 'are missing'), ('added', 'are not in the manifest')):
            if result['files'][key]:
                self.logger.error(f"{len(result['files'][key])} files {what}: {_listed(result['files'][key])}")
        if result['settings']:
            self.logger.error(f"settings files differ: {_listed(result['settings'])}")
        if result['model_data']:
            self.logger.error(f"model data files differ: {_listed(result['model_data'])}")
        for key, diff in result['environment'].items():
            self.logger.info(f"environment {key}: {diff['expected']!r} in {reference}, {diff['actual']!r} here")

        if not result['match']:
            raise OasisException(f"run '{self.model_run_dir}' does not match {reference}")
        self.logger.info(f"run '{self.model_run_dir}' matches {reference}: {result['files']['matched']} files")
        return result
