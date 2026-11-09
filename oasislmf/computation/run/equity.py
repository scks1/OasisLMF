__all__ = ['EquityReport']

import os

from ...equity import run_equity_report
from ...utils.data import analysis_settings_loader
from ...utils.exceptions import OasisException
from ..base import ComputationStep


class EquityReport(ComputationStep):
    """Report how the losses of a finished analysis fall on different groups, from its ORD outputs.

    The report is set in the ``equity_report`` section of the analysis settings. It only reads the outputs: it does
    not change any loss.
    """
    settings_params = [{'name': 'analysis_settings_json', 'loader': analysis_settings_loader, 'user_role': 'user'}]

    step_params = [
        {'name': 'model_run_dir', 'flag': '-r', 'is_path': True, 'pre_exist': True, 'required': True,
         'help': 'Model run directory of a finished analysis'},
        {'name': 'analysis_settings_json', 'flag': '-a', 'is_path': True, 'pre_exist': True,
         'help': 'Analysis settings JSON file path (default: the copy in the model run directory)'},
        {'name': 'equity_report_dir', 'is_path': True, 'pre_exist': False,
         'help': 'Directory to write the report to (default: <model_run_dir>/output/equity)'},
        {'name': 'user_data_dir', 'flag': '-D', 'is_path': True, 'pre_exist': False,
         'help': 'Directory to look for relative exposure_attributes table paths in, for the data-gap audit'},
    ]

    def run(self):
        settings = self.settings
        settings_dir = self.analysis_settings_json and os.path.dirname(os.path.abspath(self.analysis_settings_json))
        if not self.analysis_settings_json:
            run_settings = os.path.join(self.model_run_dir, 'input', 'analysis_settings.json')
            if not os.path.isfile(run_settings):
                raise OasisException(f"no analysis settings given, and none in '{run_settings}'")
            settings = analysis_settings_loader(run_settings)
        search_dirs = [d for d in (self.user_data_dir, settings_dir) if d]
        return run_equity_report(os.path.join(self.model_run_dir, 'output'), settings, self.equity_report_dir,
                                 input_dir=os.path.join(self.model_run_dir, 'input'), search_dirs=search_dirs)
