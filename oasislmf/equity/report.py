"""Write the equity report of a finished analysis."""

__all__ = ['run_equity_report']

import json
import logging
import os

import pandas as pd

from ..utils.exceptions import OasisException
from .assistance import assistance_tables
from .basis import basis_risk_table
from .burden import burden_table
from .data_gaps import data_gap_table
from .ord import ord_path
from .premiums import premium_table
from .rows import report_rows
from .settings import equity_report_settings
from .sharing import loss_sharing_table
from .weighting import equity_weights, weighted_aal_table, weighted_ep_table

logger = logging.getLogger(__name__)


def run_equity_report(output_dir, analysis_settings, report_dir=None, input_dir=None, search_dirs=()):
    """Build the equity report from the ORD outputs in ``output_dir`` and write it to ``report_dir``.

    Writes, for perspective ``P`` and summary set ``N``:

    - ``P_SN_equity-burden.csv``: see :func:`oasislmf.equity.burden.burden_table`.
    - ``P_SN_equity-weighted-aal.csv`` and ``P_SN_equity-weighted-ep.csv``, with ``equity_weighting``: see
      :mod:`oasislmf.equity.weighting`. The EP table needs the summary set's MPLT (``plt_moment``) and is skipped
      with a warning without it.
    - ``P_SN_equity-premiums.csv``, with ``premiums``: see :func:`oasislmf.equity.premiums.premium_table`.
    - ``equity-data-gaps.csv``, with ``data_gaps``: see :func:`oasislmf.equity.data_gaps.data_gap_table`. It reads the
      run's inputs, not its losses, so it is the same for every perspective.
    - ``P_SN_equity-report.json``: the settings used, with the ALT sample type and reference income.

    Args:
        output_dir (str): Analysis output directory.
        analysis_settings (dict): Analysis settings, with an ``equity_report`` section.
        report_dir (str): Where to write the report. Defaults to ``output_dir/equity``.
        input_dir (str): The run's input directory, for ``data_gaps``. Defaults to ``input`` next to ``output_dir``.
        search_dirs (list): Directories to look for relative ``exposure_attributes`` table paths in, for ``data_gaps``.

    Returns:
        dict: The tables written, by file name.

    Raises:
        OasisException: If there is no ``equity_report`` section, a setting is wrong, or a needed output is missing.
    """
    settings = equity_report_settings(analysis_settings)
    if settings is None:
        raise OasisException("the analysis settings have no 'equity_report' section")
    report_dir = report_dir or os.path.join(output_dir, 'equity')
    stem = f"{settings['runtype']}_S{settings['summary_set']}_equity"

    rows, sample_type = report_rows(output_dir, settings)
    used = {**settings, 'sample_type': sample_type}
    tables = {f'{stem}-burden.csv': burden_table(rows, settings)}

    weighting = settings['equity_weighting']
    if weighting:
        weights, reference = equity_weights(rows['income'], weighting['eta'], weighting['reference_income'])
        used['equity_weighting'] = {**weighting, 'reference_income_used': reference}
        tables[f'{stem}-weighted-aal.csv'] = weighted_aal_table(rows, weights, settings)
        if ord_path(output_dir, settings['runtype'], settings['summary_set'], 'mplt'):
            by_summary_id = pd.Series(weights.to_numpy(), index=rows['SummaryId'].to_numpy())
            tables[f'{stem}-weighted-ep.csv'] = weighted_ep_table(output_dir, by_summary_id, sample_type, settings)
        else:
            logger.warning("equity report: no %s_S%s_mplt in '%s', so no equity-weighted EP; turn on plt_moment for it",
                           settings['runtype'], settings['summary_set'], output_dir)

    if settings['premiums']:
        tables[f'{stem}-premiums.csv'] = premium_table(rows, settings)

    if settings['loss_sharing']:
        tables['equity-loss-sharing.csv'] = loss_sharing_table(output_dir, settings, sample_type)

    if settings['public_assistance']:
        assistance, budget = assistance_tables(output_dir, settings, sample_type, analysis_settings)
        tables['equity-public-assistance.csv'] = assistance
        tables['equity-public-assistance-budget-ep.csv'] = budget

    if settings['basis_risk']:
        tables['equity-basis-risk.csv'] = basis_risk_table(output_dir, settings, sample_type, analysis_settings)

    if settings['data_gaps']:
        input_dir = input_dir or os.path.join(os.path.dirname(os.path.abspath(output_dir)), 'input')
        tables['equity-data-gaps.csv'] = data_gap_table(input_dir, analysis_settings, settings, search_dirs)

    os.makedirs(report_dir, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(os.path.join(report_dir, name), index=False)
    with open(os.path.join(report_dir, f'{stem}-report.json'), 'w') as f:
        json.dump(used, f, indent=2)
    logger.info("equity report: wrote %s to '%s'", ', '.join(tables), report_dir)
    return tables
