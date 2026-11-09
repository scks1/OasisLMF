"""Compare two analyses group by group: who gains and who loses between a base run and a scenario.

The scenario can be a mitigation programme, a climate-conditioned event set, a changed policy, or the same portfolio
run with another model. Each run's rows (summary ids) are grouped by the ``equity_report`` ``group_by`` columns and the
groups are then matched by value, so the two runs need not have the same summary ids or even the same locations.
"""

__all__ = ['compare_table', 'compare_ep_table', 'run_scenario_comparison']

import json
import logging
import os

import numpy as np
import pandas as pd

from ..utils.exceptions import OasisException
from .ord import read_ord
from .rows import TOTAL_LABEL, by_group, group_label, report_rows
from .settings import equity_report_settings

logger = logging.getLogger(__name__)

EP_KEYS = ['EPCalc', 'EPType', 'ReturnPeriod']


def _group_summary(rows, settings):
    def aggregate(group):
        tiv = group['tiv'].sum()
        result = {'rows': len(group), 'tiv': tiv, 'aal': group['aal'].sum(),
                  'aal_to_tiv': group['aal'].sum() / tiv if tiv > 0 else np.nan}
        if settings['income_column']:
            result['median_aal_to_income'] = group['aal_to_income'].median()
        return result

    table = by_group(rows, settings['group_by'], aggregate)
    # Group values are matched across runs as text, so 1, 1.0 and '1' (csv vs parquet summary-info) are one group
    for column in settings['group_by']:
        table[column] = table[column].map(group_label)
    return table


def compare_table(base_rows, scenario_rows, settings):
    """Compare each group's losses in the base run and the scenario.

    Columns, after the group columns:

    - ``rows_base``, ``rows_scenario``, ``tiv_base``, ``tiv_scenario``: rows (summary ids) and value in each run. A
      group found in only one run has 0 rows in the other.
    - ``aal_base``, ``aal_scenario``, ``aal_change`` (scenario minus base) and ``aal_change_pct`` (of the base).
    - ``share_of_aal_change``: the group's part of the change in the total AAL. Groups sharing a reduction fairly
      have shares close to their share of the base AAL, ``aal_share_base``.
    - ``aal_to_tiv_base``, ``aal_to_tiv_scenario``: AAL as a share of value in each run.
    - With an ``income_column``: ``median_aal_to_income_base`` and ``median_aal_to_income_scenario``.
    - ``suppressed``: True when the group has fewer than ``min_group_size`` rows in either run; its results are
      blank.

    Args:
        base_rows (pandas.DataFrame): Rows of the base run, from :func:`oasislmf.equity.rows.report_rows`.
        scenario_rows (pandas.DataFrame): Rows of the scenario.
        settings (dict): Checked ``equity_report`` settings.

    Returns:
        pandas.DataFrame: One row per group plus the total row ``All``.
    """
    group_by = settings['group_by']
    base = _group_summary(base_rows, settings)
    scenario = _group_summary(scenario_rows, settings)
    table = base.merge(scenario, on=group_by, how='outer', suffixes=('_base', '_scenario'))
    for column in ('rows', 'tiv', 'aal'):
        for suffix in ('_base', '_scenario'):
            table[column + suffix] = table[column + suffix].fillna(0)
    table[['rows_base', 'rows_scenario']] = table[['rows_base', 'rows_scenario']].astype(int)

    is_total = (table[group_by] == TOTAL_LABEL).all(axis=1)
    total = table[is_total].iloc[0]
    table['aal_change'] = table['aal_scenario'] - table['aal_base']
    table['aal_change_pct'] = table['aal_change'] / table['aal_base'].where(table['aal_base'] > 0)
    total_change = total['aal_scenario'] - total['aal_base']
    table['share_of_aal_change'] = table['aal_change'] / total_change if total_change != 0 else np.nan
    table['aal_share_base'] = table['aal_base'] / total['aal_base'] if total['aal_base'] > 0 else np.nan

    columns = group_by + ['rows_base', 'rows_scenario', 'tiv_base', 'tiv_scenario', 'aal_base', 'aal_scenario',
                          'aal_change', 'aal_change_pct', 'share_of_aal_change', 'aal_share_base',
                          'aal_to_tiv_base', 'aal_to_tiv_scenario']
    if settings['income_column']:
        columns += ['median_aal_to_income_base', 'median_aal_to_income_scenario']
    table = pd.concat([table[~is_total].sort_values(group_by), table[is_total]])[columns].reset_index(drop=True)

    small = (table[['rows_base', 'rows_scenario']].min(axis=1) < settings['min_group_size'])
    results = columns[len(group_by) + 2:]
    table[results] = table[results].astype(float)
    table.loc[small, results] = np.nan
    table['suppressed'] = small
    return table


def compare_ep_table(base_output_dir, scenario_output_dir, settings):
    """Compare the exceedance probability tables (EPT) of the whole summary set in the two runs.

    Args:
        base_output_dir (str): Base run output directory.
        scenario_output_dir (str): Scenario output directory.
        settings (dict): Checked ``equity_report`` settings.

    Returns:
        pandas.DataFrame or None: ``EPCalc``, ``EPType``, ``ReturnPeriod``, ``loss_base``, ``loss_scenario``,
        ``loss_change`` and ``loss_change_pct``, summed over the summary ids, for the return periods in both runs.
        None if either run has no EPT.
    """
    runtype, summary_set = settings['runtype'], settings['summary_set']
    base = read_ord(base_output_dir, runtype, summary_set, 'ept')
    scenario = read_ord(scenario_output_dir, runtype, summary_set, 'ept')
    if base is None or scenario is None:
        return None

    def total(ept):
        return ept.groupby(EP_KEYS, as_index=False)['Loss'].sum()

    table = total(base).merge(total(scenario), on=EP_KEYS, suffixes=('_base', '_scenario'))
    table = table.rename(columns={'Loss_base': 'loss_base', 'Loss_scenario': 'loss_scenario'})
    table['loss_change'] = table['loss_scenario'] - table['loss_base']
    table['loss_change_pct'] = table['loss_change'] / table['loss_base'].where(table['loss_base'] > 0)
    return table.sort_values(EP_KEYS, ascending=[True, True, False]).reset_index(drop=True)


def run_scenario_comparison(base_output_dir, scenario_output_dir, analysis_settings, report_dir):
    """Compare a scenario with a base run and write the tables to ``report_dir``.

    Both runs are read with the ``equity_report`` section of ``analysis_settings`` (the same runtype, summary set,
    group columns and sample type), so each run's summary-info must have the group columns. The comparison writes
    ``<runtype>_S<summary_set>_compare-aal.csv``, ``..._compare-ep.csv`` when both runs have an EPT, and
    ``..._compare.json`` with the settings used.

    Args:
        base_output_dir (str): Output directory of the base run.
        scenario_output_dir (str): Output directory of the scenario.
        analysis_settings (dict): Analysis settings with an ``equity_report`` section.
        report_dir (str): Where to write the tables.

    Returns:
        dict: ``{file name: DataFrame}`` of the tables written.
    """
    settings = equity_report_settings(analysis_settings)
    if settings is None:
        raise OasisException("the analysis settings have no 'equity_report' section, which sets how to group the runs")
    base_rows, base_sample_type = report_rows(base_output_dir, settings)
    scenario_rows, scenario_sample_type = report_rows(scenario_output_dir, settings)
    if base_sample_type != scenario_sample_type:
        raise OasisException(
            f"the base run's AAL is sample type {base_sample_type} and the scenario's is {scenario_sample_type}; set "
            "equity_report 'sample_type' to one both runs have")

    stem = f"{settings['runtype']}_S{settings['summary_set']}_compare"
    tables = {f'{stem}-aal.csv': compare_table(base_rows, scenario_rows, settings)}
    ep = compare_ep_table(base_output_dir, scenario_output_dir, settings)
    if ep is not None:
        tables[f'{stem}-ep.csv'] = ep

    os.makedirs(report_dir, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(os.path.join(report_dir, name), index=False)
    with open(os.path.join(report_dir, f'{stem}.json'), 'w') as f:
        json.dump({**settings, 'sample_type': base_sample_type, 'base_output_dir': os.path.abspath(base_output_dir),
                   'scenario_output_dir': os.path.abspath(scenario_output_dir)}, f, indent=2)
    logger.info("scenario comparison: wrote %s to '%s'", ', '.join(tables), report_dir)
    return tables
