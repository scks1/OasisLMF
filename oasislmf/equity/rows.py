"""Build the table of report rows: one per summary id, with its AAL, value, income and group."""

__all__ = ['report_rows', 'by_group', 'group_label', 'suppress_small_groups', 'TOTAL_LABEL']

import logging

import numpy as np
import pandas as pd

from ..utils.exceptions import OasisException
from .ord import read_ord

logger = logging.getLogger(__name__)

TOTAL_LABEL = 'All'


def _aal(output_dir, settings):
    runtype, summary_set = settings['runtype'], settings['summary_set']
    alt = read_ord(output_dir, runtype, summary_set, 'palt')
    if alt is None:
        alt = read_ord(output_dir, runtype, summary_set, 'altmeanonly', needed_for='The equity report (AAL per summary id)')
    sample_type = settings['sample_type']
    if sample_type is None:
        sample_type = 2 if (alt['SampleType'] == 2).any() else 1
    alt = alt[alt['SampleType'] == sample_type]
    return alt.groupby('SummaryId')['MeanLoss'].sum(), sample_type


def report_rows(output_dir, settings):
    """Return one row per summary id of the report's summary set.

    Args:
        output_dir (str): Analysis output directory.
        settings (dict): Checked ``equity_report`` settings.

    Returns:
        tuple: ``(rows, sample_type)``. ``rows`` has the summary-info columns plus ``aal`` (0 for a summary id with no
        loss), ``aal_to_tiv`` and, with an income column, ``income`` (blank where it is missing or not positive) and
        ``aal_to_income``. ``sample_type`` is the ALT sample type the AAL was taken from.

    Raises:
        OasisException: If a needed file or column is missing.
    """
    runtype, summary_set = settings['runtype'], settings['summary_set']
    info = read_ord(output_dir, runtype, summary_set, 'summary-info', needed_for='The equity report')
    needed = settings['group_by'] + ([settings['income_column']] if settings['income_column'] else [])
    missing = [c for c in needed if c not in info.columns]
    if missing:
        raise OasisException(
            f"equity report: columns {missing} are not in {runtype}_S{summary_set}_summary-info; add them to the "
            f"summary set's oed_fields (columns from a table of area attributes can be added with exposure_attributes)")

    aal, sample_type = _aal(output_dir, settings)
    rows = info.rename(columns={'summary_id': 'SummaryId'})
    rows['aal'] = rows['SummaryId'].map(aal).fillna(0.0)
    tiv = rows['tiv'].where(rows['tiv'] > 0)
    rows['aal_to_tiv'] = rows['aal'] / tiv

    income_column = settings['income_column']
    if income_column:
        income = pd.to_numeric(rows[income_column], errors='coerce')
        income = income.where(income > 0)
        no_income = int(income.isna().sum())
        if no_income:
            logger.warning("equity report: %d of %d rows have no positive '%s'; they are left out of the income measures",
                           no_income, len(rows), income_column)
        rows['income'] = income
        rows['aal_to_income'] = rows['aal'] / income
    return rows, sample_type


def group_label(value):
    """Return a group value as text, so ``1``, ``1.0`` and ``'1'`` (from csv or parquet summary-info) are one group.

    Args:
        value: A group column value.

    Returns:
        str: The value as text; ``''`` for a missing value.
    """
    if isinstance(value, str):
        return value
    if pd.isna(value):
        return ''
    if isinstance(value, (float, np.floating)) and float(value).is_integer():
        return str(int(value))
    return str(value)


def by_group(rows, group_by, aggregate):
    """Apply ``aggregate`` to each group and to all rows, returning one row per group plus a total row.

    Args:
        rows (pandas.DataFrame): Report rows.
        group_by (list): Columns to group by.
        aggregate (callable): Takes the rows of one group and returns a dict of results.

    Returns:
        pandas.DataFrame: The group columns, then the results. The total row has ``All`` in every group column.
    """
    results = [{**dict(zip(group_by, key if isinstance(key, tuple) else (key,))), **aggregate(group)}
               for key, group in rows.groupby(group_by, observed=True, dropna=False, sort=True)]
    results.append({**{c: TOTAL_LABEL for c in group_by}, **aggregate(rows)})
    return pd.DataFrame(results)


def suppress_small_groups(table, group_by, min_group_size):
    """Blank the results of groups with fewer than ``min_group_size`` rows.

    The group and its row count are kept so the reader can see what was left out and why.

    Args:
        table (pandas.DataFrame): Table from :func:`by_group`, with a ``rows`` column.
        group_by (list): The group columns.
        min_group_size (int): Smallest group to report. 0 reports every group.

    Returns:
        pandas.DataFrame: ``table`` with a ``suppressed`` column added.
    """
    table = table.copy()
    small = table['rows'] < min_group_size
    results = [c for c in table.columns if c not in group_by and c != 'rows']
    table[results] = table[results].astype(float)
    table.loc[small, results] = np.nan
    table['suppressed'] = small
    return table
