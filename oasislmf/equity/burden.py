"""Relative burden and affordability of expected losses, by group."""

__all__ = ['burden_table']

import numpy as np

from .rows import by_group, suppress_small_groups


def _share(mask, weights=None):
    if weights is None:
        return float(mask.mean()) if len(mask) else np.nan
    total = weights.sum()
    return float(weights[mask].sum() / total) if total > 0 else np.nan


def burden_table(rows, settings):
    """Compare each group's expected annual loss with its value and, if known, its income.

    Columns:

    - ``rows``, ``tiv``, ``aal``: number of rows (summary ids) in the group, their total value and total AAL.
    - ``aal_to_tiv``: the group's AAL as a share of its value.
    - ``relative_aal_to_tiv``: ``aal_to_tiv`` divided by that of all rows. Above 1, the group carries more expected
      loss for its value than the portfolio as a whole.
    - ``median_aal_to_tiv``: the middle row's AAL as a share of its value.
    - ``share_rows_over_burden_threshold``, ``share_tiv_over_burden_threshold``: the share of rows, and of value,
      whose AAL is more than ``burden_threshold`` of their value.

    With an ``income_column``:

    - ``rows_with_income``, ``mean_income``: rows with a positive income, and their mean income.
    - ``median_aal_to_income``: the middle row's AAL as a share of its income.
    - ``share_rows_over_income_burden_threshold``: the share of rows with an income whose AAL is more than
      ``income_burden_threshold`` of it.

    Args:
        rows (pandas.DataFrame): Report rows from :func:`oasislmf.equity.rows.report_rows`.
        settings (dict): Checked ``equity_report`` settings.

    Returns:
        pandas.DataFrame: One row per group and a total row, with small groups suppressed.
    """
    group_by = settings['group_by']
    overall = rows['aal'].sum() / rows['tiv'].sum() if rows['tiv'].sum() > 0 else np.nan
    with_income = 'income' in rows.columns

    def aggregate(group):
        tiv, aal = group['tiv'].sum(), group['aal'].sum()
        over = group['aal_to_tiv'] > settings['burden_threshold']
        result = {
            'rows': len(group),
            'tiv': tiv,
            'aal': aal,
            'aal_to_tiv': aal / tiv if tiv > 0 else np.nan,
            'relative_aal_to_tiv': (aal / tiv) / overall if tiv > 0 and overall > 0 else np.nan,
            'median_aal_to_tiv': group['aal_to_tiv'].median(),
            'share_rows_over_burden_threshold': _share(over),
            'share_tiv_over_burden_threshold': _share(over, group['tiv']),
        }
        if with_income:
            has_income = group['income'].notna()
            result.update({
                'rows_with_income': int(has_income.sum()),
                'mean_income': group['income'].mean(),
                'median_aal_to_income': group['aal_to_income'].median(),
                'share_rows_over_income_burden_threshold': _share(
                    group.loc[has_income, 'aal_to_income'] > settings['income_burden_threshold']),
            })
        return result

    table = by_group(rows, group_by, aggregate)
    return suppress_small_groups(table, group_by, settings['min_group_size'])
