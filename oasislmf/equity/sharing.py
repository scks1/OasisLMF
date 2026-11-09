"""Who bears the loss: how each group's ground-up loss splits between policyholders, insurers and reinsurers.

The split compares the AAL of the same groups in the ground-up (``gul``), insured (``il``) and net of reinsurance
(``ri``) outputs:

- ``retained = gul - il``: kept by policyholders, through deductibles, losses above limits, and locations with no
  insurance at all (they have a ground-up loss but no insured loss).
- ``insurer_net = ri``: paid by insurers after reinsurance, or ``il`` without reinsurance outputs.
- ``ceded = il - ri``: paid by reinsurers.

Each perspective's summary set must group by the ``group_by`` columns. Rows are matched between perspectives by the
summary-info columns they share (other than ``summary_id``), so with summary sets grouped by location the split is
also known for each location, and income-based measures can be given.
"""

__all__ = ['loss_sharing_table', 'perspective_rows', 'summary_id_map']

import logging

import numpy as np
import pandas as pd

from ..utils.exceptions import OasisException
from .rows import by_group, group_label, report_rows, suppress_small_groups

logger = logging.getLogger(__name__)

PERSPECTIVES = ('gul', 'il', 'ri')


ROW_VALUE_COLUMNS = ('SummaryId', 'tiv', 'aal', 'aal_to_tiv', 'income', 'aal_to_income')


def perspective_rows(output_dir, settings, runtype, summary_set, sample_type):
    """Return the report rows of one perspective's summary set, with group values as text.

    Only the ground-up rows need the income column; the other perspectives are matched to them.

    Args:
        output_dir (str): Analysis output directory.
        settings (dict): Checked ``equity_report`` settings.
        runtype (str): Perspective, ``gul``, ``il`` or ``ri``.
        summary_set (int): Summary set id.
        sample_type (int): ALT sample type.

    Returns:
        pandas.DataFrame: Rows from :func:`oasislmf.equity.rows.report_rows`.
    """
    income_column = settings['income_column'] if runtype == 'gul' else None
    rows, _ = report_rows(output_dir, {**settings, 'runtype': runtype, 'summary_set': summary_set,
                                       'sample_type': sample_type, 'income_column': income_column})
    for column in settings['group_by']:
        rows[column] = rows[column].map(group_label)
    return rows


def summary_id_map(gul, other):
    """Map ``other``'s summary ids to the ground-up rows with the same summary-info values.

    Args:
        gul (pandas.DataFrame): Ground-up rows.
        other (pandas.DataFrame): Rows of another perspective.

    Returns:
        pandas.Series or None: Ground-up ``SummaryId`` indexed by ``other``'s ``SummaryId`` (rows with no ground-up match
        are left out), or None when the two have no shared summary-info column that picks out one row each.
    """
    keys = [c for c in gul.columns.intersection(other.columns) if c not in ROW_VALUE_COLUMNS]
    if not keys or other.duplicated(keys).any() or gul.duplicated(keys).any():
        return None
    left = gul[keys].astype(str).assign(GulSummaryId=gul['SummaryId'].to_numpy())
    right = other[keys].astype(str).assign(SummaryId=other['SummaryId'].to_numpy())
    matched = right.merge(left, on=keys, how='inner', validate='one_to_one')
    return pd.Series(matched['GulSummaryId'].to_numpy(), index=matched['SummaryId'].to_numpy())


def _match_rows(gul, other, name):
    """Return ``other``'s AAL for each ``gul`` row, or None when the rows don't match one to one."""
    id_map = summary_id_map(gul, other)
    if id_map is None:
        return None
    by_gul_id = other.set_index('SummaryId')['aal'].reindex(id_map.index).groupby(id_map.to_numpy()).sum()
    matched = gul['SummaryId'].map(by_gul_id)
    unmatched = int(matched.isna().sum())
    if unmatched:
        logger.info("loss sharing: %d of %d ground-up rows have no %s row; they count as having no %s loss",
                    unmatched, len(gul), name, name)
    return matched.fillna(0.0).to_numpy()


def loss_sharing_table(output_dir, settings, sample_type):
    """Split each group's ground-up AAL into the parts kept by policyholders, insurers and reinsurers.

    Columns, after the group columns:

    - ``rows``, ``tiv``: ground-up rows (summary ids) and value.
    - ``gul_aal``, ``il_aal`` and, with reinsurance outputs, ``ri_aal``.
    - ``retained_aal`` (``gul - il``) and ``retained_share`` (of ``gul``): kept by policyholders.
    - ``insurer_net_aal`` and ``insurer_net_share``; with reinsurance outputs, ``ceded_aal`` and ``ceded_share``.
    - ``retained_to_tiv``: the retained AAL as a share of value.
    - When rows match between perspectives and there is an ``income_column``: ``median_retained_to_income`` and
      ``share_rows_over_income_burden_threshold`` (rows whose retained AAL is more than ``income_burden_threshold`` of
      their income).
    - ``suppressed``: see :func:`oasislmf.equity.rows.suppress_small_groups`.

    Args:
        output_dir (str): Analysis output directory.
        settings (dict): Checked ``equity_report`` settings, with a ``loss_sharing`` section.
        sample_type (int): ALT sample type to use, as for the rest of the report.

    Returns:
        pandas.DataFrame: One row per group plus the total row ``All``.

    Raises:
        OasisException: If the ground-up or insured outputs are missing.
    """
    sets = settings['loss_sharing']['summary_sets']
    group_by = settings['group_by']
    gul = perspective_rows(output_dir, settings, 'gul', sets['gul'], sample_type)
    il = perspective_rows(output_dir, settings, 'il', sets['il'], sample_type)
    try:
        ri = perspective_rows(output_dir, settings, 'ri', sets['ri'], sample_type)
    except OasisException:
        ri = None
        logger.info("loss sharing: no ri_S%s outputs, so insurers' losses are shown before reinsurance", sets['ri'])

    def group_aal(rows):
        return rows.groupby(group_by, observed=True, dropna=False)['aal'].sum()

    il_by_row = _match_rows(gul, il, 'il')
    ri_by_row = _match_rows(gul, ri, 'ri') if ri is not None else None
    row_level = il_by_row is not None and (ri is None or ri_by_row is not None)
    if row_level:
        gul = gul.assign(il_aal=il_by_row)
        if ri is not None:
            gul = gul.assign(ri_aal=ri_by_row)
    else:
        logger.info("loss sharing: gul, il and ri rows don't match one to one, so the split is given by group only")
        il_by_group, ri_by_group = group_aal(il), group_aal(ri) if ri is not None else None

    income_column = settings['income_column']
    threshold = settings['income_burden_threshold']

    def aggregate(group):
        gul_aal = group['aal'].sum()
        if row_level:
            il_aal = group['il_aal'].sum()
            ri_aal = group['ri_aal'].sum() if ri is not None else None
        else:
            # by_group passes all the rows for the total row
            whole = group is gul
            key = tuple(group[group_by].iloc[0]) if len(group_by) > 1 else group[group_by[0]].iloc[0]
            il_aal = il_by_group.sum() if whole else il_by_group.get(key, 0.0)
            ri_aal = None if ri is None else (ri_by_group.sum() if whole else ri_by_group.get(key, 0.0))
        tiv = group['tiv'].sum()
        retained = gul_aal - il_aal
        net = il_aal if ri_aal is None else ri_aal
        result = {'rows': len(group), 'tiv': tiv, 'gul_aal': gul_aal, 'il_aal': il_aal}
        if ri is not None:
            result['ri_aal'] = ri_aal
        result.update({
            'retained_aal': retained,
            'retained_share': retained / gul_aal if gul_aal > 0 else np.nan,
            'insurer_net_aal': net,
            'insurer_net_share': net / gul_aal if gul_aal > 0 else np.nan,
        })
        if ri is not None:
            result['ceded_aal'] = il_aal - ri_aal
            result['ceded_share'] = (il_aal - ri_aal) / gul_aal if gul_aal > 0 else np.nan
        result['retained_to_tiv'] = retained / tiv if tiv > 0 else np.nan
        if row_level and income_column:
            retained_to_income = (group['aal'] - group['il_aal']) / group['income']
            known = retained_to_income.dropna()
            result['median_retained_to_income'] = known.median()
            result['share_rows_over_income_burden_threshold'] = (known > threshold).mean() if len(known) else np.nan
        return result

    gul_total, il_total = gul['aal'].sum(), (gul['il_aal'].sum() if row_level else il['aal'].sum())
    if il_total > gul_total * (1 + 1e-6) + 1e-9:
        logger.warning("loss sharing: the insured AAL (%.6g) is more than the ground-up AAL (%.6g); check that the gul "
                       "and il summary sets cover the same exposure", il_total, gul_total)
    table = by_group(gul, group_by, aggregate)
    return suppress_small_groups(table, group_by, settings['min_group_size'])
