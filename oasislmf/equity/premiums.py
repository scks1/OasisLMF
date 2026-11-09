"""Compare ways of setting premiums: who pays for whom, and who can afford it.

Every scheme starts from each row's risk premium, its AAL times ``1 + loading``, and raises the same total unless
it is funded by a public subsidy:

- ``risk_based``: each row pays its own risk premium.
- ``flat``: the total is shared equally per row (``basis: row``) or at the same rate per unit of value
  (``basis: tiv``).
- ``capped``: no row pays more than ``cap`` times its income. The shortfall is paid by a public ``subsidy``, or by
  a ``surcharge`` that scales up the other rows' premiums, never past their own cap.
- ``means_tested``: rows with an income below ``income_threshold`` get ``discount`` off their risk premium, paid by a
  ``subsidy`` or a ``surcharge`` on the other rows.

The schemes only use each row's AAL, value and income. They take no other field as a rating factor, so a scheme
can't charge a group more for who its members are.
"""

__all__ = ['scheme_premiums', 'premium_table']

import logging

import numpy as np
import pandas as pd

from .rows import by_group, suppress_small_groups

logger = logging.getLogger(__name__)


def _scale_to_total(risk, cap, total):
    """Find premiums ``min(k * risk, cap)`` that add up to ``total``, with ``k >= 1``; return them and any shortfall."""
    def capped_total(k):
        return np.minimum(k * risk, cap).sum()

    # the most the rows can pay however far they are scaled up: their cap, or without one, any amount
    most = np.where(risk > 0, cap, 0.0)
    if most.sum() < total:
        return most, total - most.sum()
    low, high = 1.0, 2.0
    while capped_total(high) < total:
        low, high = high, high * 2
    for _ in range(200):
        mid = (low + high) / 2
        low, high = (mid, high) if capped_total(mid) < total else (low, mid)
    return np.minimum(high * risk, cap), 0.0


def _attribute(shortfall, owed):
    """Split ``shortfall`` between rows in proportion to ``owed``."""
    owed = np.clip(owed, 0, None)
    return owed * (shortfall / owed.sum()) if shortfall > 0 and owed.sum() > 0 else np.zeros_like(owed)


def scheme_premiums(rows, scheme, loading=0.0):
    """Return each row's risk premium, premium and public subsidy under a scheme.

    Args:
        rows (pandas.DataFrame): Report rows, with ``aal``, ``tiv`` and, for capped and means-tested schemes,
            ``income``.
        scheme (dict): A checked scheme from the ``equity_report.premiums.schemes`` settings.
        loading (float): Share added to the AAL to give the risk premium.

    Returns:
        pandas.DataFrame: Columns ``risk_premium``, ``premium`` and ``subsidy``, indexed like ``rows``.
    """
    risk = rows['aal'].to_numpy(dtype=float) * (1 + loading)
    total = risk.sum()
    subsidy = np.zeros_like(risk)
    kind = scheme['type']

    if kind == 'risk_based':
        premium = risk.copy()
    elif kind == 'flat':
        if scheme['basis'] == 'tiv':
            tiv = rows['tiv'].to_numpy(dtype=float)
            premium = total * tiv / tiv.sum() if tiv.sum() > 0 else np.full_like(risk, total / len(risk))
        else:
            premium = np.full_like(risk, total / len(risk) if len(risk) else 0.0)
    elif kind == 'capped':
        income = rows['income'].to_numpy(dtype=float)
        cap = np.where(np.isnan(income), np.inf, scheme['cap'] * income)
        if scheme['funding'] == 'subsidy':
            premium = np.minimum(risk, cap)
            subsidy = risk - premium
        else:
            premium, shortfall = _scale_to_total(risk, cap, total)
            if shortfall > 0:
                logger.warning("premium scheme '%s': the caps can't raise the total; %.2f is left to a subsidy",
                               scheme['name'], shortfall)
            subsidy = _attribute(shortfall, risk - premium)
    elif kind == 'means_tested':
        income = rows['income'].to_numpy(dtype=float)
        eligible = income < scheme['income_threshold']  # False where income is unknown
        discount = np.where(eligible, risk * scheme['discount'], 0.0)
        premium = risk - discount
        others = risk[~eligible].sum()
        if scheme['funding'] == 'surcharge' and others > 0:
            premium = np.where(eligible, premium, risk * (1 + discount.sum() / others))
        else:
            if scheme['funding'] == 'surcharge':
                logger.warning("premium scheme '%s': no rows outside the means test to surcharge; the discount is left "
                               "to a subsidy", scheme['name'])
            subsidy = discount
    else:  # pragma: no cover - settings are checked first
        raise ValueError(kind)
    return pd.DataFrame({'risk_premium': risk, 'premium': premium, 'subsidy': subsidy}, index=rows.index)


def premium_table(rows, settings):
    """Premiums of each group under each scheme.

    Columns:

    - ``scheme``, then the group columns, ``rows``.
    - ``risk_premium``, ``premium``: the group's total risk premium and what it pays under the scheme.
    - ``cross_subsidy``: ``premium - risk_premium``. Above 0, the group pays towards other groups' risk; below 0,
      other groups (or the subsidy) pay towards its risk.
    - ``subsidy``: public money paid towards the group's premiums.
    - ``mean_premium``: premium per row.
    - ``median_premium_to_income``, ``share_rows_over_income_burden_threshold``: with an income column, the middle
      row's premium as a share of its income, and the share of rows paying more than ``income_burden_threshold``
      of it.

    Args:
        rows (pandas.DataFrame): Report rows.
        settings (dict): Checked ``equity_report`` settings, with ``premiums``.

    Returns:
        pandas.DataFrame: One row per scheme and group, plus a total row per scheme, with small groups suppressed.
    """
    group_by, premiums = settings['group_by'], settings['premiums']
    with_income = 'income' in rows.columns
    tables = []
    for scheme in premiums['schemes']:
        scheme_rows = rows.join(scheme_premiums(rows, scheme, premiums['loading']))
        if with_income:
            scheme_rows['premium_to_income'] = scheme_rows['premium'] / scheme_rows['income']

        def aggregate(group):
            premium, risk = group['premium'].sum(), group['risk_premium'].sum()
            result = {
                'rows': len(group),
                'risk_premium': risk,
                'premium': premium,
                'cross_subsidy': premium - risk,
                'subsidy': group['subsidy'].sum(),
                'mean_premium': premium / len(group),
            }
            if with_income:
                known = group['premium_to_income'].dropna()
                result['median_premium_to_income'] = known.median()
                result['share_rows_over_income_burden_threshold'] = (
                    float((known > settings['income_burden_threshold']).mean()) if len(known) else np.nan)
            return result

        table = suppress_small_groups(by_group(scheme_rows, group_by, aggregate), group_by, settings['min_group_size'])
        table.insert(0, 'scheme', scheme['name'])
        tables.append(table)
    return pd.concat(tables, ignore_index=True)
