"""Basis risk: how well a policy's payouts follow the losses they are meant to cover, group by group.

Step, parametric and index policies (for example calcrules 28, 281 and 32) pay on a trigger rather than on the loss
itself, so a household can have a loss and no payout, or a payout with no loss. This table compares, for each event
(and sample) and row, the ground-up loss (``gul``) with the payout (``il``). It works for any policy; for indemnity
policies payouts never exceed losses, so only the shortfall columns are of interest.
"""

__all__ = ['basis_risk_table']

import numpy as np
import pandas as pd

from .events import event_losses
from .rows import by_group, suppress_small_groups


def basis_risk_table(output_dir, settings, sample_type, analysis_settings=None):
    """Compare each group's payouts with its losses event by event.

    An event is a loss event for a row when its ground-up loss is above ``loss_threshold``, and paid when its payout
    is above ``payout_threshold``. Counts are expected numbers a year, over all the group's rows.

    Columns, after the group columns:

    - ``rows``; ``gul_aal`` and ``payout_aal``, and ``payout_to_loss``.
    - ``loss_events``: expected loss events a year; ``share_loss_events_unpaid``: the share of them with no payout.
    - ``unpaid_loss_aal``: expected annual loss in loss events with no payout, and ``unpaid_loss_share`` of
      ``gul_aal``.
    - ``shortfall_aal``: expected annual loss not paid (``max(loss - payout, 0)`` summed over events).
    - ``payouts_without_loss``: expected payouts a year in events that aren't loss events, and
      ``overpayment_aal`` (``max(payout - loss, 0)``).
    - With an ``income_column``: ``median_shortfall_to_income``.
    - ``suppressed``: see :func:`oasislmf.equity.rows.suppress_small_groups`.

    Args:
        output_dir (str): Analysis output directory.
        settings (dict): Checked ``equity_report`` settings, with a ``basis_risk`` section.
        sample_type (int): MPLT sample type when mean losses are used.
        analysis_settings (dict): Analysis settings, for ``number_of_samples``.

    Returns:
        pandas.DataFrame: One row per group plus the total row ``All``.
    """
    terms = settings['basis_risk']
    losses = event_losses(output_dir, settings, sample_type, analysis_settings, 'basis risk', require_il=True,
                          nonlinear_note='which hides loss events a trigger misses in some samples but not on average')
    rows, weights, samples = losses['rows'], losses['weights'], losses['samples']
    events = pd.concat({'loss': losses['gul'], 'payout': losses['il']}, axis=1).fillna(0.0)

    weight = weights.reindex(events.index.get_level_values('Period')).to_numpy() / samples
    loss, payout = events['loss'].to_numpy(), events['payout'].to_numpy()
    is_loss = loss > terms['loss_threshold']
    is_paid = payout > terms['payout_threshold']
    per_event = pd.DataFrame({
        'SummaryId': events.index.get_level_values('SummaryId'),
        'gul': loss * weight,
        'payout': payout * weight,
        'loss_events': is_loss * weight,
        'unpaid_events': (is_loss & ~is_paid) * weight,
        'unpaid_loss': np.where(is_loss & ~is_paid, loss, 0.0) * weight,
        'shortfall': np.clip(loss - payout, 0.0, None) * weight,
        'payouts_without_loss': (is_paid & ~is_loss) * weight,
        'overpayment': np.clip(payout - loss, 0.0, None) * weight,
    })
    by_row = per_event.groupby('SummaryId').sum()
    rows = rows.assign(**{c: rows['SummaryId'].map(by_row[c]).fillna(0.0) for c in by_row.columns})
    income_column = settings['income_column']

    def aggregate(group):
        gul, paid, loss_events = group['gul'].sum(), group['payout'].sum(), group['loss_events'].sum()
        result = {
            'rows': len(group),
            'gul_aal': gul,
            'payout_aal': paid,
            'payout_to_loss': paid / gul if gul > 0 else np.nan,
            'loss_events': loss_events,
            'share_loss_events_unpaid': group['unpaid_events'].sum() / loss_events if loss_events > 0 else np.nan,
            'unpaid_loss_aal': group['unpaid_loss'].sum(),
            'unpaid_loss_share': group['unpaid_loss'].sum() / gul if gul > 0 else np.nan,
            'shortfall_aal': group['shortfall'].sum(),
            'payouts_without_loss': group['payouts_without_loss'].sum(),
            'overpayment_aal': group['overpayment'].sum(),
        }
        if income_column:
            result['median_shortfall_to_income'] = (group['shortfall'] / group['income']).median()
        return result

    table = by_group(rows, settings['group_by'], aggregate)
    return suppress_small_groups(table, settings['group_by'], settings['min_group_size'])
