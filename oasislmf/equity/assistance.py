"""Public assistance: what a programme paying towards the losses people keep would cost, and whom it would reach.

A programme pays each eligible row (household or location), for each event, ``share`` of its retained loss above
``deductible``, up to ``limit``. The retained loss is ``gul - il``: deductibles, losses above limits and the losses of
uninsured locations. Without insured outputs every loss counts as retained. With ``income_threshold`` only rows whose
income is below it are eligible.

The programme's terms aren't linear, so applying them to an event's mean loss would understate what the programme
pays. When both perspectives have a sample period loss table (SPLT, ``plt_sample``) and the analysis settings give
``number_of_samples``, the terms are applied to each sample; otherwise to the mean loss from the moment period loss
table (MPLT, ``plt_moment``), with a note in the log.
"""

__all__ = ['assistance_tables']

import logging

import numpy as np
import pandas as pd

from .events import event_losses
from .rows import by_group, suppress_small_groups
from .weighting import losses_at_return_periods

logger = logging.getLogger(__name__)


def assistance_tables(output_dir, settings, sample_type, analysis_settings=None):
    """Cost and reach of a public assistance programme, by group and as an exceedance curve of its budget.

    The group table has, after the group columns:

    - ``rows`` and ``eligible_rows``.
    - ``retained_aal``: expected annual retained loss, from the event losses.
    - ``assistance_aal``: expected annual assistance paid, and ``share_of_retained_covered``.
    - ``assistance_share``: the group's part of the whole programme's spending, next to ``retained_share``, its part
      of all retained loss.
    - With an ``income_column``: ``median_retained_to_income_before`` and ``median_retained_to_income_after`` the
      assistance.
    - ``suppressed``: see :func:`oasislmf.equity.rows.suppress_small_groups`.

    The budget table has ``ReturnPeriod``, ``OEP`` (largest single-event spending in a year) and ``AEP`` (spending in
    a year). The expected annual spending is the total row's ``assistance_aal``. Rows with an unknown income are not
    eligible for a means-tested programme.

    Args:
        output_dir (str): Analysis output directory.
        settings (dict): Checked ``equity_report`` settings, with a ``public_assistance`` section.
        sample_type (int): ALT sample type the rest of the report uses, and the MPLT sample type when the programme
            is applied to mean losses.
        analysis_settings (dict): Analysis settings, for ``number_of_samples``.

    Returns:
        tuple: ``(group table, budget table)``.
    """
    programme = settings['public_assistance']
    losses = event_losses(output_dir, settings, sample_type, analysis_settings, 'public assistance', require_il=False,
                          nonlinear_note='which understates what a programme with a deductible or limit pays')
    gul, weights, samples = losses['rows'], losses['weights'], losses['samples']
    retained = losses['gul']
    if losses['il'] is not None:
        retained = losses['gul'].sub(losses['il'], fill_value=0.0).clip(lower=0.0)

    eligible = pd.Series(True, index=gul['SummaryId'].to_numpy())
    if programme['income_threshold'] is not None:
        income = gul.set_index('SummaryId')['income']
        eligible = (income < programme['income_threshold']).reindex(eligible.index, fill_value=False)
    row_eligible = eligible.reindex(retained.index.get_level_values('SummaryId'), fill_value=False).to_numpy()
    paid = ((retained - programme['deductible']).clip(lower=0.0) * programme['share'])
    if programme['limit'] is not None:
        paid = paid.clip(upper=programme['limit'])
    paid = paid.where(row_eligible, 0.0)

    # expected annual amounts per row: each sample's loss weighs PeriodWeight / number of samples
    period_weight = weights.reindex(retained.index.get_level_values('Period')).to_numpy() / samples
    by_row = pd.DataFrame({'retained': retained.to_numpy() * period_weight, 'paid': paid.to_numpy() * period_weight,
                           'SummaryId': retained.index.get_level_values('SummaryId')}).groupby('SummaryId').sum()
    rows = gul.assign(retained_aal=gul['SummaryId'].map(by_row['retained']).fillna(0.0),
                      assistance_aal=gul['SummaryId'].map(by_row['paid']).fillna(0.0),
                      eligible=eligible.reindex(gul['SummaryId']).to_numpy())
    total_retained, total_paid = rows['retained_aal'].sum(), rows['assistance_aal'].sum()
    income_column = settings['income_column']

    def aggregate(group):
        retained_aal, paid_aal = group['retained_aal'].sum(), group['assistance_aal'].sum()
        result = {
            'rows': len(group),
            'eligible_rows': int(group['eligible'].sum()),
            'retained_aal': retained_aal,
            'assistance_aal': paid_aal,
            'share_of_retained_covered': paid_aal / retained_aal if retained_aal > 0 else np.nan,
            'assistance_share': paid_aal / total_paid if total_paid > 0 else np.nan,
            'retained_share': retained_aal / total_retained if total_retained > 0 else np.nan,
        }
        if income_column:
            result['median_retained_to_income_before'] = (group['retained_aal'] / group['income']).median()
            result['median_retained_to_income_after'] = (
                (group['retained_aal'] - group['assistance_aal']) / group['income']).median()
        return result

    group_table = suppress_small_groups(by_group(rows, settings['group_by'], aggregate), settings['group_by'],
                                        settings['min_group_size'])
    group_table['eligible_rows'] = group_table['eligible_rows'].where(~group_table['suppressed'])

    # budget of each event: the programme's spending summed over rows, averaged over samples
    events = paid.groupby(level=['Period', 'EventId']).sum() / samples
    periods = events.groupby(level='Period')
    return_periods = programme['return_periods'][::-1]
    budget = pd.DataFrame({
        'ReturnPeriod': return_periods,
        'OEP': losses_at_return_periods(periods.max(), weights, return_periods),
        'AEP': losses_at_return_periods(periods.sum(), weights, return_periods),
    })
    return group_table, budget
