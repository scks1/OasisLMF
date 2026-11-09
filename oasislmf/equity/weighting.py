"""Equity-weighted losses: a loss counts for more the lower the income of whoever bears it.

Each row's losses are multiplied by ``(reference_income / income) ** eta``. With ``eta = 0`` every weight is 1 and
the weighted losses equal the standard ones. ``eta`` is the elasticity of the marginal utility of income; values
between 1 and 2 are common in the literature on well-being losses from disasters (e.g. Hallegatte, Vogt-Schilb,
Bangalore and Rozenberg, *Unbreakable*, World Bank, 2017) and in welfare-weighted cost-benefit analysis (HM Treasury,
*The Green Book*, 2022, which uses 1.3).
"""

__all__ = ['equity_weights', 'weighted_aal_table', 'weighted_ep_table', 'losses_at_return_periods']

import logging

import numpy as np
import pandas as pd

from .ord import read_ord
from .rows import by_group, suppress_small_groups

logger = logging.getLogger(__name__)


def equity_weights(income, eta, reference_income='mean'):
    """Return the equity weight of each row.

    Args:
        income (pandas.Series): Income of each row; blank where unknown.
        eta (float): Inequality aversion. 0 gives every row a weight of 1.
        reference_income (str or float): ``mean`` or ``median`` of the known incomes, or a number. A row with this
            income has a weight of 1.

    Returns:
        tuple: ``(weights, reference)``. Rows with no income get a weight of 1.
    """
    if reference_income in ('mean', 'median'):
        reference = float(getattr(income.dropna(), reference_income)())
    else:
        reference = float(reference_income)
    weights = (reference / income) ** eta
    unknown = int(weights.isna().sum())
    if unknown:
        logger.warning("equity weighting: %d rows have no income and are given a weight of 1", unknown)
    return weights.fillna(1.0), reference


def weighted_aal_table(rows, weights, settings):
    """AAL of each group before and after equity weighting.

    Columns: ``rows``, ``aal``, ``aal_share`` (of all rows), ``mean_weight``, ``weighted_aal`` and
    ``weighted_aal_share``. A group whose weighted share is above its plain share bears losses that weigh more than
    their money value.

    Args:
        rows (pandas.DataFrame): Report rows.
        weights (pandas.Series): Weight of each row, from :func:`equity_weights`.
        settings (dict): Checked ``equity_report`` settings.

    Returns:
        pandas.DataFrame: One row per group and a total row, with small groups suppressed.
    """
    rows = rows.assign(weight=weights, weighted_aal=rows['aal'] * weights)
    total_aal, total_weighted = rows['aal'].sum(), rows['weighted_aal'].sum()

    def aggregate(group):
        aal, weighted = group['aal'].sum(), group['weighted_aal'].sum()
        return {
            'rows': len(group),
            'aal': aal,
            'aal_share': aal / total_aal if total_aal > 0 else np.nan,
            'mean_weight': group['weight'].mean(),
            'weighted_aal': weighted,
            'weighted_aal_share': weighted / total_weighted if total_weighted > 0 else np.nan,
        }

    table = by_group(rows, settings['group_by'], aggregate)
    return suppress_small_groups(table, settings['group_by'], settings['min_group_size'])


def losses_at_return_periods(period_losses, period_weights, return_periods):
    """Return the loss exceeded with probability ``1 / return_period`` for each return period.

    Args:
        period_losses (pandas.Series): Loss of each period that has one, indexed by period.
        period_weights (pandas.Series): Weight (probability) of each of those periods, same index. Periods with no
            loss are left out; together all periods weigh 1.
        return_periods (list): Return periods, in years.

    Returns:
        numpy.ndarray: The loss at each return period; 0 where the chance of any loss is lower than 1 / return period.
    """
    order = np.argsort(-period_losses.to_numpy(), kind='stable')
    losses = period_losses.to_numpy()[order]
    exceedance = np.cumsum(period_weights.reindex(period_losses.index).to_numpy()[order])
    # a small relative tolerance, so a return period written as 333.333333 still finds the 3rd of 1000 periods
    at = np.searchsorted(exceedance, (1.0 / np.asarray(return_periods, dtype=float)) * (1 - 1e-6), side='left')
    return np.where(at < len(losses), losses[np.minimum(at, len(losses) - 1)], 0.0)


def weighted_ep_table(output_dir, weights_by_summary_id, sample_type, settings):
    """Occurrence and aggregate exceedance losses of the whole summary set, before and after equity weighting.

    Losses are the mean loss of each event from the moment period loss table (MPLT), summed over the rows.

    Args:
        output_dir (str): Analysis output directory.
        weights_by_summary_id (pandas.Series): Weight of each summary id.
        sample_type (int): MPLT sample type to use, 1 (analytical) or 2 (sample mean).
        settings (dict): Checked ``equity_report`` settings.

    Returns:
        pandas.DataFrame: Columns ``ReturnPeriod``, ``OEP``, ``OEPWeighted``, ``AEP`` and ``AEPWeighted``.
    """
    mplt = read_ord(output_dir, settings['runtype'], settings['summary_set'], 'mplt',
                    needed_for='Equity-weighted exceedance probabilities (plt_moment)')
    mplt = mplt[mplt['SampleType'] == sample_type]
    weight = mplt['SummaryId'].map(weights_by_summary_id).fillna(1.0)
    events = (mplt.assign(Weighted=mplt['MeanLoss'] * weight)
              .groupby(['Period', 'EventId'], sort=False)
              .agg(Loss=('MeanLoss', 'sum'), Weighted=('Weighted', 'sum'), PeriodWeight=('PeriodWeight', 'first')))
    periods = events.groupby(level='Period')
    period_weights = periods['PeriodWeight'].first()
    return_periods = settings['equity_weighting']['return_periods'][::-1]
    return pd.DataFrame({
        'ReturnPeriod': return_periods,
        'OEP': losses_at_return_periods(periods['Loss'].max(), period_weights, return_periods),
        'OEPWeighted': losses_at_return_periods(periods['Weighted'].max(), period_weights, return_periods),
        'AEP': losses_at_return_periods(periods['Loss'].sum(), period_weights, return_periods),
        'AEPWeighted': losses_at_return_periods(periods['Weighted'].sum(), period_weights, return_periods),
    })
