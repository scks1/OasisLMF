import numpy as np
import pandas as pd
import pytest

from oasislmf.equity import equity_report_settings, run_equity_report
from oasislmf.equity.assistance import assistance_tables
from oasislmf.utils.exceptions import OasisException

# Four locations in two bands, 4 periods of weight 0.25. il summary ids differ from gul ones and L4 is uninsured.
GUL_INFO = pd.DataFrame({'summary_id': [1, 2, 3, 4], 'LocNumber': ['L1', 'L2', 'L3', 'L4'], 'Band': ['A', 'A', 'B', 'B'],
                         'Income': [10_000, 20_000, 40_000, 50_000], 'tiv': [100.0, 100.0, 300.0, 300.0]})
IL_INFO = GUL_INFO.iloc[:3].assign(summary_id=[11, 12, 13])
# (period, event): {summary id: loss}
GUL_EVENTS = {(1, 1): {1: 100.0, 2: 50.0, 3: 200.0}, (2, 2): {1: 10.0, 3: 100.0}, (2, 3): {2: 30.0}}
IL_EVENTS = {(1, 1): {11: 20.0, 13: 180.0}, (2, 2): {13: 100.0}}
# retained: (1,1): L1 80, L2 50, L3 20; (2,2): L1 10; (2,3): L2 30
PROGRAMME = {'deductible': 15.0, 'share': 0.5, 'limit': 30.0, 'return_periods': [1, 2, 4]}


def mplt(events):
    return pd.DataFrame([{'Period': p, 'PeriodWeight': 0.25, 'EventId': e, 'SummaryId': s, 'SampleType': t,
                          'MeanLoss': loss} for (p, e), losses in events.items() for s, loss in losses.items()
                         for t in (1, 2)])


def palt(events):
    aal = {}
    for losses in events.values():
        for s, loss in losses.items():
            aal[s] = aal.get(s, 0.0) + loss * 0.25
    return pd.DataFrame([{'SummaryId': s, 'SampleType': t, 'MeanLoss': v, 'SDLoss': 0.0} for s, v in aal.items()
                         for t in (1, 2)])


def write(out, runtype, info, events, splt=None):
    out.mkdir(parents=True, exist_ok=True)
    info.to_csv(out / f'{runtype}_S1_summary-info.csv', index=False)
    palt(events).to_csv(out / f'{runtype}_S1_palt.csv', index=False)
    mplt(events).to_csv(out / f'{runtype}_S1_mplt.csv', index=False)
    if splt is not None:
        splt.to_csv(out / f'{runtype}_S1_splt.csv', index=False)


@pytest.fixture
def outputs(tmp_path):
    out = tmp_path / 'output'
    write(out, 'gul', GUL_INFO, GUL_EVENTS)
    write(out, 'il', IL_INFO, IL_EVENTS)
    return out


def checked(programme=PROGRAMME, **equity_report):
    return equity_report_settings({'equity_report': {'group_by': ['Band'], 'min_group_size': 0,
                                                     'public_assistance': programme, **equity_report}})


def test_programme_cost_and_reach_by_group(outputs):
    groups, budget = assistance_tables(str(outputs), checked(), 2)
    groups = groups.set_index('Band')

    # paid: (1,1) L1 min(32.5, 30) = 30, L2 17.5, L3 2.5; (2,2) L1 0 (below the deductible); (2,3) L2 7.5
    assert groups.loc['A', 'retained_aal'] == pytest.approx((80 + 10 + 50 + 30) * 0.25)
    assert groups.loc['A', 'assistance_aal'] == pytest.approx((30 + 17.5 + 7.5) * 0.25)
    assert groups.loc['B', 'retained_aal'] == pytest.approx(20 * 0.25)
    assert groups.loc['B', 'assistance_aal'] == pytest.approx(2.5 * 0.25)
    assert groups.loc['All', 'assistance_aal'] == pytest.approx(57.5 * 0.25)
    assert groups.loc['A', 'share_of_retained_covered'] == pytest.approx(55 / 170)
    assert groups.loc['A', 'assistance_share'] == pytest.approx(55 / 57.5)
    assert groups.loc['A', 'retained_share'] == pytest.approx(170 / 190)
    assert list(groups['eligible_rows']) == [2, 2, 4]
    assert 'median_retained_to_income_before' not in groups

    # budget: period 1 spends 50 on one event; period 2 spends 0 + 7.5
    assert list(budget['ReturnPeriod']) == [4, 2, 1]
    assert list(budget['OEP']) == [50.0, 7.5, 0.0]
    assert list(budget['AEP']) == [50.0, 7.5, 0.0]


def test_means_tested_programme(outputs):
    groups, budget = assistance_tables(
        str(outputs), checked({**PROGRAMME, 'income_threshold': 30_000}, income_column='Income'), 2)
    groups = groups.set_index('Band')

    assert groups.loc['B', 'eligible_rows'] == 0 and groups.loc['B', 'assistance_aal'] == 0.0
    assert groups.loc['A', 'assistance_aal'] == pytest.approx(55 * 0.25)
    # L1 retains 22.5 a year on 10k, L2 20 on 20k; after assistance 22.5 - 7.5 and 20 - 6.25
    assert groups.loc['A', 'median_retained_to_income_before'] == pytest.approx((22.5 / 10_000 + 20 / 20_000) / 2)
    assert groups.loc['A', 'median_retained_to_income_after'] == pytest.approx((15 / 10_000 + 13.75 / 20_000) / 2)
    assert list(budget['OEP']) == [47.5, 7.5, 0.0]


def test_without_insured_outputs_every_loss_is_retained(outputs):
    for f in outputs.glob('il_*'):
        f.unlink()

    groups, _ = assistance_tables(str(outputs), checked({'return_periods': [4]}), 2)

    assert groups.set_index('Band').loc['All', 'retained_aal'] == pytest.approx(490 * 0.25)
    assert groups.set_index('Band').loc['All', 'assistance_aal'] == pytest.approx(490 * 0.25)


def splt(events):
    return pd.DataFrame([{'Period': p, 'PeriodWeight': 0.25, 'EventId': e, 'SummaryId': s, 'SampleId': i, 'Loss': loss}
                         for (p, e), samples in events.items() for s, losses in samples.items()
                         for i, loss in enumerate(losses, start=1) if loss])


def test_programme_applied_to_each_sample(tmp_path):
    # L1 loses 200 in one of two samples: a mean of 100. With a deductible of 50, each sample pays 150 and 0, a mean of
    # 75, where the mean loss would pay 50.
    out = tmp_path / 'output'
    events = {(1, 1): {1: 100.0}}
    gul_splt = splt({(1, 1): {1: [200.0, 0.0]}})
    # pltpy also writes the analytical mean as sample -1, which isn't one of the samples
    gul_splt = pd.concat([gul_splt.iloc[:1].assign(SampleId=-1, Loss=100.0), gul_splt])
    write(out, 'gul', GUL_INFO, events, splt=gul_splt)
    # il has no loss: one zero row keeps its files' headers
    write(out, 'il', IL_INFO, {(1, 1): {12: 0.0}},
          splt=splt({}).reindex(columns=['Period', 'PeriodWeight', 'EventId', 'SummaryId', 'SampleId', 'Loss']))
    programme = {'deductible': 50.0, 'return_periods': [4]}

    sampled, budget = assistance_tables(str(out), checked(programme), 2, {'number_of_samples': 2})
    mean, _ = assistance_tables(str(out), checked(programme), 2, {})

    assert sampled.set_index('Band').loc['A', 'assistance_aal'] == pytest.approx(75 * 0.25)
    assert sampled.set_index('Band').loc['A', 'retained_aal'] == pytest.approx(100 * 0.25)
    assert list(budget['OEP']) == [75.0]
    assert mean.set_index('Band').loc['A', 'assistance_aal'] == pytest.approx(50 * 0.25)


def test_rows_must_match_between_perspectives(outputs):
    for f in outputs.glob('il_*'):
        f.unlink()
    write(outputs, 'il', pd.DataFrame({'summary_id': [1], 'Band': ['A'], 'tiv': [1.0]}), {(1, 1): {1: 1.0}})

    with pytest.raises(OasisException, match='same rows'):
        assistance_tables(str(outputs), checked(), 2)


def test_needs_the_plt(outputs):
    (outputs / 'gul_S1_mplt.csv').unlink()
    with pytest.raises(OasisException, match='plt_moment'):
        assistance_tables(str(outputs), checked(), 2)


def test_small_groups_are_suppressed(outputs):
    groups, _ = assistance_tables(str(outputs), checked(min_group_size=3), 2)
    groups = groups.set_index('Band')

    assert groups.loc['A', 'suppressed'] and np.isnan(groups.loc['A', 'eligible_rows'])
    assert groups.loc['All', 'eligible_rows'] == 4


@pytest.mark.parametrize('bad, message', [
    ({'share': 1.5}, "'share' is the share"),
    ({'share': 0}, "'share' must be a positive"),
    ({'deductible': -1}, "'deductible' must be a non-negative"),
    ({'limit': 0}, "'limit' must be a positive"),
    ({'income_threshold': 10}, "needs equity_report.income_column"),
    ({'return_periods': []}, "'return_periods' must be"),
    ({'cap': 1}, r"unknown keys \['cap'\]"),
])
def test_settings_are_checked(bad, message):
    with pytest.raises(OasisException, match=message):
        checked(bad)


def test_report_writes_the_tables(outputs, tmp_path):
    tables = run_equity_report(str(outputs), {'equity_report': {'group_by': ['Band'], 'min_group_size': 0,
                                                                'public_assistance': True}}, str(tmp_path / 'r'))

    assert {'equity-public-assistance.csv', 'equity-public-assistance-budget-ep.csv'} <= set(tables)
    budget = pd.read_csv(tmp_path / 'r' / 'equity-public-assistance-budget-ep.csv')
    assert list(budget['ReturnPeriod']) == [1000, 500, 250, 200, 100, 50, 25, 10]
