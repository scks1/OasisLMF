import numpy as np
import pandas as pd
import pytest

from oasislmf.equity import equity_report_settings, run_equity_report
from oasislmf.equity.basis import basis_risk_table
from oasislmf.utils.exceptions import OasisException

from .test_assistance import GUL_INFO, splt, write

# A parametric policy on four locations, 4 periods of weight 0.25. il rows are numbered 11-14.
IL_INFO = GUL_INFO.assign(summary_id=[11, 12, 13, 14])
GUL_EVENTS = {(1, 1): {1: 100.0, 2: 50.0, 3: 80.0}, (2, 2): {4: 40.0}, (2, 3): {2: 20.0}}
PAYOUTS = {
    (1, 1): {12: 60.0, 13: 80.0},  # L1's loss isn't paid; L2 is paid more than its loss
    (2, 2): {11: 30.0},  # L1 is paid with no loss; L4's loss isn't paid
    (2, 3): {12: 10.0},
    (3, 4): {13: 25.0},  # a payout in a period with no ground-up loss at all
}


@pytest.fixture
def outputs(tmp_path):
    out = tmp_path / 'output'
    write(out, 'gul', GUL_INFO, GUL_EVENTS)
    write(out, 'il', IL_INFO, PAYOUTS)
    return out


def checked(basis_risk=True, **equity_report):
    return equity_report_settings({'equity_report': {'group_by': ['Band'], 'min_group_size': 0,
                                                     'basis_risk': basis_risk, **equity_report}})


def test_basis_risk_by_group(outputs):
    table = basis_risk_table(str(outputs), checked(), 2).set_index('Band')
    a, b = table.loc['A'], table.loc['B']

    assert a['gul_aal'] == pytest.approx(170 * 0.25) and a['payout_aal'] == pytest.approx(100 * 0.25)
    assert a['payout_to_loss'] == pytest.approx(100 / 170)
    assert a['loss_events'] == pytest.approx(3 * 0.25)
    assert a['share_loss_events_unpaid'] == pytest.approx(1 / 3)
    assert a['unpaid_loss_aal'] == pytest.approx(100 * 0.25) and a['unpaid_loss_share'] == pytest.approx(100 / 170)
    assert a['shortfall_aal'] == pytest.approx(110 * 0.25)
    assert a['payouts_without_loss'] == pytest.approx(0.25) and a['overpayment_aal'] == pytest.approx(40 * 0.25)

    assert b['gul_aal'] == pytest.approx(120 * 0.25) and b['payout_aal'] == pytest.approx(105 * 0.25)
    assert b['share_loss_events_unpaid'] == pytest.approx(0.5)
    assert b['unpaid_loss_aal'] == pytest.approx(40 * 0.25) and b['shortfall_aal'] == pytest.approx(40 * 0.25)
    # the period-3 payout has no ground-up loss in its period, and still counts with its period's weight
    assert b['payouts_without_loss'] == pytest.approx(0.25) and b['overpayment_aal'] == pytest.approx(25 * 0.25)

    assert table.loc['All', 'loss_events'] == pytest.approx(5 * 0.25)
    assert table.loc['All', 'share_loss_events_unpaid'] == pytest.approx(2 / 5)
    assert 'median_shortfall_to_income' not in table


def test_basis_risk_thresholds(outputs):
    table = basis_risk_table(str(outputs), checked({'loss_threshold': 60, 'payout_threshold': 50}), 2).set_index('Band')

    # loss events are now L1 (100) and L3 (80); L1's is unpaid, L3's is paid (80 > 50)
    assert table.loc['All', 'loss_events'] == pytest.approx(2 * 0.25)
    assert table.loc['All', 'share_loss_events_unpaid'] == pytest.approx(0.5)
    # L2's payout of 60 for a 50 loss now counts as a payout without a loss event
    assert table.loc['A', 'payouts_without_loss'] == pytest.approx(0.25)


def test_basis_risk_with_income(outputs):
    table = basis_risk_table(str(outputs), checked(income_column='Income'), 2).set_index('Band')

    # B: L3 has no shortfall, L4 a shortfall of 10 a year on 50k
    assert table.loc['B', 'median_shortfall_to_income'] == pytest.approx((0 + 10 / 50_000) / 2)


def test_basis_risk_uses_samples(tmp_path):
    # L1 loses 100 in both samples; the trigger pays 100 in one. On the mean (a payout of 50) the event looks paid.
    out = tmp_path / 'output'
    write(out, 'gul', GUL_INFO, {(1, 1): {1: 100.0}}, splt=splt({(1, 1): {1: [100.0, 100.0]}}))
    write(out, 'il', IL_INFO, {(1, 1): {11: 50.0}}, splt=splt({(1, 1): {11: [100.0, 0.0]}}))

    sampled = basis_risk_table(str(out), checked(), 2, {'number_of_samples': 2}).set_index('Band')
    mean = basis_risk_table(str(out), checked(), 2).set_index('Band')

    assert sampled.loc['A', 'share_loss_events_unpaid'] == pytest.approx(0.5)
    assert sampled.loc['A', 'shortfall_aal'] == pytest.approx(50 * 0.25)
    assert mean.loc['A', 'share_loss_events_unpaid'] == 0.0


def test_basis_risk_needs_insured_outputs(outputs):
    for f in outputs.glob('il_*'):
        f.unlink()
    with pytest.raises(OasisException, match='il_S1_summary-info'):
        basis_risk_table(str(outputs), checked(), 2)


def test_basis_risk_suppresses_small_groups(outputs):
    table = basis_risk_table(str(outputs), checked(min_group_size=3), 2).set_index('Band')
    assert table.loc['A', 'suppressed'] and np.isnan(table.loc['A', 'gul_aal'])


@pytest.mark.parametrize('bad, message', [
    ({'loss_threshold': -1}, "'loss_threshold' must be a non-negative"),
    ({'trigger': 1}, r"unknown keys \['trigger'\]"),
])
def test_basis_risk_settings_are_checked(bad, message):
    with pytest.raises(OasisException, match=message):
        checked(bad)


def test_report_writes_the_basis_risk_table(outputs, tmp_path):
    tables = run_equity_report(str(outputs), {'equity_report': {'group_by': ['Band'], 'min_group_size': 0,
                                                                'basis_risk': True}}, str(tmp_path / 'r'))
    assert 'equity-basis-risk.csv' in tables
    assert list(pd.read_csv(tmp_path / 'r' / 'equity-basis-risk.csv')['Band']) == ['A', 'B', 'All']
