import numpy as np
import pandas as pd
import pytest

from oasislmf.equity import equity_report_settings, run_equity_report
from oasislmf.equity.sharing import loss_sharing_table
from oasislmf.utils.exceptions import OasisException

# Band A (poorer) has high deductibles and one uninsured location (L3); band B is well insured and reinsured.
GUL_INFO = pd.DataFrame({
    'summary_id': [1, 2, 3, 4, 5, 6],
    'LocNumber': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'],
    'Band': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Income': [10_000, 20_000, 30_000, 40_000, 50_000, 60_000],
    'tiv': [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
})
GUL_AAL = {1: 4.0, 2: 3.0, 3: 3.0, 4: 2.0, 5: 2.0, 6: 1.0}
# il and ri summary ids are numbered differently, and the uninsured L3 has no il row
IL_INFO = GUL_INFO[GUL_INFO['LocNumber'] != 'L3'].assign(summary_id=[11, 12, 14, 15, 16])
IL_AAL = {11: 1.0, 12: 1.0, 14: 1.8, 15: 1.8, 16: 0.9}
RI_AAL = {11: 1.0, 12: 1.0, 14: 0.9, 15: 0.9, 16: 0.45}


def write(output_dir, runtype, info, aal, summary_set=1):
    output_dir.mkdir(parents=True, exist_ok=True)
    info.to_csv(output_dir / f'{runtype}_S{summary_set}_summary-info.csv', index=False)
    pd.DataFrame([{'SummaryId': s, 'SampleType': t, 'MeanLoss': loss, 'SDLoss': 0.0} for s, loss in aal.items()
                  for t in (1, 2)]).to_csv(output_dir / f'{runtype}_S{summary_set}_palt.csv', index=False)


@pytest.fixture
def outputs(tmp_path):
    out = tmp_path / 'output'
    write(out, 'gul', GUL_INFO, GUL_AAL)
    write(out, 'il', IL_INFO, IL_AAL)
    write(out, 'ri', IL_INFO, RI_AAL)
    return out


def checked(**equity_report):
    return equity_report_settings({'equity_report': {'group_by': ['Band'], 'min_group_size': 0, 'loss_sharing': True,
                                                     **equity_report}})


def by_band(table):
    return table.set_index('Band')


def test_loss_sharing_splits_each_groups_loss(outputs):
    table = by_band(loss_sharing_table(str(outputs), checked(), 2))

    a, b, total = table.loc['A'], table.loc['B'], table.loc['All']
    assert (a['gul_aal'], a['il_aal'], a['ri_aal']) == (10.0, 2.0, 2.0)
    assert a['retained_aal'] == 8.0 and a['retained_share'] == pytest.approx(0.8)
    assert a['insurer_net_aal'] == 2.0 and a['ceded_aal'] == 0.0
    assert b['retained_aal'] == pytest.approx(0.5) and b['retained_share'] == pytest.approx(0.1)
    assert b['ceded_aal'] == pytest.approx(2.25) and b['ceded_share'] == pytest.approx(0.45)
    assert total['gul_aal'] == 15.0 and total['retained_aal'] == pytest.approx(8.5)
    assert a['retained_to_tiv'] == pytest.approx(8 / 300)
    for row in (a, b, total):
        assert row['retained_share'] + row['insurer_net_share'] + row['ceded_share'] == pytest.approx(1.0)
    assert 'median_retained_to_income' not in table


def test_loss_sharing_with_income_uses_each_rows_retained_loss(outputs):
    table = by_band(loss_sharing_table(str(outputs), checked(income_column='Income', income_burden_threshold=1e-4), 2))

    # band A rows retain 3, 2 and 3 (uninsured) on incomes 10k, 20k and 30k
    assert table.loc['A', 'median_retained_to_income'] == pytest.approx(1e-4)
    assert table.loc['A', 'share_rows_over_income_burden_threshold'] == pytest.approx(1 / 3)
    assert table.loc['B', 'share_rows_over_income_burden_threshold'] == 0.0


def test_loss_sharing_without_reinsurance_outputs(outputs):
    for f in outputs.glob('ri_*'):
        f.unlink()

    table = by_band(loss_sharing_table(str(outputs), checked(), 2))

    assert 'ri_aal' not in table and 'ceded_aal' not in table
    assert table.loc['B', 'insurer_net_aal'] == pytest.approx(4.5)


def test_loss_sharing_by_group_when_rows_do_not_match(outputs):
    # the il summary set groups by band only, so rows can't be matched to the ground-up locations
    for f in outputs.glob('il_*'):
        f.unlink()
    write(outputs, 'il', pd.DataFrame({'summary_id': [1, 2], 'Band': ['A', 'B'], 'tiv': [200.0, 600.0]}), {1: 2.0, 2: 4.5})

    table = by_band(loss_sharing_table(str(outputs), checked(income_column='Income'), 2))

    assert table.loc['A', 'il_aal'] == 2.0 and table.loc['B', 'il_aal'] == 4.5
    assert table.loc['All', 'il_aal'] == 6.5
    assert 'median_retained_to_income' not in table


def test_loss_sharing_reads_other_summary_sets(outputs):
    write(outputs, 'il', IL_INFO, {k: v / 2 for k, v in IL_AAL.items()}, summary_set=2)

    table = by_band(loss_sharing_table(str(outputs), checked(loss_sharing={'summary_sets': {'il': 2}}), 2))

    assert table.loc['A', 'il_aal'] == 1.0


def test_loss_sharing_warns_when_insured_exceeds_ground_up(outputs, caplog):
    write(outputs, 'il', IL_INFO, {k: v * 10 for k, v in IL_AAL.items()})

    loss_sharing_table(str(outputs), checked(), 2)

    assert 'more than the ground-up AAL' in caplog.text


def test_loss_sharing_suppresses_small_groups(outputs):
    table = by_band(loss_sharing_table(str(outputs), checked(min_group_size=4), 2))

    assert table.loc['A', 'suppressed'] and np.isnan(table.loc['A', 'retained_aal'])
    assert not table.loc['All', 'suppressed']


def test_loss_sharing_needs_insured_outputs(outputs):
    for f in outputs.glob('il_*'):
        f.unlink()
    with pytest.raises(OasisException, match='il_S1_summary-info'):
        loss_sharing_table(str(outputs), checked(), 2)


@pytest.mark.parametrize('bad, message', [
    ({'summary_sets': {'xx': 1}}, r"unknown keys \['xx'\]"),
    ({'summary_sets': {'il': 0}}, "summary set for 'il'"),
    ({'summary_sets': [1]}, 'must map perspectives'),
    ({'sets': {}}, r"unknown keys \['sets'\]"),
])
def test_loss_sharing_settings_are_checked(bad, message):
    with pytest.raises(OasisException, match=message):
        checked(loss_sharing=bad)


def test_loss_sharing_settings_default_to_the_reports_summary_set():
    assert checked(summary_set=3)['loss_sharing'] == {'summary_sets': {'gul': 3, 'il': 3, 'ri': 3}}
    assert checked(loss_sharing=None)['loss_sharing'] is None


def test_report_writes_the_loss_sharing_table(outputs, tmp_path):
    tables = run_equity_report(str(outputs), {'equity_report': {'group_by': ['Band'], 'min_group_size': 0,
                                                                'loss_sharing': True}}, str(tmp_path / 'report'))

    assert 'equity-loss-sharing.csv' in tables
    written = pd.read_csv(tmp_path / 'report' / 'equity-loss-sharing.csv')
    assert list(written['Band']) == ['A', 'B', 'All']
