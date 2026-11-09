import json
import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oasislmf.computation.run.equity import EquityReport
from oasislmf.equity import equity_report_settings, run_equity_report
from oasislmf.equity.premiums import scheme_premiums
from oasislmf.equity.rows import report_rows
from oasislmf.equity.weighting import equity_weights, losses_at_return_periods
from oasislmf.utils.exceptions import OasisException

TESTS_ASSETS_DIR = Path(__file__).parent.parent / 'assets'

# six locations in two income bands; band A is poorer and carries more loss for its value
SUMMARY_INFO = pd.DataFrame({
    'summary_id': [1, 2, 3, 4, 5, 6],
    'LocNumber': ['L1', 'L2', 'L3', 'L4', 'L5', 'L6'],
    'Band': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Income': [10_000, 20_000, '', 40_000, 50_000, 60_000],
    'tiv': [100.0, 100.0, 100.0, 200.0, 200.0, 0.0],
})
AAL = {1: 4.0, 2: 1.0, 3: 0.5, 4: 1.0, 5: 0.0}  # summary id 6 has no loss, so no PALT row


def write_outputs(output_dir, mplt=None):
    output_dir.mkdir(parents=True, exist_ok=True)
    SUMMARY_INFO.to_csv(output_dir / 'gul_S1_summary-info.csv', index=False)
    palt = pd.DataFrame([{'SummaryId': s, 'SampleType': t, 'MeanLoss': aal * (1 if t == 2 else 1.1), 'SDLoss': 0.0}
                         for s, aal in AAL.items() for t in (1, 2)])
    palt.to_csv(output_dir / 'gul_S1_palt.csv', index=False)
    if mplt is not None:
        mplt.to_csv(output_dir / 'gul_S1_mplt.csv', index=False)
    return str(output_dir)


def settings(**equity_report):
    return {'equity_report': {'group_by': ['Band'], 'min_group_size': 0, **equity_report}}


def checked(**equity_report):
    return equity_report_settings(settings(**equity_report))


def by_band(table):
    return table.set_index('Band')


# settings

def test_no_section_means_no_report():
    assert equity_report_settings({}) is None


def test_defaults_are_filled_in():
    s = equity_report_settings({'equity_report': {'group_by': 'Band'}})
    assert s['group_by'] == ['Band'] and s['runtype'] == 'gul' and s['summary_set'] == 1 and s['min_group_size'] == 10
    assert s['equity_weighting'] is None and s['premiums'] is None


@pytest.mark.parametrize('bad, message', [
    ({'runtype': 'gul'}, "needs 'group_by'"),
    ({'group_by': ['Band'], 'runtype': 'xx'}, "'runtype' must be one of"),
    ({'group_by': ['Band'], 'burden': 0.1}, r"unknown keys \['burden'\]"),
    ({'group_by': ['Band'], 'min_group_size': -1}, "'min_group_size' must be a whole number"),
    ({'group_by': ['Band'], 'sample_type': 3}, "'sample_type' must be 1"),
    ({'group_by': ['Band'], 'equity_weighting': {'eta': 1}}, "equity_weighting needs equity_report.income_column"),
    ({'group_by': ['Band'], 'income_column': 'Income', 'equity_weighting': {'eta': -1}}, "'eta' must be a non-negative"),
    ({'group_by': ['Band'], 'income_column': 'Income', 'equity_weighting': {'reference_income': 'mode'}}, "'reference_income' must be"),
    ({'group_by': ['Band'], 'premiums': {'schemes': ['lottery']}}, "must name a 'type' out of"),
    ({'group_by': ['Band'], 'premiums': {'schemes': [{'type': 'capped', 'cap': 0.05}]}}, r"\(capped\) needs equity_report.income_column"),
    ({'group_by': ['Band'], 'income_column': 'Income', 'premiums': {'schemes': [{'type': 'capped'}]}}, "'cap' must be a positive"),
    ({'group_by': ['Band'], 'income_column': 'Income',
      'premiums': {'schemes': [{'type': 'means_tested', 'income_threshold': 1, 'discount': 2}]}}, "'discount' is a share"),
    ({'group_by': ['Band'], 'premiums': {'schemes': [{'type': 'flat', 'basis': 'area'}]}}, "'basis' must be 'row'"),
    ({'group_by': ['Band'], 'premiums': {'schemes': ['flat', 'flat']}}, "schemes need different names"),
])
def test_bad_settings(bad, message):
    with pytest.raises(OasisException, match=message):
        equity_report_settings({'equity_report': bad})


# rows and burden

def test_rows_take_the_sample_mean_aal_and_skip_missing_income(tmp_path):
    rows, sample_type = report_rows(write_outputs(tmp_path), checked(income_column='Income'))
    assert sample_type == 2
    assert rows['aal'].tolist() == [4.0, 1.0, 0.5, 1.0, 0.0, 0.0]
    assert rows['income'].isna().tolist() == [False, False, True, False, False, False]
    assert np.isnan(rows['aal_to_tiv'].iloc[5])  # no value


def test_rows_can_use_the_analytical_mean(tmp_path):
    rows, sample_type = report_rows(write_outputs(tmp_path), checked(sample_type=1))
    assert sample_type == 1 and rows['aal'].iloc[0] == pytest.approx(4.4)


def test_missing_group_column_says_how_to_add_it(tmp_path):
    with pytest.raises(OasisException, match=r"columns \['Ward'\] are not in gul_S1_summary-info.*exposure_attributes"):
        report_rows(write_outputs(tmp_path), checked(group_by=['Ward']))


def test_missing_alt_says_which_output_to_turn_on(tmp_path):
    output_dir = write_outputs(tmp_path)
    (tmp_path / 'gul_S1_palt.csv').unlink()
    with pytest.raises(OasisException, match="needs gul_S1_altmeanonly.csv .*turn on the matching ord_output"):
        report_rows(output_dir, checked())


def test_burden_table(tmp_path):
    tables = run_equity_report(write_outputs(tmp_path), settings(income_column='Income', burden_threshold=0.009,
                                                                 income_burden_threshold=0.0001))
    burden = by_band(tables['gul_S1_equity-burden.csv'])
    a, b, total = burden.loc['A'], burden.loc['B'], burden.loc['All']
    assert (a['rows'], a['tiv'], a['aal']) == (3, 300.0, 5.5)
    assert a['aal_to_tiv'] == pytest.approx(5.5 / 300)
    assert total['aal_to_tiv'] == pytest.approx(6.5 / 700)
    assert a['relative_aal_to_tiv'] == pytest.approx((5.5 / 300) / (6.5 / 700))
    assert b['relative_aal_to_tiv'] == pytest.approx((1 / 400) / (6.5 / 700))
    assert a['median_aal_to_tiv'] == pytest.approx(0.01)
    # rows over 0.9% of their value: L1 (4%), L2 (1%) of band A; none of band B
    assert a['share_rows_over_burden_threshold'] == pytest.approx(2 / 3)
    assert a['share_tiv_over_burden_threshold'] == pytest.approx(2 / 3)
    assert b['share_rows_over_burden_threshold'] == 0
    assert a['rows_with_income'] == 2 and a['mean_income'] == pytest.approx(15_000)
    assert a['median_aal_to_income'] == pytest.approx((4 / 10_000 + 1 / 20_000) / 2)
    # over 0.01% of income: L1 (0.04%) only
    assert a['share_rows_over_income_burden_threshold'] == pytest.approx(0.5)
    assert not burden['suppressed'].any()
    assert list(tables) == ['gul_S1_equity-burden.csv']
    assert (tmp_path / 'equity' / 'gul_S1_equity-burden.csv').is_file()
    used = json.loads((tmp_path / 'equity' / 'gul_S1_equity-report.json').read_text())
    assert used['sample_type'] == 2 and used['group_by'] == ['Band']


def test_small_groups_are_suppressed(tmp_path):
    info = pd.concat([SUMMARY_INFO, SUMMARY_INFO.iloc[[0]].assign(summary_id=7, Band='C')])
    output_dir = write_outputs(tmp_path)
    info.to_csv(tmp_path / 'gul_S1_summary-info.csv', index=False)
    burden = by_band(run_equity_report(output_dir, settings(min_group_size=2))['gul_S1_equity-burden.csv'])
    assert burden.loc['C', 'suppressed'] and burden.loc['C', 'rows'] == 1
    assert burden.loc['C', ['tiv', 'aal', 'aal_to_tiv', 'median_aal_to_tiv']].isna().all()
    assert not burden.loc['A', 'suppressed'] and burden.loc['A', 'aal'] == 5.5
    assert burden.loc['All', 'tiv'] == 800.0  # the total still counts every row


# equity weighting

def test_weights():
    income = pd.Series([10.0, 20.0, np.nan, 40.0])
    weights, reference = equity_weights(income, eta=1, reference_income=20)
    assert reference == 20 and weights.tolist() == [2.0, 1.0, 1.0, 0.5]
    weights, reference = equity_weights(income, eta=0)
    assert weights.tolist() == [1.0] * 4
    _, reference = equity_weights(income, eta=1, reference_income='median')
    assert reference == 20
    _, reference = equity_weights(income, eta=1, reference_income='mean')
    assert reference == pytest.approx(70 / 3)


def test_losses_at_return_periods():
    losses = pd.Series([5.0, 50.0, 20.0], index=[3, 1, 2])
    weights = pd.Series([0.1, 0.1, 0.1], index=[1, 2, 3])
    # exceeded with probability 0.1 (50), 0.2 (20), 0.3 (5); beyond that there is no loss
    assert losses_at_return_periods(losses, weights, [10, 5, 1 / 0.3, 2]).tolist() == [50.0, 20.0, 5.0, 0.0]
    assert losses_at_return_periods(losses, weights, [333.333333]).tolist() == [50.0]


MPLT = pd.DataFrame({
    'Period': [1, 1, 2, 3],
    'PeriodWeight': [0.25] * 4,
    'EventId': [1, 2, 3, 4],
    'SummaryId': [1, 4, 1, 4],
    'SampleType': [2] * 4,
    'MeanLoss': [8.0, 4.0, 4.0, 2.0],
})


def test_weighted_aal_and_ep(tmp_path):
    report = settings(income_column='Income', equity_weighting={'eta': 1, 'reference_income': 20_000, 'return_periods': [2, 4]})
    tables = run_equity_report(write_outputs(tmp_path, MPLT), report)

    aal = by_band(tables['gul_S1_equity-weighted-aal.csv'])
    # weights: L1 2, L2 1, L3 1 (no income), L4 0.5, L5 0.4, L6 1/3
    assert aal.loc['A', 'weighted_aal'] == pytest.approx(4 * 2 + 1 + 0.5)
    assert aal.loc['B', 'weighted_aal'] == pytest.approx(1 * 0.5)
    assert aal.loc['All', 'weighted_aal_share'] == pytest.approx(1.0)
    assert aal.loc['A', 'aal_share'] == pytest.approx(5.5 / 6.5)
    assert aal.loc['A', 'weighted_aal_share'] == pytest.approx(9.5 / 10)
    assert aal.loc['A', 'mean_weight'] == pytest.approx(4 / 3)

    ep = tables['gul_S1_equity-weighted-ep.csv'].set_index('ReturnPeriod')
    # period losses: 1 -> events of 8 (w 2) and 4 (w 0.5); 2 -> 4 (w 2); 3 -> 2 (w 0.5)
    assert ep.loc[4].tolist() == [8.0, 16.0, 12.0, 18.0]
    assert ep.loc[2].tolist() == [4.0, 8.0, 4.0, 8.0]
    used = json.loads((tmp_path / 'equity' / 'gul_S1_equity-report.json').read_text())
    assert used['equity_weighting']['reference_income_used'] == 20_000


def test_weighted_ep_is_skipped_without_mplt(tmp_path, caplog):
    tables = run_equity_report(write_outputs(tmp_path), settings(income_column='Income', equity_weighting=True))
    assert 'gul_S1_equity-weighted-aal.csv' in tables and 'gul_S1_equity-weighted-ep.csv' not in tables
    assert 'turn on plt_moment' in caplog.text


def test_unweighted_ep_matches_lecpy(tmp_path):
    """With eta = 0, the report's EP is the mean damage EP that lecpy writes for the same losses."""
    from oasislmf.pytools.lec.manager import main as lecpy
    from oasislmf.pytools.plt.manager import main as pltpy

    run_dir = tmp_path / 'run'
    shutil.copytree(TESTS_ASSETS_DIR / 'test_lecpy' / 'lec', run_dir)
    output_dir = run_dir / 'output'
    output_dir.mkdir()
    summary_files = sorted(str(p) for p in (run_dir / 'work' / 'gul').glob('summarypy*.bin'))
    pltpy(run_dir=str(run_dir), files_in=summary_files, mplt=str(output_dir / 'gul_S1_mplt.csv'))
    lecpy(run_dir=str(run_dir), subfolder='gul', ept=str(tmp_path / 'ept.csv'), agg_sample_mean=True, occ_sample_mean=True)
    ept = pd.read_csv(tmp_path / 'ept.csv')
    ept = ept[(ept['SummaryId'] == 1) & (ept['EPCalc'] == 1)]

    pd.DataFrame({'summary_id': [1], 'Band': ['A'], 'Income': [1.0], 'tiv': [1.0]}).to_csv(
        output_dir / 'gul_S1_summary-info.csv', index=False)
    pd.DataFrame({'SummaryId': [1], 'SampleType': [1], 'MeanLoss': [1.0], 'SDLoss': [0.0]}).to_csv(
        output_dir / 'gul_S1_palt.csv', index=False)
    return_periods = ept['ReturnPeriod'].unique().tolist()
    report = settings(sample_type=1, income_column='Income', equity_weighting={'eta': 0, 'return_periods': return_periods})
    ep = run_equity_report(str(output_dir), report)['gul_S1_equity-weighted-ep.csv'].set_index('ReturnPeriod')

    for ep_type, column in ((1, 'OEP'), (3, 'AEP')):
        expected = ept[ept['EPType'] == ep_type].set_index('ReturnPeriod')['Loss']
        np.testing.assert_allclose(ep.loc[expected.index, column], expected, atol=0.1)
        np.testing.assert_allclose(ep.loc[expected.index, column + 'Weighted'], expected, atol=0.1)


# premiums

def premium_rows():
    return pd.DataFrame({'aal': [4.0, 1.0, 0.5, 1.0], 'tiv': [100.0, 100.0, 100.0, 200.0],
                         'income': [10.0, 20.0, np.nan, 100.0]})


def scheme(**kwargs):
    return equity_report_settings(settings(income_column='Income', premiums={'schemes': [kwargs]}))['premiums']['schemes'][0]


def test_risk_based_and_flat_premiums():
    rows = premium_rows()
    assert scheme_premiums(rows, scheme(type='risk_based'), loading=0.5)['premium'].tolist() == [6.0, 1.5, 0.75, 1.5]
    assert scheme_premiums(rows, scheme(type='flat'))['premium'].tolist() == [1.625] * 4
    assert scheme_premiums(rows, scheme(type='flat', basis='tiv'))['premium'].tolist() == [1.3, 1.3, 1.3, 2.6]


def test_capped_premiums_with_a_subsidy():
    result = scheme_premiums(premium_rows(), scheme(type='capped', cap=0.1))
    # caps: 1, 2, none, 10
    assert result['premium'].tolist() == [1.0, 1.0, 0.5, 1.0]
    assert result['subsidy'].tolist() == [3.0, 0.0, 0.0, 0.0]


def test_capped_premiums_with_a_surcharge_raise_the_same_total_within_the_caps():
    rows = premium_rows()
    result = scheme_premiums(rows, scheme(type='capped', cap=0.1, funding='surcharge'))
    assert result['premium'].sum() == pytest.approx(6.5)
    assert result['premium'].iloc[0] == pytest.approx(1.0)
    assert result['premium'].iloc[1] == pytest.approx(2.0)  # scaled up to its own cap
    assert (result['premium'] <= np.where(rows['income'].isna(), np.inf, 0.1 * rows['income']) + 1e-9).all()
    assert result['subsidy'].sum() == 0


def test_capped_premiums_that_cannot_raise_the_total_leave_the_rest_to_a_subsidy(caplog):
    rows = premium_rows().assign(income=[10.0, 20.0, 5.0, 10.0])
    result = scheme_premiums(rows, scheme(type='capped', cap=0.1, funding='surcharge'))
    # every row pays up to its cap; the rest of the first row's risk is left to the subsidy
    assert result['premium'].tolist() == pytest.approx([1.0, 2.0, 0.5, 1.0])
    assert result['subsidy'].tolist() == pytest.approx([2.0, 0.0, 0.0, 0.0])
    assert "left to a subsidy" in caplog.text


def test_means_tested_premiums():
    rows = premium_rows()
    subsidised = scheme_premiums(rows, scheme(type='means_tested', income_threshold=15, discount=0.5))
    assert subsidised['premium'].tolist() == [2.0, 1.0, 0.5, 1.0]
    assert subsidised['subsidy'].tolist() == [2.0, 0.0, 0.0, 0.0]
    surcharged = scheme_premiums(rows, scheme(type='means_tested', income_threshold=15, discount=0.5, funding='surcharge'))
    assert surcharged['premium'].sum() == pytest.approx(6.5)
    assert surcharged['premium'].tolist() == pytest.approx([2.0, 1.8, 0.9, 1.8])


def test_premium_table(tmp_path):
    report = settings(income_column='Income', income_burden_threshold=0.0001, premiums={
        'schemes': ['risk_based', 'flat', {'type': 'means_tested', 'name': 'half off under 15k',
                                           'income_threshold': 15_000, 'discount': 0.5}]})
    table = run_equity_report(write_outputs(tmp_path), report)['gul_S1_equity-premiums.csv']
    assert table['scheme'].unique().tolist() == ['risk_based', 'flat', 'half off under 15k']
    flat = table[table['scheme'] == 'flat'].set_index('Band')
    assert flat.loc['A', 'premium'] == pytest.approx(6.5 / 2)
    assert flat.loc['A', 'cross_subsidy'] == pytest.approx(6.5 / 2 - 5.5)
    assert flat.loc[['A', 'B'], 'cross_subsidy'].sum() == pytest.approx(0)
    means = table[table['scheme'] == 'half off under 15k'].set_index('Band')
    assert means.loc['A', 'subsidy'] == 2.0 and means.loc['All', 'subsidy'] == 2.0
    risk = table[table['scheme'] == 'risk_based'].set_index('Band')
    assert risk.loc['A', 'median_premium_to_income'] == pytest.approx((4 / 10_000 + 1 / 20_000) / 2)


# computation step

def test_step_reads_the_settings_copied_into_the_run(tmp_path):
    write_outputs(tmp_path / 'run' / 'output')
    (tmp_path / 'run' / 'input').mkdir()
    (tmp_path / 'run' / 'input' / 'analysis_settings.json').write_text(json.dumps(
        {'model_supplier_id': 'S', 'model_name_id': 'M', 'model_settings': {}, 'gul_output': True,
         'gul_summaries': [{'id': 1}], **settings()}))
    tables = EquityReport(model_run_dir=str(tmp_path / 'run'), equity_report_dir=str(tmp_path / 'report')).run()
    assert list(tables) == ['gul_S1_equity-burden.csv']
    assert (tmp_path / 'report' / 'gul_S1_equity-burden.csv').is_file()


def test_step_without_settings_raises(tmp_path):
    (tmp_path / 'run').mkdir()
    with pytest.raises(OasisException, match="no analysis settings given"):
        EquityReport(model_run_dir=str(tmp_path / 'run')).run()
