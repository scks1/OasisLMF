import json

import numpy as np
import pandas as pd
import pytest

from oasislmf.computation.run.equity import CompareScenarios
from oasislmf.equity.compare import compare_ep_table, compare_table, run_scenario_comparison
from oasislmf.equity.rows import report_rows
from oasislmf.manager import OasisManager
from oasislmf.utils.exceptions import OasisException

from .test_equity_report import checked, settings

# Two bands of three locations. In the scenario, a retrofit programme reaches band B (richer) but only one location
# in band A, so band B gets most of the reduction although band A carried most of the base loss.
INFO = pd.DataFrame({
    'summary_id': [1, 2, 3, 4, 5, 6],
    'Band': ['A', 'A', 'A', 'B', 'B', 'B'],
    'Income': [10_000, 20_000, 30_000, 40_000, 50_000, 60_000],
    'tiv': [100.0, 100.0, 100.0, 200.0, 200.0, 200.0],
})
BASE_AAL = {1: 4.0, 2: 3.0, 3: 3.0, 4: 2.0, 5: 2.0, 6: 1.0}
SCENARIO_AAL = {1: 4.0, 2: 3.0, 3: 1.0, 4: 0.5, 5: 0.5, 6: 0.0}


def write_run(output_dir, aal, info=INFO, ept=None, sample_types=(1, 2)):
    output_dir.mkdir(parents=True, exist_ok=True)
    info.to_csv(output_dir / 'gul_S1_summary-info.csv', index=False)
    pd.DataFrame([{'SummaryId': s, 'SampleType': t, 'MeanLoss': loss, 'SDLoss': 0.0}
                  for s, loss in aal.items() if loss for t in sample_types]).to_csv(output_dir / 'gul_S1_palt.csv', index=False)
    if ept is not None:
        ept.to_csv(output_dir / 'gul_S1_ept.csv', index=False)
    return str(output_dir)


def rows(output_dir, **equity_report):
    return report_rows(output_dir, checked(**equity_report))[0]


def by_band(table):
    return table.set_index('Band')


def test_compare_table_shows_who_gets_the_reduction(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL)
    scenario = write_run(tmp_path / 'scenario', SCENARIO_AAL)

    table = by_band(compare_table(rows(base), rows(scenario), checked()))

    assert list(table.index) == ['A', 'B', 'All']
    assert table.loc['A', 'aal_base'] == 10.0 and table.loc['A', 'aal_scenario'] == 8.0
    assert table.loc['B', 'aal_change'] == -4.0
    assert table.loc['All', 'aal_change'] == -6.0
    assert table.loc['A', 'aal_change_pct'] == pytest.approx(-0.2)
    assert table.loc['B', 'aal_change_pct'] == pytest.approx(-0.8)
    # band A carried 10/15 of the base loss but got 2/6 of the reduction
    assert table.loc['A', 'aal_share_base'] == pytest.approx(10 / 15)
    assert table.loc['A', 'share_of_aal_change'] == pytest.approx(2 / 6)
    assert table.loc['B', 'share_of_aal_change'] == pytest.approx(4 / 6)
    assert table.loc['All', 'share_of_aal_change'] == pytest.approx(1.0)
    assert table.loc['A', 'aal_to_tiv_base'] == pytest.approx(10 / 300)
    assert table.loc['B', 'aal_to_tiv_scenario'] == pytest.approx(1 / 600)
    assert not table['suppressed'].any()
    assert 'median_aal_to_income_base' not in table


def test_compare_table_with_income(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL)
    scenario = write_run(tmp_path / 'scenario', SCENARIO_AAL)

    table = by_band(compare_table(rows(base, income_column='Income'), rows(scenario, income_column='Income'),
                                  checked(income_column='Income')))

    assert table.loc['A', 'median_aal_to_income_base'] == pytest.approx(3 / 20_000)
    assert table.loc['A', 'median_aal_to_income_scenario'] == pytest.approx(3 / 20_000)
    assert table.loc['B', 'median_aal_to_income_scenario'] == pytest.approx(0.5 / 50_000)


def test_groups_in_only_one_run_and_different_summary_ids(tmp_path):
    # the scenario has bought out band B and renumbered the summary ids
    base = write_run(tmp_path / 'base', BASE_AAL)
    info = pd.DataFrame({'summary_id': [7, 8], 'Band': ['A', 'C'], 'Income': [1, 1], 'tiv': [100.0, 50.0]})
    scenario = write_run(tmp_path / 'scenario', {7: 2.0, 8: 1.0}, info=info)

    table = by_band(compare_table(rows(base), rows(scenario), checked()))

    assert list(table.index) == ['A', 'B', 'C', 'All']
    assert table.loc['B', 'rows_scenario'] == 0 and table.loc['B', 'aal_scenario'] == 0.0
    assert table.loc['C', 'rows_base'] == 0 and np.isnan(table.loc['C', 'aal_change_pct'])
    assert table.loc['C', 'aal_change'] == 1.0


def test_group_values_match_across_csv_and_parquet_types(tmp_path):
    info = INFO.assign(Band=[1, 1, 1, 2, 2, 2])
    base = write_run(tmp_path / 'base', BASE_AAL, info=info)
    scenario_dir = tmp_path / 'scenario'
    write_run(scenario_dir, SCENARIO_AAL, info=info)
    (scenario_dir / 'gul_S1_summary-info.csv').unlink()
    info.assign(Band=info['Band'].astype(float)).to_parquet(scenario_dir / 'gul_S1_summary-info.parquet')

    table = by_band(compare_table(rows(base), rows(str(scenario_dir)), checked()))

    assert list(table.index) == ['1', '2', 'All']
    assert table.loc['1', 'aal_scenario'] == 8.0


def test_small_groups_in_either_run_are_suppressed(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL)
    info = INFO.assign(Band=['A', 'A', 'A', 'B', 'B', 'D'])
    scenario = write_run(tmp_path / 'scenario', SCENARIO_AAL, info=info)

    table = by_band(compare_table(rows(base), rows(scenario), checked(min_group_size=3)))

    assert table.loc['A', 'suppressed'] == False  # noqa: E712
    assert table.loc['B', 'suppressed'] and np.isnan(table.loc['B', 'aal_base'])
    assert table.loc['B', 'rows_base'] == 3 and table.loc['B', 'rows_scenario'] == 2
    assert table.loc['D', 'suppressed']
    assert table.loc['All', 'suppressed'] == False  # noqa: E712


def test_no_change_gives_blank_shares(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL)
    scenario = write_run(tmp_path / 'scenario', BASE_AAL)

    table = by_band(compare_table(rows(base), rows(scenario), checked()))

    assert (table['aal_change'] == 0).all() and table['share_of_aal_change'].isna().all()


def _ept(scale):
    return pd.DataFrame([{'SummaryId': s, 'EPCalc': 1, 'EPType': t, 'ReturnPeriod': rp, 'Loss': scale * s * rp}
                         for s in (1, 2) for t in (1, 3) for rp in (10, 100)])


def test_compare_ep_table(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL, ept=_ept(1.0))
    scenario = write_run(tmp_path / 'scenario', SCENARIO_AAL, ept=_ept(0.5))

    table = compare_ep_table(base, scenario, checked())

    assert list(table['ReturnPeriod']) == [100, 10, 100, 10]
    row = table[(table['EPType'] == 1) & (table['ReturnPeriod'] == 100)].iloc[0]
    assert row['loss_base'] == 300.0 and row['loss_scenario'] == 150.0
    assert row['loss_change'] == -150.0 and row['loss_change_pct'] == pytest.approx(-0.5)


def test_compare_ep_table_needs_both_epts(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL, ept=_ept(1.0))
    scenario = write_run(tmp_path / 'scenario', SCENARIO_AAL)

    assert compare_ep_table(base, scenario, checked()) is None


def test_run_scenario_comparison_writes_the_tables(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL, ept=_ept(1.0))
    scenario = write_run(tmp_path / 'scenario', SCENARIO_AAL, ept=_ept(0.5))
    report = tmp_path / 'report'

    tables = run_scenario_comparison(base, scenario, settings(), str(report))

    assert sorted(tables) == ['gul_S1_compare-aal.csv', 'gul_S1_compare-ep.csv']
    written = pd.read_csv(report / 'gul_S1_compare-aal.csv')
    assert list(written['Band']) == ['A', 'B', 'All']
    used = json.loads((report / 'gul_S1_compare.json').read_text())
    assert used['sample_type'] == 2 and used['base_output_dir'] == base


def test_run_scenario_comparison_needs_settings_and_matching_sample_types(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL)
    scenario = write_run(tmp_path / 'scenario', SCENARIO_AAL, sample_types=(1,))
    with pytest.raises(OasisException, match="no 'equity_report' section"):
        run_scenario_comparison(base, scenario, {}, str(tmp_path / 'r'))
    with pytest.raises(OasisException, match='sample type 2 and the scenario.s is 1'):
        run_scenario_comparison(base, scenario, settings(), str(tmp_path / 'r'))
    assert run_scenario_comparison(base, scenario, settings(sample_type=1), str(tmp_path / 'r'))


def test_run_scenario_comparison_needs_the_group_columns_in_both_runs(tmp_path):
    base = write_run(tmp_path / 'base', BASE_AAL)
    scenario = write_run(tmp_path / 'scenario', SCENARIO_AAL, info=INFO.drop(columns='Band'))
    with pytest.raises(OasisException, match=r"columns \['Band'\] are not in gul_S1_summary-info"):
        run_scenario_comparison(base, scenario, settings(), str(tmp_path / 'r'))


def test_step_uses_the_scenarios_saved_settings(tmp_path):
    write_run(tmp_path / 'base' / 'output', BASE_AAL)
    write_run(tmp_path / 'scenario' / 'output', SCENARIO_AAL)
    (tmp_path / 'scenario' / 'input').mkdir()
    run_settings = {'model_supplier_id': 'S', 'model_name_id': 'M', 'model_settings': {}, 'gul_output': True,
                    'gul_summaries': [{'id': 1}], **settings()}
    (tmp_path / 'scenario' / 'input' / 'analysis_settings.json').write_text(json.dumps(run_settings))

    tables = CompareScenarios(model_run_dir=str(tmp_path / 'scenario'), compare_base_run_dir=str(tmp_path / 'base')).run()

    assert 'gul_S1_compare-aal.csv' in tables
    assert (tmp_path / 'scenario' / 'output' / 'compare' / 'gul_S1_compare-aal.csv').is_file()


def test_step_needs_settings(tmp_path):
    (tmp_path / 'base').mkdir()
    (tmp_path / 'scenario').mkdir()
    with pytest.raises(OasisException, match='no analysis settings given'):
        CompareScenarios(model_run_dir=str(tmp_path / 'scenario'), compare_base_run_dir=str(tmp_path / 'base')).run()


def test_manager_exposes_the_step():
    assert hasattr(OasisManager(), 'compare_scenarios')
