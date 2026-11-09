import json

import pandas as pd
import pytest
from ods_tools.oed import OedExposure

from oasislmf.computation.run.equity import EquityReport
from oasislmf.equity import equity_report_settings, run_equity_report
from oasislmf.equity.data_gaps import data_gap_table
from oasislmf.utils.exceptions import OasisException

# Four locations in two postcodes. loc_id follows the file order: L1 = 1 ... L4 = 4.
LOCATIONS = pd.DataFrame({
    'PortNumber': 1, 'AccNumber': 1, 'LocNumber': ['L1', 'L2', 'L3', 'L4'], 'CountryCode': 'GB',
    'LocPerilsCovered': 'WTC', 'LocCurrency': 'GBP', 'PostalCode': ['A', 'A', 'B', 'B'],
    'BuildingTIV': [100.0, 200.0, 300.0, 400.0], 'ContentsTIV': [10.0, 0.0, 30.0, 40.0],
    'Latitude': [51.0, 0.0, 52.0, None], 'Longitude': [0.1, 0.0, 0.2, 0.3],
    'GeocodeQuality': [0.9, 0.2, None, 0.8],
})
# L1 is keyed; L2's building has no match; L3 is keyed for wind but its contents failed for flood;
# L4 is keyed, on the model's default vulnerability 99.
KEYS = pd.DataFrame({'LocID': [1, 1, 3, 3, 4, 4], 'PerilID': 'WTC', 'CoverageTypeID': [1, 3, 1, 3, 1, 3],
                     'AreaPerilID': 1, 'VulnerabilityID': [5, 5, 5, 5, 99, 99]})
KEYS_ERRORS = pd.DataFrame({'LocID': [2, 3], 'PerilID': ['WTC', 'ORF'], 'CoverageTypeID': [1, 3],
                            'Status': ['nomatch', 'fail'], 'Message': ['no match', 'outside the model']})


def write_inputs(input_dir, keys_format='csv', errors=KEYS_ERRORS, locations=LOCATIONS):
    input_dir.mkdir(parents=True, exist_ok=True)
    OedExposure(location=locations.copy(), check_oed=False).save(str(input_dir), save_config=True)
    for name, frame in (('keys', KEYS), ('keys-errors', errors)):
        if frame is not None:
            getattr(frame, f'to_{keys_format}')(input_dir / f'keys{name[4:]}.{keys_format}', index=False)
    return str(input_dir)


def checked(**data_gaps):
    return equity_report_settings({'equity_report': {'group_by': 'PostalCode', 'min_group_size': 0,
                                                     'data_gaps': data_gaps or True}})


def by_postcode(table):
    return table.set_index('PostalCode')


@pytest.mark.parametrize('keys_format', ['csv', 'parquet'])
def test_lookup_failures_and_unmodelled_value(tmp_path, keys_format):
    table = by_postcode(data_gap_table(write_inputs(tmp_path / 'input', keys_format), {}, checked()))
    assert table['rows'].to_dict() == {'A': 2, 'B': 2, 'All': 4}
    assert table['tiv'].to_dict() == {'A': 310.0, 'B': 770.0, 'All': 1080.0}
    assert table['locations_failed_lookup'].to_dict() == {'A': 1, 'B': 1, 'All': 2}
    # L3's contents failed for flood but have a wind key, so none of B's value is lost
    assert table['tiv_unmodelled'].to_dict() == {'A': 200.0, 'B': 0.0, 'All': 200.0}
    assert table.loc['A', 'share_tiv_unmodelled'] == pytest.approx(200 / 310)
    assert 'share_tiv_default_vulnerability' not in table


def test_not_at_risk_and_not_modelled_are_not_gaps(tmp_path):
    errors = KEYS_ERRORS.assign(Status=['notatrisk', 'notmodelled'])
    table = by_postcode(data_gap_table(write_inputs(tmp_path / 'input', errors=errors), {}, checked()))
    assert table['locations_failed_lookup'].sum() == 0
    assert table['tiv_unmodelled'].sum() == 0


def test_default_vulnerabilities(tmp_path):
    table = by_postcode(data_gap_table(write_inputs(tmp_path / 'input'), {}, checked(default_vulnerability_ids=[99])))
    assert table['locations_default_vulnerability'].to_dict() == {'A': 0, 'B': 1, 'All': 1}
    assert table.loc['B', 'share_tiv_default_vulnerability'] == pytest.approx(440 / 770)


def test_geocodes(tmp_path):
    table = by_postcode(data_gap_table(write_inputs(tmp_path / 'input'), {}, checked(geocode_quality_threshold=0.5)))
    # L2 sits at (0, 0) and L4 has no latitude
    assert table['share_locations_no_coordinates'].to_dict() == {'A': 0.5, 'B': 0.5, 'All': 0.5}
    assert table.loc['A', 'mean_geocode_quality'] == pytest.approx(0.55)
    assert table['share_geocode_quality_missing'].to_dict() == {'A': 0.0, 'B': 0.5, 'All': 0.25}
    # below the threshold (L2) or missing (L3)
    assert table['share_low_geocode_quality'].to_dict() == {'A': 0.5, 'B': 0.5, 'All': 0.5}


def test_geocode_columns_are_left_out_without_the_fields(tmp_path):
    locations = LOCATIONS.drop(columns=['Latitude', 'Longitude', 'GeocodeQuality'])
    table = data_gap_table(write_inputs(tmp_path / 'input', locations=locations), {}, checked(geocode_quality_threshold=0.5))
    assert not {'share_locations_no_coordinates', 'mean_geocode_quality', 'share_low_geocode_quality'} & set(table)


def test_small_groups_are_suppressed(tmp_path):
    report = equity_report_settings({'equity_report': {'group_by': 'PostalCode', 'min_group_size': 3, 'data_gaps': True}})
    table = by_postcode(data_gap_table(write_inputs(tmp_path / 'input'), {}, report))
    assert table['suppressed'].to_dict() == {'A': True, 'B': True, 'All': False}
    assert table.loc[['A', 'B'], 'tiv_unmodelled'].isna().all()
    assert table.loc['All', 'tiv_unmodelled'] == 200.0


def test_groups_from_an_attribute_table(tmp_path):
    (tmp_path / 'data').mkdir()
    pd.DataFrame({'PostalCode': ['A', 'B'], 'IncomeQuintile': [1, 5]}).to_csv(tmp_path / 'data' / 'areas.csv', index=False)
    analysis_settings = {'exposure_attributes': [{'file': 'areas.csv', 'join_on': 'PostalCode', 'columns': ['IncomeQuintile']}]}
    report = equity_report_settings({'equity_report': {'group_by': 'IncomeQuintile', 'min_group_size': 0, 'data_gaps': True}})
    table = data_gap_table(write_inputs(tmp_path / 'input'), analysis_settings, report, [str(tmp_path / 'data')])
    # attribute table values are read as text
    assert table.set_index('IncomeQuintile')['tiv_unmodelled'].to_dict() == {'1': 200.0, '5': 0.0, 'All': 200.0}


def test_no_keys_errors_file_counts_no_failures(tmp_path, caplog):
    table = by_postcode(data_gap_table(write_inputs(tmp_path / 'input', errors=None), {}, checked()))
    assert table['locations_failed_lookup'].sum() == 0
    assert 'no keys errors file' in caplog.text


@pytest.mark.parametrize('remove, message', [
    ('exposure_info.json', "no 'exposure_info.json'"),
    ('keys.csv', 'no keys file'),
])
def test_missing_inputs(tmp_path, remove, message):
    input_dir = write_inputs(tmp_path / 'input')
    (tmp_path / 'input' / remove).unlink()
    with pytest.raises(OasisException, match=message):
        data_gap_table(input_dir, {}, checked())


def test_group_column_must_be_a_location_column(tmp_path):
    report = equity_report_settings({'equity_report': {'group_by': 'AccNumberGroup', 'data_gaps': True}})
    with pytest.raises(OasisException, match='not location columns'):
        data_gap_table(write_inputs(tmp_path / 'input'), {}, report)


@pytest.mark.parametrize('data_gaps, message', [
    ({'default_vulnerability_ids': 99}, 'list of vulnerability ids'),
    ({'geocode_quality_threshold': -1}, 'geocode_quality_threshold'),
    ({'precision': 1}, 'unknown keys'),
])
def test_bad_settings(data_gaps, message):
    with pytest.raises(OasisException, match=message):
        equity_report_settings({'equity_report': {'group_by': 'PostalCode', 'data_gaps': data_gaps}})


def write_run(run_dir, analysis_settings):
    write_inputs(run_dir / 'input')
    output_dir = run_dir / 'output'
    output_dir.mkdir()
    pd.DataFrame({'summary_id': [1, 2], 'PostalCode': ['A', 'B'], 'Band': ['low', 'high'], 'tiv': [310.0, 770.0]}).to_csv(
        output_dir / 'gul_S1_summary-info.csv', index=False)
    pd.DataFrame({'SummaryId': [1, 2], 'SampleType': 2, 'MeanLoss': [3.0, 1.0], 'SDLoss': 0.0}).to_csv(
        output_dir / 'gul_S1_palt.csv', index=False)
    (run_dir / 'input' / 'analysis_settings.json').write_text(json.dumps(analysis_settings))
    return str(output_dir)


def test_report_writes_the_audit_from_the_run_inputs(tmp_path):
    analysis_settings = {'equity_report': {'group_by': 'PostalCode', 'min_group_size': 0, 'data_gaps': True}}
    output_dir = write_run(tmp_path / 'run', analysis_settings)
    tables = run_equity_report(output_dir, analysis_settings)
    assert 'equity-data-gaps.csv' in tables
    written = pd.read_csv(tmp_path / 'run' / 'output' / 'equity' / 'equity-data-gaps.csv')
    assert written['tiv_unmodelled'].tolist() == [200.0, 0.0, 200.0]


def test_step_finds_attribute_tables_in_the_user_data_dir(tmp_path):
    (tmp_path / 'data').mkdir()
    pd.DataFrame({'PostalCode': ['A', 'B'], 'Band': ['low', 'high']}).to_csv(tmp_path / 'data' / 'areas.csv', index=False)
    write_run(tmp_path / 'run', {
        'model_supplier_id': 'S', 'model_name_id': 'M', 'model_settings': {}, 'gul_output': True,
        'gul_summaries': [{'id': 1, 'oed_fields': ['Band']}],
        'exposure_attributes': [{'file': 'areas.csv', 'join_on': 'PostalCode', 'columns': ['Band']}],
        'equity_report': {'group_by': 'Band', 'min_group_size': 0, 'data_gaps': True},
    })
    tables = EquityReport(model_run_dir=str(tmp_path / 'run'), user_data_dir=str(tmp_path / 'data')).run()
    gaps = tables['equity-data-gaps.csv'].set_index('Band')
    assert gaps['tiv_unmodelled'].to_dict() == {'low': 200.0, 'high': 0.0, 'All': 200.0}
