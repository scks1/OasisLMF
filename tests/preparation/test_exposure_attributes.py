import logging

import pandas as pd
import pytest

from oasislmf.preparation.exposure_attributes import attach_exposure_attributes, exposure_attribute_tables
from oasislmf.utils.exceptions import OasisException


@pytest.fixture
def locations():
    return pd.DataFrame({
        'loc_id': [1, 2, 3, 4],
        'PostalCode': pd.Series(['AB1 1AA', 'AB1 1AB', ' AB1 1AA', 'ZZ9 9ZZ'], dtype='category'),
        'AreaCode': [10, 20, 10, 30],
        'BuildingTIV': [100.0, 200.0, 300.0, 400.0],
    }).set_axis([5, 6, 7, 8])


def write(tmp_path, name, text):
    path = tmp_path / name
    path.write_text(text)
    return str(path)


def table(file, **kwargs):
    return exposure_attribute_tables({'exposure_attributes': [{'file': file, **kwargs}]})


def test_settings_none_means_no_tables():
    assert exposure_attribute_tables({}) == []
    assert exposure_attribute_tables(None) == []


def test_settings_single_table_and_string_fields_are_normalised():
    assert exposure_attribute_tables({'exposure_attributes': {'file': 'a.csv', 'join_on': 'PostalCode', 'columns': 'Band'}}) == [
        {'file': 'a.csv', 'join_on': ['PostalCode'], 'columns': ['Band'], 'default': {}}]


@pytest.mark.parametrize('bad, message', [
    ({'join_on': ['PostalCode']}, "needs a 'file'"),
    ({'file': 'a.csv'}, "needs 'join_on'"),
    ({'file': 'a.csv', 'join_on': ['PostalCode'], 'colums': ['Band']}, "unknown keys \\['colums'\\]"),
    ('a.csv', 'must be an object'),
])
def test_settings_errors(bad, message):
    with pytest.raises(OasisException, match=message):
        exposure_attribute_tables({'exposure_attributes': [bad]})


def test_no_tables_returns_the_locations_unchanged(locations):
    assert attach_exposure_attributes(locations, []) is locations


def test_join_on_text_field(tmp_path, locations, caplog):
    path = write(tmp_path, 'pc.csv', "PostalCode,IncomeQuintile,SVI\nAB1 1AA,1,0.9\nAB1 1AB ,5,0.1\n")
    original = locations.copy()
    with caplog.at_level(logging.WARNING):
        result = attach_exposure_attributes(locations, table(path, join_on=['PostalCode']))

    # surrounding spaces are ignored on both sides; rows and index are kept
    assert list(result.index) == [5, 6, 7, 8]
    assert result['IncomeQuintile'].tolist()[:3] == ['1', '5', '1']
    assert pd.isna(result['IncomeQuintile'].iloc[3])
    assert result['SVI'].tolist()[:3] == ['0.9', '0.1', '0.9']
    assert "1 of 4 locations have no row in 'pc.csv'" in caplog.text
    pd.testing.assert_frame_equal(locations, original)


def test_columns_selects_and_default_fills_unmatched(tmp_path, locations):
    path = write(tmp_path, 'pc.csv', "PostalCode,IncomeQuintile,SVI\nAB1 1AA,1,0.9\nAB1 1AB,5,0.1\n")
    result = attach_exposure_attributes(locations, table(path, join_on='PostalCode', columns=['IncomeQuintile'],
                                                         default={'IncomeQuintile': 'Unknown'}))
    assert 'SVI' not in result.columns
    assert result['IncomeQuintile'].tolist() == ['1', '5', '1', 'Unknown']


def test_join_on_numeric_field(tmp_path, locations):
    path = write(tmp_path, 'areas.csv', "AreaCode,Band\n10,low\n20.0,high\n")
    result = attach_exposure_attributes(locations, table(path, join_on=['AreaCode']))
    assert result['Band'].tolist()[:3] == ['low', 'high', 'low']


def test_numeric_field_with_text_key_raises(tmp_path, locations):
    path = write(tmp_path, 'areas.csv', "AreaCode,Band\n10,low\nten,high\n")
    with pytest.raises(OasisException, match="'ten' of 'AreaCode' is not a number"):
        attach_exposure_attributes(locations, table(path, join_on=['AreaCode']))


def test_join_on_several_fields_and_parquet(tmp_path, locations):
    path = str(tmp_path / 'areas.parquet')
    pd.DataFrame({'PostalCode': ['AB1 1AA', 'AB1 1AA'], 'AreaCode': [10, 99], 'Band': [3, 4]}).to_parquet(path)
    result = attach_exposure_attributes(locations, table(path, join_on=['PostalCode', 'AreaCode']))
    assert result['Band'].tolist()[0] == 3 and result['Band'].tolist()[2] == 3
    assert result['Band'].isna().tolist() == [False, True, False, True]


def test_duplicate_join_keys_raise(tmp_path, locations):
    path = write(tmp_path, 'pc.csv', "PostalCode,Band\nAB1 1AA,1\nAB1 1AA,2\n")
    with pytest.raises(OasisException, match="2 rows of 'pc.csv' share their join_on values, e.g. {'PostalCode': 'AB1 1AA'}"):
        attach_exposure_attributes(locations, table(path, join_on=['PostalCode']))


def test_blank_key_rows_do_not_match_blank_locations(tmp_path, locations):
    locations['PostalCode'] = pd.Series([None, 'AB1 1AB', None, None], index=locations.index, dtype='category')
    path = write(tmp_path, 'pc.csv', "PostalCode,Band\n,blank\nAB1 1AB,2\n")
    result = attach_exposure_attributes(locations, table(path, join_on=['PostalCode']))
    assert result['Band'].isna().tolist() == [True, False, True, True]


def test_missing_join_field_suggests_spelling(tmp_path, locations):
    path = write(tmp_path, 'pc.csv', "postalcode,Band\nAB1 1AA,1\n")
    with pytest.raises(OasisException, match=r"join_on fields \['PostalCode'\] are not in 'pc.csv' \('PostalCode' is spelt \['postalcode'\]\)"):
        attach_exposure_attributes(locations, table(path, join_on=['PostalCode']))


def test_join_field_missing_from_locations(tmp_path, locations):
    path = write(tmp_path, 'pc.csv', "Ward,Band\nW1,1\n")
    with pytest.raises(OasisException, match=r"join_on fields \['Ward'\] are not in the locations"):
        attach_exposure_attributes(locations, table(path, join_on=['Ward']))


def test_column_that_is_already_a_location_field_raises(tmp_path, locations):
    path = write(tmp_path, 'pc.csv', "PostalCode,BuildingTIV\nAB1 1AA,1\n")
    with pytest.raises(OasisException, match=r"columns \['BuildingTIV'\] of 'pc.csv' are already location fields"):
        attach_exposure_attributes(locations, table(path, join_on=['PostalCode']))


def test_unknown_column_and_default_raise(tmp_path, locations):
    path = write(tmp_path, 'pc.csv', "PostalCode,Band\nAB1 1AA,1\n")
    with pytest.raises(OasisException, match=r"columns \['Quintile'\] are not in 'pc.csv'"):
        attach_exposure_attributes(locations, table(path, join_on=['PostalCode'], columns=['Quintile']))
    with pytest.raises(OasisException, match=r"'default' names \['Quintile'\]"):
        attach_exposure_attributes(locations, table(path, join_on=['PostalCode'], default={'Quintile': 0}))


def test_relative_file_is_found_in_search_dirs(tmp_path, locations):
    (tmp_path / 'data').mkdir()
    write(tmp_path / 'data', 'pc.csv', "PostalCode,Band\nAB1 1AA,1\n")
    result = attach_exposure_attributes(locations, table('pc.csv', join_on=['PostalCode']),
                                        search_dirs=[None, str(tmp_path / 'nothing'), str(tmp_path / 'data')])
    assert result['Band'].iloc[0] == '1'
    with pytest.raises(OasisException, match="exposure attribute table 'other.csv' not found; looked in: .*nothing"):
        attach_exposure_attributes(locations, table('other.csv', join_on=['PostalCode']), search_dirs=[str(tmp_path / 'nothing')])


def test_several_tables(tmp_path, locations):
    a = write(tmp_path, 'pc.csv', "PostalCode,Band\nAB1 1AA,1\n")
    b = write(tmp_path, 'areas.csv', "AreaCode,Region\n30,north\n")
    tables = exposure_attribute_tables({'exposure_attributes': [{'file': a, 'join_on': 'PostalCode'}, {'file': b, 'join_on': 'AreaCode'}]})
    result = attach_exposure_attributes(locations, tables)
    assert result['Band'].tolist()[0] == '1'
    assert result['Region'].tolist()[3] == 'north'
