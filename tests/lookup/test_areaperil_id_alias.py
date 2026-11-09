"""A lookup may name its areaperil column 'areaperil_id', as the model files do (#2216)."""
import pandas as pd
import pytest

from oasislmf.lookup.base import with_area_peril_id
from oasislmf.lookup.builtin import Lookup
from oasislmf.lookup.factory import BasicKeyServer
from oasislmf.utils.status import OASIS_KEYS_STATUS

SUCCESS = OASIS_KEYS_STATUS['success']['id']
FAIL = OASIS_KEYS_STATUS['fail']['id']


def _results(areaperil_column):
    return pd.DataFrame({
        'loc_id': [1, 2],
        'peril_id': ['WTC', 'WTC'],
        'coverage_type': [1, 1],
        areaperil_column: [11, -1],
        'vulnerability_id': [5, -1],
        'status': [SUCCESS, FAIL],
        'message': ['', 'no match'],
    })


def test_with_area_peril_id_renames_alias():
    df = with_area_peril_id(_results('areaperil_id'))
    assert 'area_peril_id' in df.columns and 'areaperil_id' not in df.columns
    assert df['area_peril_id'].tolist() == [11, -1]


def test_with_area_peril_id_leaves_area_peril_id_alone():
    """If both spellings are there, 'area_peril_id' is the one used and nothing is renamed."""
    df = _results('area_peril_id').assign(areaperil_id=[99, 99])
    assert with_area_peril_id(df) is df


def test_with_area_peril_id_without_either_column():
    df = pd.DataFrame({'loc_id': [1]})
    assert with_area_peril_id(df) is df


@pytest.mark.parametrize('output_format, keys_file', [('oasis', 'keys.csv'), ('parquet', 'keys.parquet')])
def test_write_keys_file_accepts_areaperil_id(tmp_path, output_format, keys_file):
    """Results from any lookup (not only the built-in one) can use 'areaperil_id'."""
    key_server = BasicKeyServer({'builtin_lookup_type': 'peril_covered_deterministic'})
    successes_fp = str(tmp_path / keys_file)
    _, successes_count = key_server.write_keys_file(
        [_results('areaperil_id')], successes_fp=successes_fp, errors_fp=None,
        output_format=output_format, keys_success_msg=False)
    written = pd.read_csv(successes_fp) if output_format == 'oasis' else pd.read_parquet(successes_fp)
    assert successes_count == 1
    assert written['AreaPerilID'].tolist() == [11]


def test_builtin_lookup_step_can_set_areaperil_id():
    """A built-in lookup whose steps produce 'areaperil_id' returns it as 'area_peril_id'."""
    config = {
        'strategy': ['areaperil', 'vulnerability'],
        'step_definition': {
            'areaperil': {'type': 'prepare', 'columns': ['peril_id', 'coverage_type'],
                          'parameters': {'areaperil_id': {'default': 11}}},
            'vulnerability': {'type': 'prepare', 'parameters': {'vulnerability_id': {'default': 5}}},
        },
    }
    locations = pd.DataFrame({'loc_id': [1, 2], 'peril_id': ['WTC', 'WTC'], 'coverage_type': [1, 3]})
    keys = Lookup(config=config).process_locations(locations)
    assert 'areaperil_id' not in keys.columns
    assert keys['area_peril_id'].tolist() == [11, 11]
    assert (keys['status'] == SUCCESS).all()
