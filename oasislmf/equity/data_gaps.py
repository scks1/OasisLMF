"""Data-gap audit: how much of each group's exposure the model could not see, or saw through defaults.

Communities with poor address data or informal housing tend to fail the lookup, fall back to a default
vulnerability or carry low quality geocodes. Their losses are then left out of, or blurred in, every modelled number
without any warning. This table shows, per group, how often that happened and how much value it touched.
"""

__all__ = ['data_gap_table', 'read_run_locations', 'GAP_STATUSES']

import logging
import os

import numpy as np
import pandas as pd
from ods_tools.oed import OedExposure

from ..preparation.exposure_attributes import attach_exposure_attributes, exposure_attribute_tables
from ..utils.data import get_dataframe, prepare_oed_exposure
from ..utils.defaults import SAR_ID, get_default_exposure_profile
from ..utils.exceptions import OasisException
from ..utils.status import OASIS_KEYS_STATUS
from .rows import by_group, suppress_small_groups

logger = logging.getLogger(__name__)

#: Key statuses that mean the model could not place a location, as opposed to having placed it and found it not at
#: risk ('notatrisk') or not covering that peril or coverage at all ('notmodelled').
GAP_STATUSES = tuple(OASIS_KEYS_STATUS[s]['id'] for s in ('fail', 'nomatch', 'fail_ap', 'fail_v', 'noreturn'))
EXPOSURE_CONFIG = OedExposure.DEFAULT_EXPOSURE_CONFIG_NAME
KEYS_FILES = ('keys.parquet', 'keys.csv')
KEYS_ERRORS_FILES = ('keys-errors.parquet', 'keys-errors.csv')


def read_run_locations(input_dir, analysis_settings, search_dirs=()):
    """Return the locations of a run, with their ``loc_id`` and any exposure attributes attached.

    Args:
        input_dir (str): The run's input directory, holding the exposure saved by ``generate-oasis-files``.
        analysis_settings (dict): Analysis settings, for their ``exposure_attributes``.
        search_dirs (list): Directories to look for relative attribute table paths in.

    Returns:
        pandas.DataFrame: One row per location.

    Raises:
        OasisException: If the run has no saved exposure or no location file.
    """
    config = os.path.join(input_dir, EXPOSURE_CONFIG)
    if not os.path.isfile(config):
        raise OasisException(f"data-gap audit: no '{EXPOSURE_CONFIG}' in '{input_dir}'; it is written by generate-oasis-files")
    exposure = OedExposure.from_config(config)
    if exposure.location is None:
        raise OasisException(f"data-gap audit: the exposure in '{input_dir}' has no location file")
    prepare_oed_exposure(exposure)
    tables = exposure_attribute_tables(analysis_settings)
    return attach_exposure_attributes(exposure.location.dataframe, tables, search_dirs)


def _read_keys(input_dir, names, columns):
    for name in names:
        path = os.path.join(input_dir, name)
        if os.path.isfile(path):
            try:
                keys = get_dataframe(src_fp=path, lowercase_cols=False)
            except OasisException:  # an empty keys file
                return pd.DataFrame(columns=columns)
            missing = [c for c in columns if c not in keys.columns]
            if missing:
                raise OasisException(f"data-gap audit: '{path}' has no columns {missing}")
            return keys[columns]
    return None


def _coverage_values(locations):
    """Return one row per location and coverage with a positive value."""
    fields = {term['CoverageTypeID']: term['ProfileElementName'] for term in get_default_exposure_profile().values()
              if term.get('FMTermType') == 'TIV' and term['ProfileElementName'] in locations.columns}
    parts = [pd.DataFrame({SAR_ID: locations[SAR_ID].to_numpy(),
                           'coverage_type_id': coverage,
                           'tiv': pd.to_numeric(locations[field], errors='coerce').fillna(0).to_numpy()})
             for coverage, field in fields.items()]
    if not parts:
        raise OasisException('data-gap audit: the locations have no TIV columns')
    values = pd.concat(parts, ignore_index=True)
    return values[values['tiv'] > 0]


def _location_gaps(locations, keys, errors, default_vulnerability_ids):
    """Return, per location: whether any key failed, and the value with no key or only default vulnerabilities."""
    values = _coverage_values(locations)
    keys = keys.rename(columns={'LocID': SAR_ID, 'CoverageTypeID': 'coverage_type_id'})
    errors = errors.rename(columns={'LocID': SAR_ID, 'CoverageTypeID': 'coverage_type_id'})
    ids = [SAR_ID, 'coverage_type_id']
    for frame in (keys, errors):
        for c in ids:
            frame[c] = pd.to_numeric(frame[c], errors='coerce')

    keys['default_vulnerability'] = pd.to_numeric(keys['VulnerabilityID'], errors='coerce').isin(default_vulnerability_ids)
    keyed = keys.groupby(ids)['default_vulnerability'].all().rename('all_default').reset_index()
    errors = errors[errors['Status'].isin(GAP_STATUSES)]
    failed = errors[ids].drop_duplicates().assign(failed=True)

    coverages = values.merge(keyed, on=ids, how='left').merge(failed, on=ids, how='left')
    keyed_any = coverages['all_default'].notna()
    coverages['failed'] = coverages['failed'].astype('boolean').fillna(False).astype(bool)
    # a coverage is lost when it failed for every peril it was tried for: no peril returned a key
    coverages['tiv_unmodelled'] = coverages['tiv'].where(coverages['failed'] & ~keyed_any, 0.0)
    all_default = coverages['all_default'].astype('boolean').fillna(False).astype(bool)
    coverages['tiv_default_vulnerability'] = coverages['tiv'].where(all_default, 0.0)

    per_location = coverages.groupby(SAR_ID).agg(
        tiv=('tiv', 'sum'), tiv_unmodelled=('tiv_unmodelled', 'sum'),
        tiv_default_vulnerability=('tiv_default_vulnerability', 'sum'))
    failed_locations = set(errors[SAR_ID].dropna())
    default_locations = set(keys.loc[keys['default_vulnerability'], SAR_ID].dropna())
    out = pd.DataFrame({SAR_ID: locations[SAR_ID].to_numpy()})
    out = out.merge(per_location, left_on=SAR_ID, right_index=True, how='left').fillna(
        {'tiv': 0.0, 'tiv_unmodelled': 0.0, 'tiv_default_vulnerability': 0.0})
    out['failed_lookup'] = out[SAR_ID].isin(failed_locations).to_numpy()
    out['default_vulnerability'] = out[SAR_ID].isin(default_locations).to_numpy()
    return out


def _geocode(locations, threshold):
    out = pd.DataFrame(index=locations.index)
    if 'Latitude' in locations and 'Longitude' in locations:
        lat = pd.to_numeric(locations['Latitude'], errors='coerce')
        lon = pd.to_numeric(locations['Longitude'], errors='coerce')
        out['no_coordinates'] = (lat.isna() | lon.isna() | ((lat == 0) & (lon == 0))).to_numpy()
    if 'GeocodeQuality' in locations:
        quality = pd.to_numeric(locations['GeocodeQuality'], errors='coerce')
        out['geocode_quality'] = quality.to_numpy()
        if threshold is not None:
            out['low_geocode_quality'] = (quality.isna() | (quality < threshold)).to_numpy()
    return out


def _share(part, whole):
    return part / whole if whole > 0 else np.nan


def data_gap_table(input_dir, analysis_settings, settings, search_dirs=()):
    """Return the data-gap audit, one row per group plus a total row.

    The value is the location's site coverage TIV. A coverage counts as unmodelled when the lookup failed for it
    (status fail, nomatch, fail_ap, fail_v or noreturn) and returned no key for any peril. Coverages the model does not
    cover, and keys that say the location is not at risk, are not gaps.

    Columns, after the group columns:

    - ``rows``: locations; ``tiv``.
    - ``locations_failed_lookup`` and ``share_locations_failed_lookup``: locations where any key failed.
    - ``tiv_unmodelled`` and ``share_tiv_unmodelled``.
    - With ``default_vulnerability_ids``: ``locations_default_vulnerability`` (any key on a default id) and
      ``share_tiv_default_vulnerability`` (value whose every key is on a default id).
    - With ``Latitude`` and ``Longitude``: ``share_locations_no_coordinates`` (blank, or both 0).
    - With ``GeocodeQuality``: ``mean_geocode_quality``, ``share_geocode_quality_missing``, and with
      ``geocode_quality_threshold``, ``share_low_geocode_quality`` (below the threshold, or missing).
    - ``suppressed``: see :func:`oasislmf.equity.rows.suppress_small_groups`.

    Args:
        input_dir (str): The run's input directory.
        analysis_settings (dict): Analysis settings, for their ``exposure_attributes``.
        settings (dict): Checked ``equity_report`` settings, with a ``data_gaps`` section.
        search_dirs (list): Directories to look for relative attribute table paths in.

    Returns:
        pandas.DataFrame: The audit.

    Raises:
        OasisException: If the run's exposure or keys are missing, or a group column is not a location column.
    """
    gaps = settings['data_gaps']
    group_by = settings['group_by']
    locations = read_run_locations(input_dir, analysis_settings, search_dirs)
    missing = [c for c in group_by if c not in locations.columns]
    if missing:
        raise OasisException(f"data-gap audit: group_by columns {missing} are not location columns; add them with "
                             f"exposure_attributes, or leave data_gaps out")

    keys = _read_keys(input_dir, KEYS_FILES, ['LocID', 'CoverageTypeID', 'VulnerabilityID'])
    if keys is None:
        raise OasisException(f"data-gap audit: no keys file ({' or '.join(KEYS_FILES)}) in '{input_dir}'")
    errors = _read_keys(input_dir, KEYS_ERRORS_FILES, ['LocID', 'CoverageTypeID', 'Status'])
    if errors is None:
        logger.warning("data-gap audit: no keys errors file in '%s'; counting no lookup failures", input_dir)
        errors = pd.DataFrame(columns=['LocID', 'CoverageTypeID', 'Status'])

    default_ids = gaps['default_vulnerability_ids']
    per_location = _location_gaps(locations, keys, errors, default_ids)
    rows = pd.concat([locations[group_by].reset_index(drop=True),
                      per_location.reset_index(drop=True),
                      _geocode(locations, gaps['geocode_quality_threshold']).reset_index(drop=True)], axis=1)

    def aggregate(group):
        tiv = group['tiv'].sum()
        result = {
            'rows': len(group),
            'tiv': tiv,
            'locations_failed_lookup': int(group['failed_lookup'].sum()),
            'share_locations_failed_lookup': group['failed_lookup'].mean(),
            'tiv_unmodelled': group['tiv_unmodelled'].sum(),
            'share_tiv_unmodelled': _share(group['tiv_unmodelled'].sum(), tiv),
        }
        if default_ids:
            result['locations_default_vulnerability'] = int(group['default_vulnerability'].sum())
            result['share_tiv_default_vulnerability'] = _share(group['tiv_default_vulnerability'].sum(), tiv)
        if 'no_coordinates' in group:
            result['share_locations_no_coordinates'] = group['no_coordinates'].mean()
        if 'geocode_quality' in group:
            result['mean_geocode_quality'] = group['geocode_quality'].mean()
            result['share_geocode_quality_missing'] = group['geocode_quality'].isna().mean()
        if 'low_geocode_quality' in group:
            result['share_low_geocode_quality'] = group['low_geocode_quality'].mean()
        return result

    return suppress_small_groups(by_group(rows, group_by, aggregate), group_by, settings['min_group_size'])
