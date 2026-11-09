"""Join tables of extra per-location attributes onto the exposure so summary sets can group by them.

An analysis can attach one or more tables keyed on location fields, for example a table of postcodes with an
income band or a social vulnerability index. Each table is set in the analysis settings under
``exposure_attributes``::

    "exposure_attributes": [
        {
            "file": "postcode_deprivation.csv",
            "join_on": ["PostalCode"],
            "columns": ["IncomeQuintile", "SVIBand"],
            "default": {"IncomeQuintile": "Unknown"}
        }
    ]

The table's columns are then available to ``oed_fields`` in any summary set, the same as an OED location field,
and are written to that set's summary-info file.
"""

__all__ = [
    'attach_exposure_attributes',
    'exposure_attribute_tables',
    'read_attribute_table',
]

import logging
import os

import pandas as pd

from ..utils.exceptions import OasisException

logger = logging.getLogger(__name__)

SETTINGS_KEY = 'exposure_attributes'
TABLE_KEYS = {'file', 'join_on', 'columns', 'default'}


def exposure_attribute_tables(settings):
    """Return the ``exposure_attributes`` tables from analysis settings as a list of checked dicts.

    Args:
        settings (dict): Analysis settings. A single table may be given as a dict instead of a list.

    Returns:
        list: One dict per table, each with ``file`` (str), ``join_on`` (list), ``columns`` (list or None) and
        ``default`` (dict).

    Raises:
        OasisException: If a table is missing ``file`` or ``join_on``, or has an unknown key.
    """
    tables = (settings or {}).get(SETTINGS_KEY) or []
    if isinstance(tables, dict):
        tables = [tables]
    checked = []
    for i, table in enumerate(tables):
        where = f"{SETTINGS_KEY}[{i}]"
        if not isinstance(table, dict):
            raise OasisException(f"{where} must be an object with 'file' and 'join_on', got {table!r}")
        unknown = set(table) - TABLE_KEYS
        if unknown:
            raise OasisException(f"{where} has unknown keys {sorted(unknown)}; expected some of {sorted(TABLE_KEYS)}")
        if not table.get('file'):
            raise OasisException(f"{where} needs a 'file'")
        join_on = table.get('join_on')
        join_on = [join_on] if isinstance(join_on, str) else list(join_on or [])
        if not join_on:
            raise OasisException(f"{where} needs 'join_on', the location field(s) to match the table on")
        columns = table.get('columns')
        columns = [columns] if isinstance(columns, str) else (list(columns) if columns is not None else None)
        checked.append({'file': table['file'], 'join_on': join_on, 'columns': columns, 'default': dict(table.get('default') or {})})
    return checked


def _resolve(file, search_dirs):
    if os.path.isabs(file):
        candidates = [file]
    else:
        candidates = [os.path.join(d, file) for d in search_dirs if d] + [os.path.abspath(file)]
    for path in candidates:
        if os.path.isfile(path):
            return path
    raise OasisException(f"exposure attribute table '{file}' not found; looked in: {', '.join(candidates)}")


def read_attribute_table(file, search_dirs=()):
    """Read an attribute table from csv or parquet.

    Args:
        file (str): Path to the table. A relative path is looked up in each of ``search_dirs`` in turn, then in the
            working directory.
        search_dirs (iterable): Directories to look for a relative ``file`` in.

    Returns:
        tuple: ``(path, dataframe)``, where ``path`` is the file that was read.
    """
    path = _resolve(file, search_dirs)
    if path.lower().endswith('.parquet'):
        return path, pd.read_parquet(path)
    return path, pd.read_csv(path, dtype=str, keep_default_na=False, na_values=[''])


def _join_key(series, like):
    """Convert a join column of the table to compare equal to the matching location column."""
    if pd.api.types.is_numeric_dtype(like):
        converted = pd.to_numeric(series, errors='coerce')
        bad = series.notna() & converted.isna()
        if bad.any():
            raise OasisException(f"value {series[bad].iloc[0]!r} of '{series.name}' is not a number, "
                                 "but the location field it joins on is numeric")
        return converted
    return series.astype('string').str.strip()


def _as_text(series):
    if pd.api.types.is_numeric_dtype(series):
        return series
    return series.astype('string').str.strip()


def attach_exposure_attributes(location_df, tables, search_dirs=()):
    """Join attribute tables onto the locations.

    Each table is left-joined on its ``join_on`` fields, so every location is kept. A location with no matching row
    gets the table's ``default`` for each column (blank if there is none) and is counted in a warning.

    Args:
        location_df (pandas.DataFrame): Location (subject at risk) dataframe.
        tables (list): Tables as returned by :func:`exposure_attribute_tables`.
        search_dirs (iterable): Directories to look for relative table paths in.

    Returns:
        pandas.DataFrame: ``location_df`` with the tables' columns added. ``location_df`` itself is not modified.

    Raises:
        OasisException: If a join field or column is missing, a column is already a location field, or the table
            has more than one row for the same join key.
    """
    if not tables:
        return location_df

    result = location_df.copy()
    for table in tables:
        path, table_df = read_attribute_table(table['file'], search_dirs)
        name = os.path.basename(path)
        join_on = table['join_on']

        for where, df in (('the locations', result), (f"'{name}'", table_df)):
            missing = [c for c in join_on if c not in df.columns]
            if missing:
                hint = {c: [d for d in df.columns if str(d).lower() == c.lower()] for c in missing}
                hint = '; '.join(f"'{c}' is spelt {spelt}" for c, spelt in hint.items() if spelt)
                raise OasisException(f"exposure attributes: join_on fields {missing} are not in {where}" + (f" ({hint})" if hint else ''))

        columns = table['columns'] if table['columns'] is not None else [c for c in table_df.columns if c not in join_on]
        missing = [c for c in columns if c not in table_df.columns]
        if missing:
            raise OasisException(f"exposure attributes: columns {missing} are not in '{name}'")
        clash = [c for c in columns if c in result.columns]
        if clash:
            raise OasisException(f"exposure attributes: columns {clash} of '{name}' are already location fields; "
                                 "rename them in the table")
        unknown_defaults = set(table['default']) - set(columns)
        if unknown_defaults:
            raise OasisException(f"exposure attributes: 'default' names {sorted(unknown_defaults)}, which are not columns taken from '{name}'")

        keys = [f'__exposure_attribute_key_{i}' for i in range(len(join_on))]
        right = table_df[columns].copy()
        for key, col in zip(keys, join_on):
            right[key] = _join_key(table_df[col], result[col])
        # a row with a blank key can't identify an area, and would otherwise match locations with that field blank
        right = right.dropna(subset=keys)
        duplicated = right.duplicated(subset=keys, keep=False)
        if duplicated.any():
            example = table_df.loc[duplicated[duplicated].index, join_on].iloc[0].to_dict()
            raise OasisException(f"exposure attributes: {int(duplicated.sum())} rows of '{name}' share their join_on "
                                 f"values, e.g. {example}; each location must match at most one row")

        left = pd.DataFrame({key: _as_text(result[col]).reset_index(drop=True) for key, col in zip(keys, join_on)})
        joined = left.merge(right, on=keys, how='left', indicator=True, validate='many_to_one')
        unmatched = joined['_merge'] == 'left_only'
        if unmatched.any():
            logger.warning(
                "exposure attributes: %d of %d locations have no row in '%s' for their %s; %s",
                int(unmatched.sum()), len(joined), name, join_on,
                f"using the default {table['default']}" if table['default'] else 'their attributes are blank')
        for col in columns:
            values = joined[col]
            if col in table['default']:
                values = values.where(~unmatched, table['default'][col])
            result[col] = values.to_numpy()
        logger.info("exposure attributes: added %s from '%s', joined on %s", columns, path, join_on)
    return result
