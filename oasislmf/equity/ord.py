"""Find and read the ORD output files of one summary set."""

__all__ = ['ord_path', 'read_ord']

import os

import pandas as pd

from ..utils.exceptions import OasisException

EXTENSIONS = ('csv', 'parquet')


def ord_path(output_dir, runtype, summary_set, table):
    """Return the path of an ORD output file, or None if it doesn't exist.

    Args:
        output_dir (str): Analysis output directory.
        runtype (str): Perspective, e.g. ``gul`` or ``il``.
        summary_set (int): Summary set id.
        table (str): ORD table name as used in the file name, e.g. ``palt``, ``mplt`` or ``summary-info``.

    Returns:
        str or None: The csv file if there is one, else the parquet file, else None.
    """
    for ext in EXTENSIONS:
        path = os.path.join(output_dir, f'{runtype}_S{summary_set}_{table}.{ext}')
        if os.path.isfile(path):
            return path
    return None


def read_ord(output_dir, runtype, summary_set, table, needed_for=None):
    """Read an ORD output file into a dataframe.

    Args:
        output_dir (str): Analysis output directory.
        runtype (str): Perspective, e.g. ``gul`` or ``il``.
        summary_set (int): Summary set id.
        table (str): ORD table name as used in the file name.
        needed_for (str): What the file is needed for. If given, a missing file raises an error that says so;
            otherwise a missing file returns None.

    Returns:
        pandas.DataFrame or None
    """
    path = ord_path(output_dir, runtype, summary_set, table)
    if path is None:
        if needed_for is None:
            return None
        raise OasisException(
            f"{needed_for} needs {runtype}_S{summary_set}_{table}.csv (or .parquet) in '{output_dir}'; "
            f"turn on the matching ord_output option for {runtype} summary set {summary_set}")
    if path.endswith('.parquet'):
        return pd.read_parquet(path)
    return pd.read_csv(path)
