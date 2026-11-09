"""Distributional (equity) reports on Oasis results.

These reports read the ORD outputs and summary-info of a finished analysis and show how modelled losses fall on
different groups of policyholders, for example by income band or social vulnerability index. They only report: they
never change the losses, the financial terms or any premium used elsewhere.

Group fields usually come from :mod:`oasislmf.preparation.exposure_attributes`, which joins a table of area attributes
onto the locations so that summary sets can be grouped by them.
"""

__all__ = ['run_equity_report', 'equity_report_settings']

from .report import run_equity_report  # noqa: E402
from .settings import equity_report_settings  # noqa: E402
