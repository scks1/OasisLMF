.. _equity_report:

Equity report
=============

.. versionadded:: NEXT

The equity report shows how the losses of a finished analysis fall on different groups, such as income bands
or neighbourhoods. It only reads the analysis outputs and inputs. It doesn't change the losses, and no group
field is used to price or rate anything.

Run it after an analysis:

.. code-block:: bash

    oasislmf model equity-report -r <model_run_dir> -a analysis_settings.json [-D <user_data_dir>]

The report reads the ``equity_report`` section of the analysis settings. The groups usually come from an
:ref:`exposure attribute table <exposure_attributes>`, such as an income quintile or deprivation index per
postcode, and must be one of the summary set's ``oed_fields``.

.. code-block:: json

    "exposure_attributes": [
        {"file": "postcodes.csv", "join_on": "PostalCode", "columns": ["IncomeQuintile", "MedianIncome"]}
    ],
    "gul_summaries": [{"id": 1, "ord_output": {"alt_period": true, "plt_moment": true},
                       "oed_fields": ["LocNumber", "IncomeQuintile", "MedianIncome"]}],
    "equity_report": {
        "runtype": "gul",
        "summary_set": 1,
        "group_by": ["IncomeQuintile"],
        "income_column": "MedianIncome",
        "min_group_size": 10,
        "equity_weighting": {"eta": 1.5},
        "premiums": {"loading": 0.3, "schemes": ["risk_based", {"type": "capped", "cap": 0.02}]},
        "data_gaps": {"default_vulnerability_ids": [999], "geocode_quality_threshold": 0.7}
    }

Settings
--------

``runtype`` (default ``gul``) and ``summary_set`` (default ``1``)
    The outputs to report on. Each summary id of the set is one row of the report, so a set grouped by
    ``LocNumber`` reports per location.
``group_by``
    Summary-info columns to group the rows by. Without it, only the total row ``All`` is written.
``income_column``
    Summary-info column holding each row's income. Needed for the income-based columns, equity weighting and the
    ``capped`` and ``means_tested`` premium schemes.
``burden_threshold`` (default ``0.01``) and ``income_burden_threshold`` (default ``0.05``)
    A row is over the burden threshold when its AAL is more than this share of its value, or of its income.
``min_group_size`` (default ``10``)
    Groups with fewer rows have their results blanked and ``suppressed`` set, so small groups can't be singled
    out. ``0`` reports every group.
``sample_type``
    ``1`` (analytical) or ``2`` (sample mean). Defaults to the sample mean when the outputs have it, otherwise
    the analytical mean.

Tables
------

The tables are written to ``<output_dir>/equity`` (or ``--equity-report-dir``), named
``<runtype>_S<summary_set>_equity-<table>.csv``, with the settings used in ``..._equity-report.json``. Each table
has one row per group and a total row ``All``.

``burden``
    Expected annual loss (AAL) against value and income: ``aal_to_tiv``, ``relative_aal_to_tiv`` (above 1, the group
    carries more expected loss for its value than the portfolio), medians, and the share of rows and value over
    the burden thresholds. AAL comes from the period ALT (``alt_period``), or from ``alt_meanonly`` when there is
    no period ALT.
``weighted-aal`` and ``weighted-ep`` (with ``equity_weighting``)
    Each row's losses are multiplied by ``(reference_income / income) ** eta``, so a loss counts for more the lower
    the income of whoever bears it. ``reference_income`` is ``mean`` (default), ``median`` or a number;
    ``eta = 0`` gives the standard losses. The AAL table compares each group's share of the AAL before and after
    weighting; the EP table (which needs the MPLT, ``plt_moment``) gives the occurrence and aggregate losses at
    ``return_periods`` before and after weighting.
``premiums`` (with ``premiums``)
    What each group would pay under each scheme, with ``cross_subsidy`` (premium minus risk premium) and public
    ``subsidy``. The risk premium is AAL times ``1 + loading``. Schemes: ``risk_based``; ``flat`` (``basis``
    ``row`` or ``tiv``); ``capped`` (no row pays more than ``cap`` times its income); ``means_tested`` (rows below
    ``income_threshold`` get ``discount`` off). ``capped`` and ``means_tested`` are paid for by a public ``subsidy``
    (default) or a ``surcharge`` on the other rows, which never pushes a row past its own cap. Schemes use only AAL,
    value and income.

The data-gap audit, ``equity-data-gaps.csv`` (with ``data_gaps``), shows where the model sees less of a group's
exposure. It groups the run's locations, read from the run's ``input`` directory, and reports per group: locations
whose lookup failed and value left unmodelled; with ``default_vulnerability_ids``, locations and value on a default
or fallback vulnerability; locations without coordinates; and, from ``GeocodeQuality``, mean geocode quality and the
share below ``geocode_quality_threshold``. Here ``group_by`` columns are location or exposure attribute columns.

.. note::

    Group results are only as good as the attribute table and the model. Use area-level data rather than data
    about individuals, keep ``min_group_size`` high enough for the data you hold, and read small differences
    between groups with the model's uncertainty in mind.
