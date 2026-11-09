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
        "data_gaps": {"default_vulnerability_ids": [999], "geocode_quality_threshold": 0.7},
        "loss_sharing": true
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

The loss-sharing table, ``equity-loss-sharing.csv`` (with ``loss_sharing``), shows who bears each group's ground-up
loss. It compares the group's AAL in the ``gul``, ``il`` and, if there are reinsurance outputs, ``ri`` outputs:
``retained_aal`` (``gul - il``) is kept by policyholders, through deductibles, losses above limits and locations
with no insurance; ``insurer_net_aal`` is paid by insurers after reinsurance; and ``ceded_aal`` (``il - ri``) is paid
by reinsurers, each also as a share of the ground-up loss. Each perspective's summary set needs the ``group_by``
columns. By default it is the report's ``summary_set``, and ``"loss_sharing": {"summary_sets": {"il": 2}}`` picks
another. When the summary sets are grouped by location, rows are matched across perspectives and, with an
``income_column``, the table also gives the median retained AAL as a share of income.

The public-assistance tables (with ``public_assistance``) show what a programme paying towards the losses people
keep would cost and whom it would reach. For each event, each eligible row is paid ``share`` (default 1) of its
retained loss (``gul - il``, or the whole ground-up loss without insured outputs) above ``deductible`` (default 0), up
to ``limit``. With ``income_threshold``, only rows with a known income below it are eligible.
``equity-public-assistance.csv`` gives, per group, the expected annual retained loss and assistance, the share of
retained loss covered, the group's share of the programme's spending next to its share of retained loss, and, with an
income column, the median retained loss as a share of income before and after assistance.
``equity-public-assistance-budget-ep.csv`` gives the programme's OEP and AEP spending at ``return_periods``. The rows
must match between the ``gul`` and ``il`` summary sets (for example, both grouped by ``LocNumber``). The programme's
terms are applied to each sample when both have a sample PLT (``plt_sample``) and the analysis settings give
``number_of_samples``. Otherwise they are applied to each event's mean loss (``plt_moment``), which understates what a
programme with a deductible or limit pays.

.. code-block:: json

    "public_assistance": {"deductible": 1000, "share": 0.75, "limit": 25000, "income_threshold": 30000}

The data-gap audit, ``equity-data-gaps.csv`` (with ``data_gaps``), shows where the model sees less of a group's
exposure. It groups the run's locations, read from the run's ``input`` directory, and reports per group: locations
whose lookup failed and value left unmodelled; with ``default_vulnerability_ids``, locations and value on a default
or fallback vulnerability; locations without coordinates; and, from ``GeocodeQuality``, mean geocode quality and the
share below ``geocode_quality_threshold``. Here ``group_by`` columns are location or exposure attribute columns.

Comparing scenarios
-------------------

To see who gains and who loses between two analyses, such as a base run and a mitigation programme, a
climate-conditioned event set or a policy change, run:

.. code-block:: bash

    oasislmf model compare -r <scenario_run_dir> -b <base_run_dir> [-a analysis_settings.json]

Both runs are read with the same ``equity_report`` settings (by default those saved in the scenario's run
directory), and their rows are grouped and then matched by group value, so the runs may have different summary ids
or locations. ``<runtype>_S<summary_set>_compare-aal.csv`` has, per group, the rows, value and AAL in each run,
``aal_change`` and ``aal_change_pct``, and ``share_of_aal_change`` next to ``aal_share_base``. A group that carried
two thirds of the base loss but gets a third of the reduction is easy to spot. With an ``income_column`` it also has
the median AAL-to-income ratio in each run. A group is suppressed when it has fewer than ``min_group_size`` rows in
either run. When both runs have an EPT (``ept_*`` outputs), ``..._compare-ep.csv`` compares the portfolio's loss at
each return period. The tables are written to ``<scenario_run_dir>/output/compare`` unless
``--compare-report-dir`` is given.

.. note::

    Group results are only as good as the attribute table and the model. Use area-level data rather than data
    about individuals, keep ``min_group_size`` high enough for the data you hold, and read small differences
    between groups with the model's uncertainty in mind.
