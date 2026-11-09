Verifying a model run
=====================

.. versionadded:: NEXT

A run manifest records what went into and came out of a run, so anyone holding the run directory can check later
that its results haven't changed, or that a second run reproduced the first exactly. This is useful for audits,
regulatory review, regression tests, and for sharing results with people who need to trust them.

Writing a manifest
------------------

Ask for one when you run the model:

.. code-block:: bash

    oasislmf model run --config oasislmf.json --run-manifest

or write one for a finished run:

.. code-block:: bash

    oasislmf model manifest -r <model_run_dir> -a analysis_settings.json -M model_settings.json

Either way the run directory gets a ``run_manifest.json`` holding:

- the SHA-256 hash and size of every file under the run's ``input`` and ``output`` directories (the Oasis files,
  the copy of the analysis settings and all the results);
- the hashes of the analysis and model settings files;
- the model data directory (the run's ``static`` directory, unless ``-d`` names another) and, with
  ``--run-manifest-model-data``, the hash of each model data file. Hashing a large model's footprint can take a
  while, so this is off by default;
- the Python, platform and package versions (oasislmf, ods-tools, numpy, numba, pandas, pyarrow, scipy) and the
  ``OASIS_*`` environment variables.

Checking a run
--------------

.. code-block:: bash

    oasislmf model verify -r <model_run_dir>

hashes the run's files again and compares them with its manifest. It exits with an error, listing the files that
changed, are missing or are new, if anything differs, and logs any difference in versions or environment without
failing on it.

To check that a second run reproduced a first one exactly:

.. code-block:: bash

    oasislmf model verify -r <second_run_dir> --verify-against <first_run_dir> --verify-report-json diff.json

``--verify-report-json`` writes the full comparison, including the environment differences, to a file.

.. note::

    Runs with the same inputs only give byte-identical outputs when the run is deterministic. With the default
    multiprocessed keys lookup the item order, and so float rounding, can change between runs (see
    `issue 2211 <https://github.com/OasisLMF/OasisLMF/issues/2211>`_); ``--lookup-multiprocessing False`` avoids
    that. Changing the number of kernel processes (``-n``) or the machine's numeric libraries can also change the
    last digits of some results. ``model verify`` reports such runs as different, and the version and environment
    differences it logs show where to look.
