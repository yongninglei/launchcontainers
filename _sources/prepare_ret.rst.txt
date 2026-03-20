.. _prepare_ret:

Prepare — Retinotopy (PRF)
==========================

What are we preparing?
-----------------------

Population receptive field (PRF) / retinotopy analyses also rely on
vistadisplog ``.mat`` files to record what stimulus was shown during each
acquisition.  The prepare step for PRF is simpler than for GLM: there is no
first-level model to fit at prepare time, so the only goal is to produce a
clean **mapping TSV** that links every log file to its corresponding BIDS run.

This mapping TSV is the single output of the PRF prepare stage.  It is used
downstream by:

* the **GLM prepare** pipeline (to match logs → NIfTIs via acquisition time);
* any custom analysis script that needs to know which stimulus condition
  corresponds to which bold file.

How the pipeline works
-----------------------

Scan for log files
~~~~~~~~~~~~~~~~~~~

:class:`~launchcontainers.prepare.prf_prepare.PRFPrepare` looks for
``20*.mat`` files under
``<bidsdir>/sourcedata/vistadisplog/sub-<sub>/ses-<ses>/`` and sorts them in
ascending filename order (= acquisition order).

Parse each log
~~~~~~~~~~~~~~~

For every ``.mat`` file:

* ``params.loadMatrix`` is read to retrieve the basename of the stimulus file
  used during that run.
* The stimulus basename is parsed to extract the original task label (the
  second ``_``-separated token, e.g. ``fixRW``) and a per-task run counter.
* The log filename encodes the run *end* time; 6 minutes are subtracted to
  recover the approximate *start* time that will match the BIDS
  ``AcquisitionTime``.

When called with ``lc_glm=True`` (from the GLM prepare pipeline), the method
additionally computes a normalised GLM task-run label (``fixnonstop`` or
``fixblock``) and appends it as a ``glm_task_run`` column.

Write the mapping TSV
~~~~~~~~~~~~~~~~~~~~~~

The resulting table is written to the vistadisplog directory as
``sub-<sub>_ses-<ses>_desc-mapping_PRF_acqtime.tsv`` with the following
columns:

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Column
     - Description
   * - ``log_file_path``
     - Full path to the ``.mat`` log file
   * - ``log_file_name``
     - Basename of the ``.mat`` log file
   * - ``stim_name``
     - Basename of the stimulus ``.mat`` (``params.loadMatrix``)
   * - ``task_run``
     - Original task label + per-original-task run counter,
       e.g. ``task-fixRW_run-01``
   * - ``acq_time``
     - Estimated acquisition start time (``HH:MM:SS``)
   * - ``glm_task_run``
     - *(GLM mode only)* Normalised task label + counter,
       e.g. ``task-fixnonstop_run-01``

API reference
--------------

See :ref:`api_ref` for the full auto-generated documentation of all classes
and functions in this module:

* :class:`launchcontainers.prepare.prf_prepare.PRFPrepare`
