.. _prepare_dwi:

Prepare — DWI
=============

What are we preparing?
-----------------------

Diffusion-weighted imaging (DWI) pipelines (``rtppreproc``, ``rtp-pipeline``,
``rtp2-preproc``, ``rtp2-pipeline``, ``anatrois``, ``freesurferator``) need
their inputs organised in a very specific directory structure before they can
be submitted to the cluster.  The prepare step:

1. **Validates** that all required files exist for each subject/session (T1
   anatomical, DWI NIfTI, bval, bvec, FreeSurfer outputs, etc.).
2. **Creates the analysis directory** under
   ``<bidsdir>/derivatives/<container>-<version>/analysis-<name>/`` and
   writes a frozen copy of every config file into it — so the exact
   parameters used for a run are always traceable.
3. **Generates per-subject launch scripts** (``job_scripts_<timestamp>/``)
   ready to be submitted with ``sbatch``, ``qsub``, or run locally.
4. **Copies auxiliary files** (annotation files, ROI archives, tract
   parameter tables, brain masks) into the analysis directory so the container
   has everything it needs in one place.

No DWI data are modified.  The prepare stage only reads from your BIDS tree
and writes into the derivatives analysis directory.

How the pipeline works
-----------------------

Config building (``gen_config_dict_and_copy``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:func:`~launchcontainers.prepare.dwi_prepare.gen_config_dict_and_copy` reads
the launchcontainers YAML and builds the ``inputs`` dictionary that the
container ``config.json`` expects.  Depending on the active container it also
calls
:func:`~launchcontainers.prepare.dwi_prepare.copy_rtp2_configs` to copy any
optional auxiliary files (annotation ``.zip``, ROI ``.zip``, tract parameter
``.csv``, FreeSurfer mask ``.nii.gz``) into the analysis directory.

Subject-level input preparation (``RTP2_prepare_input``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

:mod:`~launchcontainers.prepare.RTP2_prepare_input` handles per-subject
input resolution — locating the correct DWI file (or RPE pair), T1, bval /
bvec, and FreeSurfer directory — and writes the paths into the per-subject
``config.json`` that the container will read at runtime.

Supported containers and their required inputs
-----------------------------------------------

.. list-table::
   :header-rows: 1
   :widths: 25 75

   * - Container
     - Key inputs validated at prepare time
   * - ``anatrois``
     - T1 NIfTI (or existing FreeSurfer zip), optional annotation file,
       optional MNI ROI zip
   * - ``freesurferator``
     - T1 NIfTI (or existing FreeSurfer zip), optional control points,
       annotation file, MNI ROI zip
   * - ``rtppreproc``
     - T1, DWI NIfTI, bval, bvec, FreeSurfer mask
   * - ``rtp-pipeline``
     - T1, DWI, bval, bvec, FreeSurfer mask, tract parameter CSV,
       optional FreeSurfer mask NIfTI
   * - ``rtp2-preproc``
     - T1, DWI, bval, bvec, FreeSurfer mask
   * - ``rtp2-pipeline``
     - T1, DWI, bval, bvec, FreeSurfer mask, tract parameter CSV,
       optional FreeSurfer mask NIfTI

API reference
--------------

See :ref:`api_ref` for the full auto-generated documentation of all classes
and functions in this module:

* :mod:`launchcontainers.prepare.dwi_prepare`
* :mod:`launchcontainers.prepare.RTP2_prepare_input`
