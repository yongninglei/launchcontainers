Core Concepts
=============

The three-phase workflow
------------------------

Every analysis in ``launchcontainers`` follows the same three phases:

.. code-block:: text

   lc prepare  →  lc run  →  lc qc

**prepare**
    Reads your config files and subject/session list from ``basedir/code/``.
    Validates that all required input files exist for each subject/session.
    Creates the analysis directory under ``BIDS/derivatives/``, writes frozen
    copies of your configs into it, and generates per-subject HPC launch scripts.

**run**
    Reads the *prepared analysis directory* — not your original config files.
    Submits the generated scripts to the HPC scheduler (SLURM, SGE, or local).

**qc**
    After jobs finish, checks whether all expected output files were produced.
    Prints a pass/fail table, writes logs, and produces a ``failed_subseslist.tsv``
    so that failed subjects can be re-submitted immediately.

The analysis directory as the source of truth
---------------------------------------------

A key design principle is that the **analysis directory** produced by
``prepare`` is fully self-contained and reproducible. Once ``prepare`` runs,
``run`` and ``qc`` only need its path — they never re-read
your original ``basedir/code/`` files.

.. code-block:: text

   BIDS/derivatives/
     rtppreproc-1.2.0-3.0.3/
       analysis-main/
         config/
           lc_config.yaml          ← frozen copy of your input config
           subseslist_input.tsv    ← original input subseslist
           subseslist.tsv          ← cleaned (blocking failures excluded)
         logs/
           prepare_TIMESTAMP.log
           run_TIMESTAMP.log
           qc_TIMESTAMP.log
         scripts/
           run_sub-01_ses-T01.sh
           run_sub-01_ses-T02.sh
         results/
           sub-01/ses-T01/
           ...
         failed_subseslist.tsv     ← written by qc on any failures

Issue severity levels
---------------------

During ``prepare``, each subject/session check returns a list of issues
with one of three severity levels:

.. list-table::
   :header-rows: 1
   :widths: 15 40 30

   * - Severity
     - Meaning
     - Behaviour
   * - ``blocking``
     - A required file is missing; cannot proceed
     - Subject/session excluded from the run
   * - ``warn``
     - Suspicious but not fatal
     - Subject/session included; issue logged
   * - ``auto_fix``
     - Issue can be resolved programmatically
     - Fix function is called automatically

Supported pipelines
-------------------

Container-based (DWI / structural)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

These pipelines run inside Apptainer/Singularity containers.

.. list-table::
   :header-rows: 1
   :widths: 25 55

   * - Container key
     - Purpose
   * - ``anatrois``
     - DWI preprocessing (legacy RTP)
   * - ``freesurferator``
     - FreeSurfer reconstruction and ROI operation
   * - ``rtppreproc``
     - DWI preprocessing (legacy RTP)
   * - ``rtp-pipeline``
     - DWI tractography (legacy RTP)
   * - ``rtp2-preproc``
     - DWI preprocessing (RTP2)
   * - ``rtp2-pipeline``
     - DWI tractography (RTP2)
   * - ``fMRIprep``
     - `fMRI Preprocessing <https://fmriprep.org>`_, including anatomical MRI processing with FreeSurfer
   * - ``prfprepare``
     - `PRF prepare <https://github.com/fMRIat/prfprepare>`_ , preparing the events.tsv, getting preprocessed bold time series
   * - ``prfanalyze-vista``
     - PRF modeling, based on `mrVista <https://github.com/fMRIat/mrvista>`_ from Stanford
   * - ``prfresult``
     - Visulizing `PRFresult <https://github.com/fMRIat/prfresult>`_ on spheres and surface

Analysis-based (fMRI)
~~~~~~~~~~~~~~~~~~~~~~

These pipelines are managed by the ``PREPARER_REGISTRY`` and do not
require a container.

.. list-table::
   :header-rows: 1
   :widths: 15 55

   * - Type key
     - Purpose
   * - ``glm``
     - First-level GLM (nilearn-based)
   * - ``presurfer``
     - MP2RAGE T1w image processing `before freesurfer <https://github.com/srikash/presurfer>`_
   * - ``nordic_fmri``
     - NORDIC denoising on fMRI data `using nordic_raw repo <https://github.com/SteenMoeller/NORDIC_Raw>`_
   * - ``nordic_dwi``
     - NORDIC denoising on DWI data `using nordic_raw repo <https://github.com/SteenMoeller/NORDIC_Raw>`_
   * - ``to be added``
     - any analysis can be coded into single session running warappers.

Supported hosts
---------------

.. list-table::
   :header-rows: 1
   :widths: 15 20 45

   * - Host key
     - Scheduler
     - Notes
   * - ``DIPC``
     - SLURM
     - ``sbatch``, scratch filesystem at ``/scratch``
   * - ``BCBL``
     - SGE
     - ``qsub``, ``long.q`` queue, ``/bcbl`` filesystem
   * - ``local``
     - bash
     - Serial or parallel via concurrent.futures
