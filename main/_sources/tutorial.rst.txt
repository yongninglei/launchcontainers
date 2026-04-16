Tutorial
========

.. toctree::
   :maxdepth: 2
   :caption: Prepare tutorials

   prepare_glm
   prepare_dwi
   prepare_ret

----

This page walks through a complete run from scratch using ``rtppreproc``
as the example pipeline. The same steps apply to all other pipelines —
only the config values change.

Step 1 — Copy the example configs
----------------------------------

Copy the bundled example config files to your working directory
(we recommend ``basedir/code/``):

.. code-block:: console

   lc copy_configs --output /scratch/tlei/VOTCLOC/code/

This gives you:

- ``lc_config.yaml`` — main config file
- ``subseslist.txt`` — subject/session list
- ``rtppreproc.json`` (and others) — container-specific parameter files

Step 2 — Edit ``lc_config.yaml``
----------------------------------

Open ``lc_config.yaml`` and set at minimum:

.. code-block:: yaml

   general:
     basedir: /scratch/tlei/VOTCLOC
     bidsdir_name: BIDS
     container: rtppreproc
     analysis_name: main
     host: DIPC
     deriv_layout: legacy
     force: True

   container_specific:
     rtppreproc:
       version: 1.2.0-3.0.3
       precontainer_anat: freesurferator_0.2.1_7.4.1
       anat_analysis_name: main
       rpe: True

   host_options:
     DIPC:
       manager: slurm
       cores: 20
       memory: 32G
       partition: general
       qos: regular
       walltime: '10:00:00'

See :doc:`configuration` for a full reference of all keys.

Step 3 — Edit ``subseslist.txt``
----------------------------------

Set ``RUN = True`` for every subject/session you want to process:

.. code-block:: text

   sub,ses,RUN,dwi
   01,T01,True,True
   01,T02,True,True
   02,T01,True,True

.. tip::

   To generate a subseslist automatically from an existing BIDS directory:

   .. code-block:: console

      lc gen_subses --basedir /scratch/tlei/VOTCLOC/BIDS --name subseslist.txt

Step 4 — Prepare
-----------------

.. code-block:: console

   lc prepare \
     --lc_config or -lcc               /scratch/tlei/VOTCLOC/code/lc_config.yaml \
     --sub_ses_list or -ssl            /scratch/tlei/VOTCLOC/code/subseslist.txt \
     --container_specific_config or -cc /scratch/tlei/VOTCLOC/code/rtppreproc.json

``prepare`` will:

1. Validate that all required input files exist for each session.
2. Create the analysis directory under ``BIDS/derivatives/rtppreproc-1.2.0-3.0.3/analysis-main/``.
3. Freeze copies of all your configs into ``analysis-dir``.
4. Generate per-subject launch scripts in ``analysis-dir/job_scripts_<timing>/``.
5. Write a cleaned ``subseslist.txt`` (blocking failures excluded).
6. Print a Rich summary table showing the status of every subject/session.

If any subjects have blocking issues they will be listed clearly and
excluded from the cleaned ``subseslist.txt`` — all other subjects proceed.

Step 5 — Run
-------------

First do a dry run to verify what would be submitted:

.. code-block:: console

   lc run --workdir or -w /scratch/tlei/VOTCLOC/BIDS/derivatives/rtppreproc-1.2.0-3.0.3/analysis-main

Then submit for real:

.. code-block:: console

   lc run --workdir or -w /scratch/tlei/VOTCLOC/BIDS/derivatives/rtppreproc-1.2.0-3.0.3/analysis-main \
          --run_lc or -R

Jobs are submitted with ``sbatch`` (SLURM), ``qsub`` (SGE), or run directly
via ``bash`` (local), depending on ``host_options.manager``.

Step 6 — QC
------------

After jobs finish:

.. code-block:: console

   lc qc --workdir or -w /scratch/tlei/VOTCLOC/BIDS/derivatives/rtppreproc-1.2.0-3.0.3/analysis-main

This will:

- Print a pass/fail table for every subject/session.
- Write a ``qc_TIMESTAMP.log`` under ``analysis-dir/logs/``.
- Write ``analysis-dir/failed_subseslist.txt`` for any failures.

Re-running failures
--------------------

Pass the QC-generated failed list straight back into ``prepare``:

.. code-block:: console

   lc prepare \
     --lc_config                /scratch/tlei/VOTCLOC/code/lc_config.yaml \
     --sub_ses_list             /scratch/.../analysis-main/failed_subseslist.txt \
     --container_specific_config /scratch/tlei/VOTCLOC/code/rtppreproc.json

Then run and QC again as normal.

Using the integrity checker
-----------------------------

The ``checker`` tool validates completeness across your whole dataset
independently of the launcher:

.. code-block:: console

   checker --spec fmriprep \
           --bids-dir /scratch/tlei/VOTCLOC/BIDS \
           --sub-ses-list /scratch/tlei/VOTCLOC/code/subseslist.txt \
           --output-dir /scratch/tlei/VOTCLOC/code/qc/

It produces three output files:

- **Brief CSV** — one row per subject/session, pass/fail summary
- **Detailed log** — indexed list of all missing files
- **Matrix CSV** — subjects × sessions pivot table (``1`` = complete,
  ``0.5`` = partial, ``0`` = missing), ready to import into Google Sheets

Supported specs: ``bids``, ``fmriprep``, ``glm``, ``prf``, ``prfprepare``,
``prfanalyze``, ``bidsdwi``, ``rtp``.
