.. _api_ref:

API Reference
=============

This page documents the main public modules in ``launchcontainers`` using the
package docstrings and function docstrings pulled directly from the source.

Core CLI
--------

.. automodule:: launchcontainers.cli
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.do_prepare
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.do_launch
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.do_qc
   :members:
   :undoc-members: False

Utilities
---------

.. automodule:: launchcontainers.utils
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.log_setup
   :members:
   :undoc-members: False

Job-script generation
---------------------

.. automodule:: launchcontainers.gen_jobscript
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.gen_jobscript.gen_container_cmd
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.gen_jobscript.gen_matlab_cmd
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.gen_jobscript.gen_py_cmd
   :members:
   :undoc-members: False

Preparation
-----------

For full documentation of the prepare pipelines see the tutorial pages:

* :ref:`prepare_glm` — fMRI GLM preparation (``GLMPrepare``, ``PRFPrepare``,
  ``BasePrepare``)
* :ref:`prepare_dwi` — DWI pipeline preparation
* :ref:`prepare_ret` — Retinotopy / PRF preparation

Schedulers
----------

.. automodule:: launchcontainers.clusters.local
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.clusters.slurm
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.clusters.sge
   :members:
   :undoc-members: False

Checks And QC
-------------

.. automodule:: launchcontainers.check.check_dwi_pipelines
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.check.general_checks
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.quality_control.continue_run_rtp2pipeline
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.quality_control.qc_rtp2preproc_output
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.quality_control.qc_tract_finish_rtp2pipeline
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.quality_control.rtp2pipelne_unzip_output
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.quality_control.batch_sync_tract_and_mrtrix
   :members:
   :undoc-members: False


Helper Scripts
--------------

.. automodule:: launchcontainers.helper_function.copy_configs
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.helper_function.create_bids
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.helper_function.gen_subses
   :members:
   :undoc-members: False

.. automodule:: launchcontainers.helper_function.zip_example_config
   :members:
   :undoc-members: False
