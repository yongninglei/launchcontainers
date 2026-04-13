.. image:: https://img.shields.io/badge/License-MIT-4c4c4c.svg
   :target: https://github.com/garikoitz/launchcontainers/blob/main/LICENSE
   :alt: License: MIT

.. image:: https://img.shields.io/badge/Documentation-online-4caf50.svg
   :target: https://garikoitz.github.io/launchcontainers
   :alt: Documentation online

.. image:: https://img.shields.io/github/actions/workflow/status/garikoitz/launchcontainers/ci.yml?label=CI&logo=github
   :target: https://github.com/garikoitz/launchcontainers/actions
   :alt: CI passing

|

.. raw:: html

   <h1 style="font-size:2.8em; font-weight:700; margin-top:0.2em;">launchcontainers</h1>

**launchcontainers** is an open-source Python framework for running
neuroimaging analysis pipelines on high-performance computing (HPC) clusters
in a reproducible, BIDS-compliant way. It was developed at the
`Basque Center on Cognition, Brain and Language (BCBL) - LMC Group
<https://www.bcbl.eu/en/research/research-groups/language-memory-control>`_
to address the practical challenges of managing large multi-subject,
multi-session neuroimaging datasets across heterogeneous computing environments.

The framework wraps complex container-based and Python-based pipelines — from
DWI preprocessing and tractography to fMRI first-level GLM and population
receptive field (PRF) analysis — behind a single, consistent command-line
interface. Rather than requiring researchers to manually track config files,
input paths, and job scripts for dozens of subjects, launchcontainers handles
all of that automatically: it validates inputs before any job is submitted,
freezes a snapshot of every config used, and generates ready-to-submit HPC
scripts tailored to your cluster's scheduler.

A key design principle is that the **analysis directory** produced at the
prepare stage is fully self-contained. Every run is fully traceable — you can
always inspect exactly which config, which software version, and which
subject list was used to produce a given result. Quality control is built
into the workflow as a first-class step, generating per-subject pass/fail
reports and a ``failed_subseslist.tsv`` that feeds directly back into the
next prepare cycle with no manual editing required.

----

Key features
------------

.. list-table::
   :widths: 5 95

   * - 🔁
     - **Three-phase workflow** — ``prepare → run → qc`` keeps validation,
       execution, and quality control cleanly separated; each phase reads only
       from the self-contained analysis directory, never from your original
       config files.
   * - 📦
     - **Container and analysis pipelines** — supports six DWI/structural
       Apptainer containers (``anatrois``, ``freesurferator``, ``rtppreproc``,
       ``rtp-pipeline``, ``rtp2-preproc``, ``rtp2-pipeline``, ``fMRIPrep``) and fMRI
       analysis types (``glm``, ``prf``) through an extensible registry.
   * - 🖥️
     - **Multi-cluster support** — works with SLURM (DIPC), SGE (BCBL), and
       local execution out of the box; adding a new host requires only one
       config block.
   * - 🔒
     - **Reproducibility by design** — configs, subject lists, and software
       versions are frozen into the analysis directory at prepare time,
       making every result fully auditable.
   * - ✅
     - **Built-in QC** — outputs are validated after every run; failures are
       reported and exported as a ready-to-resubmit subject list.
   * - 🧩
     - **Extensible** — adding a new analysis type requires only a subclass
       and one line in the registry; no existing code needs to change.

----

Installation
------------

For **users** — install the latest release from PyPI:

.. code-block:: console

   pip install launchcontainers

For **developers** — clone the repo and install with Poetry:

.. code-block:: console

   pip install pipx && pipx install poetry
   git clone https://github.com/garikoitz/launchcontainers.git
   cd launchcontainers
   poetry env use python3 && poetry install
   poetry shell

See the :doc:`installation` page for full requirements, verification steps,
and instructions for building the documentation locally.

----

How to cite
-----------

If you use launchcontainers in your research, please cite:

.. code-block:: text

   Lerma-Usabiaga, G., Lei, Y., Liu, M., Lecca, L., Linhardt, D.,
   Tellaetxe, I., & others (2023). launchcontainers: a Python framework
   for reproducible neuroimaging pipeline execution on HPC clusters.
   [Software]. https://github.com/garikoitz/launchcontainers

----


.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Getting Started

   bcbl_computing
   installation
   concepts

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: User Guide

   tutorial
   cli_reference
   configuration
   prepare_fmriprep

.. toctree::
   :hidden:
   :maxdepth: 2
   :caption: Developer Guide

   extending
   code_structure

.. toctree::
   :hidden:
   :maxdepth: 1
   :caption: Other

   changelog
   api
