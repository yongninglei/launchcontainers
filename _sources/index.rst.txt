launchcontainers
================

.. raw:: html

   <section class="lc-hero">
     <img
       class="lc-logo"
       src="https://user-images.githubusercontent.com/48440236/262432254-c7b53943-7c90-489c-933c-5f5a32510db4.png"
       alt="launchcontainers logo"
     />
     <p class="lc-eyebrow">Neuroimaging pipeline orchestration</p>
     <h1>launchcontainers</h1>
     <p class="lc-lead">
       Prepare analysis folders, preserve configuration state, and launch
       containerized workflows on local machines or HPC schedulers.
     </p>
     <div class="lc-actions">
       <a class="lc-button lc-button-primary" href="api.html">Browse API Reference</a>
       <a class="lc-button lc-button-secondary" href="https://github.com/garikoitz/launchcontainers">View on GitHub</a>
     </div>
   </section>

.. raw:: html

   <section class="lc-card-grid">
     <article class="lc-card">
       <h2>Prepare</h2>
       <p>
         Build per-subject and per-session input trees from BIDS data,
         container configuration, and a curated <code>subseslist</code>.
       </p>
     </article>
     <article class="lc-card">
       <h2>Launch</h2>
       <p>
         Generate and submit reproducible container commands for local runs,
         Dask clusters, SLURM, or SGE.
       </p>
     </article>
     <article class="lc-card">
       <h2>Validate</h2>
       <p>
         Run project quality-control helpers and analysis integrity checks on
         generated outputs.
       </p>
     </article>
   </section>

Overview
--------

``launchcontainers`` is a Python package for organizing and launching
containerized neuroimaging workflows. The repository centers on three practical
jobs:

1. prepare an analysis directory with the expected input structure,
2. generate launch commands for each ``sub`` and ``ses`` pair,
3. support validation and quality-control workflows after processing.

The current pipeline support in the package focuses on diffusion MRI workflows,
including `anatROIs <https://github.com/garikoitz/anatROIs>`_,
`RTP-preproc <https://github.com/garikoitz/rtp-preproc>`_, and
`RTP2-pipeline <https://github.com/garikoitz/rtp-pipeline>`_.

.. raw:: html

   <section class="lc-section-intro">
     <h2>Documentation</h2>
     <p>
       The API reference is generated directly from the package docstrings so
       the published documentation follows the codebase as it evolves.
     </p>
   </section>

Main Modules
------------

.. list-table::
   :widths: 24 76
   :header-rows: 1

   * - Area
     - Purpose
   * - ``launchcontainers.cli``
     - Entry points for preparing analyses, launching jobs, and related helper commands.
   * - ``launchcontainers.prepare``
     - Build input folder structures and link container-specific assets.
   * - ``launchcontainers.clusters``
     - Submit or coordinate jobs on Dask, SLURM, and SGE backends.
   * - ``launchcontainers.quality_control``
     - Run helper utilities for post-run checks and workflow bookkeeping.

Next Step
---------

Start with the :doc:`API reference <api>` to see the documented functions and
modules available in the package.

.. toctree::
   :hidden:
   :maxdepth: 2

   api
