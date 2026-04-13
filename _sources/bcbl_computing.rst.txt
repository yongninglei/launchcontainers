BCBL Computing Environment
==========================

This page describes the computing infrastructure at BCBL and the
recommended first-time setup for running launchcontainers on it.

----

1. Computing systems overview
------------------------------

1.1 Login node: brodmann
~~~~~~~~~~~~~~~~~~~~~~~~

``brodmann`` is the gateway machine you log into first.  It is used
only as an entry point — do not run heavy computations on it.

- **Home directory**: ``/home/<user>``

1.2 Compute resources
~~~~~~~~~~~~~~~~~~~~~~

**Cajal** (interactive computing)
   Four interactive nodes (``cajal01``–``cajal04``).  Cajal03 and
   Cajal04 each have 96 cores and 500 GB RAM.

   .. code-block:: console

      # Connect to Cajal03
      ssh cajal03

      # Connect with display forwarding (e.g. for Freeview)
      ssh -X cajal03

   There is also ``lmc02``, a node reserved for the LMC group.  Use it
   when the Cajal nodes are heavily loaded.

**IPS** (SGE-based HPC cluster)
   The main cluster for large-scale batch jobs.

   .. code-block:: console

      ssh ips-0-3

   .. note::
      MATLAB does not work on IPS.

**DIPC** (additional HPC)
   Accessible through ``brodmann`` only.
   See the `DIPC documentation <https://scc.dipc.org/docs/>`_ for
   details.

.. important::

   **Home directory paths differ between systems:**

   +--------------------+------------------------+
   | System             | Home path              |
   +====================+========================+
   | brodmann           | ``/home/<user>``       |
   +--------------------+------------------------+
   | Cajal / IPS        | ``/exporthome/<user>`` |
   +--------------------+------------------------+

   Always store your work **under your user directory** so it is
   available across systems.

**Check system load:**

.. code-block:: console

   htop

Or consult the IPS wiki for queue status.

----

2. First-time setup
--------------------

2.1 Configure your shell
~~~~~~~~~~~~~~~~~~~~~~~~~

Edit ``~/.bashrc`` (or ``~/.bash_profile``) on **both** brodmann
and the export filesystem to set your ``PATH``, module loads, and
any environment variables your pipelines need.

2.2 Set up Git credentials
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Configure Git on both brodmann and the export filesystem:

.. code-block:: console

   git config --global user.name  "Your Name"
   git config --global user.email "you@example.com"

   # Cache credentials so you are not prompted repeatedly
   git config --global credential.helper cache

2.3 Set up your Python environment
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We use three tools together:

+--------------------+--------------------------------------------------+-----------------------------------------+
| Tool               | Purpose                                          | Install location                        |
+====================+==================================================+=========================================+
| **micromamba**     | Environment management (fast conda replacement)  | ``~/soft/``                             |
+--------------------+--------------------------------------------------+-----------------------------------------+
| **pipx**           | Isolated tool installation                       | ``/exporthome/<user>`` (export home)    |
+--------------------+--------------------------------------------------+-----------------------------------------+
| **poetry**         | Dependency management and packaging              | installed via pipx                      |
+--------------------+--------------------------------------------------+-----------------------------------------+

**Step 1 — Install micromamba** into ``~/soft/``:

.. code-block:: console

   mkdir -p ~/soft
   cd ~/soft
   curl -L micro.mamba.pm/install.sh | bash

**Step 2 — Install pipx** (on the export filesystem):

.. code-block:: console

   pip install --user pipx

**Step 3 — Install Poetry** via pipx:

.. code-block:: console

   pipx install poetry
   poetry --version   # verify

**Step 4 — Create a Python environment**:

.. code-block:: console

   micromamba create -n myenv python=3.10
   micromamba activate myenv

**Step 5 — Install project dependencies** with Poetry:

.. code-block:: console

   poetry install

For launchcontainers specifically, see :doc:`installation` for the
full developer install instructions.

2.4 FreeSurfer setup
~~~~~~~~~~~~~~~~~~~~~

You need a valid FreeSurfer licence file.  Request one from
`<https://surfer.nmr.mgh.harvard.edu/registration.html>`_ and place
it at ``~/.license`` or set ``FREESURFER_HOME`` and
``FS_LICENSE`` in your shell config.

----

3. When to contact IT
----------------------

Contact the BCBL IT helpdesk if you need:

- System tools installed (e.g. ``htop``, ``tree``, compilers)
- Admin / sudo privileges
- New compute allocations or queue adjustments
- Network or storage issues

**Ticket system**: https://support.bcbl.eu/userui/welcome.php
