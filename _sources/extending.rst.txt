Extending launchcontainers — Adding a new Prepare pipeline
==========================================================

This page describes the **design pattern** used by all prepare pipelines in
launchcontainers and explains how to add a new one.  The pattern was
established while building the :ref:`prepare_glm` pipeline and applies to
every future pipeline (DWI, PRF, and beyond).

----

The pattern in one sentence
----------------------------

*Be clear about what goes in and what comes out, expose every option as a
typed property, let the class generate its own example config, implement
one method per prepare step, and put a single wrapper function at the bottom
of the module for* ``do_prepare.py`` *to call.*

----

Step 0 — Think before you code
--------------------------------

Before writing any code, answer these four questions:

1. **What are the inputs?** — raw data files, derivatives, log files, config
   values.
2. **What are the outputs?** — files written, symlinks created, directories
   produced.
3. **What are the options?** — every tunable parameter (space, task filter,
   block duration, …).
4. **What are the steps?** — break the prepare work into discrete, testable
   operations.

Only once these are clear should you open a new file.

----

Step 1 — Create the module and subclass ``BasePrepare``
--------------------------------------------------------

Create ``launchcontainers/prepare/<name>_prepare.py`` and subclass
:class:`~launchcontainers.prepare.base_prepare.BasePrepare`:

.. code-block:: python

   from launchcontainers.prepare.base_prepare import BasePrepare

   class MyPrepare(BasePrepare):

       def __init__(self, lc_config: dict | None = None):
           super().__init__(lc_config)
           # pull your pipeline's sub-dict from container_specific
           self._cfg = self.lc_config.get("container_specific", {}).get("MyPipeline", {})

``BasePrepare`` provides for free:

* :attr:`~launchcontainers.prepare.base_prepare.BasePrepare.basedir` —
  ``general.basedir``
* :attr:`~launchcontainers.prepare.base_prepare.BasePrepare.bidsdir` —
  ``<basedir>/<bidsdir_name>``
* :meth:`~launchcontainers.prepare.base_prepare.BasePrepare.write_example_config` —
  writes a YAML file; delegates content to :meth:`_example_config_dict`

----

Step 2 — Expose every option as a typed ``@property``
------------------------------------------------------

One property per config key.  Always provide a sensible default via
``.get(key, default)`` so the class can be instantiated with a minimal config
during testing:

.. code-block:: python

   @property
   def space(self) -> str:
       """Output space (e.g. ``fsnative``, ``T1w``)."""
       return self._cfg.get("space", "fsnative")

   @property
   def dry_run(self) -> bool:
       """If ``True``, log actions without writing files."""
       return bool(self._cfg.get("dry_run", False))

Properties that derive a **path** from another property (e.g.
``fmriprep_dir``) should be properties too, not computed inside methods:

.. code-block:: python

   @property
   def fmriprep_dir(self) -> str:
       return op.join(self.bidsdir, "derivatives", self.fmriprep_analysis_name)

----

Step 3 — Override ``_example_config_dict``
-------------------------------------------

Return a plain Python ``dict`` that represents a fully annotated
``lc_config.yaml`` for your pipeline.
:meth:`~launchcontainers.prepare.base_prepare.BasePrepare.write_example_config`
(inherited from ``BasePrepare``) handles the YAML serialisation:

.. code-block:: python

   @classmethod
   def _example_config_dict(cls) -> dict:
       return {
           "general": {
               "basedir": "/path/to/basedir",
               "bidsdir_name": "BIDS",
               "container": "MyPipeline",
               "host": "local",
               "force": True,
           },
           "container_specific": {
               "MyPipeline": {
                   "space": "fsnative",
                   "dry_run": False,
                   # ... all keys with sensible defaults
               }
           },
           "host_options": {"local": {}},
       }

Users can then auto-generate a starter config:

.. code-block:: console

   python -c "from launchcontainers.prepare.my_prepare import MyPrepare; MyPrepare.write_example_config()"

----

Step 4 — Implement one method per prepare step
-----------------------------------------------

Name methods after what they produce, not after implementation details.
Each method should:

* accept ``sub`` and ``ses`` as its first arguments;
* accept an optional ``output_dir`` so tests can redirect output;
* return the paths (or a structured list) of everything written.

.. code-block:: python

   def gen_events_tsv(self, sub: str, ses: str, output_dir=None) -> list[str]:
       """Write one events.tsv per run and return their paths."""
       ...

   def gen_bold_symlinks(self, sub, ses, layout, output_dir=None) -> list[dict]:
       """Symlink bold NIfTIs with normalised names; return matched list."""
       ...

.. note::

   **Module-level helper functions** (pure utilities that do not need
   ``self``) may live outside the class at the top of the module.  This is
   fine — especially when the class is still small.  Note it in a comment so
   the next developer knows it is intentional and not forgotten refactoring.

----

Step 5 — Add a module-level wrapper function
---------------------------------------------

After the class definition, add a standalone function that ``do_prepare.py``
can call.  It instantiates the class, iterates over ``df_subses``, and calls
the step methods in order:

.. code-block:: python

   def run_my_prepare(lc_config: dict, df_subses, layout=None) -> bool:
       """
       Entry point called by do_prepare.main when container == 'MyPipeline'.
       """
       prep = MyPrepare(lc_config)
       for row in df_subses.itertuples():
           sub, ses = str(row.sub), str(row.ses)
           prep.gen_events_tsv(sub, ses)
           matched = prep.gen_bold_symlinks(sub, ses, layout)
           prep.gen_preprocessed_symlinks(sub, ses, matched)
       return True

This keeps ``do_prepare.py`` free of pipeline-specific logic — it only needs
to know the function name.

----

Step 6 — Register in ``do_prepare.py``
---------------------------------------

Add one import and one dispatch entry:

.. code-block:: python

   from launchcontainers.prepare.my_prepare import run_my_prepare

   # inside the dispatch block:
   elif container == "MyPipeline":
       run_my_prepare(lc_config, df_subses, layout=layout)

----

Summary — anatomy of a prepare module
---------------------------------------

.. code-block:: text

   my_prepare.py
   ├── module-level helper functions   ← pure utils, no self needed (note intentional)
   ├── class MyPrepare(BasePrepare)
   │   ├── __init__                    ← super().__init__ + pull container_specific sub-dict
   │   ├── @property …                 ← one per config key; path derivations too
   │   ├── _example_config_dict        ← full lc_config.yaml as a dict
   │   └── methods: gen_*              ← one per prepare step; return paths written
   └── run_my_prepare(…)               ← wrapper called by do_prepare.py

----

``BasePrepare`` API
--------------------

See :ref:`api_ref` for the full auto-generated documentation:

* :class:`launchcontainers.prepare.base_prepare.BasePrepare`
