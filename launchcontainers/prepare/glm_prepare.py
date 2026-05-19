# """
# MIT License
# Copyright (c) 2020-2025 Garikoitz Lerma-Usabiaga
# Copyright (c) 2020-2022 Mengxing Liu
# Copyright (c) 2022-2023 Leandro Lecca
# Copyright (c) 2022-2025 Yongning Lei
# Copyright (c) 2023 David Linhardt
# Copyright (c) 2023 Iñigo Tellaetxe
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software
# and associated documentation files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Software, and to permit persons to
# whom the Software is furnished to do so, subject to the following conditions:
# The above copyright notice and this permission notice shall be included in all copies or
# substantial portions of the Software.
# """
from __future__ import annotations

import csv
import glob
import json
import os
import os.path as op
import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import bids  # pybids — BIDSLayout lives here

from launchcontainers.log_setup import console
from launchcontainers.prepare.base_prepare import BasePrepare


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_events_from_stim_name(stim_basename: str, block_time: float):
    """
    Derive BIDS event columns from the stim-file name stored in a vistadisplog
    ``.mat`` file (``params.loadMatrix``).

    The stim basename is expected to follow the convention::

        <lgn>_<task>_duration-<N>.mat

    where:

    * ``lgn``  — language / condition prefix (e.g. ``en``, ``eu``, ``cb``)
    * ``task`` — task label (e.g. ``fixFF``, ``fixRW``, ``fixRWblock``, or any
      custom string).  If ``block`` appears anywhere in the task label the run
      is treated as a block design; otherwise it is treated as a single event
      spanning the whole run.
    * ``duration-<N>`` — total run duration in seconds.

    Parameters
    ----------
    stim_basename : str
        Basename of the stim ``.mat`` file, e.g.
        ``en_fixFF_duration-300.mat`` or ``eu_fixRWblock_duration-200.mat``.
    block_time : float
        Duration of each block epoch in seconds (default 10).  Only used when
        the task label contains ``block``.

    Returns
    -------
    tuple[list, list, list, str, str]
        ``(onset, duration, trial_type, lgn, task)``

    Raises
    ------
    ValueError
        If the ``duration-<N>`` token is missing from the stim basename.
    """
    parts = stim_basename.split("_")
    lgn = parts[0]
    raw_task = parts[1] if len(parts) > 1 else "unknown"

    # Warn and skip if 'fix' not in task — unexpected stim naming convention
    if "fix" not in raw_task:
        raise ValueError(
            f"'fix' not found in task label '{raw_task}' (stim: {stim_basename}). "
            "Skipping this run."
        )

    # Extract stimulus condition (RW / FF) from the raw task label before normalising
    if "RW" in raw_task:
        condition = "RW"
    elif "FF" in raw_task:
        condition = "FF"
    else:
        condition = raw_task.replace("fix", "").replace("block", "") or "unknown"

    # Normalise task name (used for the BIDS filename, not the trial_type)
    if "block" in raw_task:
        task = "WCblock"
    else:
        task = "WCnonestop"

    # Total run duration from "duration-<N>" token
    match = re.search(r"duration-(\d+)", stim_basename)
    if not match:
        raise ValueError(
            f"Could not find 'duration-<N>' token in stim name: {stim_basename}"
        )
    total_duration = int(match.group(1))

    if task == "WCblock":
        # Block design: alternating baseline / active epochs
        n_epochs = int(total_duration / block_time)
        onset = [float(block_time * i) for i in range(n_epochs)]
        duration = [float(block_time)] * n_epochs
        trial_type = [
            "baseline" if i % 2 == 0 else f"{lgn}_{condition}" for i in range(n_epochs)
        ]
    else:
        # Event design: single event spanning the whole run
        onset = [0.0]
        duration = [float(total_duration)]
        trial_type = [f"{lgn}_{condition}"]

    return onset, duration, trial_type, lgn, task


# ---------------------------------------------------------------------------
# GLMPrepare
# ---------------------------------------------------------------------------


class GLMPrepare(BasePrepare):
    """
    Prepare GLM inputs for one subject / session.

    The four key pieces this class manages:

    1. **events.tsv** — written to the BIDS ``func/`` directory by default;
       discoverable via a standard BIDS query.
    2. **Raw BIDS bold** — ``<bidsdir>/sub-X/ses-X/func/*_bold.nii.gz``;
       read by nilearn to build the first-level model.
    3. **Preprocessed bold** — ``<fmriprep_dir>/sub-X/ses-X/func/*_bold.nii.gz``;
       symlinked with normalised GLM task-run names.
    4. **Contrast config** — path to contrast YAML under ``container_specific.l1_surface``.

    Inherits :attr:`~launchcontainers.prepare.base_prepare.BasePrepare.basedir`,
    :attr:`~launchcontainers.prepare.base_prepare.BasePrepare.bidsdir`, and
    :meth:`~launchcontainers.prepare.base_prepare.BasePrepare.write_example_config`
    from :class:`~launchcontainers.prepare.base_prepare.BasePrepare`.

    Parameters
    ----------
    lc_config : dict or None
        Parsed launchcontainers YAML configuration.  Pass ``None`` to use
        :meth:`write_example_config` without needing a real config.
    """

    def __init__(self, lc_config: dict | None = None):
        super().__init__(lc_config)
        self._glm_cfg = self.lc_config.get("container_specific", {}).get(
            "l1_surface", {}
        )

    # ------------------------------------------------------------------
    # Convenience properties — from container_specific.l1_surface
    # ------------------------------------------------------------------
    @property
    def is_WC(self) -> bool:
        """Whether the GLM is run with the PRF Word Center case."""
        return self._glm_cfg["is_WC"]

    @property
    def output_bids(self) -> str | None:
        """Name of the output BIDS directory used in WC mode (e.g. ``BIDS_WC``).
        Only relevant when ``is_WC=True``."""
        return self._glm_cfg.get("output_bids")

    @property
    def output_bids_dir(self) -> str | None:
        """Absolute path to the output BIDS directory (``<basedir>/<output_bids>``).
        Returns ``None`` when ``output_bids`` is not set."""
        if self.output_bids is None:
            return None
        return op.join(self.basedir, self.output_bids)

    @property
    def block_time(self) -> float:
        """For the WC confition, the block_time is 10.0)."""
        return float(self._glm_cfg.get("block_time", 10.0))

    @property
    def fmriprep_analysis_name(self) -> str:
        """Analysis name of the fMRIprep run (used to locate the derivatives dir)."""
        return self._glm_cfg["fmriprep_analysis_name"]

    @property
    def fmriprep_dir(self) -> str:
        """Path to the fMRIprep derivatives directory, built from ``fmriprep_analysis_name``."""
        return op.join(self.bidsdir, "derivatives", self.fmriprep_analysis_name)

    @property
    def task(self) -> str | None:
        """Task name of the fMRI time series."""
        return self._glm_cfg.get("task")

    @property
    def start_scans(self) -> int:
        """Number of non-steady TRs to discard at the start of each run."""
        return int(self._glm_cfg.get("start_scans", 0))

    @property
    def space(self) -> str:
        """Output space for the GLM (e.g. T1w, MNI152NLin2009cAsym, fsnative)."""
        return self._glm_cfg.get("space", "fsnative")

    @property
    def contrast_yaml(self) -> str | None:
        """Path to the contrast YAML file. Returns ``None`` if not set."""
        return self._glm_cfg.get("contrast_yaml")

    @property
    def analysis_name(self) -> str:
        """Analysis name for the GLM results, taken from general.analysis_name."""
        return self.lc_config["general"]["analysis_name"]

    @property
    def slice_timing_ref(self) -> float:
        """Slice timing reference (fMRIprep default is 0.5)."""
        return float(self._glm_cfg.get("slice_timing_ref", 0.5))

    ##
    ## Below are addtional properties
    # if we only run on certain runs
    @property
    def selected_runs(self) -> list | None:
        """List of run numbers to use selectively. ``None`` means all runs."""
        return self._glm_cfg.get("selected_runs")

    # Below are the options for smoothing and masking — not needed for a basic GLM but useful for some cases
    @property
    def use_smoothed(self) -> bool:
        """Whether to use smoothed bold files."""
        return bool(self._glm_cfg.get("use_smoothed", False))

    @property
    def sm(self) -> str | None:
        """FreeSurfer FWHM smooth factor (e.g. '05')."""
        return self._glm_cfg.get("sm")

    @property
    def force(self) -> bool:
        """If ``True``, overwrite existing outputs (from ``general.force``)."""
        return bool(self.lc_config.get("general", {}).get("force", False))

    @property
    def dry_run(self) -> bool:
        """If ``True``, print actions without executing them."""
        return bool(self._glm_cfg.get("dry_run", False))

    @property
    def mask(self) -> str | None:
        """Label file name to apply as a mask (searched under BIDS/freesurfer)."""
        return self._glm_cfg.get("mask")

    # Below are the options for Power analysis
    @property
    def power_analysis(self) -> bool:
        """Whether to run power analysis mode (100 GLMs across 1-10 runs)."""
        return bool(self._glm_cfg.get("power_analysis", False))

    @property
    def n_iterations(self) -> int:
        """Number of random iterations per run count in power analysis mode."""
        return int(self._glm_cfg.get("n_iterations", 10))

    @property
    def seed(self) -> int:
        """Random seed for power analysis run generation."""
        return int(self._glm_cfg.get("seed", 42))

    @property
    def total_runs(self) -> int:
        """Total number of runs available for power analysis."""
        return int(self._glm_cfg.get("total_runs", 10))

    # ------------------------------------------------------------------
    # Example config — supplies content to BasePrepare.write_example_config
    # ------------------------------------------------------------------

    @classmethod
    def _example_config_dict(cls) -> dict:
        """Return the example config dict for the l1_surface pipeline."""
        return {
            "general": {
                "basedir": "/path/to/basedir",
                "bidsdir_name": "BIDS",
                "container": "l1_surface",
                "analysis_name": "glm_01",
                "host": "local",
                "force": True,
            },
            "container_specific": {
                "l1_surface": {
                    "is_WC": False,
                    "output_bids": "BIDS_WC",
                    "fmriprep_analysis_name": "fmriprep-25.1.4",
                    "task": None,
                    "start_scans": 5,
                    "space": "fsnative",
                    "contrast_yaml": "/path/to/contrast.yaml",
                    "slice_timing_ref": 0.5,
                    "use_smoothed": False,
                    "dry_run": False,
                    "sm": None,
                    "mask": None,
                    "selected_runs": None,
                    "power_analysis": False,
                    "n_iterations": 10,
                    "seed": 42,
                    "total_runs": 10,
                }
            },
            "host_options": {
                "local": {},
            },
        }

    # ------------------------------------------------------------------
    # Step 1.a — events.tsv from vistadisplog .mat files
    # ------------------------------------------------------------------

    def gen_events_tsv_vistadisplog(
        self,
        sub: str,
        ses: str,
        output_dir: str | None = None,
    ) -> list[str]:
        """
        Generate BIDS ``events.tsv`` files from vistadisplog ``.mat`` files.

        Looks for ``20*.mat`` files inside
        ``<bidsdir>/sourcedata/vistadisplog/sub-<sub>/ses-<ses>/``, sorts them
        in ascending filename order (= acquisition order), assigns run numbers
        starting from 1, reads ``params.loadMatrix`` from each to determine
        the task and condition, then writes one ``events.tsv`` per run.

        Parameters
        ----------
        sub : str
            Subject identifier without the ``sub-`` prefix.
        ses : str
            Session identifier without the ``ses-`` prefix.
        output_dir : str or None
            Directory to write ``events.tsv`` files into.  If ``None``, files
            are written to the standard BIDS location
            ``<bidsdir>/sourcedata/vistadisplog/sub-<sub>/ses-<ses>/func/``.

        Returns
        -------
        list[str]
            Paths to all ``events.tsv`` files written.

        Raises
        ------
        FileNotFoundError
            If no ``20*.mat`` files are found in the vistadisplog directory.
        ValueError
            If the stim-name duration token is missing or arrays are
            inconsistent.
        """
        from launchcontainers.prepare.prf_prepare import (
            PRFPrepare,
        )  # avoid circular import
        from scipy.io import loadmat  # lazy import — not needed at module level

        block_time = self.block_time

        vistadisplog_dir = op.join(
            self.bidsdir,
            "sourcedata",
            "vistadisplog",
            f"sub-{sub}",
            f"ses-{ses}",
        )
        console.print(
            f"\n### Searching vistadisplog dir: {vistadisplog_dir}", style="cyan"
        )

        mat_files = sorted(glob.glob(op.join(vistadisplog_dir, "20*.mat")))
        if not mat_files:
            raise FileNotFoundError(f"No '20*.mat' files found in {vistadisplog_dir}")
        console.print(
            f"Found {len(mat_files)} vistadisplog .mat file(s).", style="cyan"
        )

        if output_dir is not None:
            func_dir = output_dir
        else:
            func_dir = vistadisplog_dir
        os.makedirs(func_dir, exist_ok=True)

        # Write the mapping TSV (log → task_run → glm_task_run)
        console.print("\n### Generating mapping TSV via PRFPrepare...", style="cyan")
        PRFPrepare(self.lc_config).parse_prf_mat(sub, ses, lc_glm=True)

        written = []
        run_counters: dict[str, int] = {}  # per-task run counter
        for mat_file in mat_files:
            console.print(f"  {op.basename(mat_file)}", style="cyan")

            # Read params.loadMatrix to get the original stim filename
            params = loadmat(mat_file, simplify_cells=True)["params"]
            stim_path = params["loadMatrix"]
            stim_basename = op.basename(stim_path)
            console.print(f"    stim name: {stim_basename}", style="cyan")

            # Derive events from stim name
            try:
                onset, duration, trial_type, lgn, task = _parse_events_from_stim_name(
                    stim_basename, block_time=block_time
                )
            except ValueError as exc:
                console.print(f"    [WARNING] {exc}", style="yellow")
                continue
            console.print(f"    lgn={lgn}  task={task}", style="cyan")

            if not (len(onset) == len(duration) == len(trial_type)):
                raise ValueError(
                    f"Inconsistent event arrays for {mat_file}: "
                    f"onset={len(onset)}, duration={len(duration)}, "
                    f"trial_type={len(trial_type)}"
                )

            run_counters[task] = run_counters.get(task, 0) + 1
            run_num = run_counters[task]

            out_file = op.join(
                func_dir,
                f"sub-{sub}_ses-{ses}_task-{task}_run-{run_num:02d}_events.tsv",
            )
            with open(out_file, "w") as fh:
                fh.write("onset\tduration\ttrial_type\n")
                for o, d, t in zip(onset, duration, trial_type):
                    fh.write(f"{o}\t{d}\t{t}\n")

            console.print(f"    written → {out_file}", style="cyan")
            written.append(out_file)

        console.print(
            f"\n### events.tsv generation complete: {len(written)} file(s) written.",
            style="bold red",
        )
        return written

    # ------------------------------------------------------------------
    # Shared helper — mapping TSV
    # ------------------------------------------------------------------

    def _load_mapping_tsv(self, sub: str, ses: str, force: bool = False) -> list[dict]:
        """
        Load (or create) the vistadisplog mapping TSV for *sub* / *ses*.

        The TSV always lives in the BIDS ``func/`` directory.  If it does not
        exist yet (or *force* is ``True``) it is regenerated via
        :class:`~launchcontainers.prepare.prf_prepare.PRFPrepare`.

        Parameters
        ----------
        sub, ses : str
            Subject / session labels without prefix.
        force : bool
            When ``True``, delete and regenerate the TSV even if it exists.
            Useful during development / testing.

        Returns
        -------
        list[dict]
            Rows from the mapping TSV.
        """
        from launchcontainers.prepare.prf_prepare import PRFPrepare

        vistadisplog_dir = op.join(
            self.bidsdir, "sourcedata", "vistadisplog", f"sub-{sub}", f"ses-{ses}"
        )
        tsv_file = op.join(
            vistadisplog_dir, f"sub-{sub}_ses-{ses}_desc-mapping_PRF_acqtime.tsv"
        )
        if force and op.exists(tsv_file):
            os.remove(tsv_file)
            console.print(
                "  force=True — deleted existing mapping TSV.", style="yellow"
            )
        if not op.exists(tsv_file):
            console.print(
                "  Mapping TSV not found — generating via PRFPrepare...", style="cyan"
            )
            PRFPrepare(self.lc_config).parse_prf_mat(sub, ses, lc_glm=True)
        with open(tsv_file, newline="") as fh:
            mapping = list(csv.DictReader(fh, delimiter="\t"))
        console.print(
            f"\n### Loaded {len(mapping)} rows from mapping TSV.", style="cyan"
        )
        return mapping

    # ------------------------------------------------------------------
    # Step 1.b — symlink BIDS bold files (pyBIDS query + mapping TSV match)
    # ------------------------------------------------------------------

    def gen_bids_bold_symlinks(
        self,
        sub: str,
        ses: str,
        layout: bids.BIDSLayout,
        output_dir: str | None = None,
    ) -> list[dict]:
        """
        Use pyBIDS to find raw bold files, match them to the mapping TSV via
        ``AcquisitionTime``, and create symlinks with normalised GLM task-run
        names.

        Parameters
        ----------
        sub : str
            Subject identifier without the ``sub-`` prefix.
        ses : str
            Session identifier without the ``ses-`` prefix.
        layout : BIDSLayout
            Pre-loaded pyBIDS layout for the dataset.
        output_dir : str or None
            Destination directory for the symlinks.  Defaults to
            ``<bidsdir>/sub-<sub>/ses-<ses>/func/``.

        Returns
        -------
        list[dict]
            One entry per matched bold, each with keys:

            * ``bids_path``    — absolute path to the original BIDS bold
            * ``glm_task_run`` — normalised GLM task-run string, e.g.
              ``task-WCnonestop_run-01``
            * ``link_path``    — absolute path to the symlink that was created
        """
        from launchcontainers.utils import parse_hms, times_match

        mapping = self._load_mapping_tsv(sub, ses)
        bids_func = op.join(self.bidsdir, f"sub-{sub}", f"ses-{ses}", "func")
        func_dir = output_dir if output_dir is not None else bids_func
        os.makedirs(func_dir, exist_ok=True)

        bids_files = sorted(
            layout.get(
                subject=sub,
                session=ses,
                suffix="bold",
                extension=".nii.gz",
                return_type="file",
            )
        )
        bids_files = [
            f
            for f in bids_files
            if "task-fLoc" not in op.basename(f) and not op.islink(f)
        ]
        if not bids_files:
            console.print(
                f"  No non-fLoc BIDS bold files found for sub-{sub} ses-{ses}",
                style="yellow",
            )
            return []
        console.print(f"  Found {len(bids_files)} BIDS bold file(s).", style="cyan")

        matched_files = []
        for bold_file in bids_files:
            basename = op.basename(bold_file)

            json_file = bold_file.replace(".nii.gz", ".json")
            if not op.exists(json_file):
                console.print(
                    f"  [WARNING] JSON sidecar missing for {basename} — skipping.",
                    style="yellow",
                )
                continue
            with open(json_file) as fh:
                metadata = json.load(fh)
            bold_acq_time = parse_hms(metadata.get("AcquisitionTime", ""))

            matched_row = next(
                (
                    row
                    for row in mapping
                    if times_match(bold_acq_time, row["acq_time"], max_diff_sec=180)
                ),
                None,
            )
            if matched_row is None:
                console.print(
                    f"  [WARNING] No TSV match for {basename} "
                    f"(acq_time={bold_acq_time}) — skipping.",
                    style="yellow",
                )
                continue

            glm_task_run = matched_row["glm_task_run"]
            if glm_task_run == "N/A":
                console.print(
                    f"  [WARNING] glm_task_run is N/A for {basename} — skipping.",
                    style="yellow",
                )
                continue

            new_basename = re.sub(r"task-\w+_run-\d+", glm_task_run, basename)
            link_path = op.join(func_dir, new_basename)
            if op.islink(link_path) or op.exists(link_path):
                os.remove(link_path)
            os.symlink(bold_file, link_path)

            # Also symlink the JSON sidecar
            new_json_basename = new_basename.replace(".nii.gz", ".json")
            json_link_path = op.join(func_dir, new_json_basename)
            if op.islink(json_link_path) or op.exists(json_link_path):
                os.remove(json_link_path)
            os.symlink(json_file, json_link_path)

            # Extract the original task-X_run-N token from the bold filename
            task_run_match = re.search(r"task-\w+_run-\d+", basename)
            orig_task_run = task_run_match.group(0) if task_run_match else None

            matched_files.append(
                {
                    "bids_path": bold_file,
                    "task_run": orig_task_run,
                    "glm_task_run": glm_task_run,
                    "link_path": link_path,
                }
            )
            console.print(
                f"  {basename}\n    → {new_basename}  (acq={bold_acq_time})",
                style="cyan",
            )

        console.print(
            f"\n### BIDS bold symlinks complete: {len(matched_files)} created.",
            style="bold red",
        )
        return matched_files

    # ------------------------------------------------------------------
    # Step 1.c — symlink fMRIprep bold (derived from BIDS matched list)
    # ------------------------------------------------------------------

    def gen_fmriprep_bold_symlinks(
        self,
        sub: str,
        ses: str,
        bids_matched: list[dict],
        output_dir: str | None = None,
    ) -> list[str]:
        """
        Create fMRIprep bold symlinks using the same run list resolved by
        :meth:`gen_bids_bold_symlinks`.

        For each entry in *bids_matched* the method locates **all** files in the
        fMRIprep ``func/`` directory that match the original ``task-<x>_run-<n>``
        token (all spaces, hemispheres, and suffixes) and creates a symlink for
        each one with the normalised ``glm_task_run`` name.  This ensures that
        both fsnative (``.func.gii``) and volumetric (``space-T1w .nii.gz``)
        files are available under the WC task name, as required by both the
        surface GLM and ``first_level_from_bids`` (which needs a T1w file to
        extract TR and events).

        Parameters
        ----------
        sub : str
            Subject identifier without the ``sub-`` prefix.
        ses : str
            Session identifier without the ``ses-`` prefix.
        bids_matched : list[dict]
            Output of :meth:`gen_bids_bold_symlinks` — list of dicts with keys
            ``bids_path``, ``task_run`` (original, e.g. ``task-fixRW_run-01``),
            and ``glm_task_run``.
        output_dir : str or None
            Destination directory for the symlinks.  Defaults to
            ``<fmriprep_dir>/sub-<sub>/ses-<ses>/func/``.

        Returns
        -------
        list[str]
            Absolute paths to all symlinks created.
        """
        fmriprep_func = op.join(self.fmriprep_dir, f"sub-{sub}", f"ses-{ses}", "func")
        func_dir = output_dir if output_dir is not None else fmriprep_func
        os.makedirs(func_dir, exist_ok=True)

        symlinks = []
        for entry in bids_matched:
            glm_task_run = entry["glm_task_run"]
            orig_task_run = entry["task_run"]

            if not orig_task_run:
                console.print(
                    f"  [WARNING] Missing task_run for {op.basename(entry['bids_path'])} — skipping.",
                    style="yellow",
                )
                continue

            task_run_match = re.search(r"task-(\w+)_run-(\d+)", orig_task_run)
            task, run = task_run_match.group(1), task_run_match.group(2)

            # Collect ALL files for this task/run — all spaces, hemis, and
            # suffixes (.func.gii, .nii.gz, .json).  run_glm.py uses
            # first_level_from_bids with space_label="T1w" to extract TR/events,
            # so both fsnative and T1w files must be symlinked with WC names.
            candidates = glob.glob(
                op.join(fmriprep_func, f"sub-{sub}_ses-{ses}_task-{task}_run-{run}*")
            )
            # Exclude files that are already symlinks (from a prior prepare run)
            candidates = [c for c in candidates if not op.islink(c)]

            if not candidates:
                console.print(
                    f"  [WARNING] No fMRIprep files for task-{task}_run-{run} — skipping.",
                    style="yellow",
                )
                continue

            for fp_bold in candidates:
                fp_basename = op.basename(fp_bold)

                new_basename = re.sub(r"task-\w+_run-\d+", glm_task_run, fp_basename)
                link_path = op.join(func_dir, new_basename)
                if op.islink(link_path) or op.exists(link_path):
                    os.remove(link_path)
                os.symlink(fp_bold, link_path)
                symlinks.append(link_path)

                # Also symlink the JSON sidecar if it exists
                # JSON sidecar: derive from bold path by replacing bold suffix
                for bold_suffix in (".func.gii", ".nii.gz"):
                    if fp_bold.endswith(bold_suffix):
                        fp_json = fp_bold[: -len(bold_suffix)] + ".json"
                        if op.exists(fp_json):
                            new_json_basename = (
                                new_basename[: -len(bold_suffix)] + ".json"
                            )
                            json_link_path = op.join(func_dir, new_json_basename)
                            if op.islink(json_link_path) or op.exists(json_link_path):
                                os.remove(json_link_path)
                            os.symlink(fp_json, json_link_path)
                        break

                console.print(
                    f"  {fp_basename}\n    → {new_basename}",
                    style="cyan",
                )

        console.print(
            f"\n### fMRIprep bold symlinks complete: {len(symlinks)} created.",
            style="bold red",
        )
        return symlinks


# ---------------------------------------------------------------------------
# Batch entry point
# ---------------------------------------------------------------------------


def run_prf_glm_prepare(
    lc_config: dict, df_subses: list[tuple[str, str]], layout: bids.BIDSLayout
) -> bool:
    """
    Run the full PRF-GLM preparation pipeline for every subject/session.

    Called by :func:`launchcontainers.do_prepare.main` when
    ``general.container`` is ``fMRI-prfGLM``.

    Calls, in order, for each row in *df_subses*:

    1. :meth:`GLMPrepare.gen_events_tsv_vistadisplog` — writes ``events.tsv``
       files to the BIDS ``func/`` directory.
    2. :meth:`GLMPrepare.gen_bids_bold_symlinks` — queries pyBIDS, matches
       bold files to the mapping TSV, and symlinks them with GLM task-run names
       into the BIDS ``func/`` directory.
    3. :meth:`GLMPrepare.gen_fmriprep_bold_symlinks` — uses the matched list
       from step 2 to locate the corresponding fMRIprep preprocessed bold files
       and symlink them with the same GLM task-run names.

    Parameters
    ----------
    lc_config : dict
        Parsed launchcontainers YAML configuration.
    df_subses : list[tuple[str, str]]
        Subject/session pairs filtered to rows where ``RUN == True``.

    Returns
    -------
    bool
        ``True`` when all subjects/sessions complete without raising.
    """
    glm = GLMPrepare(lc_config)

    # ------------------------------------------------------------------
    # One-time setup: seed output_bids_dir with BIDS root files so that
    # pyBIDS and other tools recognise it as a valid BIDS dataset.
    # ------------------------------------------------------------------
    if glm.output_bids_dir is not None:
        import shutil

        os.makedirs(glm.output_bids_dir, exist_ok=True)

        # Copy dataset_description.json (required by BIDS)
        src_desc = op.join(glm.bidsdir, "dataset_description.json")
        dst_desc = op.join(glm.output_bids_dir, "dataset_description.json")
        if op.exists(src_desc) and not op.exists(dst_desc):
            shutil.copy2(src_desc, dst_desc)
            console.print(
                f"  Copied dataset_description.json → {dst_desc}", style="cyan"
            )

        # Symlink README (any extension: README, README.md, README.txt, …)
        for fname in os.listdir(glm.bidsdir):
            if fname.upper().startswith("README"):
                src_readme = op.join(glm.bidsdir, fname)
                dst_readme = op.join(glm.output_bids_dir, fname)
                if not op.islink(dst_readme) and not op.exists(dst_readme):
                    os.symlink(src_readme, dst_readme)
                    console.print(f"  Symlinked {fname} → {dst_readme}", style="cyan")

    for sub, ses in df_subses:
        console.print(f"\n### PRF-GLM prepare  sub-{sub}  ses-{ses}", style="bold cyan")

        # Resolve output dirs: when output_bids_dir is set, mirror the full
        # BIDS + derivatives tree under it; otherwise fall back to defaults.
        if glm.output_bids_dir is not None:
            out_func = op.join(glm.output_bids_dir, f"sub-{sub}", f"ses-{ses}", "func")
            out_fmriprep_func = op.join(
                glm.output_bids_dir,
                "derivatives",
                glm.fmriprep_analysis_name,
                f"sub-{sub}",
                f"ses-{ses}",
                "func",
            )
            console.print(
                f"  output_bids active\n"
                f"    BIDS func    → {out_func}\n"
                f"    fMRIprep func → {out_fmriprep_func}",
                style="cyan",
            )
        else:
            out_func = None  # each method falls back to its own default
            out_fmriprep_func = None

        # If force, delete existing mapping TSV so it is regenerated fresh
        if glm.force:
            mapping_tsv = op.join(
                glm.bidsdir,
                "sourcedata",
                "vistadisplog",
                f"sub-{sub}",
                f"ses-{ses}",
                f"sub-{sub}_ses-{ses}_desc-mapping_PRF_acqtime.tsv",
            )
            if op.exists(mapping_tsv):
                os.remove(mapping_tsv)
                console.print(
                    f"  force=True — deleted existing mapping TSV: {mapping_tsv}",
                    style="yellow",
                )

        # 1. Generate events.tsv
        glm.gen_events_tsv_vistadisplog(sub, ses, output_dir=out_func)

        # 2. Symlink BIDS bold (+ JSON) → output func dir
        bids_matched = glm.gen_bids_bold_symlinks(sub, ses, layout, output_dir=out_func)

        # 3. Symlink fMRIprep bold (+ JSON) → output derivatives func dir
        glm.gen_fmriprep_bold_symlinks(
            sub, ses, bids_matched, output_dir=out_fmriprep_func
        )

    return True


def run_glm_prepare(
    lc_config: dict | None = None,
    df_subses: list[tuple[str, str]] | None = None,
    layout: bids.BIDSLayout | None = None,
) -> bool:
    """
    Run the full GLM preparation pipeline for every subject/session.

    Called by :func:`launchcontainers.do_prepare.main` when
    ``general.container`` is ``l1_surface``.

    If called with no arguments, writes an example config YAML to the current
    working directory and returns ``False``.

    Parameters
    ----------
    lc_config : dict or None
        Parsed launchcontainers YAML configuration.
    df_subses : list[tuple[str, str]] or None
        Subject/session pairs filtered to rows where ``RUN == True``.
    layout : BIDSLayout or None
        Pre-loaded BIDS layout for the dataset.

    Returns
    -------
    bool
        ``True`` when all subjects/sessions complete without raising.
        ``False`` when called with no arguments (example config written).
    """
    glm = GLMPrepare(lc_config)
    if glm.is_WC:
        console.print(
            "\n### Running GLM prepare in PRF Word Center mode (is_WC=True)",
            style="bold cyan",
        )
        run_prf_glm_prepare(lc_config, df_subses, layout=layout)
    else:
        console.print(
            "\n### Running GLM prepare in standard mode (is_WC=False)",
            style="bold cyan",
        )

    return True
