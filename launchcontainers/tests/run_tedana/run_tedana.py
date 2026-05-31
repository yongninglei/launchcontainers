#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
run_tedana.py

Run tedana multi-echo ICA on fMRIprep per-echo T1w nifti outputs.

Strategy
--------
fMRIprep with ``--me-output-echos`` writes individual echoes in T1w space
(no space entity, implied T1w).  This script:

  1. Discovers ``…_acq-ME_run-N_echo-M_desc-preproc_bold.nii.gz`` per run.
  2. Reads echo times from fmriprep derivative JSONs (fallback: raw BIDS).
  3. Uses the fmriprep T1w brain mask (``space-T1w_desc-brain_mask``).
  4. Runs ``tedana_workflow`` directly on the real niftis.
  5. Copies all tedana outputs (nifti, tsv, html) with a BIDS prefix.

Use ``project_to_spaces.py`` to project the denoised output to MNI,
fsnative (gifti), or fsaverage.

Expected fMRIprep echo file pattern::

    sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{run}_echo-{N}_desc-preproc_bold.nii.gz

Output (in derivatives/{out_dir}/analysis-{name}/sub-{sub}/ses-{ses}/func/)::

    sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{run}_desc-optcomDenoised_bold.nii.gz
    sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{run}_desc-optcom_bold.nii.gz
    sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{run}_desc-tedana_mixing.tsv
    sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{run}_desc-tedana_metrics.tsv

Usage
-----
Single sub-session::

    python run_tedana.py \\
        -b /data/BIDS \\
        -fp /data/BIDS/derivatives/fmriprep-25.1.4 \\
        -o tedana-26.0.3 \\
        -n pilot01 \\
        -s pilot02,01 \\
        --tasks BfLocVideo

Batch (one "sub,ses" per line in file)::

    python run_tedana.py \\
        -b /data/BIDS \\
        -fp /data/BIDS/derivatives/fmriprep-25.1.4 \\
        -o tedana-26.0.3 \\
        -n main \\
        -f subseslist.txt \\
        --tasks BfLocVideo
"""

from __future__ import annotations

import glob
import json
import os
import os.path as op
import re
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _fmt_time(seconds: float) -> str:
    """Format elapsed seconds as a human-readable string (e.g. '1m 23.4s')."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(seconds, 60)
    if m < 60:
        return f"{int(m)}m {s:.1f}s"
    h, m = divmod(m, 60)
    return f"{int(h)}h {int(m)}m {s:.1f}s"


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------


def _glob_echo_niftis(
    func_dir: str, sub: str, ses: str, task: str, acq: str, run: str | None
) -> list[str]:
    """Return sorted per-echo T1w nifti paths for one run.

    fMRIprep with --me-output-echos writes individual echoes as
    ``…_acq-ME_run-N_echo-M_desc-preproc_bold.nii.gz`` (no space entity,
    implied T1w space).
    """
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""
    stem = (
        f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}"
        f"_echo-*_desc-preproc_bold.nii.gz"
    )
    return sorted(glob.glob(op.join(func_dir, stem)))


def _find_runs(
    func_dir: str, sub: str, ses: str, task: str, acq: str
) -> list[str | None]:
    """Return sorted run labels that have ME echo niftis (or [None] if no run entity)."""
    acq_token = f"_acq-{acq}" if acq else ""
    pattern = (
        f"sub-{sub}_ses-{ses}_task-{task}{acq_token}"
        f"_run-*_echo-*_desc-preproc_bold.nii.gz"
    )
    found = glob.glob(op.join(func_dir, pattern))

    runs: set[str] = set()
    for f in found:
        m = re.search(r"_run-(\w+)[_.]", op.basename(f))
        if m:
            runs.add(m.group(1))
    if runs:
        return sorted(runs)

    # No run entity — check for single-run data
    no_run = glob.glob(
        op.join(
            func_dir,
            f"sub-{sub}_ses-{ses}_task-{task}{acq_token}_echo-*_desc-preproc_bold.nii.gz",
        )
    )
    return [None] if no_run else []


def _get_echo_times_s(
    fp_func_dir: str,
    bids_dir: str,
    sub: str,
    ses: str,
    task: str,
    acq: str,
    run: str | None,
) -> list[float]:
    """Get echo times in seconds — tries fmriprep derivative JSONs first, then raw BIDS.

    fmriprep inherits EchoTime from raw BIDS into its derivative JSON sidecars,
    so the derivative JSONs are the primary source.
    """
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""

    def _read_tes(json_paths: list[str]) -> list[float] | None:
        tes: list[float] = []
        for jf in sorted(json_paths):
            with open(jf) as fh:
                meta = json.load(fh)
            et = meta.get("EchoTime")
            if et is None:
                return None
            tes.append(float(et))
        return tes if tes else None

    # 1. fmriprep derivative JSONs
    deriv_pat = op.join(
        fp_func_dir,
        f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}_echo-*_desc-preproc_bold.json",
    )
    deriv_jsons = glob.glob(deriv_pat)
    if deriv_jsons:
        tes = _read_tes(deriv_jsons)
        if tes:
            return tes

    # 2. Raw BIDS JSON sidecars
    raw_pat = op.join(
        bids_dir,
        f"sub-{sub}",
        f"ses-{ses}",
        "func",
        f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}_echo-*_bold.json",
    )
    raw_jsons = glob.glob(raw_pat)
    if not raw_jsons:
        raise FileNotFoundError(
            f"EchoTime not found in fmriprep derivatives and no raw BIDS JSONs at: {raw_pat}"
        )
    tes = _read_tes(raw_jsons)
    if tes is None:
        raise KeyError(f"EchoTime missing in raw BIDS JSONs: {raw_pat}")
    return tes


def _find_t1w_mask(
    func_dir: str, sub: str, ses: str, task: str, acq: str, run: str | None
) -> str | None:
    """Return path to fmriprep T1w brain mask for this run, or None if absent."""
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""
    pat = op.join(
        func_dir,
        f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}_space-T1w_desc-brain_mask.nii.gz",
    )
    files = glob.glob(pat)
    return files[0] if files else None


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------


def _bids_prefix(sub: str, ses: str, task: str, acq: str, run: str | None) -> str:
    parts = [f"sub-{sub}", f"ses-{ses}", f"task-{task}"]
    if acq:
        parts.append(f"acq-{acq}")
    if run:
        parts.append(f"run-{run}")
    return "_".join(parts) + "_"


def _process_run(
    sub: str,
    ses: str,
    task: str,
    run: str | None,
    fp_func_dir: str,
    bids_dir: str,
    out_func_dir: str,
    acq: str,
    fittype: str,
    n_threads: int,
    overwrite: bool,
    dry_run: bool,
) -> tuple[bool, float]:
    """Process one task/run: T1w echo niftis → tedana → denoised nifti outputs.

    fMRIprep only projects the optimal combination to surface spaces; individual
    echoes exist only as volumetric niftis (acq-ME, no space entity = T1w space).
    tedana is therefore run directly on those niftis — no fake-nifti conversion
    is needed.

    Returns (success, elapsed_seconds).
    """
    t_run_start = time.perf_counter()
    run_label = f"run-{run}" if run else "(no run entity)"
    console.print(f"\n  [bold]task-{task}  {run_label}[/bold]")

    # ── Discover echo niftis ───────────────────────────────────────────────
    echo_files = _glob_echo_niftis(fp_func_dir, sub, ses, task, acq, run)
    if not echo_files:
        console.print(
            f"    [red]✗ No echo niftis found "
            f"(acq-{acq}, pattern: …_echo-*_desc-preproc_bold.nii.gz)[/red]"
        )
        return False, time.perf_counter() - t_run_start

    n_echoes = len(echo_files)
    console.print(f"    Echoes: {n_echoes}")
    for f in echo_files:
        console.print(f"      {op.basename(f)}")

    # ── Get echo times ─────────────────────────────────────────────────────
    try:
        tes = _get_echo_times_s(fp_func_dir, bids_dir, sub, ses, task, acq, run)
    except (FileNotFoundError, KeyError) as exc:
        console.print(f"    [red]✗ {exc}[/red]")
        return False, time.perf_counter() - t_run_start

    if len(tes) != n_echoes:
        console.print(
            f"    [red]✗ Echo time count ({len(tes)}) ≠ echo file count ({n_echoes})[/red]"
        )
        return False, time.perf_counter() - t_run_start

    console.print(f"    Echo times (s): {tes}")

    # ── Mask ───────────────────────────────────────────────────────────────
    # echo-N_desc-preproc_bold.nii.gz is in native BOLD space (oblique affine).
    # The fmriprep space-T1w_desc-brain_mask is on a different T1w grid and
    # cannot be passed to tedana. Let tedana build its own adaptive mask via
    # dropout detection across echoes, which operates in the correct BOLD space.
    console.print("    Mask: tedana adaptive (dropout across echoes)")

    # ── Skip if already done ───────────────────────────────────────────────
    prefix = _bids_prefix(sub, ses, task, acq, run)
    out_denoised = op.join(out_func_dir, f"{prefix}desc-optcomDenoised_bold.nii.gz")

    if op.exists(out_denoised) and not overwrite:
        console.print(
            "    [yellow]→ output exists, skipping (use --overwrite to redo)[/yellow]"
        )
        return True, time.perf_counter() - t_run_start

    if dry_run:
        console.print(
            f"    [yellow][dry-run] would run tedana on {n_echoes} echoes "
            f"(fittype={fittype})[/yellow]"
        )
        console.print(f"    [yellow][dry-run] output → {out_func_dir}[/yellow]")
        return True, time.perf_counter() - t_run_start

    # ── Run tedana ─────────────────────────────────────────────────────────
    # tedana writes to a persistent work dir inside out_func_dir so that a
    # crash in the copy step doesn't force a full ICA re-run.
    # The work dir is removed only after a successful copy.
    work_dir = op.join(out_func_dir, f".tedana_work_{prefix.rstrip('_')}")
    os.makedirs(work_dir, exist_ok=True)
    tmpdir = work_dir
    if True:  # keeps indentation consistent with the old context-manager block
        console.print("    Running tedana (this may take several minutes)...")
        t0 = time.perf_counter()
        try:
            from tedana.workflows import tedana_workflow

            tedana_workflow(
                data=echo_files,
                tes=tes,
                out_dir=tmpdir,
                mask=None,
                masktype=["dropout"],
                fittype=fittype,
                convention="bids",
                prefix=prefix,
                n_threads=n_threads,
                overwrite=True,
                quiet=True,
            )
        except Exception as exc:
            console.print(f"    [red]✗ tedana error: {exc}[/red]")
            return False, time.perf_counter() - t_run_start
        t_tedana = time.perf_counter() - t0
        console.print(f"    tedana finished  [dim]({_fmt_time(t_tedana)})[/dim]")

        # ── Copy outputs ───────────────────────────────────────────────────
        os.makedirs(out_func_dir, exist_ok=True)
        t0 = time.perf_counter()

        # Verify the key output exists before copying anything
        # tedana 26.x renamed: desc-optcomDenoised → desc-denoised
        denoised_candidates = glob.glob(
            op.join(tmpdir, "*desc-denoised_bold.nii.gz")
        ) or glob.glob(op.join(tmpdir, "*desc-optcomDenoised_bold.nii.gz"))
        if not denoised_candidates:
            console.print(f"    [red]✗ denoised output not found in {tmpdir}[/red]")
            console.print(f"      tedana files: {os.listdir(tmpdir)}")
            return False, time.perf_counter() - t_run_start

        n_copied = 0
        for fname in os.listdir(tmpdir):
            src = op.join(tmpdir, fname)
            dst = op.join(out_func_dir, fname)
            if op.isdir(src):
                # e.g. figures/ — contains carpet plot images referenced by the HTML
                shutil.copytree(src, dst, dirs_exist_ok=True)
                console.print(f"    [green]✓[/green] [dim]{fname}/[/dim]")
                n_copied += 1
            elif fname.endswith((".nii.gz", ".tsv", ".json", ".html", ".csv")):
                shutil.copy2(src, dst)
                console.print(f"    [green]✓[/green] [dim]{fname}[/dim]")
                n_copied += 1

        t_save = time.perf_counter() - t0
        console.print(f"    {n_copied} files saved  [dim]({_fmt_time(t_save)})[/dim]")

        # Remove work dir only after successful copy
        shutil.rmtree(work_dir, ignore_errors=True)

    t_total = time.perf_counter() - t_run_start
    console.print(
        f"    [bold]Run total: {_fmt_time(t_total)}[/bold]  "
        f"[dim](tedana {_fmt_time(t_tedana)} + save {_fmt_time(t_save)})[/dim]"
    )
    return True, t_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    bids_dir: str = typer.Option(..., "-b", help="BIDS root directory"),
    fp_dir: str = typer.Option(
        ..., "-fp", help="fMRIprep derivatives directory (absolute path)"
    ),
    out_dir_name: str = typer.Option(
        ..., "-o", help="Output derivatives dir name under bids_dir/derivatives/"
    ),
    analysis_name: str = typer.Option(
        ...,
        "-n",
        "--analysis-name",
        help="Analysis label → derivatives/{out_dir}/analysis-{name}/",
    ),
    single: Optional[str] = typer.Option(
        None, "-s", help="Single subject-session pair: sub,ses  e.g.  01,T01"
    ),
    batch_file: Optional[str] = typer.Option(
        None, "-f", help="Batch file: one 'sub,ses' per line"
    ),
    tasks: str = typer.Option(
        ...,
        "--tasks",
        "-t",
        help="Comma-separated task names  e.g. BfLocVideo,IRAKEINU",
    ),
    acq: str = typer.Option(
        "ME", "--acq", help="Acquisition label for multi-echo runs (acq-ME)"
    ),
    fittype: str = typer.Option(
        "curvefit", "--fittype", help="tedana fit type: loglin | curvefit"
    ),
    n_threads: int = typer.Option(
        1, "--n-threads", "-j", help="Parallel threads per tedana run (internal)"
    ),
    n_run_jobs: int = typer.Option(
        1,
        "--n-run-jobs",
        "-J",
        help="Runs to process in parallel (1=sequential, -1=all)",
    ),
    overwrite: bool = typer.Option(
        False, "--overwrite", help="Overwrite existing outputs"
    ),
    dry_run: bool = typer.Option(False, "--dry-run", help="Print plan without running"),
) -> None:
    """Run tedana on fMRIprep per-echo T1w niftis (acq-ME, no space entity)."""

    # ── Resolve sub/ses list ───────────────────────────────────────────────
    subses_pairs: list[tuple[str, str]] = []
    if single:
        parts = single.split(",")
        if len(parts) != 2:
            console.print("[red]✗ -s expects 'sub,ses'  e.g.  01,T01[/red]")
            raise typer.Exit(1)
        subses_pairs.append((parts[0].strip(), parts[1].strip()))
    elif batch_file:
        if not op.isfile(batch_file):
            console.print(f"[red]✗ batch file not found: {batch_file}[/red]")
            raise typer.Exit(1)
        with open(batch_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split(",")
                if len(parts) < 2:
                    parts = line.split()
                if len(parts) < 2:
                    console.print(
                        f"[yellow]⚠ skipping malformed line: {line!r}[/yellow]"
                    )
                    continue
                subses_pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        console.print("[red]✗ provide -s sub,ses  or  -f batch_file[/red]")
        raise typer.Exit(1)

    task_list = [t.strip() for t in tasks.split(",") if t.strip()]
    out_root = op.join(
        bids_dir, "derivatives", out_dir_name, f"analysis-{analysis_name}"
    )

    # ── Summary table ──────────────────────────────────────────────────────
    tbl = Table(title="tedana fsnative workflow", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("bids_dir", bids_dir)
    tbl.add_row("fp_dir", fp_dir)
    tbl.add_row("out_dir", out_root)
    tbl.add_row("analysis_name", analysis_name)
    tbl.add_row("tasks", ", ".join(task_list))
    tbl.add_row("acq", acq)
    tbl.add_row("fittype", fittype)
    tbl.add_row("n_threads", str(n_threads))
    tbl.add_row("n_run_jobs", str(n_run_jobs) + ("  (all)" if n_run_jobs == -1 else ""))
    tbl.add_row("subjects", str(len(subses_pairs)))
    tbl.add_row("dry_run", str(dry_run))
    console.print(tbl)

    # ── Main loop ──────────────────────────────────────────────────────────
    results: list[tuple[str, str, str, str | None, bool, float]] = []
    t_total_start = time.perf_counter()

    for sub, ses in subses_pairs:
        t_subses_start = time.perf_counter()
        console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/bold cyan]")
        fp_func_dir = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "func")
        out_func_dir = op.join(out_root, f"sub-{sub}", f"ses-{ses}", "func")

        if not op.isdir(fp_func_dir):
            console.print(f"  [red]✗ fmriprep func dir not found: {fp_func_dir}[/red]")
            for task in task_list:
                results.append((sub, ses, task, None, False, 0.0))
            continue

        for task in task_list:
            runs = _find_runs(fp_func_dir, sub, ses, task, acq)
            if not runs:
                console.print(
                    f"  [yellow]⚠ task-{task}: no acq-{acq} echo niftis found[/yellow]"
                )
                results.append((sub, ses, task, None, False, 0.0))
                continue

            run_kwargs = [
                dict(
                    sub=sub,
                    ses=ses,
                    task=task,
                    run=run,
                    fp_func_dir=fp_func_dir,
                    bids_dir=bids_dir,
                    out_func_dir=out_func_dir,
                    acq=acq,
                    fittype=fittype,
                    n_threads=n_threads,
                    overwrite=overwrite,
                    dry_run=dry_run,
                )
                for run in runs
            ]

            workers = len(runs) if n_run_jobs == -1 else n_run_jobs
            if workers == 1 or len(runs) == 1:
                # Sequential — preserves clean ordered console output
                for kw in run_kwargs:
                    ok, elapsed = _process_run(**kw)
                    results.append((sub, ses, task, kw["run"], ok, elapsed))
            else:
                # Parallel across runs — output from each worker will be interleaved
                console.print(
                    f"  [dim]Launching {len(runs)} run(s) across {min(workers, len(runs))} worker(s) …[/dim]"
                )
                with ProcessPoolExecutor(max_workers=min(workers, len(runs))) as pool:
                    future_to_run = {
                        pool.submit(_process_run, **kw): kw["run"] for kw in run_kwargs
                    }
                    for future in as_completed(future_to_run):
                        run = future_to_run[future]
                        try:
                            ok, elapsed = future.result()
                        except Exception as exc:
                            console.print(f"  [red]✗ run-{run} raised: {exc}[/red]")
                            ok, elapsed = False, 0.0
                        results.append((sub, ses, task, run, ok, elapsed))

        t_subses = time.perf_counter() - t_subses_start
        console.print(
            f"\n  [cyan]sub-{sub} ses-{ses} total: {_fmt_time(t_subses)}[/cyan]"
        )

    t_total = time.perf_counter() - t_total_start

    # ── Results summary ────────────────────────────────────────────────────
    console.print("\n")
    summary = Table(title="Results", show_lines=False)
    summary.add_column("sub")
    summary.add_column("ses")
    summary.add_column("task")
    summary.add_column("run")
    summary.add_column("status")
    summary.add_column("wall time", justify="right")
    for sub, ses, task, run, ok, elapsed in results:
        status = "[green]✓ done[/green]" if ok else "[red]✗ failed[/red]"
        t_str = _fmt_time(elapsed) if elapsed > 0 else "—"
        summary.add_row(sub, ses, task, run or "—", status, t_str)
    console.print(summary)
    console.print(f"Total wall time: [bold]{_fmt_time(t_total)}[/bold]")

    n_failed = sum(1 for *_, ok, _e in results if not ok)
    if n_failed:
        console.print(f"\n[red]{n_failed} run(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("\n[green]All runs completed successfully.[/green]")


if __name__ == "__main__":
    app()
