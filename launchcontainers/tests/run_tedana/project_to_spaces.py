#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
project_to_spaces.py

Project tedana outputs (native BOLD space) to other spaces using fMRIprep
transforms and system tools (ANTs + FreeSurfer loaded via environment modules).

Transform chain
---------------
  BOLD → MNI      : antsApplyTransforms  boldref_to_T1w.txt + T1w_to_MNI.h5
  BOLD → fsnative : bbregister --bold --force-ras (native boldref → FreeSurfer T1)
                    + mri_vol2surf --reg <bbr.reg> --surf midthickness
  BOLD → fsaverage: same bbregister step
                    + mri_vol2surf --reg <bbr.reg> --surf white --trgsubject fsaverage

Outputs are written back into the same sub/ses/func dir as the tedana inputs.
A ``project_to_spaces_sources.json`` is saved in the analysis root.

Usage
-----
Dry-run (print plan, no execution)::

    python project_to_spaces.py \\
        -i .../tedana-26.0.3/analysis-default_ica \\
        -fp .../fmriprep-25.1.4 \\
        --fs-subjects-dir .../derivatives/freesurfer \\
        -s pilot02,01 --tasks BfLocVideo

Execute::

    python project_to_spaces.py ... --execute
"""

from __future__ import annotations

import datetime
import glob
import json
import os.path as op
import re
import subprocess
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

VALID_SPACES = {"MNI", "fsnative", "fsaverage"}
MNI_SPACE = "MNI152NLin2009cAsym"

# Locations where Environment Modules initialisation script may live
_MODULE_INITS = [
    "/usr/share/Modules/init/bash",
    "/etc/profile.d/modules.sh",
    "/usr/local/Modules/init/bash",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    return (
        f"{int(m)}m {s:.1f}s" if m < 60 else f"{int(m // 60)}h {int(m % 60)}m {s:.1f}s"
    )


def _bids_prefix(sub, ses, task, acq, run) -> str:
    parts = [f"sub-{sub}", f"ses-{ses}", f"task-{task}"]
    if acq:
        parts.append(f"acq-{acq}")
    if run:
        parts.append(f"run-{run}")
    return "_".join(parts) + "_"


def _run_with_modules(cmd_str: str, modules: list[str]) -> None:
    """Run a shell command after loading the given environment modules."""
    # Source whichever module init file exists on this system
    init_src = " || ".join(
        f'{{ [ -f "{p}" ] && source "{p}"; }}' for p in _MODULE_INITS
    )
    module_loads = " && ".join(f"module load {m}" for m in modules)
    full = f"({init_src}) 2>/dev/null; {module_loads} && {cmd_str}"
    result = subprocess.run(
        full, shell=True, executable="/bin/bash", capture_output=True
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Command failed (exit {result.returncode})\n"
            f"CMD:    {cmd_str}\n"
            f"STDERR: {result.stderr.decode()}"
        )


# ---------------------------------------------------------------------------
# Sources recording
# ---------------------------------------------------------------------------


def _save_sources(
    analysis_root: str, fp_dir: str, spaces: list[str], fs_subjects_dir: str
) -> None:
    path = op.join(analysis_root, "project_to_spaces_sources.json")
    record = {
        "fmriprep_dir": fp_dir,
        "freesurfer_subjects_dir": fs_subjects_dir,
        "spaces": spaces,
        "date": datetime.date.today().isoformat(),
        "tool": "project_to_spaces.py",
    }
    with open(path, "w") as fh:
        json.dump(record, fh, indent=2)
    console.print(f"  [dim]sources → {path}[/dim]")


# ---------------------------------------------------------------------------
# fmriprep file discovery
# ---------------------------------------------------------------------------


def _find_in_anat(fp_anat_dir: str, suffix: str) -> str | None:
    found = sorted(glob.glob(op.join(fp_anat_dir, f"*{suffix}")))
    return found[0] if found else None


def _find_boldref_to_t1w(fp_func_dir, sub, ses, task, acq, run) -> str | None:
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""
    pat = op.join(
        fp_func_dir,
        f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}"
        f"_from-boldref_to-T1w_mode-image_desc-coreg_xfm.txt",
    )
    found = glob.glob(pat)
    return found[0] if found else None


def _find_t1w_to_mni(fp_anat_dir: str) -> str | None:
    return _find_in_anat(fp_anat_dir, f"_from-T1w_to-{MNI_SPACE}_mode-image_xfm.h5")


def _find_mni_boldref(fp_func_dir, sub, ses, task, acq, run) -> str | None:
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""
    found = glob.glob(
        op.join(
            fp_func_dir,
            f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}"
            f"_space-{MNI_SPACE}_boldref.nii.gz",
        )
    )
    return found[0] if found else None


def _find_t1w_boldref(fp_func_dir, sub, ses, task, acq, run) -> str | None:
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""
    found = glob.glob(
        op.join(
            fp_func_dir,
            f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}"
            f"_space-T1w_boldref.nii.gz",
        )
    )
    return found[0] if found else None


def _find_boldref_native(fp_func_dir, sub, ses, task, acq, run) -> str | None:
    """Find fMRIprep's native-BOLD-space boldref (no space entity) for bbregister."""
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""
    found = glob.glob(
        op.join(
            fp_func_dir,
            f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}_boldref.nii.gz",
        )
    )
    return found[0] if found else None


def _find_input_files(
    func_dir, sub, ses, task, acq, desc
) -> list[tuple[str | None, str]]:
    acq_token = f"_acq-{acq}" if acq else ""
    found = sorted(
        glob.glob(
            op.join(
                func_dir,
                f"sub-{sub}_ses-{ses}_task-{task}{acq_token}_run-*_desc-{desc}_bold.nii.gz",
            )
        )
    )
    pairs = []
    for f in found:
        m = re.search(r"_run-(\w+)[_.]", op.basename(f))
        if m:
            pairs.append((m.group(1), f))
    if not pairs:
        found2 = sorted(
            glob.glob(
                op.join(
                    func_dir,
                    f"sub-{sub}_ses-{ses}_task-{task}{acq_token}_desc-{desc}_bold.nii.gz",
                )
            )
        )
        pairs = [(None, f) for f in found2]
    return pairs


# ---------------------------------------------------------------------------
# Transform commands (also used for dry-run printing)
# ---------------------------------------------------------------------------


def _ants_cmd(
    input_nii: str, reference_nii: str, transforms: list[str], output_nii: str
) -> str:
    """Return the antsApplyTransforms shell command string."""
    xfm_flags = " ".join(f"--transform {x}" for x in transforms)
    return (
        f"antsApplyTransforms"
        f" --dimensionality 3"
        f" --input-image-type 3"
        f" --input {input_nii}"
        f" --reference-image {reference_nii}"
        f" --interpolation LanczosWindowedSinc"
        f" {xfm_flags}"
        f" --output {output_nii}"
    )


def _bbregister_cmd(boldref: str, reg_out: str, subject: str, subjects_dir: str) -> str:
    """Return bbregister command to compute a BBR registration from native BOLD to FreeSurfer T1."""
    return (
        f"SUBJECTS_DIR={subjects_dir}"
        f" bbregister"
        f" --s {subject}"
        f" --mov {boldref}"
        f" --init-header"
        f" --bold"
        f" --force-ras"
        f" --reg {reg_out}"
    )


def _vol2surf_cmd(
    src: str,
    out: str,
    hemi: str,
    surf: str,
    subject: str,
    subjects_dir: str,
    reg: str | None = None,
    trg_subject: str | None = None,
) -> str:
    """Return the mri_vol2surf shell command string."""
    fs_hemi = "lh" if hemi == "L" else "rh"
    trg = f" --trgsubject {trg_subject}" if trg_subject else ""
    reg_flag = f" --reg {reg}" if reg else f" --regheader {subject}"
    return (
        f"SUBJECTS_DIR={subjects_dir}"
        f" mri_vol2surf"
        f" --src {src}"
        f"{reg_flag}"
        f" --hemi {fs_hemi}"
        f" --surf {surf}"
        f"{trg}"
        f" --o {out}"
    )


# ---------------------------------------------------------------------------
# Core: plan + execute per run
# ---------------------------------------------------------------------------


def _process_run(
    sub: str,
    ses: str,
    task: str,
    acq: str,
    run: str | None,
    input_path: str,
    func_dir: str,
    fp_func_dir: str,
    fp_anat_dir: str,
    spaces: list[str],
    desc: str,
    fs_subject: str,
    fs_subjects_dir: str,
    ants_module: str,
    fs_module: str,
    execute: bool,
    overwrite: bool,
) -> tuple[bool, float]:
    t0_run = time.perf_counter()
    run_label = f"run-{run}" if run else "(no run entity)"
    prefix = _bids_prefix(sub, ses, task, acq, run)
    console.print(f"\n  [bold]task-{task}  {run_label}  desc-{desc}[/bold]")
    console.print(f"    Input : {op.basename(input_path)}")

    # ── Resolve transforms ─────────────────────────────────────────────────
    boldref_to_t1w = _find_boldref_to_t1w(fp_func_dir, sub, ses, task, acq, run)
    t1w_to_mni = _find_t1w_to_mni(fp_anat_dir)
    mni_boldref = _find_mni_boldref(fp_func_dir, sub, ses, task, acq, run)
    t1w_boldref = _find_t1w_boldref(fp_func_dir, sub, ses, task, acq, run)

    all_ok = True

    for space in spaces:
        # ── MNI ───────────────────────────────────────────────────────────
        if space == "MNI":
            out = op.join(
                func_dir, f"{prefix}space-{MNI_SPACE}_desc-{desc}_bold.nii.gz"
            )
            missing = [
                n
                for n, v in [
                    ("boldref_to_T1w xfm", boldref_to_t1w),
                    ("T1w_to_MNI xfm", t1w_to_mni),
                    ("MNI boldref", mni_boldref),
                ]
                if not v
            ]
            if missing:
                console.print(f"    [red]✗ MNI: missing {missing}[/red]")
                all_ok = False
                continue

            cmd = _ants_cmd(input_path, mni_boldref, [t1w_to_mni, boldref_to_t1w], out)

            console.print("    [cyan]→ MNI[/cyan]")
            console.print(f"      Module : {ants_module}")
            console.print(
                f"      XFM1   : {op.basename(t1w_to_mni)}  (T1w→MNI, applied 1st)"
            )
            console.print(
                f"      XFM2   : {op.basename(boldref_to_t1w)}  (BOLD→T1w, applied 2nd)"
            )
            console.print(f"      Ref    : {op.basename(mni_boldref)}")
            console.print(f"      Cmd    : {cmd}")
            console.print(f"      Out    : {op.basename(out)}")

            if not execute:
                continue
            if op.isfile(out) and not overwrite:
                console.print("      [yellow]→ exists, skip[/yellow]")
                continue
            t0 = time.perf_counter()
            try:
                _run_with_modules(cmd, [ants_module])
                console.print(
                    f"      [green]✓ done[/green] [dim]({_fmt_time(time.perf_counter() - t0)})[/dim]"
                )
            except RuntimeError as exc:
                console.print(f"      [red]✗ {exc}[/red]")
                all_ok = False

        # ── fsnative / fsaverage ──────────────────────────────────────────
        elif space in ("fsnative", "fsaverage"):
            trg = "fsaverage" if space == "fsaverage" else None
            surf = "white" if space == "fsaverage" else "midthickness"
            space_label = "fsaverage" if space == "fsaverage" else "fsnative"

            # ── Step 1: bbregister (native BOLD boldref → FreeSurfer T1) ──
            boldref_native = _find_boldref_native(fp_func_dir, sub, ses, task, acq, run)
            if not boldref_native:
                console.print(
                    f"    [red]✗ {space_label}: native boldref not found for bbregister[/red]"
                )
                all_ok = False
                continue
            reg_file = op.join(func_dir, f"{prefix}from-boldref_to-T1w_bbr.reg")
            bbr_cmd = _bbregister_cmd(
                boldref_native, reg_file, fs_subject, fs_subjects_dir
            )
            console.print("    [cyan]→ bbregister (BOLD→T1w, force-BBR)[/cyan]")
            console.print(f"      Module : {fs_module}")
            console.print(f"      Mov    : {op.basename(boldref_native)}")
            console.print(f"      Cmd    : {bbr_cmd}")
            console.print(f"      Reg    : {op.basename(reg_file)}")

            if execute:
                if not op.isfile(reg_file) or overwrite:
                    t0 = time.perf_counter()
                    try:
                        _run_with_modules(bbr_cmd, [fs_module])
                        console.print(
                            f"      [green]✓ done[/green] "
                            f"[dim]({_fmt_time(time.perf_counter() - t0)})[/dim]"
                        )
                    except RuntimeError as exc:
                        console.print(f"      [red]✗ bbregister failed: {exc}[/red]")
                        all_ok = False
                        continue
                else:
                    console.print("      [yellow]→ reg exists, skip[/yellow]")

            # ── Step 2: mri_vol2surf per hemi (native BOLD → surface) ─────
            for hemi in ("L", "R"):
                out = op.join(
                    func_dir,
                    f"{prefix}hemi-{hemi}_space-{space_label}_desc-{desc}_bold.func.gii",
                )
                v2s_cmd = _vol2surf_cmd(
                    input_path,
                    out,
                    hemi,
                    surf,
                    fs_subject,
                    fs_subjects_dir,
                    reg=reg_file,
                    trg_subject=trg,
                )
                console.print(f"    [cyan]→ {space_label} hemi-{hemi}[/cyan]")
                console.print(f"      Module : {fs_module}")
                console.print(f"      Cmd    : {v2s_cmd}")
                console.print(f"      Out    : {op.basename(out)}")

                if not execute:
                    continue
                if op.isfile(out) and not overwrite:
                    console.print("      [yellow]→ exists, skip[/yellow]")
                    continue

                t0 = time.perf_counter()
                try:
                    _run_with_modules(v2s_cmd, [fs_module])
                    console.print(
                        f"      [green]✓ done[/green] [dim]({_fmt_time(time.perf_counter() - t0)})[/dim]"
                    )
                except RuntimeError as exc:
                    console.print(f"      [red]✗ {exc}[/red]")
                    all_ok = False

    t_total = time.perf_counter() - t0_run
    return all_ok, t_total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    input_dir: str = typer.Option(
        ..., "-i", help="tedana analysis root  (contains sub-*/ses-*/func/)"
    ),
    fp_dir: str = typer.Option(
        ..., "-fp", help="fMRIprep derivatives directory (absolute path)"
    ),
    fs_subjects_dir: str = typer.Option(
        ...,
        "--fs-subjects-dir",
        "--fs-sd",
        help="FreeSurfer subjects directory  e.g. .../derivatives/freesurfer",
    ),
    fs_subject_template: str = typer.Option(
        "sub-{sub}",
        "--fs-subject",
        help="FreeSurfer subject name template. Placeholders: {sub} {ses}",
    ),
    single: Optional[str] = typer.Option(None, "-s", help="sub,ses  e.g.  pilot02,01"),
    batch_file: Optional[str] = typer.Option(
        None, "-f", help="Batch file: one sub,ses per line"
    ),
    tasks: str = typer.Option(
        "BfLocVideo", "--tasks", "-t", help="Comma-separated task names"
    ),
    acq: str = typer.Option("ME", "--acq", help="Acquisition label"),
    descs: str = typer.Option(
        "denoised", "--descs", help="Comma-separated desc labels  e.g.  denoised,optcom"
    ),
    spaces: str = typer.Option(
        "MNI,fsnative,fsaverage",
        "--spaces",
        help="Comma-separated target spaces: MNI, fsnative, fsaverage",
    ),
    ants_module: str = typer.Option("ants/2.5.1", "--ants-module"),
    fs_module: str = typer.Option("freesurfer/7.3.2", "--fs-module"),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Run transforms. Without this flag only the plan is printed.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    n_run_jobs: int = typer.Option(
        1, "--n-run-jobs", "-J", help="Runs in parallel (1=sequential, -1=all)"
    ),
) -> None:
    """Project tedana outputs to other spaces via ANTs + FreeSurfer modules."""

    space_list = [s.strip() for s in spaces.split(",") if s.strip()]
    invalid = set(space_list) - VALID_SPACES
    if invalid:
        console.print(
            f"[red]✗ Unknown space(s): {invalid}  Valid: {VALID_SPACES}[/red]"
        )
        raise typer.Exit(1)

    desc_list = [d.strip() for d in descs.split(",") if d.strip()]
    task_list = [t.strip() for t in tasks.split(",") if t.strip()]

    # ── Sub/ses pairs ──────────────────────────────────────────────────────
    subses_pairs: list[tuple[str, str]] = []
    if single:
        p = single.split(",")
        if len(p) != 2:
            console.print("[red]✗ -s expects sub,ses[/red]")
            raise typer.Exit(1)
        subses_pairs.append((p[0].strip(), p[1].strip()))
    elif batch_file:
        if not op.isfile(batch_file):
            console.print(f"[red]✗ batch file not found: {batch_file}[/red]")
            raise typer.Exit(1)
        with open(batch_file) as fh:
            for line in fh:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                p = line.split(",")
                if len(p) < 2:
                    p = line.split()
                if len(p) >= 2:
                    subses_pairs.append((p[0].strip(), p[1].strip()))
    else:
        console.print("[red]✗ provide -s sub,ses  or  -f batch_file[/red]")
        raise typer.Exit(1)

    # ── Banner ─────────────────────────────────────────────────────────────
    tbl = Table(title="project_to_spaces", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    tbl.add_row("input_dir", input_dir)
    tbl.add_row("fp_dir", fp_dir)
    tbl.add_row("fs_subjects_dir", fs_subjects_dir)
    tbl.add_row("fs_subject template", fs_subject_template)
    tbl.add_row("tasks", ", ".join(task_list))
    tbl.add_row("acq", acq)
    tbl.add_row("descs", ", ".join(desc_list))
    tbl.add_row("spaces", ", ".join(space_list))
    tbl.add_row("ants_module", ants_module)
    tbl.add_row("fs_module", fs_module)
    tbl.add_row("execute", str(execute))
    console.print(tbl)

    if not execute:
        console.print(
            "\n[yellow bold]DRY RUN — pass --execute to run transforms[/yellow bold]\n"
        )

    if execute:
        _save_sources(input_dir, fp_dir, space_list, fs_subjects_dir)

    # ── Main loop ──────────────────────────────────────────────────────────
    results: list[tuple[str, str, str, str | None, str, bool, float]] = []
    t_total_start = time.perf_counter()

    for sub, ses in subses_pairs:
        console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/bold cyan]")
        func_dir = op.join(input_dir, f"sub-{sub}", f"ses-{ses}", "func")
        fp_func_dir = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "func")
        fp_anat_dir = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "anat")
        fs_subject = fs_subject_template.format(sub=sub, ses=ses)

        for path, label in [
            (func_dir, "tedana func"),
            (fp_func_dir, "fmriprep func"),
            (fp_anat_dir, "fmriprep anat"),
            (fs_subjects_dir, "FS subjects"),
        ]:
            if not op.isdir(path):
                console.print(f"  [red]✗ {label} dir not found: {path}[/red]")
                for task in task_list:
                    for desc in desc_list:
                        results.append((sub, ses, task, None, desc, False, 0.0))
                continue

        for task in task_list:
            for desc in desc_list:
                pairs = _find_input_files(func_dir, sub, ses, task, acq, desc)
                if not pairs:
                    console.print(
                        f"  [yellow]⚠ task-{task} desc-{desc}: no input files found[/yellow]"
                    )
                    results.append((sub, ses, task, None, desc, False, 0.0))
                    continue

                run_kwargs = [
                    dict(
                        sub=sub,
                        ses=ses,
                        task=task,
                        acq=acq,
                        run=run,
                        input_path=path,
                        func_dir=func_dir,
                        fp_func_dir=fp_func_dir,
                        fp_anat_dir=fp_anat_dir,
                        spaces=space_list,
                        desc=desc,
                        fs_subject=fs_subject,
                        fs_subjects_dir=fs_subjects_dir,
                        ants_module=ants_module,
                        fs_module=fs_module,
                        execute=execute,
                        overwrite=overwrite,
                    )
                    for run, path in pairs
                ]

                workers = len(pairs) if n_run_jobs == -1 else n_run_jobs
                if workers == 1 or len(pairs) == 1:
                    for kw in run_kwargs:
                        ok, elapsed = _process_run(**kw)
                        results.append((sub, ses, task, kw["run"], desc, ok, elapsed))
                else:
                    console.print(
                        f"  [dim]Launching {len(pairs)} run(s) across "
                        f"{min(workers, len(pairs))} worker(s) …[/dim]"
                    )
                    with ProcessPoolExecutor(
                        max_workers=min(workers, len(pairs))
                    ) as pool:
                        future_to_run = {
                            pool.submit(_process_run, **kw): kw["run"]
                            for kw in run_kwargs
                        }
                        for future in as_completed(future_to_run):
                            run = future_to_run[future]
                            try:
                                ok, elapsed = future.result()
                            except Exception as exc:
                                console.print(f"  [red]✗ run-{run}: {exc}[/red]")
                                ok, elapsed = False, 0.0
                            results.append((sub, ses, task, run, desc, ok, elapsed))

    t_total = time.perf_counter() - t_total_start

    # ── Summary ────────────────────────────────────────────────────────────
    console.print("\n")
    summary = Table(title="Results", show_lines=False)
    summary.add_column("sub")
    summary.add_column("ses")
    summary.add_column("task")
    summary.add_column("run")
    summary.add_column("desc")
    summary.add_column("status")
    summary.add_column("wall time", justify="right")
    for sub, ses, task, run, desc, ok, elapsed in results:
        summary.add_row(
            sub,
            ses,
            task,
            run or "—",
            desc,
            "[green]✓[/green]" if ok else "[red]✗[/red]",
            _fmt_time(elapsed) if elapsed > 0 else "—",
        )
    console.print(summary)

    if not execute:
        console.print(
            "[yellow]Above is the plan. Re-run with --execute to apply.[/yellow]"
        )
        return

    console.print(f"Total wall time: [bold]{_fmt_time(t_total)}[/bold]")
    n_failed = sum(1 for *_, ok, _e in results if not ok)
    if n_failed:
        console.print(f"\n[red]{n_failed} run(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("\n[green]All runs completed successfully.[/green]")


if __name__ == "__main__":
    app()
