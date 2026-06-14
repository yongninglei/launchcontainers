#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
run_all_combos.py

Top-level driver for the EPI vol2surf pipeline. For each sub/ses, runs:

    0_check_affine - reuse a cached .lta, or convert fMRIprep's
                      from-T1w_to-fsnative xfm (--src-space T1w only)
    1_do_register  - if step 0 found nothing, compute a new .lta
                      via bbregister or ANTs (--reg-method / --mov)
    2_transform    - project every file matching --input-glob to the
                      surface, once for EACH of the 6 hardcoded
                      surf x interp combinations
                      (transform_common.SURF_INTERP_COMBOS):

          white/pial/midthickness  x  nearest/trilinear

so step 2's projection logic runs 6 times per input file (12 outputs per
file: 6 combos x {L,R}), each tagged with `_desc-{surf}{Interp}_` in the
output filename (see transform_common.desc_label / derive_output_path).

All resolved CLI parameters are written to a YAML manifest in
`--cache-dir` (the same per-sub/ses directory used for the .lta cache):

    {cache_dir}/sub-{sub}_ses-{ses}_epi_vol2surf_manifest.yaml

Note: projected outputs themselves are written next to each input file
(same directory as the matched volume), per transform_common.derive_output_path.

Usage
-----
Dry-run (print plan, no execution)::

    python run_all_combos.py \\
        --src-space T1w \\
        --fp-dir .../derivatives/fmriprep-25.1.4_IRpilot \\
        --fs-subjects-dir .../derivatives/freesurfer \\
        --cache-dir .../analysis-NAME/sub-{sub}/ses-{ses} \\
        --input-glob ".../analysis-NAME/sub-{sub}/ses-{ses}/sub-{sub}_ses-{ses}_*_space-T1w_*_statmap.nii.gz" \\
        -s pilot01,01

Execute, with native-space EPI registered via bbregister, projecting all
6 combos with 6-way parallelism::

    python run_all_combos.py \\
        --src-space native --reg-method bbregister \\
        --mov ".../sub-{sub}_ses-{ses}_task-x_boldref.nii.gz" \\
        --fs-subjects-dir .../derivatives/freesurfer \\
        --cache-dir .../analysis-NAME/sub-{sub}/ses-{ses} \\
        --input-glob ".../sub-{sub}_ses-{ses}_task-x_space-native_*.nii.gz" \\
        --space-label-in native \\
        -s pilot01,01 --execute -J 6
"""

from __future__ import annotations

import glob as globmod
import os.path as op
import time
from typing import Optional

import typer
import yaml
from rich.console import Console
from rich.table import Table

import transform_common as tc

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


@app.command()
def main(
    src_space: str = typer.Option(..., "--src-space", help="Source space of the EPI data: T1w | native"),
    fp_dir: str = typer.Option("", "--fp-dir", help="fMRIprep derivatives directory (required for --src-space T1w, or ANTs+native)"),
    fs_subjects_dir: str = typer.Option(..., "--fs-subjects-dir", "--fs-sd", help="FreeSurfer SUBJECTS_DIR"),
    fs_subject_template: str = typer.Option("sub-{sub}", "--fs-subject", help="FreeSurfer subject name template. Placeholders: {sub} {ses}"),
    cache_dir: str = typer.Option(..., "--cache-dir", help="Directory (template, placeholders {sub} {ses}) where the resolved .lta and the YAML manifest are stored"),
    input_glob: str = typer.Option(..., "--input-glob", help="Glob pattern (placeholders {sub} {ses}) for volumes to project"),
    single: Optional[str] = typer.Option(None, "-s", help="sub,ses  e.g.  pilot01,01"),
    batch_file: Optional[str] = typer.Option(None, "-f", help="Batch file: one sub,ses per line"),
    reg_method: str = typer.Option("bbregister", "--reg-method", help="Registration method if step 0 finds nothing: bbregister | ants"),
    mov: str = typer.Option("", "--mov", help="Moving volume template (placeholders {sub} {ses}) for 1_do_register. Required if --src-space native, or if step 0 finds nothing for T1w."),
    bbr_contrast: Optional[str] = typer.Option(None, "--bbr-contrast", help="bbregister contrast: t1 | t2 | bold (default: t1 for T1w, bold for native)"),
    bbr_init: str = typer.Option("header", "--bbr-init", help="bbregister init: header | fsl | coreg"),
    hemi: str = typer.Option("LR", "--hemi", help="Hemispheres to project: L | R | LR"),
    proj_frac: Optional[float] = typer.Option(None, "--proj-frac", help="--projfrac value"),
    proj_frac_avg: Optional[str] = typer.Option(None, "--proj-frac-avg", help="start,stop,step for --projfrac-avg (overrides --proj-frac)"),
    trgsubject: Optional[str] = typer.Option(None, "--trgsubject", help="Resample to this FS subject (e.g. fsaverage)"),
    space_label_in: Optional[str] = typer.Option(None, "--space-label-in", help="space-* label to replace in input filenames (default: --src-space)"),
    space_label_out: str = typer.Option("fsnative", "--space-label-out", help="space-* label to use in output filenames (overridden by --trgsubject)"),
    out_ext: str = typer.Option(".func.gii", "--out-ext", help="Output file extension"),
    fs_module: str = typer.Option("freesurfer/7.3.2", "--fs-module"),
    ants_module: str = typer.Option("ants", "--ants-module"),
    execute: bool = typer.Option(False, "--execute", help="Run all steps. Without this flag only the plan is printed."),
    overwrite: bool = typer.Option(False, "--overwrite"),
    n_jobs: int = typer.Option(1, "--n-jobs", "-J", help="(file, surf, interp) jobs to run in parallel per sub/ses (1=sequential, -1=all 6 combos x n_files)"),
) -> None:
    """Run check_affine -> do_register (if needed) -> vol2surf x 6 combos for each sub/ses."""

    if src_space not in tc.SRC_SPACES:
        console.print(f"[red]✗ Unknown --src-space: {src_space}  Valid: {tc.SRC_SPACES}[/red]")
        raise typer.Exit(1)
    if reg_method not in tc.REG_METHODS:
        console.print(f"[red]✗ Unknown --reg-method: {reg_method}  Valid: {tc.REG_METHODS}[/red]")
        raise typer.Exit(1)

    space_label_in = space_label_in or src_space

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

    pfa: Optional[tuple[float, float, float]] = None
    if proj_frac_avg:
        parts = proj_frac_avg.split(",")
        if len(parts) != 3:
            console.print("[red]✗ --proj-frac-avg expects start,stop,step[/red]")
            raise typer.Exit(1)
        pfa = tuple(float(p) for p in parts)  # type: ignore[assignment]

    hemi = hemi.upper()
    hemis = [h for h in ("L", "R") if h in hemi]
    if not hemis:
        console.print(f"[red]✗ --hemi must be L, R, or LR (got {hemi})[/red]")
        raise typer.Exit(1)

    cli_params = {
        "src_space": src_space,
        "fp_dir": fp_dir,
        "fs_subjects_dir": fs_subjects_dir,
        "fs_subject_template": fs_subject_template,
        "cache_dir": cache_dir,
        "input_glob": input_glob,
        "subses_pairs": [f"{s},{ses}" for s, ses in subses_pairs],
        "reg_method": reg_method,
        "mov": mov,
        "bbr_contrast": bbr_contrast,
        "bbr_init": bbr_init,
        "hemi": "".join(hemis),
        "surf_interp_combos": [tc.desc_label(s, i) for s, i in tc.SURF_INTERP_COMBOS],
        "proj_frac": proj_frac,
        "proj_frac_avg": list(pfa) if pfa else None,
        "trgsubject": trgsubject,
        "space_label_in": space_label_in,
        "space_label_out": space_label_out,
        "out_ext": out_ext,
        "fs_module": fs_module,
        "ants_module": ants_module,
        "execute": execute,
        "overwrite": overwrite,
        "n_jobs": n_jobs,
    }

    tbl = Table(title="run_all_combos", show_lines=False)
    tbl.add_column("Parameter", style="cyan")
    tbl.add_column("Value")
    for k, v in cli_params.items():
        tbl.add_row(k, str(v))
    console.print(tbl)

    if not execute:
        console.print("\n[yellow bold]DRY RUN — pass --execute to run all steps[/yellow bold]\n")

    results: list[tuple[str, str, str, str, bool, float]] = []
    t_total_start = time.perf_counter()

    for sub, ses in subses_pairs:
        console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/bold cyan]")
        fs_subject = fs_subject_template.format(sub=sub, ses=ses)
        sub_cache_dir = cache_dir.format(sub=sub, ses=ses)

        # ── Glob the input files up front: the first match is used as the
        #    --src geometry reference when converting fMRIprep's ITK xfm to
        #    .lta (step 0), so mri_vol2surf accepts it as --src in step 2. ──
        pattern = op.expanduser(input_glob.format(sub=sub, ses=ses))
        files = sorted(globmod.glob(pattern))
        if not files:
            console.print(f"    [yellow]⚠ no files matched: {pattern}[/yellow]")
            continue
        console.print(f"    Found {len(files)} file(s)")

        # ── Step 0: check for an existing affine ────────────────────────────
        console.print("  [bold]Step 0 - check_affine[/bold]")
        chk = tc.check_affine(
            sub, ses, src_space, fp_dir, fs_subjects_dir, fs_subject,
            sub_cache_dir, fs_module, execute, overwrite,
            lta_src_vol=files[0],
        )
        for line in chk["log"]:
            console.print(f"    {line}")

        reg_lta = chk["lta_path"]

        # ── Step 1: register if needed ───────────────────────────────────────
        if chk["needs_register"]:
            console.print("  [bold]Step 1 - do_register[/bold]")
            if not mov:
                console.print("    [red]✗ no affine found and --mov not given; cannot register[/red]")
                results.append((sub, ses, "(reg)", "-", False, 0.0))
                continue
            mov_path = op.expanduser(mov.format(sub=sub, ses=ses))
            reg = tc.do_register(
                sub, ses, src_space, reg_method, mov_path, fp_dir, fs_subjects_dir,
                fs_subject, sub_cache_dir, bbr_contrast, bbr_init, fs_module,
                ants_module, execute, overwrite,
            )
            for line in reg["log"]:
                console.print(f"    {line}")
            reg_lta = reg["lta_path"]

        if execute and not op.isfile(reg_lta):
            console.print(f"  [red]✗ registration .lta not found after step 0/1: {reg_lta}[/red]")
            results.append((sub, ses, "(reg)", "-", False, 0.0))
            continue

        # ── Step 2: project files, 6 times each (one per surf x interp combo) ──
        console.print("  [bold]Step 2 - vol2surf x 6 combos[/bold]")

        run_kwargs = [
            dict(
                src=f, reg=reg_lta, fs_subject=fs_subject, fs_subjects_dir=fs_subjects_dir,
                hemis=hemis, surf=surf, interp=interp,
                proj_frac=proj_frac, proj_frac_avg=pfa,
                trgsubject=trgsubject, space_label_in=space_label_in,
                space_label_out=space_label_out, out_ext=out_ext, fs_module=fs_module,
                execute=execute, overwrite=overwrite,
            )
            for f in files
            for surf, interp in tc.SURF_INTERP_COMBOS
        ]

        workers = len(run_kwargs) if n_jobs == -1 else n_jobs
        if workers <= 1 or len(run_kwargs) == 1:
            for kw in run_kwargs:
                desc = tc.desc_label(kw["surf"], kw["interp"])
                console.print(f"    [bold]{op.basename(kw['src'])}  desc-{desc}[/bold]")
                ok, elapsed, log = tc.project_file(**kw)
                for line in log:
                    console.print(line)
                results.append((sub, ses, op.basename(kw["src"]), desc, ok, elapsed))
        else:
            from concurrent.futures import ProcessPoolExecutor, as_completed

            console.print(f"    [dim]Projecting {len(run_kwargs)} (file, combo) job(s) across {min(workers, len(run_kwargs))} worker(s) …[/dim]")
            with ProcessPoolExecutor(max_workers=min(workers, len(run_kwargs))) as pool:
                future_to_kw = {pool.submit(tc.project_file, **kw): kw for kw in run_kwargs}
                for future in as_completed(future_to_kw):
                    kw = future_to_kw[future]
                    desc = tc.desc_label(kw["surf"], kw["interp"])
                    try:
                        ok, elapsed, log = future.result()
                    except Exception as exc:
                        console.print(f"    [red]✗ {op.basename(kw['src'])} desc-{desc}: {exc}[/red]")
                        ok, elapsed, log = False, 0.0, []
                    console.print(f"    [bold]{op.basename(kw['src'])}  desc-{desc}[/bold]")
                    for line in log:
                        console.print(line)
                    results.append((sub, ses, op.basename(kw["src"]), desc, ok, elapsed))

        # ── write the YAML manifest for this sub/ses ─────────────────────────
        manifest = dict(cli_params)
        manifest["sub"] = sub
        manifest["ses"] = ses
        manifest["fs_subject"] = fs_subject
        manifest["reg_lta"] = reg_lta
        manifest["input_files"] = files
        manifest_path = op.join(sub_cache_dir, f"sub-{sub}_ses-{ses}_epi_vol2surf_manifest.yaml")
        if execute:
            with open(manifest_path, "w") as fh:
                yaml.safe_dump(manifest, fh, sort_keys=False)
            console.print(f"    [dim]manifest written: {manifest_path}[/dim]")
        else:
            console.print(f"    [dim](dry-run: manifest would be written to {manifest_path})[/dim]")

    t_total = time.perf_counter() - t_total_start

    console.print("\n")
    summary = Table(title="Results", show_lines=False)
    summary.add_column("sub")
    summary.add_column("ses")
    summary.add_column("file")
    summary.add_column("desc")
    summary.add_column("status")
    summary.add_column("wall time", justify="right")
    for sub, ses, fname, desc, ok, elapsed in results:
        summary.add_row(
            sub, ses, fname, desc,
            "[green]✓[/green]" if ok else "[red]✗[/red]",
            tc.fmt_time(elapsed) if elapsed > 0 else "—",
        )
    console.print(summary)

    if not execute:
        console.print("[yellow]Above is the plan. Re-run with --execute to apply.[/yellow]")
        return

    console.print(f"Total wall time: [bold]{tc.fmt_time(t_total)}[/bold]")
    n_failed = sum(1 for *_, ok, _e in results if not ok)
    if n_failed:
        console.print(f"\n[red]{n_failed} item(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("\n[green]All projections completed successfully.[/green]")


if __name__ == "__main__":
    app()
