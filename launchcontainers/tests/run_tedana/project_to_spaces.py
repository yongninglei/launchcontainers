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
  BOLD → T1w      : antsApplyTransforms  boldref_to_T1w.txt
  BOLD → MNI      : antsApplyTransforms  boldref_to_T1w.txt + T1w_to_MNI.h5
  BOLD → fsnative : lta_convert boldref_to_T1w.txt + lta_convert T1w_to_fsnative.txt
                    + mri_concatenate_lta -> boldref_to_fsnative.lta (fMRIprep's own
                    transforms, no extra registration)
                    + mri_vol2surf --reg <concat.lta> --surf white --interp trilinear
                    --projfrac-avg 0 1 0.2
  BOLD → fsaverage: same boldref_to_fsnative.lta
                    + mri_vol2surf --reg <concat.lta> --surf white --interp trilinear
                    --projfrac-avg 0 1 0.2 --trgsubject fsaverage

Outputs are written back into the same sub/ses/func dir as the tedana inputs.
A ``project_to_spaces_sources.json`` is saved in the analysis root.

Parallelization
----------------
Every (sub, ses, task, run, desc, space) combination is broken down into one
or more atomic jobs (one antsApplyTransforms call, one fsnative-reg
computation, one mri_vol2surf call, ...). All jobs across the whole batch are
collected into a single flat queue and executed by a pool of ``--n-jobs``
workers, regardless of how many subs/sessions/runs were requested. The only
dependency is that the fsnative-reg job (shared by fsnative/fsaverage and by
all descs of the same run) must finish before its 4 mri_vol2surf jobs run.

Usage
-----
Dry-run (print plan, no execution)::

    python project_to_spaces.py \\
        -i .../tedana-26.0.3/analysis-default_ica \\
        -fp .../fmriprep-25.1.4 \\
        --fs-subjects-dir .../derivatives/freesurfer \\
        -s pilot02,01 --tasks BfLocVideo

Execute, using 8 parallel workers across the whole batch::

    python project_to_spaces.py ... --execute -J 8
"""

from __future__ import annotations

import datetime
import glob
import json
import os
import os.path as op
import re
import subprocess
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Callable, Optional

import typer
from rich.console import Console
from rich.table import Table

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

VALID_SPACES = {"T1w", "MNI", "fsnative", "fsaverage"}
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
    """Find fMRIprep's native-BOLD-space boldref (no space entity)."""
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""
    found = glob.glob(
        op.join(
            fp_func_dir,
            f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}_boldref.nii.gz",
        )
    )
    return found[0] if found else None


def _find_t1w_preproc(fp_anat_dir: str, sub: str, ses: str) -> str | None:
    """Locate fMRIprep's sub-*_desc-preproc_T1w.nii.gz."""
    found = sorted(
        glob.glob(op.join(fp_anat_dir, f"sub-{sub}_ses-{ses}*_desc-preproc_T1w.nii.gz"))
    )
    return found[0] if found else None


def _find_t1w_to_fsnative_xfm(fp_anat_dir: str, sub: str, ses: str) -> str | None:
    """Locate fMRIprep's sub-*_from-T1w_to-fsnative_mode-image_xfm.txt (ITK affine)."""
    found = sorted(
        glob.glob(
            op.join(
                fp_anat_dir,
                f"sub-{sub}_ses-{ses}*_from-T1w_to-fsnative_mode-image_xfm.txt",
            )
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


def _lta_convert_cmd(itk_xfm: str, src_vol: str, trg_vol: str, out_lta: str, subjects_dir: str = "") -> str:
    """Return an lta_convert command converting an ITK affine (double_3_3 header) to .lta."""
    prefix = f"SUBJECTS_DIR={subjects_dir} " if subjects_dir else ""
    return (
        f"{prefix}lta_convert"
        f" --initk {itk_xfm}"
        f" --src {src_vol}"
        f" --trg {trg_vol}"
        f" --outlta {out_lta}"
    )


def _concat_lta_cmd(lta_first: str, lta_second: str, out_lta: str) -> str:
    """Concatenate two .lta transforms: out = lta_second(lta_first(x))."""
    return f"mri_concatenate_lta {lta_first} {lta_second} {out_lta}"


def _fix_itk_xfm_precision(itk_xfm: str, out_path: str) -> None:
    """Work around `lta_convert --initk ... ERROR readITK: Transform type unknown!`.

    ANTs/fMRIPrep write ITK transforms with header type
    `[Matrix...|Affine]Transform_float_3_3`, but FreeSurfer's `lta_convert
    --initk` only recognises the `_double_3_3` variant. The transform
    parameters are stored as plain-text numbers regardless of the declared
    precision, so rewriting the header type to `_double_3_3` is sufficient.
    """
    with open(itk_xfm) as fh:
        content = fh.read()
    content = content.replace("_float_3_3", "_double_3_3")
    with open(out_path, "w") as fh:
        fh.write(content)


def _set_lta_subject(lta_path: str, subject: str) -> None:
    """Ensure the `.lta`'s `subject <name>` field is set to `subject`.

    `lta_convert --initk` leaves this field blank, which makes
    `mri_vol2surf` resolve surface files as `$SUBJECTS_DIR//surf/lh.pial`
    (empty subject) instead of `$SUBJECTS_DIR/{subject}/surf/lh.pial`,
    failing with "could not open file".
    """
    with open(lta_path) as fh:
        lines = fh.readlines()
    for i, line in enumerate(lines):
        if line.startswith("subject"):
            lines[i] = f"subject {subject}\n"
            break
    else:
        lines.append(f"subject {subject}\n")
    with open(lta_path, "w") as fh:
        fh.writelines(lines)


def _vol2surf_cmd(
    src: str,
    out: str,
    hemi: str,
    surf: str,
    subject: str,
    subjects_dir: str,
    reg: str | None = None,
    trg_subject: str | None = None,
    interp: str = "trilinear",
    proj_frac_avg: tuple[float, float, float] = (0, 1, 0.2),
) -> str:
    """Return the mri_vol2surf shell command string (fMRIprep-style surface sampling:
    --surf white --interp trilinear --projfrac-avg 0 1 0.2)."""
    fs_hemi = "lh" if hemi == "L" else "rh"
    trg = f" --trgsubject {trg_subject}" if trg_subject else ""
    reg_flag = f" --reg {reg}" if reg else f" --regheader {subject}"
    start, stop, step = proj_frac_avg
    return (
        f"SUBJECTS_DIR={subjects_dir}"
        f" mri_vol2surf"
        f" --src {src}"
        f"{reg_flag}"
        f" --hemi {fs_hemi}"
        f" --surf {surf}"
        f" --interp {interp}"
        f" --projfrac-avg {start} {stop} {step}"
        f"{trg}"
        f" --o {out}"
    )


# ---------------------------------------------------------------------------
# Job queue
# ---------------------------------------------------------------------------


@dataclass
class RunCtx:
    sub: str
    ses: str
    task: str
    acq: str
    run: str | None
    run_label: str
    desc: str
    prefix: str
    input_path: str
    func_dir: str


@dataclass
class Job:
    id: str
    sub: str
    ses: str
    task: str
    run: str | None
    desc: str
    label: str
    depends_on: list[str] = field(default_factory=list)
    planned_ok: bool = True
    run_fn: Optional[Callable[[], tuple[bool, float]]] = None


def _run_jobs(jobs: list[Job], n_workers: int, execute: bool) -> list[tuple[Job, bool, float]]:
    """Run `jobs` (a flat list with dependencies via `depends_on`) across `n_workers`
    threads. In dry-run mode (`execute=False`), nothing is run; each job's status
    reflects whether its prerequisites were found during planning."""
    if not execute:
        return [(j, j.planned_ok, 0.0) for j in jobs]

    job_by_id = {j.id: j for j in jobs}
    results: dict[str, tuple[bool, float]] = {}
    pending = dict(job_by_id)
    futures: dict = {}

    def submit_ready(pool: ThreadPoolExecutor) -> None:
        for jid in list(pending):
            j = pending[jid]
            if not all(d in results for d in j.depends_on):
                continue
            if not j.planned_ok or any(not results[d][0] for d in j.depends_on):
                results[jid] = (False, 0.0)
                del pending[jid]
                continue
            futures[pool.submit(j.run_fn)] = jid
            del pending[jid]

    with ThreadPoolExecutor(max_workers=max(1, n_workers)) as pool:
        submit_ready(pool)
        while pending or futures:
            if not futures:
                # Nothing runnable (shouldn't normally happen) — fail the rest.
                for jid in list(pending):
                    results[jid] = (False, 0.0)
                    del pending[jid]
                break
            done, _ = wait(list(futures.keys()), return_when=FIRST_COMPLETED)
            for f in done:
                jid = futures.pop(f)
                try:
                    results[jid] = f.result()
                except Exception as exc:
                    console.print(f"  [red]✗ {job_by_id[jid].label}: {exc}[/red]")
                    results[jid] = (False, 0.0)
            submit_ready(pool)

    return [(j, *results[j.id]) for j in jobs]


# ---------------------------------------------------------------------------
# Per-job planners
# ---------------------------------------------------------------------------


def _plan_t1w_job(
    ctx: RunCtx, boldref_to_t1w: str | None, t1w_boldref: str | None,
    ants_module: str, overwrite: bool,
) -> Job:
    jid = f"{ctx.prefix}{ctx.desc}_T1w"
    label = "T1w"

    missing = [
        n for n, v in [("boldref_to_T1w xfm", boldref_to_t1w), ("T1w boldref", t1w_boldref)]
        if not v
    ]
    if missing:
        console.print(f"    [red]✗ T1w: missing {missing}[/red]")
        return Job(jid, ctx.sub, ctx.ses, ctx.task, ctx.run, ctx.desc, label,
                    planned_ok=False, run_fn=lambda: (False, 0.0))

    out = op.join(ctx.func_dir, f"{ctx.prefix}space-T1w_desc-{ctx.desc}_bold.nii.gz")
    cmd = _ants_cmd(ctx.input_path, t1w_boldref, [boldref_to_t1w], out)

    console.print("    [cyan]→ T1w[/cyan]")
    console.print(f"      Module : {ants_module}")
    console.print(f"      XFM    : {op.basename(boldref_to_t1w)}  (BOLD→T1w)")
    console.print(f"      Ref    : {op.basename(t1w_boldref)}")
    console.print(f"      Cmd    : {cmd}")
    console.print(f"      Out    : {op.basename(out)}")

    tag = f"[{ctx.sub}/{ctx.ses}/{ctx.run_label}/{ctx.desc}] {label}"

    def run_fn() -> tuple[bool, float]:
        if op.isfile(out) and not overwrite:
            console.print(f"      [yellow]→ {tag} exists, skip[/yellow]")
            return True, 0.0
        t0 = time.perf_counter()
        try:
            _run_with_modules(cmd, [ants_module])
            elapsed = time.perf_counter() - t0
            console.print(f"      [green]✓ {tag} done[/green] [dim]({_fmt_time(elapsed)})[/dim]")
            return True, elapsed
        except RuntimeError as exc:
            console.print(f"      [red]✗ {tag}: {exc}[/red]")
            return False, 0.0

    return Job(jid, ctx.sub, ctx.ses, ctx.task, ctx.run, ctx.desc, label, run_fn=run_fn)


def _plan_mni_job(
    ctx: RunCtx, boldref_to_t1w: str | None, t1w_to_mni: str | None,
    mni_boldref: str | None, ants_module: str, overwrite: bool,
) -> Job:
    jid = f"{ctx.prefix}{ctx.desc}_MNI"
    label = "MNI"

    missing = [
        n for n, v in [
            ("boldref_to_T1w xfm", boldref_to_t1w),
            ("T1w_to_MNI xfm", t1w_to_mni),
            ("MNI boldref", mni_boldref),
        ] if not v
    ]
    if missing:
        console.print(f"    [red]✗ MNI: missing {missing}[/red]")
        return Job(jid, ctx.sub, ctx.ses, ctx.task, ctx.run, ctx.desc, label,
                    planned_ok=False, run_fn=lambda: (False, 0.0))

    out = op.join(ctx.func_dir, f"{ctx.prefix}space-{MNI_SPACE}_desc-{ctx.desc}_bold.nii.gz")
    cmd = _ants_cmd(ctx.input_path, mni_boldref, [t1w_to_mni, boldref_to_t1w], out)

    console.print("    [cyan]→ MNI[/cyan]")
    console.print(f"      Module : {ants_module}")
    console.print(f"      XFM1   : {op.basename(t1w_to_mni)}  (T1w→MNI, applied 1st)")
    console.print(f"      XFM2   : {op.basename(boldref_to_t1w)}  (BOLD→T1w, applied 2nd)")
    console.print(f"      Ref    : {op.basename(mni_boldref)}")
    console.print(f"      Cmd    : {cmd}")
    console.print(f"      Out    : {op.basename(out)}")

    tag = f"[{ctx.sub}/{ctx.ses}/{ctx.run_label}/{ctx.desc}] {label}"

    def run_fn() -> tuple[bool, float]:
        if op.isfile(out) and not overwrite:
            console.print(f"      [yellow]→ {tag} exists, skip[/yellow]")
            return True, 0.0
        t0 = time.perf_counter()
        try:
            _run_with_modules(cmd, [ants_module])
            elapsed = time.perf_counter() - t0
            console.print(f"      [green]✓ {tag} done[/green] [dim]({_fmt_time(elapsed)})[/dim]")
            return True, elapsed
        except RuntimeError as exc:
            console.print(f"      [red]✗ {tag}: {exc}[/red]")
            return False, 0.0

    return Job(jid, ctx.sub, ctx.ses, ctx.task, ctx.run, ctx.desc, label, run_fn=run_fn)


def _plan_fsreg_job(
    ctx: RunCtx, boldref_to_t1w: str | None, fp_func_dir: str, fp_anat_dir: str,
    fs_subject: str, fs_subjects_dir: str, fs_module: str, overwrite: bool,
) -> tuple[Job | None, str | None, bool]:
    """Plan the boldref→fsnative .lta job (shared across fsnative/fsaverage and
    across all descs of this run). Returns (job_or_None, reg_file, ok)."""
    reg_file = op.join(ctx.func_dir, f"{ctx.prefix}from-boldref_to-fsnative_reg.lta")

    if op.isfile(reg_file) and not overwrite:
        console.print("    [cyan]→ boldref→fsnative reg[/cyan]")
        console.print(f"      [yellow]→ reg exists, skip ({op.basename(reg_file)})[/yellow]")
        return None, reg_file, True

    boldref_native = _find_boldref_native(fp_func_dir, ctx.sub, ctx.ses, ctx.task, ctx.acq, ctx.run)
    t1w_preproc = _find_t1w_preproc(fp_anat_dir, ctx.sub, ctx.ses)
    t1w_to_fsnative_xfm = _find_t1w_to_fsnative_xfm(fp_anat_dir, ctx.sub, ctx.ses)

    missing = [
        n for n, v in [
            ("native boldref", boldref_native),
            ("boldref_to_T1w xfm", boldref_to_t1w),
            ("T1w preproc", t1w_preproc),
            ("T1w_to_fsnative xfm", t1w_to_fsnative_xfm),
        ] if not v
    ]
    if missing:
        console.print(f"    [red]✗ fsnative-reg: missing {missing}[/red]")
        return None, None, False

    fs_orig = op.join(fs_subjects_dir, fs_subject, "mri", "orig.mgz")

    boldref_to_t1w_fixed = op.join(ctx.func_dir, f"{ctx.prefix}from-boldref_to-T1w_xfm_double.txt")
    boldref_to_t1w_lta = op.join(ctx.func_dir, f"{ctx.prefix}from-boldref_to-T1w_reg.lta")
    t1w_to_fsnative_fixed = op.join(ctx.func_dir, f"sub-{ctx.sub}_ses-{ctx.ses}_from-T1w_to-fsnative_xfm_double.txt")
    t1w_to_fsnative_lta = op.join(ctx.func_dir, f"sub-{ctx.sub}_ses-{ctx.ses}_from-T1w_to-fsnative_reg.lta")

    cmd1 = _lta_convert_cmd(boldref_to_t1w_fixed, boldref_native, t1w_preproc, boldref_to_t1w_lta)
    cmd2 = _lta_convert_cmd(t1w_to_fsnative_fixed, t1w_preproc, fs_orig, t1w_to_fsnative_lta, fs_subjects_dir)
    cmd3 = _concat_lta_cmd(boldref_to_t1w_lta, t1w_to_fsnative_lta, reg_file)

    console.print("    [cyan]→ boldref→fsnative reg (fMRIprep transforms, concatenated)[/cyan]")
    console.print(f"      Module : {fs_module}")
    console.print(f"      XFM1   : {op.basename(boldref_to_t1w)}  (boldref→T1w)")
    console.print(f"      XFM2   : {op.basename(t1w_to_fsnative_xfm)}  (T1w→fsnative)")
    console.print(f"      Cmd1   : {cmd1}")
    console.print(f"      Cmd2   : {cmd2}")
    console.print(f"      Cmd3   : {cmd3}")
    console.print(f"      Reg    : {op.basename(reg_file)}")

    tag = f"[{ctx.sub}/{ctx.ses}/{ctx.run_label}] fsnative-reg"
    jid = f"{ctx.prefix}fsreg"

    def run_fn() -> tuple[bool, float]:
        t0 = time.perf_counter()
        try:
            _fix_itk_xfm_precision(boldref_to_t1w, boldref_to_t1w_fixed)
            _run_with_modules(cmd1, [fs_module])
            if not (op.isfile(t1w_to_fsnative_lta) and not overwrite):
                _fix_itk_xfm_precision(t1w_to_fsnative_xfm, t1w_to_fsnative_fixed)
                _run_with_modules(cmd2, [fs_module])
            _run_with_modules(cmd3, [fs_module])
            _set_lta_subject(reg_file, fs_subject)
            elapsed = time.perf_counter() - t0
            console.print(f"      [green]✓ {tag} done[/green] [dim]({_fmt_time(elapsed)})[/dim]")
            return True, elapsed
        except RuntimeError as exc:
            console.print(f"      [red]✗ {tag}: {exc}[/red]")
            return False, 0.0

    job = Job(jid, ctx.sub, ctx.ses, ctx.task, ctx.run, "—", "fsnative-reg", run_fn=run_fn)
    return job, reg_file, True


def _plan_vol2surf_job(
    ctx: RunCtx, hemi: str, surf: str, fs_subject: str, fs_subjects_dir: str,
    fs_module: str, reg_file: str | None, reg_ok: bool, trg: str | None,
    space_label: str, depends_on: list[str], overwrite: bool,
) -> Job:
    label = f"{space_label} hemi-{hemi}"
    jid = f"{ctx.prefix}{ctx.desc}_{label}"

    if not reg_ok:
        console.print(f"    [red]✗ {label}: fsnative-reg prerequisites missing[/red]")
        return Job(jid, ctx.sub, ctx.ses, ctx.task, ctx.run, ctx.desc, label,
                    planned_ok=False, run_fn=lambda: (False, 0.0))

    out = op.join(
        ctx.func_dir,
        f"{ctx.prefix}hemi-{hemi}_space-{space_label}_desc-{ctx.desc}_bold.func.gii",
    )
    v2s_cmd = _vol2surf_cmd(ctx.input_path, out, hemi, surf, fs_subject, fs_subjects_dir, reg=reg_file, trg_subject=trg)

    console.print(f"    [cyan]→ {label}[/cyan]")
    console.print(f"      Module : {fs_module}")
    console.print(f"      Cmd    : {v2s_cmd}")
    console.print(f"      Out    : {op.basename(out)}")

    tag = f"[{ctx.sub}/{ctx.ses}/{ctx.run_label}/{ctx.desc}] {label}"

    def run_fn() -> tuple[bool, float]:
        if op.isfile(out) and not overwrite:
            console.print(f"      [yellow]→ {tag} exists, skip[/yellow]")
            return True, 0.0
        t0 = time.perf_counter()
        try:
            _run_with_modules(v2s_cmd, [fs_module])
            elapsed = time.perf_counter() - t0
            console.print(f"      [green]✓ {tag} done[/green] [dim]({_fmt_time(elapsed)})[/dim]")
            return True, elapsed
        except RuntimeError as exc:
            console.print(f"      [red]✗ {tag}: {exc}[/red]")
            return False, 0.0

    return Job(jid, ctx.sub, ctx.ses, ctx.task, ctx.run, ctx.desc, label,
               depends_on=depends_on, run_fn=run_fn)


def _plan_jobs_for_run(
    ctx: RunCtx, fp_func_dir: str, fp_anat_dir: str, fs_subject: str, fs_subjects_dir: str,
    spaces: list[str], ants_module: str, fs_module: str, overwrite: bool,
    fsreg_cache: dict[str, tuple[Job | None, str | None, bool]], jobs: list[Job],
) -> None:
    console.print(f"\n  [bold]task-{ctx.task}  {ctx.run_label}  desc-{ctx.desc}[/bold]")
    console.print(f"    Input : {op.basename(ctx.input_path)}")

    boldref_to_t1w = _find_boldref_to_t1w(fp_func_dir, ctx.sub, ctx.ses, ctx.task, ctx.acq, ctx.run)
    t1w_to_mni = _find_t1w_to_mni(fp_anat_dir)
    mni_boldref = _find_mni_boldref(fp_func_dir, ctx.sub, ctx.ses, ctx.task, ctx.acq, ctx.run)
    t1w_boldref = _find_t1w_boldref(fp_func_dir, ctx.sub, ctx.ses, ctx.task, ctx.acq, ctx.run)

    for space in spaces:
        if space == "T1w":
            jobs.append(_plan_t1w_job(ctx, boldref_to_t1w, t1w_boldref, ants_module, overwrite))

        elif space == "MNI":
            jobs.append(_plan_mni_job(ctx, boldref_to_t1w, t1w_to_mni, mni_boldref, ants_module, overwrite))

        elif space in ("fsnative", "fsaverage"):
            if ctx.prefix not in fsreg_cache:
                fsreg_cache[ctx.prefix] = _plan_fsreg_job(
                    ctx, boldref_to_t1w, fp_func_dir, fp_anat_dir, fs_subject, fs_subjects_dir, fs_module, overwrite
                )
                reg_job, _, _ = fsreg_cache[ctx.prefix]
                if reg_job is not None:
                    jobs.append(reg_job)

            reg_job, reg_file, reg_ok = fsreg_cache[ctx.prefix]
            depends_on = [reg_job.id] if reg_job is not None else []
            trg = "fsaverage" if space == "fsaverage" else None
            space_label = "fsaverage" if space == "fsaverage" else "fsnative"

            for hemi in ("L", "R"):
                jobs.append(_plan_vol2surf_job(
                    ctx, hemi, "white", fs_subject, fs_subjects_dir, fs_module,
                    reg_file, reg_ok, trg, space_label, depends_on, overwrite,
                ))


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
        "T1w,MNI,fsnative,fsaverage",
        "--spaces",
        help="Comma-separated target spaces: T1w, MNI, fsnative, fsaverage",
    ),
    ants_module: str = typer.Option("ants/2.5.1", "--ants-module"),
    fs_module: str = typer.Option("freesurfer/7.3.2", "--fs-module"),
    execute: bool = typer.Option(
        False,
        "--execute",
        help="Run transforms. Without this flag only the plan is printed.",
    ),
    overwrite: bool = typer.Option(False, "--overwrite"),
    n_jobs: int = typer.Option(
        1, "--n-jobs", "-J",
        help="Total parallel workers for the whole batch's job queue "
             "(ANTs/vol2surf calls across all subs/sessions/runs/spaces). "
             "1=sequential, -1=use all CPU cores.",
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

    workers = os.cpu_count() or 1 if n_jobs == -1 else max(1, n_jobs)

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
    tbl.add_row("n_jobs (workers)", str(workers))
    tbl.add_row("execute", str(execute))
    console.print(tbl)

    if not execute:
        console.print(
            "\n[yellow bold]DRY RUN — pass --execute to run transforms[/yellow bold]\n"
        )

    # ── Plan: build a flat job queue across the whole batch ─────────────────
    jobs: list[Job] = []
    fsreg_cache: dict[str, tuple[Job | None, str | None, bool]] = {}

    for sub, ses in subses_pairs:
        console.print(f"\n[bold cyan]sub-{sub}  ses-{ses}[/bold cyan]")
        func_dir = op.join(input_dir, f"sub-{sub}", f"ses-{ses}", "func")
        fp_func_dir = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "func")
        fp_anat_dir = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "anat")
        fs_subject = fs_subject_template.format(sub=sub, ses=ses)

        dirs_ok = True
        for path, label in [
            (func_dir, "tedana func"),
            (fp_func_dir, "fmriprep func"),
            (fp_anat_dir, "fmriprep anat"),
            (fs_subjects_dir, "FS subjects"),
        ]:
            if not op.isdir(path):
                console.print(f"  [red]✗ {label} dir not found: {path}[/red]")
                dirs_ok = False

        if not dirs_ok:
            for task in task_list:
                for desc in desc_list:
                    jid = f"sub-{sub}_ses-{ses}_task-{task}_desc-{desc}_setup"
                    jobs.append(Job(jid, sub, ses, task, None, desc, "(missing dirs)",
                                     planned_ok=False, run_fn=lambda: (False, 0.0)))
            continue

        for task in task_list:
            for desc in desc_list:
                pairs = _find_input_files(func_dir, sub, ses, task, acq, desc)
                if not pairs:
                    console.print(
                        f"  [yellow]⚠ task-{task} desc-{desc}: no input files found[/yellow]"
                    )
                    jid = f"sub-{sub}_ses-{ses}_task-{task}_desc-{desc}_noinput"
                    jobs.append(Job(jid, sub, ses, task, None, desc, "(no input)",
                                     planned_ok=False, run_fn=lambda: (False, 0.0)))
                    continue

                for run, path in pairs:
                    run_label = f"run-{run}" if run else "(no run entity)"
                    prefix = _bids_prefix(sub, ses, task, acq, run)
                    ctx = RunCtx(sub, ses, task, acq, run, run_label, desc, prefix, path, func_dir)
                    _plan_jobs_for_run(
                        ctx, fp_func_dir, fp_anat_dir, fs_subject, fs_subjects_dir,
                        space_list, ants_module, fs_module, overwrite, fsreg_cache, jobs,
                    )

    console.print(f"\n[bold]Total jobs in queue: {len(jobs)}[/bold]")

    # ── Execute: run the flat job queue across `workers` threads ────────────
    if execute:
        _save_sources(input_dir, fp_dir, space_list, fs_subjects_dir)
        console.print(f"[dim]Running {len(jobs)} job(s) across up to {workers} worker(s) …[/dim]\n")

    t_total_start = time.perf_counter()
    results = _run_jobs(jobs, workers, execute)
    t_total = time.perf_counter() - t_total_start

    # ── Summary ────────────────────────────────────────────────────────────
    console.print("\n")
    summary = Table(title="Results", show_lines=False)
    summary.add_column("sub")
    summary.add_column("ses")
    summary.add_column("task")
    summary.add_column("run")
    summary.add_column("desc")
    summary.add_column("job")
    summary.add_column("status")
    summary.add_column("wall time", justify="right")
    for job, ok, elapsed in results:
        summary.add_row(
            job.sub, job.ses, job.task, job.run or "—", job.desc, job.label,
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
    n_failed = sum(1 for _job, ok, _e in results if not ok)
    if n_failed:
        console.print(f"\n[red]{n_failed} job(s) failed.[/red]")
        raise typer.Exit(1)
    console.print("\n[green]All jobs completed successfully.[/green]")


if __name__ == "__main__":
    app()
