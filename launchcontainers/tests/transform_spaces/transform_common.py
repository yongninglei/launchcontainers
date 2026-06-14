#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2025-2026
# Apache-2.0 license
# -----------------------------------------------------------------------------
"""
transform_common.py

Shared helpers for the EPI vol2surf pipeline:

    0_check_affine.py     - is there already a usable volume->fsnative .lta?
    1_do_register.py      - if not, compute one (bbregister or ANTs)
    2_do_transformation.py - run mri_vol2surf with the resolved .lta

All three steps (and the `epi_vol2surf.py` orchestrator) import from here so
that command-building / module-loading logic lives in one place.
"""

from __future__ import annotations

import glob
import os.path as op
import subprocess
import time
from typing import Optional

SRC_SPACES = {"T1w", "native"}
REG_METHODS = {"bbregister", "ants"}

# Locations where Environment Modules initialisation script may live
_MODULE_INITS = [
    "/usr/share/Modules/init/bash",
    "/etc/profile.d/modules.sh",
    "/usr/local/Modules/init/bash",
]


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------


def fmt_time(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    m, s = divmod(s, 60)
    return (
        f"{int(m)}m {s:.1f}s" if m < 60 else f"{int(m // 60)}h {int(m % 60)}m {s:.1f}s"
    )


def run_with_modules(cmd_str: str, modules: list[str]) -> None:
    """Run a shell command after loading the given environment modules."""
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
            f"STDOUT: {result.stdout.decode()}\n"
            f"STDERR: {result.stderr.decode()}"
        )


# ---------------------------------------------------------------------------
# fMRIprep discovery
# ---------------------------------------------------------------------------


def find_fp_xfm_t1w_to_fsnative(fp_anat_dir: str, sub: str, ses: str) -> Optional[str]:
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


def find_fp_t1w_preproc(fp_anat_dir: str, sub: str, ses: str) -> Optional[str]:
    """Locate fMRIprep's sub-*_desc-preproc_T1w.nii.gz."""
    found = sorted(
        glob.glob(op.join(fp_anat_dir, f"sub-{sub}_ses-{ses}*_desc-preproc_T1w.nii.gz"))
    )
    return found[0] if found else None


# ---------------------------------------------------------------------------
# Step 0 - affine cache path / lookup
# ---------------------------------------------------------------------------


def affine_cache_path(cache_dir: str, sub: str, ses: str, src_space: str) -> str:
    """Where the resolved volume->fsnative .lta for this src_space is cached."""
    return op.join(cache_dir, f"sub-{sub}_ses-{ses}_from-{src_space}_to-fsnative_reg.lta")


# ---------------------------------------------------------------------------
# lta_convert / bbregister / ANTs command builders
# ---------------------------------------------------------------------------


def fix_itk_xfm_precision(itk_xfm: str, out_path: str) -> None:
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


def lta_convert_cmd(itk_xfm: str, src_vol: str, trg_vol: str, out_lta: str, subjects_dir: str = "") -> str:
    """Convert an ITK-format affine (double_3_3 header) to a FreeSurfer .lta.

    `src_vol`/`trg_vol` are arbitrary volumes defining the affine's
    source/target geometry (not necessarily FreeSurfer subject volumes).
    """
    prefix = f"SUBJECTS_DIR={subjects_dir} " if subjects_dir else ""
    return (
        f"{prefix}lta_convert"
        f" --initk {itk_xfm}"
        f" --src {src_vol}"
        f" --trg {trg_vol}"
        f" --outlta {out_lta}"
    )


def lta_convert_itk_cmd(
    itk_xfm: str, src_vol: str, fs_subject: str, subjects_dir: str, out_lta: str
) -> str:
    """Convert an ITK-format affine (e.g. fMRIprep's *_xfm.txt) to a FreeSurfer .lta,
    targeting the given FreeSurfer subject's orig.mgz."""
    trg = op.join(subjects_dir, fs_subject, "mri", "orig.mgz")
    return lta_convert_cmd(itk_xfm, src_vol, trg, out_lta, subjects_dir)


def bbregister_cmd(
    mov: str,
    reg_out: str,
    fs_subject: str,
    subjects_dir: str,
    contrast: str = "t1",
    init: str = "header",
) -> str:
    """Build a bbregister command: register `mov` onto the FreeSurfer subject.

    contrast: "t1" | "t2" | "bold"  (selects --t1 / --t2 / --bold)
    init:     "header" | "fsl" | "coreg"  (selects --init-header / --init-fsl / --init-coreg)
    """
    init_flag = {
        "header": "--init-header",
        "fsl": "--init-fsl",
        "coreg": "--init-coreg",
    }[init]
    return (
        f"SUBJECTS_DIR={subjects_dir}"
        f" bbregister"
        f" --s {fs_subject}"
        f" --mov {mov}"
        f" {init_flag}"
        f" --{contrast}"
        f" --reg {reg_out}"
    )


def ants_register_cmd(mov: str, fixed: str, out_prefix: str) -> str:
    """Rigid-body ANTs registration of `mov` onto `fixed`. Writes
    {out_prefix}0GenericAffine.mat plus a {out_prefix}Warped.nii.gz."""
    return (
        f"antsRegistration --dimensionality 3 --float 0"
        f" --output [{out_prefix},{out_prefix}Warped.nii.gz]"
        f" --interpolation Linear"
        f" --winsorize-image-intensities [0.005,0.995]"
        f" --use-histogram-matching 0"
        f" --transform Rigid[0.1]"
        f" --metric MI[{fixed},{mov},1,32,Regular,0.25]"
        f" --convergence [1000x500x250x100,1e-6,10]"
        f" --shrink-factors 8x4x2x1"
        f" --smoothing-sigmas 3x2x1x0vox"
    )


def ants_affine_to_lta_cmds(
    ants_affine_mat: str,
    mov: str,
    fixed: str,
    fs_subject: str,
    subjects_dir: str,
    out_lta: str,
) -> list[str]:
    """Convert an ANTs `*0GenericAffine.mat` (mov->fixed) to a FreeSurfer .lta
    (mov->fixed). Two steps: ANTs binary .mat -> ITK text, then lta_convert."""
    itk_txt = ants_affine_mat.replace(".mat", ".txt")
    convert_cmd = f"ConvertTransformFile 3 {ants_affine_mat} {itk_txt}"
    lta_cmd = (
        f"SUBJECTS_DIR={subjects_dir}"
        f" lta_convert"
        f" --initk {itk_txt}"
        f" --src {mov}"
        f" --trg {fixed}"
        f" --outlta {out_lta}"
    )
    return [convert_cmd, lta_cmd]


def set_lta_subject(lta_path: str, subject: str) -> None:
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


def concat_lta_cmd(lta_first: str, lta_second: str, out_lta: str) -> str:
    """Concatenate two .lta transforms: out = lta_second(lta_first(x)).

    i.e. `lta_first` maps src->intermediate, `lta_second` maps intermediate->dst.
    """
    return f"mri_concatenate_lta {lta_first} {lta_second} {out_lta}"


# ---------------------------------------------------------------------------
# Step 2 - mri_vol2surf
# ---------------------------------------------------------------------------


# The 6 surf x interp combinations projected for every input file.
SURF_INTERP_COMBOS: list[tuple[str, str]] = [
    ("white", "nearest"),
    ("white", "trilinear"),
    ("pial", "nearest"),
    ("pial", "trilinear"),
    ("midthickness", "nearest"),
    ("midthickness", "trilinear"),
]


def desc_label(surf: str, interp: str) -> str:
    """e.g. ("white", "nearest") -> "whiteNearest" """
    return f"{surf}{interp.capitalize()}"


def derive_output_path(
    src: str,
    hemi: str,
    space_label_in: str,
    space_label_out: str,
    out_ext: str,
    desc: Optional[str] = None,
) -> str:
    """Derive a surface-space output path from a volume-space input path.

    ..._space-{space_label_in}_...{ext} -> ..._hemi-{H}_space-{space_label_out}_...{out_ext}
    If `_space-{space_label_in}_` is absent, append `_hemi-{H}_space-{space_label_out}`
    before the extension.

    If `desc` is given, `_desc-{desc}_` is inserted before the final
    underscore-separated component (the BIDS suffix, e.g. "statmap").
    """
    base = op.basename(src)
    in_token = f"_space-{space_label_in}_"
    out_token = f"_hemi-{hemi}_space-{space_label_out}_"

    if in_token in base:
        new_base = base.replace(in_token, out_token)
    else:
        new_base = f"{base}{out_token.rstrip('_')}"

    for ext in (".nii.gz", ".mgz", ".nii"):
        if new_base.endswith(ext):
            new_base = new_base[: -len(ext)]
            break

    if desc:
        if "_" in new_base:
            prefix, suffix = new_base.rsplit("_", 1)
            new_base = f"{prefix}_desc-{desc}_{suffix}"
        else:
            new_base = f"{new_base}_desc-{desc}"

    new_base += out_ext

    return op.join(op.dirname(src), new_base)


def vol2surf_cmd(
    src: str,
    out: str,
    hemi: str,
    fs_subject: str,
    subjects_dir: str,
    reg: str,
    surf: str = "white",
    proj_frac: Optional[float] = None,
    proj_frac_avg: Optional[tuple[float, float, float]] = None,
    interp: str = "trilinear",
    trgsubject: Optional[str] = None,
) -> str:
    """Build an `mri_vol2surf` command.

    surf:           "white" | "pial" | "midthickness" | ...
    proj_frac:      single fractional projection along the surface normal (--projfrac)
    proj_frac_avg:  (start, stop, step) for --projfrac-avg (overrides proj_frac)
    interp:         "nearest" | "trilinear"
    trgsubject:     resample to this FreeSurfer subject (e.g. "fsaverage")
    """
    fs_hemi = "lh" if hemi == "L" else "rh"
    cmd = (
        f"SUBJECTS_DIR={subjects_dir}"
        f" mri_vol2surf"
        f" --src {src}"
        f" --reg {reg}"
        f" --hemi {fs_hemi}"
        f" --surf {surf}"
        f" --interp {interp}"
        f" --o {out}"
    )
    if proj_frac_avg is not None:
        start, stop, step = proj_frac_avg
        cmd += f" --projfrac-avg {start} {stop} {step}"
    elif proj_frac is not None:
        cmd += f" --projfrac {proj_frac}"
    if trgsubject:
        cmd += f" --trgsubject {trgsubject}"
    return cmd


# ---------------------------------------------------------------------------
# Step 0 - check_affine
# ---------------------------------------------------------------------------


def check_affine(
    sub: str,
    ses: str,
    src_space: str,
    fp_dir: str,
    fs_subjects_dir: str,
    fs_subject: str,
    cache_dir: str,
    fs_module: str,
    execute: bool,
    overwrite: bool,
    lta_src_vol: Optional[str] = None,
) -> dict:
    """Return a dict describing whether a volume->fsnative .lta is available.

    `lta_src_vol`, if given, is used as the `--src` volume when converting
    fMRIprep's ITK xfm to .lta (instead of fMRIprep's desc-preproc_T1w.nii.gz).
    The resulting .lta then has its "src" geometry recorded from this volume,
    which `mri_vol2surf` requires to match the volume passed as `--src` in
    step 2 (otherwise: "ERROR: source volume is neither source nor target of
    the registration"). Pass one of the actual files that will be projected
    in step 2 (they all share the same space-T1w grid).

    Keys: found (bool), lta_path (str), source (str), log (list[str]),
          needs_register (bool)
    """
    log: list[str] = []
    out_lta = affine_cache_path(cache_dir, sub, ses, src_space)

    if op.isfile(out_lta) and not overwrite:
        log.append(f"cached .lta found: {out_lta}")
        return {"found": True, "lta_path": out_lta, "source": "cached", "log": log, "needs_register": False}

    if src_space == "T1w":
        fp_anat_dir = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "anat")
        fp_xfm = find_fp_xfm_t1w_to_fsnative(fp_anat_dir, sub, ses)
        if fp_xfm:
            t1w_vol = lta_src_vol or find_fp_t1w_preproc(fp_anat_dir, sub, ses)
            if not t1w_vol:
                log.append(f"[red]found fMRIprep xfm but no desc-preproc_T1w.nii.gz in {fp_anat_dir}[/red]")
                return {"found": False, "lta_path": out_lta, "source": "none", "log": log, "needs_register": True}

            fixed_xfm = op.join(cache_dir, op.basename(fp_xfm).replace(".txt", "_double.txt"))
            cmd = lta_convert_itk_cmd(fixed_xfm, t1w_vol, fs_subject, fs_subjects_dir, out_lta)
            log.append("source       : fMRIprep from-T1w_to-fsnative xfm (lta_convert)")
            log.append(f"  ITK xfm    : {fp_xfm}")
            log.append(f"  fixed xfm  : {fixed_xfm}  (float_3_3 -> double_3_3 header fix)")
            log.append(f"  T1w vol    : {t1w_vol}")
            log.append(f"  Cmd        : {cmd}")
            log.append(f"  Out lta    : {out_lta}")

            if execute:
                t0 = time.perf_counter()
                fix_itk_xfm_precision(fp_xfm, fixed_xfm)
                run_with_modules(cmd, [fs_module])
                set_lta_subject(out_lta, fs_subject)
                log.append(f"  [green]done[/green] ({fmt_time(time.perf_counter() - t0)})")
                return {"found": True, "lta_path": out_lta, "source": "fmriprep", "log": log, "needs_register": False}
            else:
                log.append("  [yellow](dry-run: not converted yet)[/yellow]")
                return {"found": True, "lta_path": out_lta, "source": "fmriprep", "log": log, "needs_register": False}

    log.append(f"no existing transform found for src-space={src_space}; run 1_do_register.py")
    return {"found": False, "lta_path": out_lta, "source": "none", "log": log, "needs_register": True}


# ---------------------------------------------------------------------------
# Step 1 - do_register
# ---------------------------------------------------------------------------


def do_register(
    sub: str,
    ses: str,
    src_space: str,
    method: str,
    mov: str,
    fp_dir: str,
    fs_subjects_dir: str,
    fs_subject: str,
    cache_dir: str,
    bbr_contrast: Optional[str],
    bbr_init: str,
    fs_module: str,
    ants_module: str,
    execute: bool,
    overwrite: bool,
) -> dict:
    """Compute (or print the plan for) a volume->fsnative .lta.

    Returns dict: lta_path (str), cmds (list[str]), log (list[str])
    """
    log: list[str] = []
    out_lta = affine_cache_path(cache_dir, sub, ses, src_space)

    if op.isfile(out_lta) and not overwrite:
        log.append(f"[yellow]cached .lta already exists, skipping ({out_lta})[/yellow]")
        return {"lta_path": out_lta, "cmds": [], "log": log}

    cmds: list[str] = []

    if method == "bbregister":
        contrast = bbr_contrast or {"T1w": "t1", "native": "bold"}[src_space]
        cmd = bbregister_cmd(mov, out_lta, fs_subject, fs_subjects_dir, contrast=contrast, init=bbr_init)
        cmds.append(cmd)
        log.append(f"method       : bbregister --{contrast} --init-{bbr_init}")
        log.append(f"  mov        : {mov}")
        log.append(f"  Cmd        : {cmd}")

    elif method == "ants":
        out_prefix = op.join(cache_dir, f"sub-{sub}_ses-{ses}_from-{src_space}_to-")

        if src_space == "T1w":
            fixed = op.join(fs_subjects_dir, fs_subject, "mri", "orig.mgz")
            prefix = out_prefix + "fsnative_"
            cmds.append(ants_register_cmd(mov, fixed, prefix))
            cmds += ants_affine_to_lta_cmds(
                prefix + "0GenericAffine.mat", mov, fixed, fs_subject, fs_subjects_dir, out_lta
            )
            log.append("method       : ANTs rigid (T1w -> FS orig.mgz)")
            log.append(f"  mov        : {mov}")
            log.append(f"  fixed      : {fixed}")
            for c in cmds:
                log.append(f"  Cmd        : {c}")

        else:  # native
            fp_anat_dir = op.join(fp_dir, f"sub-{sub}", f"ses-{ses}", "anat")
            fixed = find_fp_t1w_preproc(fp_anat_dir, sub, ses)
            if not fixed:
                log.append(f"[red]No desc-preproc_T1w.nii.gz found in {fp_anat_dir}[/red]")
                return {"lta_path": out_lta, "cmds": [], "log": log}

            t1w_to_fsnative = affine_cache_path(cache_dir, sub, ses, "T1w")
            if not op.isfile(t1w_to_fsnative):
                log.append(
                    f"[red]No T1w->fsnative .lta found at {t1w_to_fsnative}.[/red]\n"
                    "  Run 0_check_affine.py --src-space T1w first."
                )
                return {"lta_path": out_lta, "cmds": [], "log": log}

            prefix = out_prefix + "T1w_"
            mov_to_t1w_lta = prefix + "reg.lta"
            cmds.append(ants_register_cmd(mov, fixed, prefix))
            cmds += ants_affine_to_lta_cmds(
                prefix + "0GenericAffine.mat", mov, fixed, fs_subject, fs_subjects_dir, mov_to_t1w_lta
            )
            cmds.append(concat_lta_cmd(mov_to_t1w_lta, t1w_to_fsnative, out_lta))

            log.append("method       : ANTs rigid (native -> T1w), concatenated with cached T1w->fsnative")
            log.append(f"  mov        : {mov}")
            log.append(f"  fixed (T1w): {fixed}")
            log.append(f"  T1w->fsnative lta: {t1w_to_fsnative}")
            for c in cmds:
                log.append(f"  Cmd        : {c}")

    else:
        raise RuntimeError(f"Unknown --method: {method}")

    log.append(f"  Out lta    : {out_lta}")

    if execute:
        modules = [fs_module] if method == "bbregister" else [ants_module, fs_module]
        for cmd in cmds:
            t0 = time.perf_counter()
            run_with_modules(cmd, modules)
            log.append(f"  [green]done[/green] ({fmt_time(time.perf_counter() - t0)})  {cmd.split()[0]}")
        set_lta_subject(out_lta, fs_subject)

    return {"lta_path": out_lta, "cmds": cmds, "log": log}


# ---------------------------------------------------------------------------
# Step 2 - project_file
# ---------------------------------------------------------------------------


def project_file(
    src: str,
    reg: str,
    fs_subject: str,
    fs_subjects_dir: str,
    hemis: list[str],
    surf: str,
    interp: str,
    proj_frac: Optional[float],
    proj_frac_avg: Optional[tuple[float, float, float]],
    trgsubject: Optional[str],
    space_label_in: str,
    space_label_out: str,
    out_ext: str,
    fs_module: str,
    execute: bool,
    overwrite: bool,
) -> tuple[bool, float, list[str]]:
    """Project one volume to the surface for every hemi, using a single
    surf/interp combination. The output filename gets a
    `_desc-{surf}{Interp}_` label (see `desc_label`) so that different
    surf/interp combinations can coexist side by side.

    Returns (ok, elapsed, log) - `log` is a list of (rich-markup) lines
    describing what was/would be done, safe to print from the caller even
    when this ran in a worker process.
    """
    t0_run = time.perf_counter()
    ok = True
    log: list[str] = []
    space_out = trgsubject or space_label_out
    desc = desc_label(surf, interp)

    for hemi in hemis:
        out = derive_output_path(src, hemi, space_label_in, space_out, out_ext, desc=desc)
        cmd = vol2surf_cmd(
            src, out, hemi, fs_subject, fs_subjects_dir, reg,
            surf=surf, proj_frac=proj_frac, proj_frac_avg=proj_frac_avg,
            interp=interp, trgsubject=trgsubject,
        )

        log.append(f"    [cyan]hemi-{hemi} desc-{desc}[/cyan]  {op.basename(out)}")
        log.append(f"      Cmd : {cmd}")

        if not execute:
            continue
        if op.isfile(out) and not overwrite:
            log.append("      [yellow]→ exists, skip[/yellow]")
            continue

        t0 = time.perf_counter()
        try:
            run_with_modules(cmd, [fs_module])
            log.append(f"      [green]✓ done[/green] [dim]({fmt_time(time.perf_counter() - t0)})[/dim]")
        except RuntimeError as exc:
            log.append(f"      [red]✗ {exc}[/red]")
            ok = False

    return ok, time.perf_counter() - t0_run, log


# ---------------------------------------------------------------------------
# 01_register.py - generic registration (mov -> fixed/fs_subject)
# ---------------------------------------------------------------------------


def register_generic(
    mov: str,
    method: str,
    out_prefix: str,
    fixed: Optional[str] = None,
    fs_subject: Optional[str] = None,
    fs_subjects_dir: Optional[str] = None,
    bbr_contrast: str = "t1",
    bbr_init: str = "header",
    fs_module: str = "freesurfer/7.3.2",
    ants_module: str = "ants",
    execute: bool = False,
    overwrite: bool = False,
) -> dict:
    """Register `mov` onto a target, tagging outputs with `_desc-{method}_`.

    method == "bbregister":
        Registers `mov` directly onto `fs_subject` (requires `fs_subject` and
        `fs_subjects_dir`). Produces a direct mov->fsnative .lta:
            {out_prefix}_desc-bbregister_reg.lta

    method == "ants":
        Rigid ANTs registration of `mov` onto `fixed` (any volume), then
        converted to .lta:
            {out_prefix}_desc-ants_0GenericAffine.mat
            {out_prefix}_desc-ants_reg.lta
        If `fs_subject`/`fs_subjects_dir` are also given, the .lta's
        `subject` field is set (required by mri_vol2surf when `fixed` is a
        FreeSurfer volume, e.g. orig.mgz).

    Returns dict: lta_path (str), cmds (list[str]), log (list[str])
    """
    if method not in REG_METHODS:
        raise ValueError(f"Unknown method: {method}  Valid: {REG_METHODS}")

    log: list[str] = []
    cmds: list[str] = []

    if method == "bbregister":
        if not (fs_subject and fs_subjects_dir):
            raise ValueError("--method bbregister requires fs_subject and fs_subjects_dir")

        out_lta = f"{out_prefix}_desc-bbregister_reg.lta"
        if op.isfile(out_lta) and not overwrite:
            log.append(f"[yellow]cached .lta already exists, skipping ({out_lta})[/yellow]")
            return {"lta_path": out_lta, "cmds": [], "log": log}

        cmd = bbregister_cmd(mov, out_lta, fs_subject, fs_subjects_dir, contrast=bbr_contrast, init=bbr_init)
        cmds.append(cmd)
        log.append(f"method       : bbregister --{bbr_contrast} --init-{bbr_init}")
        log.append(f"  mov        : {mov}")
        log.append(f"  fs_subject : {fs_subject}")
        log.append(f"  Cmd        : {cmd}")
        log.append(f"  Out lta    : {out_lta}")

        if execute:
            t0 = time.perf_counter()
            run_with_modules(cmd, [fs_module])
            set_lta_subject(out_lta, fs_subject)
            log.append(f"  [green]done[/green] ({fmt_time(time.perf_counter() - t0)})")

        return {"lta_path": out_lta, "cmds": cmds, "log": log}

    # method == "ants"
    if not fixed:
        raise ValueError("--method ants requires --fixed")

    out_lta = f"{out_prefix}_desc-ants_reg.lta"
    if op.isfile(out_lta) and not overwrite:
        log.append(f"[yellow]cached .lta already exists, skipping ({out_lta})[/yellow]")
        return {"lta_path": out_lta, "cmds": [], "log": log}

    ants_prefix = f"{out_prefix}_desc-ants_"
    ants_mat = ants_prefix + "0GenericAffine.mat"
    itk_txt = ants_prefix + "0GenericAffine.txt"
    itk_txt_fixed = ants_prefix + "0GenericAffine_double.txt"

    register_cmd = ants_register_cmd(mov, fixed, ants_prefix)
    convert_cmd = f"ConvertTransformFile 3 {ants_mat} {itk_txt}"
    lta_cmd = lta_convert_cmd(itk_txt_fixed, mov, fixed, out_lta, fs_subjects_dir or "")
    cmds = [register_cmd, convert_cmd, lta_cmd]

    log.append("method       : ANTs rigid registration")
    log.append(f"  mov        : {mov}")
    log.append(f"  fixed      : {fixed}")
    for c in cmds:
        log.append(f"  Cmd        : {c}")
    log.append(f"  Out lta    : {out_lta}")

    if execute:
        t0 = time.perf_counter()
        run_with_modules(register_cmd, [ants_module])
        run_with_modules(convert_cmd, [ants_module])
        fix_itk_xfm_precision(itk_txt, itk_txt_fixed)
        run_with_modules(lta_cmd, [fs_module])
        if fs_subject:
            set_lta_subject(out_lta, fs_subject)
        log.append(f"  [green]done[/green] ({fmt_time(time.perf_counter() - t0)})")

    return {"lta_path": out_lta, "cmds": cmds, "log": log}


# ---------------------------------------------------------------------------
# 02_transform.py - generic affine resolution (any affine -> .lta)
# ---------------------------------------------------------------------------


def resolve_affine_to_lta(
    affine: str,
    out_dir: str,
    src_vol: Optional[str] = None,
    trg_vol: Optional[str] = None,
    fs_subject: Optional[str] = None,
    fs_subjects_dir: Optional[str] = None,
    fs_module: str = "freesurfer/7.3.2",
    ants_module: str = "ants",
    execute: bool = False,
    overwrite: bool = False,
) -> dict:
    """Ensure `affine` is usable by mri_vol2surf as a .lta, converting if needed.

    `affine` may be:
        *.lta   - used as-is (returned unchanged)
        *.mat   - ANTs `*0GenericAffine.mat`, converted via ConvertTransformFile + lta_convert
        *.txt   - ITK-format affine (e.g. fMRIprep's *_xfm.txt), converted via lta_convert

    `src_vol` defines the affine's source geometry and must match the volume
    that will later be passed to mri_vol2surf as `--src` (required when
    converting). `trg_vol` defines the target geometry; if not given, defaults
    to `{fs_subjects_dir}/{fs_subject}/mri/orig.mgz`.

    Returns dict: lta_path (str), cmds (list[str]), log (list[str])
    """
    log: list[str] = []

    if affine.endswith(".lta"):
        log.append(f"affine is already .lta: {affine}")
        return {"lta_path": affine, "cmds": [], "log": log}

    if affine.endswith(".mat"):
        kind = "ants"
    elif affine.endswith(".txt"):
        kind = "itk"
    else:
        raise ValueError(f"Unsupported affine format (expected .lta, .mat, or .txt): {affine}")

    if trg_vol is None:
        if fs_subject and fs_subjects_dir:
            trg_vol = op.join(fs_subjects_dir, fs_subject, "mri", "orig.mgz")
        else:
            raise ValueError("resolve_affine_to_lta: need trg_vol or fs_subject+fs_subjects_dir")
    if src_vol is None:
        raise ValueError("resolve_affine_to_lta: need src_vol to convert affine to .lta")

    base = op.basename(affine)
    for ext in (".mat", ".txt"):
        if base.endswith(ext):
            base = base[: -len(ext)]
            break
    out_lta = op.join(out_dir, f"{base}_reg.lta")

    if op.isfile(out_lta) and not overwrite:
        log.append(f"cached .lta found: {out_lta}")
        return {"lta_path": out_lta, "cmds": [], "log": log}

    cmds: list[str] = []
    if kind == "ants":
        itk_txt = op.join(out_dir, f"{base}.txt")
        convert_cmd = f"ConvertTransformFile 3 {affine} {itk_txt}"
        cmds.append(convert_cmd)
        log.append("source       : ANTs affine (ConvertTransformFile + lta_convert)")
        log.append(f"  ANTs mat   : {affine}")
    else:
        itk_txt = affine
        log.append("source       : ITK affine xfm (lta_convert)")
        log.append(f"  ITK xfm    : {affine}")

    itk_txt_fixed = op.join(out_dir, f"{base}_double.txt")
    lta_cmd = lta_convert_cmd(itk_txt_fixed, src_vol, trg_vol, out_lta, fs_subjects_dir or "")
    cmds.append(lta_cmd)

    log.append(f"  src vol    : {src_vol}")
    log.append(f"  trg vol    : {trg_vol}")
    for c in cmds:
        log.append(f"  Cmd        : {c}")
    log.append(f"  Out lta    : {out_lta}")

    if execute:
        t0 = time.perf_counter()
        if kind == "ants":
            run_with_modules(cmds[0], [ants_module])
        fix_itk_xfm_precision(itk_txt, itk_txt_fixed)
        run_with_modules(lta_cmd, [fs_module])
        if fs_subject:
            set_lta_subject(out_lta, fs_subject)
        log.append(f"  [green]done[/green] ({fmt_time(time.perf_counter() - t0)})")

    return {"lta_path": out_lta, "cmds": cmds, "log": log}
