# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2024-2025
# All rights reserved.
#
# This script is distributed under the Apache-2.0 license.
#
# power_analysis_local_parallel.py
# --------------------------------
# Local (single big node, e.g. BCBL cajal04) parallel version of
# power_analysis_ohbm.py.  Same GLM logic, but all subjects in one process
# pool so every core is busy:
#
#   Stage A  load      one job per (sub, ses, run): read fsnative GIFTI, keep only
#                      the ROI-union vertices, drop start_scans, z-score,
#                      read confounds/events.  ~1100 jobs -> all workers busy.
#   assemble           per subject (main process): concatenate runs, build the
#                      design matrix exactly as power_analysis_ohbm.py, save a
#                      small ROI-only cache (.npz) + DM diagnostics.
#   Stage B  fit       one job per (sub, n_runs, iter): fit run_glm on the row-
#                      slice of the selected runs, mean T per ROI per contrast.
#                      ~3300 jobs (11 subs x ~100 run counts x 3 iters).
#
# Differences from power_analysis_ohbm.py
#   - only ROI-union vertices are ever held in memory or cached (the original
#     cached the whole hemisphere, ~10 GB/subject; /bcbl/home is full)
#   - run discovery by path instead of BIDSLayout (no SQLite index, no races)
#   - z-scoring a vertex row is independent of other rows, so slicing before
#     z-scoring gives identical values
#   - random subsets: same generator sequence as the original (fresh
#     default_rng(seed) per subject/hemi, drawn n_runs = 1..total in order)
#   - saves a long CSV with every (n_runs, iter, selected runs, contrast, roi)
#
# Outputs  <base>/BIDS/derivatives/l1_surface/analysis-<output_name>/sub-XX/
#   sub-XX_hemi-L_roicache.npz                 ROI-only Y, X, contrasts, runs
#   allses/design_matrix.png, *_design_matrix.csv, *_contrast_matrix.png
#   sub-XX_hemi-L_power_results.npz            (n_runs, n_iter, n_contrast, n_roi)
#   sub-XX_hemi-L_power_long.csv               one row per GLM x contrast x roi
#   sub-XX_hemi-L_power_summary.tsv            mean/std/median over iters
#   sub-XX_hemi-L_power_plot.png
# and at the analysis root: group_power_long.csv, run_inventory.csv
#
# Resume: a subject whose power_results.npz exists is skipped (unless --force);
# a subject whose roicache.npz exists skips Stage A.
# -----------------------------------------------------------------------------
from __future__ import annotations

import os

# One BLAS/OpenMP thread per worker process — parallelism comes from the pool.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import csv
import glob
import multiprocessing as mp
import importlib.util as _ilu
import json
import os.path as op
import re
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import typer
import yaml
from nilearn.glm.contrasts import compute_contrast
from nilearn.glm.first_level import make_first_level_design_matrix
from nilearn.glm.first_level.first_level import run_glm
from nilearn.plotting import plot_contrast_matrix, plot_design_matrix
from nilearn.surface import load_surf_data
from rich.console import Console
from scipy import stats

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)

# ---------------------------------------------------------------------------
# Shared helpers from glm_strategy (same source as power_analysis_ohbm.py)
# ---------------------------------------------------------------------------
_THIS_DIR = op.dirname(op.abspath(__file__))
_GLM_STRATEGY = op.join(_THIS_DIR, "..", "glm_strategy", "glm_surface_check_model_strategy.py")
_spec = _ilu.spec_from_file_location("_glm_strategy", _GLM_STRATEGY)
_glm_mod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_glm_mod)

replace_prefix_and_suffix = _glm_mod.replace_prefix_and_suffix
load_contrasts            = _glm_mod.load_contrasts
load_confound_strategy    = _glm_mod.load_confound_strategy
_resolve_confound_columns = _glm_mod._resolve_confound_columns
_load_rerun_exclusions    = _glm_mod._load_rerun_exclusions

HELPER_LABEL_ROOT = op.expanduser(
    "~/tlei/soft/app_manual_delineation_helper/example_dataset/freesurfer-with_t2"
)
DEFAULT_EXCLUDE = {"02": {"07", "08", "09", "10"}}   # same exclusion as analysis-final_v3 figures


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def sessions_from_l1(l1_ref_dir: str, sub: str) -> list[str]:
    """Sessions with an individual-session GLM in the reference l1 analysis."""
    out = []
    for d in sorted(glob.glob(op.join(l1_ref_dir, f"sub-{sub}", "ses-[0-9][0-9]"))):
        out.append(op.basename(d).replace("ses-", ""))
    return [s for s in out if s not in DEFAULT_EXCLUDE.get(sub, set())]


def sessions_from_file(path: str, sub: str) -> list[str]:
    delim = "\t" if path.endswith(".tsv") else ","
    seen: dict[str, None] = {}
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter=delim):
            if str(row.get("sub", "")).strip().zfill(2) != sub:
                continue
            if "RUN" in row and str(row["RUN"]).strip() != "True":
                continue
            seen[str(row["ses"]).strip().zfill(2)] = None
    return list(seen)


def discover_runs(bids_dir, fmriprep_dir, sub, ses, task, hemi, space, rerun_excl) -> tuple[list[dict], list[dict]]:
    """Runs with raw events + fMRIPrep surface bold + confounds + sidecar json."""
    ok, missing = [], []
    ev_glob = op.join(bids_dir, f"sub-{sub}", f"ses-{ses}", "func",
                      f"sub-{sub}_ses-{ses}_task-{task}_run-*_events.tsv")
    excl = rerun_excl.get((sub, ses, task), set())
    for ev in sorted(glob.glob(ev_glob)):
        run = re.search(r"_run-(\d+)_", op.basename(ev)).group(1).zfill(2)
        if run in excl:
            missing.append({"sub": sub, "ses": ses, "run": run, "reason": "rerun-compensated"})
            continue
        fdir = op.join(fmriprep_dir, f"sub-{sub}", f"ses-{ses}", "func")
        stem = f"sub-{sub}_ses-{ses}_task-{task}_run-{run}"
        func = op.join(fdir, f"{stem}_hemi-{hemi}_space-{space}_bold.func.gii")
        side = op.join(fdir, f"{stem}_hemi-{hemi}_space-{space}_bold.json")
        conf = op.join(fdir, f"{stem}_desc-confounds_timeseries.tsv")
        lacking = [n for n, p in (("func", func), ("json", side), ("confounds", conf)) if not op.exists(p)]
        if lacking:
            missing.append({"sub": sub, "ses": ses, "run": run, "reason": "missing " + "+".join(lacking)})
            continue
        ok.append({"sub": sub, "ses": ses, "run": run, "func": func, "json": side,
                   "conf": conf, "events": ev})
    return ok, missing


def load_roi_masks(roi_yaml: str, hemi: str, label_dir: str) -> dict[str, np.ndarray]:
    with open(roi_yaml) as f:
        cfg = yaml.safe_load(f)
    masks = {}
    for roi, paths in cfg.get("rois", {}).items():
        lf = (paths or {}).get(hemi)
        if not lf:
            continue
        p = op.join(label_dir, lf)
        if not op.exists(p):
            console.print(f"  [yellow]WARNING[/yellow] missing label {p} — skipping {roi}")
            continue
        masks[roi] = load_surf_data(p).astype(int)
    return masks


# ---------------------------------------------------------------------------
# Stage A worker: one run
# ---------------------------------------------------------------------------

def _load_run(job: dict) -> dict:
    t0 = time.time()
    data = np.asarray(load_surf_data(job["func"]), dtype=np.float64)       # (n_verts, n_t)
    data = data[job["roi_union"], job["start_scans"]:]
    y = stats.zscore(data, axis=1).astype(np.float32)
    with open(job["json"]) as jf:
        t_r = float(json.load(jf)["RepetitionTime"])
    events = pd.read_csv(job["events"], sep="\t")
    confounds = pd.read_csv(job["conf"], sep="\t")
    keys = _resolve_confound_columns(job["confound_strategy"], list(confounds.columns))
    ck = confounds[keys].copy()
    if "framewise_displacement" in ck.columns:
        ck.loc[ck.index[0], "framewise_displacement"] = np.nanmean(ck["framewise_displacement"])
    ck = ck.iloc[job["start_scans"]:].reset_index(drop=True)
    return {"key": (job["sub"], job["ses"], job["run"]), "y": y, "t_r": t_r,
            "events": events, "confounds": ck, "secs": time.time() - t0}


# ---------------------------------------------------------------------------
# Assemble one subject (main process) — mirrors prepare_allses_data()
# ---------------------------------------------------------------------------

def assemble_subject(loaded: list[dict], contrast_yaml: str, slice_time_ref: float):
    loaded = sorted(loaded, key=lambda d: d["key"])
    Ys, fts, evs, cfs, bounds, labels = [], [], [], [], [], []
    total = 0
    for d in loaded:
        n = d["y"].shape[1]
        t_r = d["t_r"]
        ev = d["events"].copy()
        ev.loc[:, "onset"] = ev["onset"] + total * t_r
        evs.append(ev[ev["trial_type"] != "baseline"])
        fts.append(t_r * ((np.arange(n) + slice_time_ref) + total))
        cfs.append(d["confounds"])
        Ys.append(d["y"])
        bounds.append((total, total + n))
        labels.append(f"ses-{d['key'][1]}_run-{d['key'][2]}")
        total += n
    Y = np.concatenate(Ys, axis=1)
    events = pd.concat(evs).map(replace_prefix_and_suffix)
    conf = pd.concat(cfs, axis=0).dropna(axis=1, how="any")
    dm = make_first_level_design_matrix(np.concatenate(fts), events=events, hrf_model="spm",
                                        drift_model=None, add_regs=conf)
    dm_std = dm.apply(stats.zscore, axis=0)
    dm_std["constant"] = np.ones(len(dm_std)).astype(int)
    contrasts = load_contrasts(contrast_yaml, dm)
    return Y, np.asarray(dm_std, dtype=np.float32), dm_std, contrasts, bounds, labels


def save_dm_diagnostics(dm_std, contrasts, ses_dir, sub, task, hemi):
    os.makedirs(ses_dir, exist_ok=True)
    ax = plot_design_matrix(dm_std)
    fig = ax.get_figure()
    fig.suptitle(f"sub-{sub}  all-sessions  task-{task}  hemi-{hemi}", fontsize=9)
    fig.savefig(op.join(ses_dir, "design_matrix.png"), bbox_inches="tight", dpi=120)
    plt.close(fig)
    dm_std.to_csv(op.join(ses_dir, f"sub-{sub}_ses-allses_task-{task}_hemi-{hemi}_design_matrix.csv"))
    for k, v in contrasts.items():
        ax = plot_contrast_matrix(v, design_matrix=dm_std)
        fig = ax.get_figure()
        fig.suptitle(f"sub-{sub}  {k}  hemi-{hemi}", fontsize=9)
        fig.savefig(op.join(ses_dir, f"sub-{sub}_ses-allses_task-{task}_hemi-{hemi}"
                                     f"_contrast-{k}_contrast_matrix.png"), bbox_inches="tight", dpi=100)
        plt.close(fig)


def cache_path(outdir: str, sub: str, hemi: str) -> str:
    return op.join(outdir, f"sub-{sub}", f"sub-{sub}_hemi-{hemi}_roicache.npz")


def save_roi_cache(path, Y, X, contrasts, bounds, labels, roi_names, roi_idx):
    os.makedirs(op.dirname(path), exist_ok=True)
    np.savez(
        path, Y=Y, X=X,
        contrast_names=np.array(list(contrasts)),
        contrast_vecs=np.array(list(contrasts.values()), dtype=np.float32),
        rb=np.array(bounds, dtype=np.int64), run_labels=np.array(labels),
        roi_names=np.array(roi_names),
        **{f"roi_idx_{i}": idx for i, idx in enumerate(roi_idx)},
    )


# ---------------------------------------------------------------------------
# Stage B worker: one GLM
# ---------------------------------------------------------------------------

# Filled in the main process before the Stage B pool forks; workers read it
# copy-on-write, so the ROI caches are loaded from disk exactly once.
_CACHES: dict[str, dict] = {}


def _read_cache(path: str) -> dict:
    z = np.load(path, allow_pickle=False)
    roi_names = z["roi_names"].tolist()
    return {
        "Y": z["Y"], "X": z["X"], "rb": z["rb"],
        "cvecs": z["contrast_vecs"], "cnames": z["contrast_names"].tolist(),
        "roi_idx": [z[f"roi_idx_{i}"] for i in range(len(roi_names))],
    }


def _fit_one(job: dict) -> dict:
    t0 = time.time()
    c = _CACHES[job["sub"]]
    keep = [c["cnames"].index(n) for n in job["contrasts"]]
    rows = np.concatenate([np.arange(c["rb"][i, 0], c["rb"][i, 1]) for i in job["selected"]])
    labels, estimates = run_glm(c["Y"][:, rows].T, c["X"][rows, :])
    out = np.full((len(keep), len(c["roi_idx"])), np.nan, dtype=np.float32)
    for ci, k in enumerate(keep):
        t_map = compute_contrast(labels, estimates, c["cvecs"][k]).stat()
        for ri, idx in enumerate(c["roi_idx"]):
            out[ci, ri] = np.nanmean(t_map[idx])
    return {"sub": job["sub"], "n_idx": job["n_idx"], "iter": job["iter"],
            "selected": job["selected"], "mean_t": out, "secs": time.time() - t0}


def draw_subsets(total_runs: int, n_iter: int, seed: int) -> list[list[list[int]]]:
    """Identical sequence to power_analysis_ohbm.run_power_loop."""
    rng = np.random.default_rng(seed)
    return [[sorted(rng.choice(total_runs, size=n, replace=False).tolist()) for _ in range(n_iter)]
            for n in range(1, total_runs + 1)]


# ---------------------------------------------------------------------------
# Per-subject outputs
# ---------------------------------------------------------------------------

def save_subject_results(sub_dir, sub, hemi, results, selected, cnames, rnames, run_labels, n_iter, seed):
    total = results.shape[0]
    run_counts = np.arange(1, total + 1)
    np.savez_compressed(op.join(sub_dir, f"sub-{sub}_hemi-{hemi}_power_results.npz"),
                        mean_t=results, run_counts=run_counts, roi_names=np.array(rnames),
                        contrast_names=np.array(cnames), seed=np.array(seed))
    long_rows, summ_rows = [], []
    for ni in range(total):
        for it in range(n_iter):
            sel = selected[ni][it]
            sel_str = ";".join(run_labels[i] for i in sel)
            for ci, cn in enumerate(cnames):
                for ri, rn in enumerate(rnames):
                    long_rows.append({"subject": sub, "hemi": hemi, "n_runs": ni + 1, "iter": it + 1,
                                      "contrast": cn, "roi": rn,
                                      "mean_t": float(results[ni, it, ci, ri]),
                                      "selected_runs": sel_str})
        for ci, cn in enumerate(cnames):
            for ri, rn in enumerate(rnames):
                v = results[ni, :, ci, ri]
                summ_rows.append({"n_runs": ni + 1, "contrast": cn, "roi": rn, "hemi": hemi,
                                  "mean_t_mean": float(np.nanmean(v)), "mean_t_std": float(np.nanstd(v)),
                                  "mean_t_median": float(np.nanmedian(v)), "n_iter": n_iter, "seed": seed})
    pd.DataFrame(long_rows).to_csv(op.join(sub_dir, f"sub-{sub}_hemi-{hemi}_power_long.csv"), index=False)
    pd.DataFrame(summ_rows).to_csv(op.join(sub_dir, f"sub-{sub}_hemi-{hemi}_power_summary.tsv"),
                                   sep="\t", index=False)

    ncols = min(len(rnames), 3)
    nrows = (len(rnames) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(cnames), 1)))
    for ri, rn in enumerate(rnames):
        ax = axes[ri // ncols][ri % ncols]
        for ci, cn in enumerate(cnames):
            mu = np.nanmean(results[:, :, ci, ri], axis=1)
            sd = np.nanstd(results[:, :, ci, ri], axis=1)
            ax.plot(run_counts, mu, label=cn, color=colors[ci], lw=1.5)
            ax.fill_between(run_counts, mu - sd, mu + sd, alpha=0.15, color=colors[ci])
        ax.axhline(3, color="r", lw=0.6, ls=":")
        ax.axhline(0, color="k", lw=0.5, ls="--")
        ax.set_title(f"{rn} (hemi-{hemi})", fontsize=10)
        ax.set_xlabel("n runs")
        ax.set_ylabel("mean T")
        ax.legend(fontsize=6)
    for i in range(len(rnames), nrows * ncols):
        axes[i // ncols][i % ncols].set_visible(False)
    fig.suptitle(f"sub-{sub} power analysis — hemi-{hemi}", fontsize=12)
    fig.tight_layout()
    fig.savefig(op.join(sub_dir, f"sub-{sub}_hemi-{hemi}_power_plot.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    base: str = typer.Option("/bcbl/home/public/Gari/VOTCLOC/main_exp", "--base"),
    subs: Optional[str] = typer.Option(None, "--subs", help="Comma list, e.g. 01,02 (default: all with labels)"),
    sessions_file: Optional[str] = typer.Option(None, "-f", help="Subseslist (sub,ses,RUN). Default: sessions of --l1-ref"),
    l1_ref: str = typer.Option("analysis-final_v3", "--l1-ref", help="l1_surface analysis whose ses-XX dirs define sessions"),
    fp_ana_name: str = typer.Option("25.1.4_t2w_fmapsbref_newest", "--fp-ana-name"),
    task: str = typer.Option("fLoc", "--task"),
    space: str = typer.Option("fsnative", "--space"),
    hemi: str = typer.Option("L", "--hemi"),
    start_scans: int = typer.Option(6, "--start-scans"),
    slice_time_ref: float = typer.Option(0.5, "--slice-time-ref"),
    contrast: str = typer.Option(op.join(_THIS_DIR, "contrast_VOTCLOC_power.yaml"), "--contrast"),
    roi_yaml: str = typer.Option(op.join(_THIS_DIR, "roi_config_tiger_ROI_auto.yaml"), "--roi-yaml"),
    label_root: str = typer.Option(HELPER_LABEL_ROOT, "--label-root"),
    label_subdir: str = typer.Option("tiger_ROI_auto", "--label-subdir"),
    strategy_yaml: str = typer.Option(op.join(_THIS_DIR, "..", "glm_strategy", "strategy.yaml"), "--strategy-yaml"),
    strategy: str = typer.Option("basic_MC", "--strategy"),
    rerun_map: Optional[str] = typer.Option(None, "--rerun-map", help="Default: <base>/BIDS/sourcedata/qc/rerun_check.tsv"),
    output_name: str = typer.Option("final_v3_poweranalysis", "--output-name"),
    n_iter: int = typer.Option(3, "--n-iter"),
    seed: int = typer.Option(42, "--seed"),
    n_workers: int = typer.Option(90, "--n-workers"),
    max_runs: Optional[int] = typer.Option(None, "--max-runs", help="Test only: stop the run-count loop at this n"),
    stage: str = typer.Option("all", "--stage", help="all | load (Stage A + assemble only)"),
    force: bool = typer.Option(False, "--force", help="Recompute caches and results"),
) -> None:
    t_start = time.time()
    if n_workers < 1:
        console.print(f"[red]ERROR[/red] --n-workers must be >= 1 (got {n_workers})")
        raise typer.Exit(1)
    bids_dir = op.join(base, "BIDS")
    fmriprep_dir = op.join(bids_dir, "derivatives", f"fmriprep-{fp_ana_name}")
    l1_dir = op.join(bids_dir, "derivatives", "l1_surface")
    outdir = op.join(l1_dir, f"analysis-{output_name}")
    rerun_map = rerun_map or op.join(bids_dir, "sourcedata", "qc", "rerun_check.tsv")
    os.makedirs(outdir, exist_ok=True)

    rerun_excl = _load_rerun_exclusions(rerun_map) if op.exists(rerun_map) else {}
    cstrat = load_confound_strategy(strategy_yaml, strategy)
    with open(contrast) as f:
        wanted = list(yaml.safe_load(f).keys())

    if subs:
        sub_list = [s.strip().zfill(2) for s in subs.split(",")]
    else:
        sub_list = sorted(op.basename(d).replace("sub-", "")
                          for d in glob.glob(op.join(label_root, "sub-*"))
                          if op.isdir(op.join(d, "label", label_subdir)))

    console.rule("[bold cyan]Power analysis — local parallel[/bold cyan]")
    for k, v in [("subjects", ",".join(sub_list)), ("fmriprep", fmriprep_dir),
                 ("sessions", sessions_file or f"ses dirs of {l1_ref}"), ("labels", f"{label_root}/sub-*/label/{label_subdir}"),
                 ("contrasts", ",".join(wanted)), ("strategy", strategy), ("rerun map", rerun_map),
                 ("hemi", hemi), ("n_iter", n_iter), ("seed", seed), ("workers", n_workers),
                 ("max_runs", max_runs), ("output", outdir)]:
        console.print(f"  {k:10s} {v}")

    # ── discovery ────────────────────────────────────────────────────────────
    inventory, todo_subs, roi_info = [], [], {}
    load_jobs: list[dict] = []
    for sub in sub_list:
        sub_dir = op.join(outdir, f"sub-{sub}")
        if not force and op.exists(op.join(sub_dir, f"sub-{sub}_hemi-{hemi}_power_results.npz")) and stage == "all":
            console.print(f"  sub-{sub}: results exist — skip")
            continue
        masks = load_roi_masks(roi_yaml, hemi, op.join(label_root, f"sub-{sub}", "label", label_subdir))
        if not masks:
            console.print(f"  [red]sub-{sub}: no ROI labels — skip[/red]")
            continue
        union = np.unique(np.concatenate(list(masks.values())))
        roi_info[sub] = (list(masks), [np.searchsorted(union, v) for v in masks.values()], union)
        todo_subs.append(sub)
        if not force and op.exists(cache_path(outdir, sub, hemi)):
            console.print(f"  sub-{sub}: ROI cache exists — Stage A skipped")
            continue
        sess = sessions_from_file(sessions_file, sub) if sessions_file else sessions_from_l1(op.join(l1_dir, l1_ref), sub)
        n_ok = 0
        for ses in sess:
            ok, miss = discover_runs(bids_dir, fmriprep_dir, sub, ses, task, hemi, space, rerun_excl)
            inventory += [{**r, "status": "used"} for r in ({"sub": o["sub"], "ses": o["ses"], "run": o["run"]} for o in ok)]
            inventory += [{**m, "status": "skipped"} for m in miss]
            for o in ok:
                load_jobs.append({**o, "roi_union": union, "start_scans": start_scans, "confound_strategy": cstrat})
            n_ok += len(ok)
        console.print(f"  sub-{sub}: {len(sess)} sessions, {n_ok} runs, {len(union)} ROI vertices "
                      f"({', '.join(f'{r}={len(v)}' for r, v in masks.items())})")
    if inventory:
        pd.DataFrame(inventory).to_csv(op.join(outdir, "run_inventory.csv"), index=False)

    # ── Stage A: load runs in parallel ───────────────────────────────────────
    if load_jobs:
        console.rule(f"Stage A — loading {len(load_jobs)} runs with {n_workers} workers")
        per_sub: dict[str, list] = {}
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork")) as pool:
            futs = {pool.submit(_load_run, j): j for j in load_jobs}
            for i, fut in enumerate(as_completed(futs), 1):
                d = fut.result()
                per_sub.setdefault(d["key"][0], []).append(d)
                if i % max(1, len(load_jobs) // 20) == 0 or i == len(load_jobs):
                    console.print(f"  [{i}/{len(load_jobs)}] {time.time() - t0:.0f}s elapsed")
        for sub, loaded in sorted(per_sub.items()):
            Y, X, dm_std, contrasts, bounds, labels = assemble_subject(loaded, contrast, slice_time_ref)
            missing_c = [c for c in wanted if c not in contrasts]
            if missing_c:
                raise RuntimeError(f"sub-{sub}: contrasts {missing_c} not built")
            save_dm_diagnostics(dm_std, contrasts, op.join(outdir, f"sub-{sub}", "allses"), sub, task, hemi)
            rnames, ridx, _ = roi_info[sub]
            save_roi_cache(cache_path(outdir, sub, hemi), Y, X, contrasts, bounds, labels, rnames, ridx)
            console.print(f"  sub-{sub}: Y={Y.shape} X={X.shape} runs={len(bounds)} -> cache saved")
            del Y, X, dm_std
        del per_sub

    if stage == "load":
        console.print(f"[green]Stage A done[/green] ({(time.time() - t_start) / 60:.1f} min)")
        return

    # ── Stage B: all GLMs of all subjects in one pool ────────────────────────
    fit_jobs, meta = [], {}
    for sub in todo_subs:
        cp = cache_path(outdir, sub, hemi)
        z = np.load(cp, allow_pickle=False)
        _CACHES[sub] = _read_cache(cp)
        total = int(z["rb"].shape[0])
        labels = z["run_labels"].tolist()
        cnames = z["contrast_names"].tolist()
        miss = [c for c in wanted if c not in cnames]
        if miss:
            raise RuntimeError(f"sub-{sub}: cache lacks {miss}; rerun with --force")
        subsets = draw_subsets(total, n_iter, seed)
        n_max = min(total, max_runs) if max_runs else total
        meta[sub] = {"total": n_max, "labels": labels, "subsets": subsets[:n_max],
                     "rnames": z["roi_names"].tolist()}
        for ni in range(n_max):
            for it in range(n_iter):
                fit_jobs.append({"sub": sub, "n_idx": ni, "iter": it,
                                 "selected": subsets[ni][it], "contrasts": wanted})
    # biggest GLMs first so the long ones don't straggle at the end
    fit_jobs.sort(key=lambda j: -len(j["selected"]))

    console.rule(f"Stage B — {len(fit_jobs)} GLMs over {len(meta)} subjects with {n_workers} workers")
    results = {s: np.full((m["total"], n_iter, len(wanted), len(m["rnames"])), np.nan, dtype=np.float32)
               for s, m in meta.items()}
    left = {s: m["total"] * n_iter for s, m in meta.items()}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, mp_context=mp.get_context("fork")) as pool:
        futs = [pool.submit(_fit_one, j) for j in fit_jobs]
        for i, fut in enumerate(as_completed(futs), 1):
            r = fut.result()
            results[r["sub"]][r["n_idx"], r["iter"]] = r["mean_t"]
            left[r["sub"]] -= 1
            if left[r["sub"]] == 0:
                m = meta[r["sub"]]
                save_subject_results(op.join(outdir, f"sub-{r['sub']}"), r["sub"], hemi, results[r["sub"]],
                                     m["subsets"], wanted, m["rnames"], m["labels"], n_iter, seed)
                console.print(f"  [green]sub-{r['sub']} done[/green] -> results saved")
            if i % max(1, len(fit_jobs) // 50) == 0 or i == len(fit_jobs):
                el = time.time() - t0
                console.print(f"  [{i}/{len(fit_jobs)}] {el / 60:.1f} min  ETA {el / i * (len(fit_jobs) - i) / 60:.1f} min")

    # ── group long CSV ───────────────────────────────────────────────────────
    parts = sorted(glob.glob(op.join(outdir, "sub-*", f"sub-*_hemi-{hemi}_power_long.csv")))
    if parts:
        pd.concat([pd.read_csv(p, dtype={"subject": str}) for p in parts]).to_csv(
            op.join(outdir, "group_power_long.csv"), index=False)
        console.print(f"  group_power_long.csv <- {len(parts)} subjects")
    console.rule(f"Done in {(time.time() - t_start) / 60:.1f} min")


if __name__ == "__main__":
    app()
