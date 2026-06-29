# -----------------------------------------------------------------------------
# Copyright (c) Yongning Lei 2024-2025
# All rights reserved.
#
# This script is distributed under the Apache-2.0 license.
# You may use, distribute, and modify this code under the terms of the Apache-2.0 license.
# See the LICENSE file for details.
#
# THIS SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE, AND NONINFRINGEMENT.
#
# Author: Yongning Lei
# Email: yl4874@nyu.edu
# GitHub: https://github.com/yongninglei
#
# fir_curve_metrics.py
# ---------------------
# Per-vertex curve analysis of the pooled "AllStim" FIR response produced by
# run_allses_glm_FIR.py.  Reads:
#
#   sub-XX_ses-allses_task-<task>_hemi-<H>_space-<space>_desc-AllStimFIR0-18s_timeseries.func.gii
#
# (n_vertices x n_delays GIFTI), and for each hemisphere writes single-frame
# GIFTI maps:
#
#   desc-FIRpeakamp        peak amplitude (signed)
#   desc-FIRtimetopeak     time-to-peak (s)
#   desc-FIRauc            area under the curve (0..(n_delays-1)*tr s)
#   desc-FIRfwhm           full-width-at-half-max (s)
#   desc-FIRfitR2          R^2 of best-fitting candidate gamma HRF
#   desc-FIRfitamp         best-fit amplitude scale
#   desc-FIRfitcandidate   nominal peak time (s) of the best-fitting candidate
#
# Usage::
#
#     python fir_curve_metrics.py \
#         --betas-dir /bcbl/home/public/Gari/VOTCLOC/main_exp/BIDS/derivatives/l1_surface_fir/analysis-FIR_v1/sub-03/allses \
#         --sub 03 --task fLoc --space fsnative --tr 2.0
# -----------------------------------------------------------------------------

from __future__ import annotations

import os.path as op

import numpy as np
import typer
from nilearn.surface import load_surf_data
from rich.console import Console
from scipy import stats
from scipy.interpolate import CubicSpline
from scipy.signal import savgol_filter

# Reuse the GIFTI writer from the shared helpers module.
import importlib.util as _ilu
_helpers_path = op.join(op.dirname(op.abspath(__file__)), "glm_surface_check_model_strategy.py")
_spec = _ilu.spec_from_file_location("_glm_helpers_mod", _helpers_path)
_glm_helpers = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_glm_helpers)
save_statmap_to_gifti = _glm_helpers.save_statmap_to_gifti

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


# ---------------------------------------------------------------------------
# Candidate canonical HRFs (gamma-shaped), sampled at the FIR delay grid
# ---------------------------------------------------------------------------

def _candidate_hrfs(t: np.ndarray, peak_times: list[float]) -> dict[float, np.ndarray]:
    """
    Return {peak_time: hrf_curve} for a small bank of gamma-shaped canonical
    HRFs sampled at ``t``, each normalized to a max of 1.
    """
    candidates = {}
    for peak in peak_times:
        a = peak + 1.0  # gamma pdf mode = (a-1)*scale, scale=1
        curve = stats.gamma.pdf(t, a=a, scale=1.0)
        peak_val = curve.max()
        if peak_val > 0:
            curve = curve / peak_val
        candidates[peak] = curve
    return candidates


# ---------------------------------------------------------------------------
# Per-vertex curve metrics (vectorized)
# ---------------------------------------------------------------------------

def compute_curve_metrics(
    curve: np.ndarray,
    t: np.ndarray,
    smooth: bool,
    fine_step: float,
    peak_times: list[float],
) -> dict[str, np.ndarray]:
    """
    Parameters
    ----------
    curve : (n_vertices, n_delays) array — AllStim FIR betas.
    t     : (n_delays,) array — delay times in seconds (e.g. 0,2,...,18).

    Returns
    -------
    dict of (n_vertices,) arrays: peakamp, timetopeak, auc, fwhm, fitR2, fitamp, fitcandidate
    """
    n_vtx, n_delays = curve.shape

    # ── Step 6: smoothing ───────────────────────────────────────────────────
    work_curve = curve
    if smooth:
        window = min(5, n_delays if n_delays % 2 == 1 else n_delays - 1)
        if window >= 3:
            work_curve = savgol_filter(curve, window_length=window, polyorder=2, axis=1)

    # ── Upsample via cubic spline for sub-TR precision ──────────────────────
    spline = CubicSpline(t, work_curve, axis=1)
    fine_t = np.arange(t[0], t[-1] + fine_step, fine_step)
    fine_t = fine_t[fine_t <= t[-1]]
    curve_fine = spline(fine_t)  # (n_vtx, n_fine)

    # ── peak amplitude + time-to-peak (step 7) ───────────────────────────────
    peak_idx = np.argmax(curve_fine, axis=1)
    peak_amp = curve_fine[np.arange(n_vtx), peak_idx]
    time_to_peak = fine_t[peak_idx]

    # ── AUC (step 7) ──────────────────────────────────────────────────────────
    auc = np.trapz(curve_fine, fine_t, axis=1)

    # ── FWHM width (step 7) ───────────────────────────────────────────────────
    baseline = work_curve[:, 0]
    half_max = baseline + (peak_amp - baseline) / 2.0
    above_half = curve_fine >= half_max[:, None]
    has_peak = (peak_amp > baseline) & above_half.any(axis=1)

    first_idx = np.argmax(above_half, axis=1)
    last_idx = curve_fine.shape[1] - 1 - np.argmax(above_half[:, ::-1], axis=1)
    fwhm = np.where(has_peak, fine_t[last_idx] - fine_t[first_idx], 0.0)

    # ── Step 8: fit quality against candidate gamma HRFs ─────────────────────
    candidates = _candidate_hrfs(t, peak_times)
    n_cand = len(candidates)
    cand_peaks = np.array(list(candidates.keys()))
    cand_hrfs = np.stack(list(candidates.values()), axis=0)  # (n_cand, n_delays)

    r2_all = np.zeros((n_cand, n_vtx), dtype=np.float64)
    scale_all = np.zeros((n_cand, n_vtx), dtype=np.float64)

    curve_mean = work_curve.mean(axis=1, keepdims=True)
    ss_tot = np.sum((work_curve - curve_mean) ** 2, axis=1)

    for i, hrf in enumerate(cand_hrfs):
        denom = np.dot(hrf, hrf)
        scale = (work_curve @ hrf) / denom if denom > 0 else np.zeros(n_vtx)
        pred = scale[:, None] * hrf[None, :]
        ss_res = np.sum((work_curve - pred) ** 2, axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            r2 = np.where(ss_tot > 0, 1.0 - ss_res / ss_tot, 0.0)
        r2_all[i] = r2
        scale_all[i] = scale

    best_idx = np.argmax(r2_all, axis=0)
    cols = np.arange(n_vtx)
    fit_r2 = r2_all[best_idx, cols]
    fit_amp = scale_all[best_idx, cols]
    fit_candidate = cand_peaks[best_idx]

    return {
        "peakamp": peak_amp.astype(np.float32),
        "timetopeak": time_to_peak.astype(np.float32),
        "auc": auc.astype(np.float32),
        "fwhm": fwhm.astype(np.float32),
        "fitR2": fit_r2.astype(np.float32),
        "fitamp": fit_amp.astype(np.float32),
        "fitcandidate": fit_candidate.astype(np.float32),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@app.command()
def main(
    betas_dir: str = typer.Option(
        ..., "--betas-dir",
        help="allses output dir from run_allses_glm_FIR.py "
             "(<bids_dir>/derivatives/l1_surface_fir/analysis-<name>/sub-<sub>/allses)",
    ),
    sub: str = typer.Option(..., "--sub", help="Subject label without 'sub-'"),
    task: str = typer.Option(..., "--task", help="Task name, e.g. fLoc"),
    space: str = typer.Option("fsnative", "--space"),
    tr: float = typer.Option(2.0, "--tr", help="Repetition time (s) — FIR delay step."),
    n_delays: int = typer.Option(10, "--n-delays", help="Number of FIR delays (must match run_allses_glm_FIR.py)."),
    smooth: bool = typer.Option(True, "--smooth/--no-smooth", help="Savitzky-Golay smooth the FIR curve before metrics."),
    fine_step: float = typer.Option(0.1, "--fine-step", help="Spline upsampling step (s) for peak/AUC/width."),
    desc_suffix: str = typer.Option("", "--desc-suffix", help="Extra desc suffix matching the input file, e.g. '_desc-smoothed05'."),
    peak_times: str = typer.Option(
        "4,6,8,10", "--peak-times",
        help="Comma-separated candidate gamma-HRF peak times (s) for the fit-quality step.",
    ),
) -> None:
    sub = sub.strip().zfill(2)
    peak_times_list = [float(p.strip()) for p in peak_times.split(",") if p.strip()]
    t = np.arange(n_delays) * tr

    for hemi in ["L", "R"]:
        infile = op.join(
            betas_dir,
            f"sub-{sub}_ses-allses_task-{task}_hemi-{hemi}_space-{space}"
            f"{desc_suffix}_desc-AllStimFIR0-18s_timeseries.func.gii",
        )
        if not op.exists(infile):
            console.print(f"[yellow]WARNING[/yellow]: not found, skipping: {infile}")
            continue

        console.print(f"[bold]hemi-{hemi}[/bold]  loading {op.basename(infile)}")
        data = load_surf_data(infile)
        curve = np.vstack(data[:, :]).astype(np.float64)  # (n_vertices, n_delays)
        if curve.shape[1] != n_delays:
            console.print(
                f"[red]ERROR[/red]: expected {n_delays} frames, got {curve.shape[1]} in {infile}"
            )
            raise typer.Exit(1)

        metrics = compute_curve_metrics(curve, t, smooth, fine_step, peak_times_list)

        for name, arr in metrics.items():
            outname = op.join(
                betas_dir,
                f"sub-{sub}_ses-allses_task-{task}_hemi-{hemi}_space-{space}"
                f"{desc_suffix}_desc-FIR{name}_statmap.func.gii",
            )
            save_statmap_to_gifti(arr, outname)

        console.print(f"  [green]done[/green]  hemi-{hemi}  ({curve.shape[0]} vertices)")


if __name__ == "__main__":
    app()
