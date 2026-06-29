#!/usr/bin/env python3
"""
regenerate_report.py

Re-run tedana using an existing ICA mixing matrix to regenerate the HTML
report and figures/ directory without repeating the full ICA (~1 hr).

Uses the desc-ICA_mixing.tsv already in the tedana output to skip ICA.

Usage:
    micromamba run -n lc python regenerate_report.py \\
        -b /data/BIDS \\
        -fp /data/BIDS/derivatives/fmriprep-25.1.4 \\
        -tedana tedana-26.0.3 \\
        -n default_ica \\
        -s pilot02,01 \\
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
from typing import Optional

import typer
from rich.console import Console

console = Console()
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _bids_prefix(sub, ses, task, acq, run):
    parts = [f"sub-{sub}", f"ses-{ses}", f"task-{task}"]
    if acq:
        parts.append(f"acq-{acq}")
    if run:
        parts.append(f"run-{run}")
    return "_".join(parts) + "_"


def _get_echo_times_s(fp_func_dir, bids_dir, sub, ses, task, acq, run):
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""

    def _read(paths):
        tes = []
        for jf in sorted(paths):
            with open(jf) as f:
                meta = json.load(f)
            et = meta.get("EchoTime")
            if et is None:
                return None
            tes.append(float(et))
        return tes or None

    deriv = glob.glob(
        op.join(
            fp_func_dir,
            f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}_echo-*_desc-preproc_bold.json",
        )
    )
    if deriv:
        tes = _read(deriv)
        if tes:
            return tes

    raw = glob.glob(
        op.join(
            bids_dir,
            f"sub-{sub}",
            f"ses-{ses}",
            "func",
            f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}_echo-*_bold.json",
        )
    )
    if not raw:
        raise FileNotFoundError(f"No echo JSONs for {task} run-{run}")
    tes = _read(raw)
    if not tes:
        raise KeyError("EchoTime missing in raw BIDS JSONs")
    return tes


def _glob_echo_niftis(func_dir, sub, ses, task, acq, run):
    acq_token = f"_acq-{acq}" if acq else ""
    run_token = f"_run-{run}" if run else ""
    return sorted(
        glob.glob(
            op.join(
                func_dir,
                f"sub-{sub}_ses-{ses}_task-{task}{acq_token}{run_token}_echo-*_desc-preproc_bold.nii.gz",
            )
        )
    )


def _find_runs(tedana_func_dir, sub, ses, task, acq):
    acq_token = f"_acq-{acq}" if acq else ""
    found = glob.glob(
        op.join(
            tedana_func_dir,
            f"sub-{sub}_ses-{ses}_task-{task}{acq_token}_run-*_desc-ICA_mixing.tsv",
        )
    )
    runs = set()
    for f in found:
        m = re.search(r"_run-(\w+)[_.]", op.basename(f))
        if m:
            runs.add(m.group(1))
    return sorted(runs) if runs else [None]


@app.command()
def main(
    bids_dir: str = typer.Option(..., "-b"),
    fp_dir: str = typer.Option(..., "-fp"),
    tedana_dir_name: str = typer.Option(..., "-tedana"),
    analysis_name: str = typer.Option(..., "-n", "--analysis-name"),
    single: Optional[str] = typer.Option(None, "-s"),
    tasks: str = typer.Option(..., "--tasks", "-t"),
    acq: str = typer.Option("ME", "--acq"),
    n_threads: int = typer.Option(4, "--n-threads", "-j"),
) -> None:
    """Regenerate tedana HTML report + figures using existing ICA mixing matrix."""

    subses_pairs = []
    if single:
        parts = single.split(",")
        subses_pairs.append((parts[0].strip(), parts[1].strip()))
    else:
        console.print("[red]✗ provide -s sub,ses[/red]")
        raise typer.Exit(1)

    task_list = [t.strip() for t in tasks.split(",")]
    tedana_root = op.join(
        bids_dir, "derivatives", tedana_dir_name, f"analysis-{analysis_name}"
    )
    fp_func_base = op.join(fp_dir)

    for sub, ses in subses_pairs:
        tedana_func = op.join(tedana_root, f"sub-{sub}", f"ses-{ses}", "func")
        fp_func_dir = op.join(fp_func_base, f"sub-{sub}", f"ses-{ses}", "func")

        for task in task_list:
            runs = _find_runs(tedana_func, sub, ses, task, acq)
            for run in runs:
                prefix = _bids_prefix(sub, ses, task, acq, run)
                mixing = op.join(tedana_func, f"{prefix}desc-ICA_mixing.tsv")

                if not op.isfile(mixing):
                    console.print(f"  [red]✗ mixing matrix not found: {mixing}[/red]")
                    continue

                echo_files = _glob_echo_niftis(fp_func_dir, sub, ses, task, acq, run)
                tes = _get_echo_times_s(fp_func_dir, bids_dir, sub, ses, task, acq, run)

                console.print(
                    f"  [bold]run-{run}[/bold]  mixing: {op.basename(mixing)}"
                )

                t0 = time.perf_counter()
                work_dir = op.join(tedana_func, f".regen_work_{prefix.rstrip('_')}")
                os.makedirs(work_dir, exist_ok=True)
                try:
                    from tedana.workflows import tedana_workflow

                    tedana_workflow(
                        data=echo_files,
                        tes=tes,
                        out_dir=work_dir,
                        mask=None,
                        masktype=["dropout"],
                        fittype="curvefit",
                        convention="bids",
                        prefix=prefix,
                        mixing_file=mixing,
                        n_threads=n_threads,
                        overwrite=True,
                        quiet=True,
                    )
                except Exception as exc:
                    console.print(f"  [red]✗ {exc}[/red]")
                    shutil.rmtree(work_dir, ignore_errors=True)
                    continue

                # Copy only figures and HTML
                for name in os.listdir(work_dir):
                    src = op.join(work_dir, name)
                    dst = op.join(tedana_func, name)
                    if op.isdir(src):
                        shutil.copytree(src, dst, dirs_exist_ok=True)
                        console.print(f"  [green]✓[/green] {name}/")
                    elif name.endswith(".html"):
                        shutil.copy2(src, dst)
                        console.print(f"  [green]✓[/green] {name}")

                shutil.rmtree(work_dir, ignore_errors=True)
                console.print(f"  done  ({time.perf_counter() - t0:.1f}s)")


if __name__ == "__main__":
    app()
