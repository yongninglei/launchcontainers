"""
MIT License
Copyright (c) 2024-2025 Yongning Lei
Permission is hereby granted, free of charge, to any person obtaining a copy of this software
and associated documentation files (the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:
The above copyright notice and this permission notice shall be included in all copies or substantial
portions of the Software.
"""

from __future__ import annotations

import copy
import json
import os
import re
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from launchcontainers.utils import parse_subses_list

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

# All task labels that are valid PRF tasks.  Only these are recognised when
# scanning BIDS func/ — any other task label in the filenames is ignored.
_ALL_VALID_TASKS: set[str] = {
    "retRW",
    "retFF",
    "retCB",
    "retfixFF",
    "retfixRW",
    "retfixRWblock",
    "retfixRWblock01",
    "retfixRWblock02",
}

# ---------------------------------------------------------------------------
# Base templates — static fields only.
# Dynamic fields (subjects, sessions, subjectName, sessionName, tasks, and the
# step-specific IDs) are inserted at runtime from CLI arguments.
# ---------------------------------------------------------------------------

# Maps the short CLI label to the value written into the JSON "model" field.
_MODEL_LABELS: dict[str, str] = {
    "og": "one gaussian",
    "css": "css",
}

_BASE: dict[str, dict] = {
    "prfprepare": {
        "etcorrection": False,
        "tasks": ["all"],
        "force": False,
        "custom_output_name": "",
        "fmriprep_legacy_layout": False,
        "forceParams": "",
        "use_numImages": False,
        "verbose": True,
        "config": {
            "average_runs": True,
            "output_only_average": False,
            "rois": "all",
            "atlases": "all",
            "fmriprep_bids_layout": False,
            # "fmriprep_analysis" is injected from --fp at runtime
        },
    },
    "prfanalyze-vista": {
        "isPRFSynthData": False,
        "options": {
            # "prfprepareAnalysis" is injected from --prepid at runtime
            # "model" is injected from --model at runtime (og → "one gaussian", css → "css")
            "grid": False,
            "wsearch": "5",
            "detrend": 1,
            "keepAllPoints": False,
            "numberStimulusGridPoints": 101,
        },
        "stimulus": {
            "stimulus_diameter": 18,
        },
    },
    "prfresult": {
        "tasks": ["all"],
        "runs": ["all"],
        # "prfanalyzeAnalysis" is injected from --analyzeid at runtime
        "masks": {
            "rois": [
                [
                    "all",
                ]
            ],
            "atlases": ["all"],
            "varianceExplained": [0.1],
            "eccentricity": False,
            "beta": False,
        },
        "coveragePlot": {
            "create": True,
            "method": ["max"],
            "minColorBar": [0],
        },
        "cortexPlot": {
            "createCortex": True,
            "createGIF": True,
            "parameter": ["ecc"],
            "hemisphere": "both",
            "surface": ["sphere"],
            "showBordersArea": ["V1"],
        },
        "saveAs3D": True,
        "verbose": True,
        "force": True,
    },
}

_VALID_STEPS = set(_BASE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _discover_tasks_for_session(bids_dir: Path, sub: str, ses: str) -> list[str]:
    """Return sorted list of valid PRF task names found in BIDS raw func for this sub/ses.

    Scans ``bids_dir/sub-{sub}/ses-{ses}/func/`` for ``*_bold.nii.gz`` files,
    extracts the task label from the filename, and filters against
    ``_ALL_VALID_TASKS``.  Unknown task labels are silently ignored.
    """
    func_dir = bids_dir / f"sub-{sub}" / f"ses-{ses}" / "func"
    if not func_dir.exists():
        return []
    pattern = re.compile(rf"sub-{sub}_ses-{ses}_task-([^_]+)_run-")
    found: set[str] = set()
    for f in func_dir.iterdir():
        if not f.name.endswith("_bold.nii.gz"):
            continue
        m = pattern.match(f.name)
        if m:
            task = m.group(1)
            if task in _ALL_VALID_TASKS:
                found.add(task)
    return sorted(found)


def _check_and_create_symlinks(fmriprep_analysis: str, bids_dir: Path) -> None:
    """Check and create necessary symlinks for fmriprep analysis directory."""
    analysis_dir = (
        bids_dir / "derivatives" / "fmriprep" / f"analysis-{fmriprep_analysis}"
    )
    source_dir = bids_dir / "derivatives" / f"fmriprep-{fmriprep_analysis}"
    freesurfer_link = analysis_dir / "sourcedata" / "freesurfer"

    if not analysis_dir.exists():
        if source_dir.exists():
            os.symlink(source_dir, analysis_dir)
            console.print(
                f"[green]✓ Created symlink:[/green] {analysis_dir} -> {source_dir}"
            )
        else:
            console.print(f"[yellow]⚠ Source dir does not exist: {source_dir}[/yellow]")
            return
    else:
        console.print(f"[green]✓ Analysis dir exists:[/green] {analysis_dir}")

    if os.path.islink(freesurfer_link):
        if freesurfer_link.exists():
            console.print(f"[green]✓ FreeSurfer link valid:[/green] {freesurfer_link}")
        else:
            console.print(
                f"[yellow]⚠ FreeSurfer symlink broken:[/yellow] {freesurfer_link}"
            )
    elif freesurfer_link.exists():
        console.print(f"[green]✓ FreeSurfer dir exists:[/green] {freesurfer_link}")
    else:
        console.print(f"[yellow]⚠ FreeSurfer not found:[/yellow] {freesurfer_link}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    bids_dir: Path = typer.Option(..., "--bids", "-b", help="BIDS root directory."),
    sub: Optional[str] = typer.Option(
        None, "--sub", "-s", help="Subject ID (e.g. 03)."
    ),
    ses: Optional[str] = typer.Option(None, "--ses", help="Session ID (e.g. 01)."),
    file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="Subseslist CSV/TSV with sub,ses columns."
    ),
    step: str = typer.Option(
        ...,
        "--step",
        help="Pipeline step: prfprepare | prfresult | prfanalyze-vista.",
    ),
    output_dir: Path = typer.Option(
        ..., "--output", "-o", help="Output directory for generated JSONs."
    ),
    fp: Optional[str] = typer.Option(
        None,
        "--fp",
        help="[prfprepare] fmriprep analysis name, e.g. 25.1.4_t2w_fmapsbref_newest.",
    ),
    prepid: Optional[str] = typer.Option(
        None,
        "--prepid",
        help="[prfanalyze-vista] prfprepare analysis ID string, e.g. 01.",
    ),
    analyzeid: Optional[str] = typer.Option(
        None,
        "--analyzeid",
        help="[prfresult] prfanalyze analysis ID string, e.g. 03.",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="[prfanalyze-vista] Model: og (one gaussian) or css.",
    ),
    force: bool = typer.Option(
        True,
        "--force/--no-force",
        help="Overwrite existing JSONs (default: True — cheap to regenerate).",
    ),
) -> None:
    """Generate per-subject/session JSON configs for PRF pipeline steps.

    \b
    Step-specific required options:
      prfprepare       --fp       fmriprep analysis name
      prfanalyze-vista --prepid   prfprepare analysis ID
      prfresult        --analyzeid prfanalyze analysis ID

    \b
    Examples:
      python 03_prepare_prf_subses_json.py \\
          -b /path/BIDS -f subseslist.txt \\
          --step prfprepare --fp 25.1.4_t2w_fmapsbref_newest -o prfprepare_jsons

      python 03_prepare_prf_subses_json.py \\
          -b /path/BIDS -f subseslist.txt \\
          --step prfanalyze-vista --prepid 01 -o prfanalyze_jsons

      python 03_prepare_prf_subses_json.py \\
          -b /path/BIDS -f subseslist.txt \\
          --step prfresult --analyzeid 03 -o prfresult_jsons
    """
    if step not in _VALID_STEPS:
        console.print(
            f"[red]Unknown step '{step}'. Choose from: {', '.join(sorted(_VALID_STEPS))}[/red]"
        )
        raise typer.Exit(1)

    # Validate step-specific required args
    if step == "prfprepare" and not fp:
        raise typer.BadParameter("--fp is required for step prfprepare")
    if step == "prfanalyze-vista" and not prepid:
        raise typer.BadParameter("--prepid is required for step prfanalyze-vista")
    if step == "prfanalyze-vista" and model not in _MODEL_LABELS:
        raise typer.BadParameter(
            f"--model is required for step prfanalyze-vista; choose: {', '.join(_MODEL_LABELS)}"
        )
    if step == "prfresult" and not analyzeid:
        raise typer.BadParameter("--analyzeid is required for step prfresult")

    if file is None and (sub is None or ses is None):
        raise typer.BadParameter("Provide either --file or both --sub and --ses")

    pairs = parse_subses_list(file) if file else [(sub.zfill(2), ses.zfill(2))]
    output_dir.mkdir(parents=True, exist_ok=True)

    if step == "prfprepare":
        console.print("\n[bold]Checking fmriprep analysis setup...[/bold]")
        _check_and_create_symlinks(fp, bids_dir)

    console.print(f"\n[dim]step={step}  → {output_dir}[/dim]")
    generated = 0

    for sub, ses in pairs:
        if step == "prfprepare":
            base = copy.deepcopy(_BASE["prfprepare"])
            base["config"]["fmriprep_analysis"] = fp
            config = {"subjects": sub, "sessions": ses, **base}
            out_path = output_dir / f"all_sub-{sub}_ses-{ses}.json"

            if out_path.exists() and not force:
                console.print(f"[yellow]Skipped (exists):[/yellow] {out_path.name}")
                continue
            with open(out_path, "w") as fh:
                json.dump(config, fh, indent=4)
            console.print(f"[green]Generated:[/green] {out_path.name}")
            generated += 1

        elif step == "prfanalyze-vista":
            tasks = _discover_tasks_for_session(bids_dir, sub, ses)
            if not tasks:
                console.print(
                    f"[yellow]sub-{sub} ses-{ses}[/yellow] — no valid PRF tasks found in "
                    f"BIDS func, skipping"
                )
                continue
            console.print(f"[dim]sub-{sub} ses-{ses}[/dim] — tasks detected: {tasks}")
            for task in tasks:
                base = copy.deepcopy(_BASE["prfanalyze-vista"])
                base["options"]["prfprepareAnalysis"] = prepid
                base["options"]["model"] = _MODEL_LABELS[model]
                config = {"subjectName": sub, "sessionName": ses, "tasks": task, **base}
                out_path = output_dir / f"{task}_{model}_sub-{sub}_ses-{ses}.json"

                if out_path.exists() and not force:
                    console.print(f"[yellow]Skipped (exists):[/yellow] {out_path.name}")
                    continue
                with open(out_path, "w") as fh:
                    json.dump(config, fh, indent=4)
                console.print(f"[green]Generated:[/green] {out_path.name}")
                generated += 1

        elif step == "prfresult":
            base = copy.deepcopy(_BASE["prfresult"])
            config = {
                "subjects": sub,
                "sessions": ses,
                "prfanalyzeAnalysis": analyzeid,
                **base,
            }
            out_path = output_dir / f"all_sub-{sub}_ses-{ses}.json"

            if out_path.exists() and not force:
                console.print(f"[yellow]Skipped (exists):[/yellow] {out_path.name}")
                continue
            with open(out_path, "w") as fh:
                json.dump(config, fh, indent=4)
            console.print(f"[green]Generated:[/green] {out_path.name}")
            generated += 1

    console.print(
        f"\n[bold]Done:[/bold] {generated} JSON file(s) written to {output_dir}"
    )


if __name__ == "__main__":
    app()
