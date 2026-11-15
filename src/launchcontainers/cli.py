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

import os
import os.path as op
import shutil
from argparse import Namespace
from datetime import datetime

import typer

from launchcontainers.log_setup import _logger, console, setup_verbosity, set_log_files

app = typer.Typer()


@app.command()
def prepare(
    lc_config: str = typer.Option(
        ..., "--lc-config", "-lcc", help="Path to launchcontainers config YAML"
    ),
    sub_ses_list: str = typer.Option(
        ..., "--sub-ses-list", "-ssl", help="Path to subject/session list"
    ),
    container_specific_config: str | None = typer.Option(
        None,
        "--container-specific-config",
        "-cc",
        help="Path to container-specific config (optional)",
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose mode"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug mode"),
):
    setup_verbosity(quiet=quiet, verbose=verbose, debug=debug)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    tmp_log_dir = op.join(".", "logs_tmp")
    os.makedirs(tmp_log_dir, exist_ok=True)
    log_fpath = op.join(tmp_log_dir, f"lc_prepare_{timestamp}.log")
    err_fpath = op.join(tmp_log_dir, f"lc_prepare_{timestamp}.err")
    set_log_files(log_fpath, err_fpath)

    console.print("....running prepare mode", style="bold cyan")
    if _uses_tool_prepare(lc_config):
        _prepare_with_tool(
            lc_config, sub_ses_list, container_specific_config, log_fpath, err_fpath
        )
        return

    # legacy path: GLM (l1_surface) until it is ported to a Tool
    parse_namespace = Namespace(
        lc_config=lc_config,
        sub_ses_list=sub_ses_list,
        container_specific_config=container_specific_config,
    )
    from launchcontainers import do_prepare

    success, analysis_dir = False, None
    try:
        success, analysis_dir = do_prepare.main(parse_namespace)
    except Exception:
        _logger.exception("lc prepare crashed")
        raise
    finally:
        # Copy logs to analysis_dir/prepare_log/ whenever an analysis dir exists,
        # including when prepare failed — that is when the logs matter most.
        if analysis_dir is None:
            analysis_dir = getattr(do_prepare, "LAST_ANALYSIS_DIR", None)
        if analysis_dir is not None and op.isdir(analysis_dir):
            prepare_log_dir = op.join(analysis_dir, "prepare_log")
            os.makedirs(prepare_log_dir, exist_ok=True)
            shutil.copy(log_fpath, op.join(prepare_log_dir, op.basename(log_fpath)))
            shutil.copy(err_fpath, op.join(prepare_log_dir, op.basename(err_fpath)))
            console.print(f"Copied prepare logs to {prepare_log_dir}", style="cyan")
    if not success:
        console.print("lc prepare finished with failures", style="red")
        raise typer.Exit(1)


_LEGACY_PREPARE = {"l1_surface"}


def _uses_tool_prepare(lc_config: str) -> bool:
    from launchcontainers import utils as do

    general = (do.read_yaml(lc_config) or {}).get("general") or {}
    return general.get("tool", general.get("container")) not in _LEGACY_PREPARE


def _prepare_with_tool(lc_config, sub_ses_list, container_config, log_fpath, err_fpath):
    """`lc prepare` through the Tool interface + analysis-core provenance."""
    from analysis_core import AnalysisCoreError

    from launchcontainers import lc_prepare

    res = None
    try:
        res = lc_prepare.prepare(
            lc_config, sub_ses_list, container_config, console=console
        )
    except AnalysisCoreError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(2) from None
    except Exception:
        _logger.exception("lc prepare crashed")
        raise
    finally:
        if res is not None and res.run_id:
            for adir in res.analysis_dirs:
                dest = adir / "logs" / res.run_id
                dest.mkdir(parents=True, exist_ok=True)
                for f in (log_fpath, err_fpath):
                    if op.exists(f):
                        shutil.copy(f, dest / op.basename(f))
                console.print(f"Prepare record and logs: {dest}", style="cyan")
    for note in res.notes:
        console.print(f"[yellow]{note}[/yellow]")
    n_bad = sum(r.status != "prepared" for r in res.results)
    for r in res.results:
        if r.status != "prepared":
            console.print(
                f"[red]FAILED[/red] sub-{r.sub} ses-{r.ses}: " + "; ".join(r.messages)
            )
    console.print(
        f"{res.tool}: {len(res.results) - n_bad}/{len(res.results)} sessions prepared in "
        + ", ".join(str(d) for d in res.analysis_dirs),
        style="green" if not n_bad else "red",
    )
    if n_bad:
        raise typer.Exit(1)


@app.command()
def run(
    workdir: str = typer.Option(..., "--workdir", "-w", help="Working directory"),
    run_lc: bool = typer.Option(
        False, "--run-lc", "-R", help="Whether to run launchcontainers"
    ),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose mode"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug mode"),
):
    setup_verbosity(quiet=quiet, verbose=verbose, debug=debug)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_log_dir = op.join(workdir, "run_log")
    os.makedirs(run_log_dir, exist_ok=True)
    set_log_files(
        op.join(run_log_dir, f"lc_run_{timestamp}.log"),
        op.join(run_log_dir, f"lc_run_{timestamp}.err"),
    )
    from launchcontainers import do_launch

    console.print("....running run mode", style="bold cyan")
    do_launch.main(workdir, run_lc)


@app.command()
def qc(
    workdir: str = typer.Option(..., "--workdir", "-w", help="Working directory"),
    log_dir: str = typer.Option(
        "./logs", "--log-dir", "-log", help="Directory to write logs"
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug mode"),
):
    setup_verbosity(debug=debug)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(log_dir, exist_ok=True)
    set_log_files(
        op.join(log_dir, f"qc_{timestamp}.log"),
        op.join(log_dir, f"qc_{timestamp}.err"),
    )
    from launchcontainers import do_qc

    console.print("\n....running quality check mode\n", style="bold red")
    do_qc.main(workdir, log_dir, debug)


# Add other commands similarly...


def main():
    app()


if __name__ == "__main__":
    main()
