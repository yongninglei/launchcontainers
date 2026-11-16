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
    if _tool_name(lc_config) not in _GLM_INPUT_PREP:
        _prepare_with_tool(
            lc_config, sub_ses_list, container_specific_config, log_fpath, err_fpath
        )
        return

    # l1_surface: GLM *input* preparation (events.tsv from vistadisp logs, BIDS /
    # fMRIPrep BOLD symlinks; prepare/glm_prepare.py). pyGLM has no equivalent;
    # the GLM itself is pyGLM (general.tool: pyGLM).
    from argparse import Namespace

    parse_namespace = Namespace(
        lc_config=lc_config,
        sub_ses_list=sub_ses_list,
        container_specific_config=container_specific_config,
    )
    from launchcontainers.core import LCError
    from launchcontainers.tools.dwi import require_extra

    try:
        require_extra(
            "fmri",
            ("bids", "nibabel", "numpy", "pandas", "scipy"),
            "l1_surface prepare",
        )
    except LCError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(2) from None
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


# `lc prepare` only: GLM input preparation (no Tool; not part of pyGLM).
_GLM_INPUT_PREP = {"l1_surface"}

# `lc run` / `lc qc`: names whose computation lives in another repo's Tool.
_MOVED = {
    "l1_surface": (
        "the GLM itself is pyGLM (MRI-hub/pyGLM): `lc prepare` with l1_surface "
        "only prepares its inputs; to fit, use an lc_config with general.tool: pyGLM "
        "and container_specific.pyGLM (see pyGLM's README), and point "
        "host_options.<host>.exe.pyGLM at pyGLM's `pyglm` program "
        "(pyGLM runs in its own environment; lc does not install it)"
    ),
}


def _tool_name(lc_config: str) -> str | None:
    from launchcontainers import utils as do

    if not op.isfile(lc_config):
        return None  # the Tool path reports the missing file
    general = (do.read_yaml(lc_config) or {}).get("general") or {}
    return general.get("tool", general.get("container"))


def _refuse_moved(lc_config: str) -> None:
    """Exit 2 with a pointer when the config names work that lives in another tool."""
    name = _tool_name(lc_config)
    if name in _MOVED:
        console.print(f"[red]ERROR[/red] {lc_config}: {name!r}: {_MOVED[name]}")
        raise typer.Exit(2)


def _prepare_with_tool(lc_config, sub_ses_list, container_config, log_fpath, err_fpath):
    """`lc prepare` through the Tool interface + launchcontainers.core provenance."""
    from launchcontainers.core import LCError

    from launchcontainers import lc_prepare

    res = None
    try:
        res = lc_prepare.prepare(
            lc_config, sub_ses_list, container_config, console=console
        )
    except LCError as exc:
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
    workdir: str = typer.Option(
        ..., "--workdir", "-w", help="Prepared analysis directory"
    ),
    run_lc: bool = typer.Option(
        False, "--run-lc", "-R", help="Submit / run the jobs (default: dry run)"
    ),
    sub_ses_list: str | None = typer.Option(
        None,
        "--sub-ses-list",
        "-ssl",
        help="Run only these sessions (default: <workdir>/subseslist.txt)",
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Do not ask before submitting"),
    quiet: bool = typer.Option(False, "--quiet", "-q", help="Quiet mode"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose mode"),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug mode"),
):
    setup_verbosity(quiet=quiet, verbose=verbose, debug=debug)
    _refuse_moved(op.join(workdir, "lc_config.yaml"))
    _run_with_tool(workdir, run_lc, sub_ses_list, yes)


def _run_with_tool(workdir, submit, sub_ses_list, yes):
    """`lc run` through the Tool interface: every job wrapped by lc-session."""
    from launchcontainers.core import LCError

    from launchcontainers import lc_run
    from launchcontainers.check import general_checks

    console.rule(f"[bold]lc run — {workdir}[/bold]")
    try:
        p = lc_run.plan(workdir, sub_ses_list)
    except LCError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(2) from None
    for n in p.notes:
        console.print(f"[yellow]{n}[/yellow]")
    for s in p.skipped:
        console.print(f"[yellow]SKIP[/yellow] {s}")
    console.print(
        f"{p.tool}: {len(p.jobs)} session(s), host {p.host} ({p.manager})", style="cyan"
    )
    sub, ses = p.jobs[0].sub, p.jobs[0].ses
    if (p.workdir / f"sub-{sub}").is_dir():
        console.print("\n### example session folder", style="bold")
        general_checks.cli_show_folder_struc(str(p.workdir), sub, ses)
    console.print("\n### first command", style="bold")
    console.print(p.jobs[0].line, markup=False, soft_wrap=True)

    if submit and not yes:
        try:
            ans = input(
                f"Launch {len(p.jobs)} job(s) on {p.host} ({p.manager})? [y/N]: "
            )
        except EOFError:
            ans = ""
        if ans.strip().lower() not in ("y", "yes"):
            console.print("Aborted.", style="cyan")
            raise typer.Exit(0)

    try:
        res = lc_run.run(p, submit, console=console)
    except LCError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(2) from None
    for n in p.notes:
        console.print(f"[yellow]{n}[/yellow]")
    if res.job_script is not None:
        console.print(f"\n### job script: {res.job_script}", style="bold")
        console.print(res.job_script.read_text(), markup=False)
    if not submit:
        console.print(
            f"[yellow]dry run[/yellow]: nothing launched; commands in {res.run_dir}. "
            "Add -R to launch."
        )
        return
    if res.submit_output:
        console.print(f"[green]submitted[/green] {res.submit_output}")
    if res.exit_codes:
        n_bad = res.n_failed
        console.print(
            f"{len(res.exit_codes) - n_bad}/{len(res.exit_codes)} sessions exited 0",
            style="green" if not n_bad else "red",
        )
    console.print(
        f"run folder: {res.run_dir}; per-session records: {p.workdir / 'logs'}",
        style="cyan",
    )
    if res.n_failed:
        raise typer.Exit(1)


@app.command()
def qc(
    workdir: str = typer.Option(..., "--workdir", "-w", help="Analysis directory"),
    sub_ses_list: str | None = typer.Option(
        None,
        "--sub-ses-list",
        "-ssl",
        help="Only these sessions (default: <workdir>/subseslist.txt)",
    ),
    debug: bool = typer.Option(False, "--debug", "-d", help="Debug mode"),
):
    """Did each launch work? Exit code, log, outputs → qc/index.html + failed list."""
    import time

    from launchcontainers import lc_qc
    from launchcontainers.core import LCError

    setup_verbosity(debug=debug)
    console.rule(f"[bold]lc qc — {workdir}[/bold]")
    _refuse_moved(op.join(workdir, "lc_config.yaml"))
    t0 = time.time()
    try:
        rep = lc_qc.qc(workdir, sub_ses_list, console=console)
    except LCError as exc:
        console.print(f"[red]ERROR[/red] {exc}")
        raise typer.Exit(2) from None
    n = {k: sum(rep.worst(r) == k for r in rep.results) for k in ("ok", "warn", "fail")}
    console.print(
        f"{rep.tool}: {len(rep.results)} sessions — ok {n['ok']}, warn {n['warn']}, "
        f"fail {n['fail']}  [dim]({time.time() - t0:.1f} s)[/dim]"
    )
    console.print(f"[green]saved[/green] {rep.index}")
    if rep.failed:
        console.print(
            f"[yellow]{len(rep.failed)} failed[/yellow] → {rep.failed_list} "
            f"(re-run: lc run -w {workdir} -ssl {rep.failed_list})"
        )


# Add other commands similarly...


def main():
    app()


if __name__ == "__main__":
    main()
