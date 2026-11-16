"""``lc-session``: run one session's command and record it.

::

    lc-session --analysis-dir D --tool T --sub S --ses E [--project-dir P] -- <cmd…>

Every job line ``lc run`` submits goes through this wrapper, so each array
task records its own start, end, exit code and log:

1. opens a run (runlog.start_run, scope ``sub-S_ses-E``): the run folder is
   ``P/logs/T/<D name>_sub-S_ses-E_<stamp>/`` with ``--project-dir``, else
   ``D/logs/<D name>_sub-S_ses-E_<stamp>/``;
2. runs ``<cmd…>`` with ``ANALYSIS_RUN_ID`` / ``ANALYSIS_RUN_DIR`` set (a tool
   CLI inside records into this run instead of opening its own), streaming
   its merged stdout/stderr to ``<run folder>/sub-S_ses-E.log`` and to stdout;
3. closes the run (runlog.finish_run): run.json, runs.tsv + sessions.tsv rows
   in D (and in any analysis dir a nested tool record names), PROVENANCE lines;
4. exits with the child's exit code (127 if it could not be started, 130 on
   Ctrl-C). It never raises because the child failed.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import typer
from rich.console import Console

from launchcontainers.core import __version__
from launchcontainers.core.io import bare
from launchcontainers.core.runlog import (
    CONFIG_NAME,
    ENV_RUN_DIR,
    ENV_RUN_ID,
    finish_run,
    nested_records,
    start_run,
)
from launchcontainers.core.tool import tool_version

console = Console(stderr=True)
app = typer.Typer(add_completion=False, pretty_exceptions_show_locals=False)


def _stream(cmd: list[str], env: dict[str, str], log_path: Path) -> int:
    """Run ``cmd``; copy its merged output to ``log_path`` and stdout. Returns the exit code."""
    out = sys.stdout.buffer
    with open(log_path, "ab") as log:
        try:
            proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env
            )
        except OSError as exc:
            msg = f"lc-session: cannot start {cmd[0]!r}: {exc}\n".encode()
            log.write(msg)
            out.write(msg)
            out.flush()
            return 127
        try:
            assert proc.stdout is not None
            for line in iter(proc.stdout.readline, b""):
                log.write(line)
                log.flush()
                out.write(line)
                out.flush()
            return proc.wait()
        except KeyboardInterrupt:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            log.write(b"lc-session: interrupted\n")
            return 130


def _session_status(exit_code: int, run_dir: Path, sub: str, ses: str) -> str:
    """Exit code → ok/failed; a nested tool record's own status for this session wins on success."""
    if exit_code != 0:
        return "failed"
    for rec in nested_records(run_dir):
        for s in rec.get("Sessions") or []:
            if (
                bare(s.get("sub", ""), "sub") == sub
                and bare(s.get("ses", ""), "ses") == ses
            ):
                return str(s.get("status") or ("ok" if s.get("ok") else "failed"))
    return "ok"


def run_session(
    analysis_dir: Path,
    tool: str,
    sub: str,
    ses: str,
    cmd: list[str],
    project_dir: Path | None = None,
) -> dict[str, Any]:
    """The unit behind the CLI. Returns {exit_code, status, run_dir, log, markers}."""
    analysis_dir = Path(analysis_dir).resolve()
    analysis_dir.mkdir(parents=True, exist_ok=True)
    sub, ses = bare(sub, "sub"), bare(ses, "ses")
    scope = f"sub-{sub}_ses-{ses}"
    config = analysis_dir / CONFIG_NAME
    # A wrapper is never nested in another wrapper's run.
    saved = {k: os.environ.pop(k) for k in (ENV_RUN_ID, ENV_RUN_DIR) if k in os.environ}
    try:
        run = start_run(
            tool,
            project_dir,
            analysis_dir.name,
            [analysis_dir],
            config if config.is_file() else None,
            None,
            version=tool_version(tool),
            scope=scope,
        )
    finally:
        os.environ.update(saved)
    run.record.update(
        Command=list(cmd),
        Wrapper={"lc-session": __version__, "argv": sys.argv},
        Sub=sub,
        Ses=ses,
    )
    log_path = run.run_dir / f"{scope}.log"
    env = {**os.environ, ENV_RUN_ID: run.run_id, ENV_RUN_DIR: str(run.run_dir)}
    exit_code = _stream(cmd, env, log_path)
    status = _session_status(exit_code, run.run_dir, sub, ses)
    finish_run(run, exit_code, [{"sub": sub, "ses": ses, "status": status}])
    return {
        "exit_code": exit_code,
        "status": status,
        "run_dir": run.run_dir,
        "log": log_path,
        "markers": run.markers(),
    }


@app.command(context_settings={"allow_interspersed_args": False})
def main(
    analysis_dir: Path = typer.Option(
        ..., "--analysis-dir", help="Analysis directory the session belongs to"
    ),
    tool: str = typer.Option(
        ..., "--tool", help="Tool name (key under container_specific)"
    ),
    sub: str = typer.Option(..., "--sub", help="Subject label, e.g. 02"),
    ses: str = typer.Option(..., "--ses", help="Session label, e.g. 05"),
    project_dir: Path | None = typer.Option(
        None, "--project-dir", help="Put the run folder under <project>/logs/<tool>/"
    ),
    cmd: list[str] = typer.Argument(..., help="The command to run, after `--`"),
) -> None:
    """Run one session's command, log it, and record start/end/exit code."""
    console.rule(
        f"[bold]lc-session — {tool} sub-{bare(sub, 'sub')} ses-{bare(ses, 'ses')}[/bold]"
    )
    res = run_session(analysis_dir, tool, sub, ses, cmd, project_dir)
    colour = "green" if res["exit_code"] == 0 else "red"
    console.print(
        f"[{colour}]exit {res['exit_code']}[/{colour}] status={res['status']}  log: {res['log']}"
    )
    for m in res["markers"]:
        print(m, flush=True)
    raise typer.Exit(res["exit_code"])


if __name__ == "__main__":
    app()
