"""`lc run` on launchcontainers.core: prepared analysis dir → one wrapped command per session → backend.

Every job line is

    lc-session --analysis-dir <workdir> --tool <tool> --sub S --ses E -- <tool.command()>

so each session records its own start, end, exit code and log in
``<workdir>/logs/<run_id>/`` and a ``sessions.tsv`` row, whatever the backend.

The backend comes from ``host_options.<general.host>.manager`` (local / sge /
slurm), never from the host's name. Each ``lc run`` writes
``<workdir>/logs/run_<stamp>/`` (``_dryrun`` suffix without ``-R``):
``commands.txt`` (one line per session = one array task), the job script,
the scheduler's .out/.err, ``exit_codes.tsv`` and the submit output.
"""

from __future__ import annotations

import csv
import datetime
import resource
import shlex
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from launchcontainers.core import LCError
from launchcontainers.core.runlog import CONFIG_NAME, SESSIONS_TSV, SUBSES_NAME

from launchcontainers import lc_prepare

MANAGERS = ("local", "sge", "slurm")


@dataclass
class Job:
    sub: str
    ses: str
    line: str  # the shell line for this session (wrapper + tool command)


@dataclass
class RunPlan:
    tool: str
    workdir: Path
    host: str
    manager: str
    opts: dict
    jobs: list[Job]
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    tool_obj: object = None
    tc: object = None


@dataclass
class RunResult:
    plan: RunPlan
    run_dir: Path
    submitted: bool
    job_script: Path | None = None
    exit_codes: dict[tuple[str, str], int] = field(default_factory=dict)
    submit_output: str = ""

    @property
    def n_failed(self) -> int:
        return sum(1 for c in self.exit_codes.values() if c != 0)


# -- which sessions -------------------------------------------------------------
def _prepare_status(workdir: Path) -> dict[tuple[str, str], str]:
    """Latest status per sub/ses written by an `lc prepare` batch."""
    tsv = workdir / "logs" / SESSIONS_TSV
    if not tsv.is_file():
        return {}
    out: dict[tuple[str, str], str] = {}
    with open(tsv, newline="") as f:
        for row in csv.DictReader(f, delimiter="\t"):
            if "_prepare_" in (row.get("run_id") or ""):
                out[(row["sub"], row["ses"])] = row["status"]
    return out


def _wrapper_exe() -> str:
    """lc-session from the interpreter running lc (jobs must use the same venv)."""
    exe = Path(sys.executable).parent / "lc-session"
    return str(exe) if exe.exists() else (shutil.which("lc-session") or "lc-session")


def plan(workdir: Path, subseslist: Path | None = None) -> RunPlan:
    """Read the prepared analysis dir and build one job line per runnable session."""
    workdir = Path(workdir).expanduser().resolve()
    lc_config = workdir / CONFIG_NAME
    if not lc_config.is_file():
        raise LCError(f"{workdir}: no {CONFIG_NAME}; run `lc prepare` first")
    ssl = (
        Path(subseslist).expanduser().resolve() if subseslist else workdir / SUBSES_NAME
    )
    if not ssl.is_file():
        raise LCError(f"work list not found: {ssl}")
    tool, tc, rows = lc_prepare.load(lc_config, ssl, None)
    cfg = tc.general

    notes: list[str] = []
    dirs = [Path(d).resolve() for d in tool.analysis_dirs(tc)]
    if workdir not in dirs:
        notes.append(
            f"{workdir} is not one of {tool.name}'s analysis dirs for this config: "
            + ", ".join(map(str, dirs))
        )

    host = cfg.host
    opts = dict(cfg.host_options(host))
    manager = str(opts.get("manager") or "").lower()
    if manager not in MANAGERS:
        raise LCError(
            f"{lc_config}: host_options.{host}.manager must be one of {MANAGERS}, "
            f"got {opts.get('manager')!r}"
        )

    prepared = _prepare_status(workdir)
    wrapper = _wrapper_exe()
    jobs, skipped = [], []
    for r in rows:
        st = prepared.get((r.sub, r.ses))
        if st is not None and st != "prepared":
            skipped.append(f"sub-{r.sub} ses-{r.ses}: last prepare status {st!r}")
            continue
        cmd = list(tool.command(tc, r.sub, r.ses))
        argv = [
            wrapper,
            "--analysis-dir",
            str(workdir),
            "--tool",
            tool.name,
            "--sub",
            r.sub,
            "--ses",
            r.ses,
            "--",
            *cmd,
        ]
        jobs.append(Job(r.sub, r.ses, shlex.join(argv)))
    if not jobs:
        raise LCError(f"{workdir}: no session to run ({len(skipped)} skipped)")
    return RunPlan(
        tool.name, workdir, host, manager, opts, jobs, skipped, notes, tool, tc
    )


# -- job scripts ------------------------------------------------------------------
def _directives(prefix: str, lines) -> str:
    return "".join(f"{prefix} {x}\n" for x in (lines or []))


def _task_body(task_var: str, commands: Path, run_dir: Path) -> str:
    return f"""
CMD=$(sed -n "${{{task_var}}}p" {shlex.quote(str(commands))})
echo "task ${task_var} on $(hostname) at $(date '+%Y-%m-%d %H:%M:%S')"
echo "$CMD"
eval "$CMD"
exitcode=$?
printf '%s\\t%s\\n' "${task_var}" "$exitcode" >> {shlex.quote(str(run_dir / "exit_codes.tsv"))}
echo "task ${task_var} finished with exit code $exitcode"
exit $exitcode
"""


def sge_script(p: RunPlan, commands: Path, run_dir: Path) -> str:
    o = p.opts
    name = f"{o.get('job_name') or p.tool}_array"
    head = (
        "#!/bin/bash\n"
        f"#$ -t 1-{len(p.jobs)}\n"
        f"#$ -N {name}\n"
        f"#$ -o {run_dir}/{name}_$JOB_ID.$TASK_ID.out\n"
        f"#$ -e {run_dir}/{name}_$JOB_ID.$TASK_ID.err\n"
        "#$ -S /bin/bash\n"
    )
    if o.get("walltime"):
        head += f"#$ -l h_rt={o['walltime']}\n"
    if o.get("queue"):
        head += f"#$ -q {o['queue']}\n"
    head += _directives("#$", o.get("sge_extra"))
    return head + _task_body("SGE_TASK_ID", commands, run_dir)


def slurm_script(p: RunPlan, commands: Path, run_dir: Path) -> str:
    o = p.opts
    name = f"{o.get('job_name') or p.tool}_array"
    head = (
        "#!/bin/bash\n"
        f"#SBATCH --array=1-{len(p.jobs)}\n"
        f"#SBATCH --job-name={name}\n"
        f"#SBATCH --output={run_dir}/{name}_%A_%a.out\n"
        f"#SBATCH --error={run_dir}/{name}_%A_%a.err\n"
        "#SBATCH --ntasks=1\n"
    )
    for key, flag in (
        ("walltime", "time"),
        ("cores", "cpus-per-task"),
        ("memory", "mem"),
        ("partition", "partition"),
        ("qos", "qos"),
    ):
        if o.get(key) not in (None, ""):
            head += f"#SBATCH --{flag}={o[key]}\n"
    head += _directives("#SBATCH", o.get("slurm_extra"))
    return head + _task_body("SLURM_ARRAY_TASK_ID", commands, run_dir)


# -- local backend ----------------------------------------------------------------
def _parse_mem(s: str) -> int:
    s = str(s).strip().lower().removesuffix("ib").removesuffix("b")
    mult = {"k": 1024, "m": 1024**2, "g": 1024**3, "t": 1024**4}
    return int(float(s[:-1]) * mult[s[-1]]) if s and s[-1] in mult else int(float(s))


def local_workers(opts: dict) -> int:
    """``n_workers``; the legacy ``launch_mode: parallel`` + ``max_workers`` still works."""
    if opts.get("n_workers"):
        return max(1, int(opts["n_workers"]))
    if str(opts.get("launch_mode", "serial")) == "parallel":
        return max(1, int(opts.get("max_workers") or 1))
    return 1


def _run_local(job: Job, run_dir: Path, mem_bytes: int | None) -> int:
    def limit():  # runs in the child before exec
        if mem_bytes:
            try:
                resource.setrlimit(resource.RLIMIT_AS, (mem_bytes, mem_bytes))
            except (ValueError, OSError):
                pass

    out = run_dir / f"sub-{job.sub}_ses-{job.ses}.out"
    with open(out, "wb") as f:
        # login shell: module systems (module load apptainer) are initialised
        return subprocess.run(
            ["bash", "-l", "-c", job.line],
            stdout=f,
            stderr=subprocess.STDOUT,
            preexec_fn=limit,
        ).returncode


# -- the step ---------------------------------------------------------------------
def run(p: RunPlan, submit: bool, console=None) -> RunResult:
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = p.workdir / "logs" / f"run_{stamp}{'' if submit else '_dryrun'}"
    run_dir.mkdir(parents=True, exist_ok=False)
    commands = run_dir / "commands.txt"
    commands.write_text("".join(j.line + "\n" for j in p.jobs))
    (run_dir / "jobs.tsv").write_text(
        "task\tsub\tses\n"
        + "".join(f"{i}\t{j.sub}\t{j.ses}\n" for i, j in enumerate(p.jobs, 1))
    )
    res = RunResult(p, run_dir, submitted=submit)

    before = getattr(p.tool_obj, "before_run", None)
    if before is not None:
        p.notes += list(before(p.tc, [(j.sub, j.ses) for j in p.jobs], submit) or [])

    if p.manager in ("sge", "slurm"):
        text = (sge_script if p.manager == "sge" else slurm_script)(
            p, commands, run_dir
        )
        script = run_dir / ("job.sh" if p.manager == "sge" else "job.slurm")
        script.write_text(text)
        script.chmod(0o755)
        res.job_script = script
        if submit:
            exe = "qsub" if p.manager == "sge" else "sbatch"
            try:
                cp = subprocess.run(
                    [exe, str(script)], capture_output=True, text=True, timeout=60
                )
            except (OSError, subprocess.TimeoutExpired) as exc:
                raise LCError(f"{exe} failed: {exc}") from None
            res.submit_output = (cp.stdout + cp.stderr).strip()
            (run_dir / "submit.txt").write_text(
                f"$ {exe} {script}\nexit {cp.returncode}\n{res.submit_output}\n"
            )
            if cp.returncode != 0:
                raise LCError(f"{exe} exited {cp.returncode}: {res.submit_output}")
        return res

    # local
    if not submit:
        return res
    n = local_workers(p.opts)
    mem = p.opts.get("mem_per_job")
    mem_bytes = _parse_mem(mem) if mem else None
    if console is not None:
        console.print(f"running {len(p.jobs)} session(s) locally, {n} at a time")

    def done(job: Job, rc: int) -> None:
        res.exit_codes[(job.sub, job.ses)] = rc
        with open(run_dir / "exit_codes.tsv", "a") as f:
            f.write(f"{p.jobs.index(job) + 1}\t{rc}\n")
        if console is not None:
            style = "green" if rc == 0 else "red"
            console.print(f"  [{style}]exit {rc}[/{style}] sub-{job.sub} ses-{job.ses}")

    if n > 1:
        with ThreadPoolExecutor(max_workers=n) as pool:
            futs = {pool.submit(_run_local, j, run_dir, mem_bytes): j for j in p.jobs}
            for fut in as_completed(futs):
                done(futs[fut], fut.result())
    else:
        for j in p.jobs:
            done(j, _run_local(j, run_dir, mem_bytes))
    return res
