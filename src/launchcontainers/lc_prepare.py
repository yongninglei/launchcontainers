"""`lc prepare` on launchcontainers.core: config → guarded analysis dir → per-session prepare.

The loop over the work list lives here (and only here); each tool stages one
sub/ses per call. Every prepare is recorded as a run in the analysis dir
(``logs/<analysis>_prepare_<stamp>/run.json``, ``logs/runs.tsv``,
``logs/sessions.tsv`` with status ``prepared`` / ``failed``).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from launchcontainers.core import LCError
from launchcontainers.core.config import load_lc_config
from launchcontainers.core.provenance import (
    guard_analysis_dir,
    software_info,
    write_dataset_description,
)
from launchcontainers.core.runlog import finish_run, start_run
from launchcontainers.core.subseslist import read_subseslist
from launchcontainers.core.tool import (
    SessionResult,
    Tool,
    ToolConfig,
    get_tool,
    prepare_analysis,
)

CODE_DIR = Path(__file__).resolve().parents[2]
PACKAGES = ("launchcontainers",)


@dataclass
class PrepareResult:
    tool: str
    analysis_dirs: list[Path]
    results: list[SessionResult] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    run_id: str | None = None

    @property
    def ok(self) -> bool:
        return all(r.status in ("prepared", "ok", "skipped") for r in self.results)


def resolve_tool(name: str) -> Tool:
    """A built-in container tool, or any tool installed under launchcontainers.tools."""
    from launchcontainers.tools import BUILTIN

    if name in BUILTIN:
        return BUILTIN[name]
    return get_tool(name)


def load(lc_config: Path, subseslist: Path, container_config: Path | None = None):
    """Validate the inputs without writing anything: (tool, tool config, rows)."""
    cfg = load_lc_config(lc_config)
    tool = resolve_tool(cfg.tool)
    tc: ToolConfig = tool.load(cfg)
    rows = read_subseslist(subseslist)
    if not rows:
        raise LCError(f"{subseslist}: no RUN=True rows")
    tc.rows = {(r.sub, r.ses): r for r in rows}
    tc.extra["subseslist"] = str(Path(subseslist).resolve())
    if container_config is not None:
        cc = Path(container_config).expanduser().resolve()
        if not cc.is_file():
            raise LCError(f"container config not found: {cc}")
        tc.extra["container_config"] = str(cc)
    return tool, tc, rows


def tool_version(tool: Tool, tc: ToolConfig) -> str:
    """The version that identifies the analysis: the container's, else the tool's."""
    return str(tc.extra.get("container_version") or tool.version)


def prepare(
    lc_config: Path,
    subseslist: Path,
    container_config: Path | None = None,
    console=None,
) -> PrepareResult:
    tool, tc, rows = load(lc_config, subseslist, container_config)
    cfg = tc.general
    version = tool_version(tool, tc)
    dirs = [Path(d) for d in tool.analysis_dirs(tc)]
    res = PrepareResult(tool.name, dirs)

    runtime_only = tuple(getattr(tool, "runtime_only", ()))
    for d in dirs:
        d.mkdir(parents=True, exist_ok=True)
        res.notes += guard_analysis_dir(
            d, cfg, tool.name, version, runtime_only=runtime_only
        )
        write_dataset_description(
            d,
            name=f"{tool.name} {cfg.analysis_name}",
            description=f"{tool.name} {version}, prepared by launchcontainers",
            generated_by=[
                {**software_info(tool.name, version), "Container": tool.name},
                software_info(
                    "launchcontainers",
                    _lc_version(),
                    "https://github.com/garikoitz/launchcontainers",
                    CODE_DIR,
                ),
            ],
            source_datasets={"raw": cfg.bids_dir},
        )

    run = start_run(
        tool.name,
        None,
        f"{cfg.analysis_name}_prepare",
        dirs,
        cfg.source_path,
        Path(subseslist),
        version=version,
        code_dir=CODE_DIR,
        packages=PACKAGES,
    )
    res.run_id = run.run_id
    exit_code = 1
    try:
        pairs = prepare_analysis(tool, tc, [(r.sub, r.ses) for r in rows])
        for sub, ses in pairs:
            if console is not None:
                console.print(f"[bold cyan]sub-{sub} ses-{ses}[/bold cyan]")
            r = tool.prepare(tc, sub, ses)
            res.results.append(r)
            if console is not None:
                style = "green" if r.status == "prepared" else "red"
                console.print(
                    f"  [{style}]{r.status}[/{style}] " + "; ".join(r.messages)
                )
        exit_code = 0 if res.ok else 1
    finally:
        finish_run(
            run,
            exit_code,
            [{"sub": r.sub, "ses": r.ses, "status": r.status} for r in res.results],
        )
    return res


def _lc_version() -> str:
    from launchcontainers import __version__

    return __version__
