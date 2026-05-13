"""
00b_copy_fmap_to_bids.py
------------------------
Copy fmap folders and scans.tsv files from a raw_nifti BIDS tree into a
target BIDS directory, then chmod the destination fmap folder and all its
JSON files to 755.

Pre-requisite
-------------
Run after ``00_merge_split_ses_reording_runs.py`` and
``01_drop_duplicated_sbrefs.py`` so that the fmap files in raw_nifti are
already clean (correct run indices, no orphan sbrefs).

What it does
------------
1. For each sub/ses pair: rsync the fmap/ subdirectory into the target BIDS
   tree, excluding ``*_orig.json`` files.
2. rsync the session-level scans.tsv.
3. chmod 755 the destination fmap/ directory and all .json / .nii.gz files
   inside it.
4. Optionally sync the top-level BIDS components (dataset_description.json,
   participants.tsv/json, README) once — use --sync-bids-components.

Run-number formatting
---------------------
  fmap   → 1-digit  (run-1, run-2, …)   [same convention as 00_ / 02_]
  others → 2-digit  (run-01, run-02, …)

Usage
-----
  # dry-run single session (default):
  python 00b_copy_fmap_to_bids.py -b /raw_nifti -t /BIDS -s 10,02

  # execute:
  python 00b_copy_fmap_to_bids.py -b /raw_nifti -t /BIDS -s 10,02 --execute

  # batch + sync top-level BIDS components:
  python 00b_copy_fmap_to_bids.py -b /raw_nifti -t /BIDS -f subseslist.tsv \\
      --execute --sync-bids-components
"""

from __future__ import annotations

import glob
import os
import os.path as op
import subprocess
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from launchcontainers.utils import parse_subses_list

console = Console()
app = typer.Typer(pretty_exceptions_show_locals=False)

# Top-level BIDS files synced by --sync-bids-components
BIDS_COMPONENTS = [
    "dataset_description.json",
    "participants.json",
    "participants.tsv",
    "README",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _rsync(src: str, dst: str, extra_args: list[str] | None = None) -> bool:
    """Run rsync -av src dst.  Returns True on success."""
    cmd = ["rsync", "-av"] + (extra_args or []) + [src, dst]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        console.print(f"  [red][ERROR] rsync failed:[/] {result.stderr.strip()}")
        return False
    return True


def _chmod755(path: str) -> None:
    """chmod 755 a file or directory."""
    try:
        os.chmod(path, 0o755)
    except Exception as exc:
        console.print(f"  [yellow][WARN] chmod failed for {path}: {exc}[/]")


# ---------------------------------------------------------------------------
# Core operations
# ---------------------------------------------------------------------------


def _sync_bids_components(src: str, dst: str, force: bool, dry_run: bool) -> None:
    """Sync top-level BIDS components (run once)."""
    os.makedirs(dst, exist_ok=True)
    for item in BIDS_COMPONENTS:
        src_item = op.join(src, item)
        dst_item = op.join(dst, item)
        if not op.exists(src_item):
            continue
        if op.exists(dst_item) and not force:
            console.print(f"  [dim]skip (exists) {item}[/]")
            continue
        if dry_run:
            console.print(f"  [dim][DRY][/] would copy {item}")
        else:
            ok = _rsync(src_item, dst_item)
            if ok:
                console.print(f"  [green]✓[/] copied {item}")


def _copy_fmap(
    src_bids: str,
    dst_bids: str,
    sub: str,
    ses: str,
    force: bool,
    dry_run: bool,
) -> dict:
    """
    Copy the fmap/ directory for sub/ses from src_bids to dst_bids.
    After copying, chmod 755 the dst fmap dir and all files inside it.
    Returns a result dict.
    """
    src_fmap = op.join(src_bids, f"sub-{sub}", f"ses-{ses}", "fmap")
    dst_fmap = op.join(dst_bids, f"sub-{sub}", f"ses-{ses}", "fmap")

    if not op.isdir(src_fmap):
        console.print(
            f"  [yellow]sub-{sub} ses-{ses}: no fmap dir in source — skipped[/]"
        )
        return {
            "sub": sub,
            "ses": ses,
            "fmap": "no_src",
            "tsv": "skip",
            "error": "no fmap src",
        }

    if op.exists(dst_fmap) and not force:
        console.print(
            f"  [dim]sub-{sub} ses-{ses}: fmap already exists — skipped (use --force)[/]"
        )
        return {"sub": sub, "ses": ses, "fmap": "skip", "tsv": "skip", "error": None}

    if dry_run:
        n_files = len(glob.glob(op.join(src_fmap, "*")))
        console.print(
            f"  [dim][DRY][/] would rsync  {src_fmap}  →  {dst_fmap}  ({n_files} files)"
        )
        return {"sub": sub, "ses": ses, "fmap": "dry", "tsv": "dry", "error": None}

    os.makedirs(dst_fmap, exist_ok=True)
    ok = _rsync(
        src_fmap + "/",
        dst_fmap + "/",
        extra_args=["--exclude=*_orig.json"],
    )
    if not ok:
        return {
            "sub": sub,
            "ses": ses,
            "fmap": "error",
            "tsv": "skip",
            "error": "rsync failed",
        }

    # chmod 755 the fmap dir itself and every file inside it
    _chmod755(dst_fmap)
    for f in glob.glob(op.join(dst_fmap, "*")):
        _chmod755(f)

    n_copied = len(glob.glob(op.join(dst_fmap, "*")))
    console.print(
        f"  [green]✓[/] sub-{sub} ses-{ses}  fmap  ({n_copied} files, chmod 755 applied)"
    )
    return {"sub": sub, "ses": ses, "fmap": "ok", "tsv": None, "error": None}


def _copy_scans_tsv(
    src_bids: str,
    dst_bids: str,
    sub: str,
    ses: str,
    force: bool,
    dry_run: bool,
) -> str:
    """
    Copy the session-level scans.tsv for sub/ses.
    Returns "ok" | "skip" | "dry" | "no_src" | "error".
    """
    # scans.tsv is at ses level: sub-XX/ses-YY/sub-XX_ses-YY_scans.tsv
    pattern = op.join(
        src_bids, f"sub-{sub}", f"ses-{ses}", f"sub-{sub}_ses-{ses}_scans.tsv"
    )
    matches = glob.glob(pattern)
    if not matches:
        console.print(f"  [yellow]sub-{sub} ses-{ses}: no scans.tsv found — skipped[/]")
        return "no_src"

    src_tsv = matches[0]
    dst_tsv = op.join(dst_bids, f"sub-{sub}", f"ses-{ses}", op.basename(src_tsv))

    if op.exists(dst_tsv) and not force:
        console.print(
            f"  [dim]sub-{sub} ses-{ses}: scans.tsv already exists — skipped[/]"
        )
        return "skip"

    if dry_run:
        console.print(f"  [dim][DRY][/] would copy scans.tsv → {dst_tsv}")
        return "dry"

    os.makedirs(op.dirname(dst_tsv), exist_ok=True)
    ok = _rsync(src_tsv, dst_tsv)
    if not ok:
        return "error"

    console.print(f"  [green]✓[/] sub-{sub} ses-{ses}  scans.tsv")
    return "ok"


# ---------------------------------------------------------------------------
# Per-session driver
# ---------------------------------------------------------------------------


def _process_session(
    src_bids: str,
    dst_bids: str,
    sub: str,
    ses: str,
    force: bool,
    dry_run: bool,
) -> dict:
    result = _copy_fmap(src_bids, dst_bids, sub, ses, force, dry_run)
    tsv_status = _copy_scans_tsv(src_bids, dst_bids, sub, ses, force, dry_run)
    result["tsv"] = tsv_status
    return result


# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------


def _print_summary(results: list[dict]) -> None:
    t = Table(
        title="copy-fmap-to-bids summary",
        show_header=True,
        header_style="bold magenta",
        box=None,
        padding=(0, 1),
    )
    t.add_column("sub", justify="right", style="cyan")
    t.add_column("ses", justify="right", style="cyan")
    t.add_column("fmap", justify="center")
    t.add_column("tsv", justify="center")
    t.add_column("status", justify="center")

    for r in results:
        fmap_cell = {
            "ok": "[green]copied[/]",
            "skip": "[dim]skip[/]",
            "dry": "[dim]dry[/]",
            "no_src": "[yellow]no src[/]",
            "error": "[red]ERROR[/]",
        }.get(r["fmap"], r["fmap"])

        tsv_cell = {
            "ok": "[green]copied[/]",
            "skip": "[dim]skip[/]",
            "dry": "[dim]dry[/]",
            "no_src": "[yellow]no src[/]",
            "error": "[red]ERROR[/]",
        }.get(r.get("tsv", ""), "—")

        st = (
            "[red]ERROR[/]"
            if r.get("error")
            else "[green]OK[/]"
            if r["fmap"] == "ok"
            else "[dim]—[/]"
        )
        t.add_row(r["sub"], r["ses"], fmap_cell, tsv_cell, st)

    console.print(t)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


@app.command()
def main(
    src_bids: Path = typer.Option(
        ..., "--src", "-b", help="Source BIDS directory (e.g. raw_nifti)."
    ),
    dst_bids: Path = typer.Option(..., "--dst", "-t", help="Target BIDS directory."),
    subses: Optional[list[str]] = typer.Option(
        None, "--subses", "-s", help="sub,ses pair e.g. -s 10,02  (repeatable)."
    ),
    subses_file: Optional[Path] = typer.Option(
        None, "--file", "-f", help="TSV/CSV with columns sub, ses."
    ),
    force: bool = typer.Option(
        False, "--force", help="Overwrite existing destination files."
    ),
    dry_run: bool = typer.Option(
        True, "--dry-run/--execute", help="Print plan only (default) or execute copies."
    ),
    sync_bids_components: bool = typer.Option(
        False,
        "--sync-bids-components",
        help="Also sync top-level BIDS files (dataset_description.json etc.).",
    ),
) -> None:
    """
    Copy fmap folders and scans.tsv from a raw_nifti BIDS tree into a target
    BIDS directory.  After copying, chmod 755 is applied to the fmap dir and
    all files inside it.

    Default: dry-run.  Use --execute to apply.

    Examples:
        python 00b_copy_fmap_to_bids.py -b /raw_nifti -t /BIDS -s 10,02
        python 00b_copy_fmap_to_bids.py -b /raw_nifti -t /BIDS -f subseslist.tsv --execute
    """
    if subses_file is not None:
        pairs = parse_subses_list(subses_file)
    elif subses:
        pairs = []
        for token in subses:
            parts = token.split(",")
            if len(parts) != 2:
                console.print(f"[red]Error:[/] -s expects 'sub,ses' (got '{token}').")
                raise typer.Exit(1)
            pairs.append((parts[0].strip().zfill(2), parts[1].strip().zfill(2)))
    else:
        console.print("[bold red]Error:[/] provide -s sub,ses  or  -f <subseslist>.")
        raise typer.Exit(1)

    mode_str = "[yellow]DRY-RUN[/]" if dry_run else "[bold red]EXECUTE[/]"
    console.print(
        f"\n[bold]00b_copy_fmap_to_bids[/]  src={src_bids}  dst={dst_bids}  mode={mode_str}\n"
    )

    if sync_bids_components:
        console.print("[bold]Syncing top-level BIDS components…[/]")
        _sync_bids_components(
            str(src_bids), str(dst_bids), force=force, dry_run=dry_run
        )
        console.print()

    results = []
    for sub, ses in pairs:
        results.append(
            _process_session(str(src_bids), str(dst_bids), sub, ses, force, dry_run)
        )

    console.print()
    _print_summary(results)

    if dry_run:
        console.print("\n[yellow]Dry-run complete.  Use --execute to apply.[/]")
    else:
        n_ok = sum(1 for r in results if r["fmap"] == "ok")
        n_err = sum(1 for r in results if r.get("error"))
        console.print(f"\n[green]Done.[/]  copied={n_ok}  errors={n_err}")


if __name__ == "__main__":
    app()
