"""Capture golden snapshots from the current (pre-restructure) code.

Run once, from the commit the snapshots should describe:

    PYTHONPATH=<repo> python tests/golden/capture_legacy.py <scratch_dir>

Writes ``tests/golden/data/<scenario>.json``. Re-running on the same commit
must reproduce them byte for byte.

The goldens describe commit a7448d9. This script drives the legacy code
(``do_prepare``, ``clusters/sge.py``, ``clusters/slurm.py``), which current
``lc`` no longer calls; to re-capture, run it from a checkout of a7448d9
(``git worktree add /tmp/lc-a7448d9 a7448d9``), not from the current tree.
"""

from __future__ import annotations

import io
import sys
from argparse import Namespace
from contextlib import redirect_stdout
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE)]


def _helpers():
    import fakedata
    import scenarios
    import snapshot

    return fakedata, scenarios, snapshot


def run_legacy(root: Path, name, container, layout, host, overrides) -> dict:
    fakedata, _, snapshot = _helpers()
    norm, snapshot_dir = snapshot.norm, snapshot.snapshot_dir
    from launchcontainers import do_prepare
    from launchcontainers import utils as do
    from launchcontainers.clusters import sge, slurm
    from launchcontainers.gen_jobscript.gen_container_cmd import gen_RTP2_cmd

    overrides = {
        k: (str(root / "configs" / "fsmask.nii.gz") if v == "FSMASK" else v)
        for k, v in overrides.items()
    }
    cfg = fakedata.make_config(root, container, name, layout, host, **overrides)
    ns = Namespace(
        lc_config=str(cfg),
        sub_ses_list=str(root / "configs" / "subseslist.txt"),
        container_specific_config=str(root / "configs" / f"{container}.json"),
    )
    buf = io.StringIO()
    with redirect_stdout(buf):
        ok, adir = do_prepare.main(ns)
    snap = snapshot_dir(Path(adir), root)
    snap["analysis_dir"] = norm(str(adir), root)
    snap["status"] = {"success": bool(ok)}
    lc = do.read_yaml(str(Path(adir) / "lc_config.yaml"))
    rows = do.parse_subses_list(str(Path(adir) / "subseslist.txt"))
    snap["commands"] = {
        f"sub-{s}_ses-{e}": norm(gen_RTP2_cmd(lc, s, e, adir), root) for s, e in rows
    }
    rns = Namespace(workdir=adir, run_lc=False)
    if host == "BCBL":
        snap["job_script"] = norm(
            sge.gen_sge_array_job_script(rns, "<LOGDIR>", len(rows)), root
        )
    elif host == "DIPC":
        snap["job_script"] = norm(
            slurm.gen_slurm_array_job_script(rns, "<LOGDIR>", len(rows)), root
        )
    return snap


def main() -> None:
    fakedata, scenarios, snapshot = _helpers()
    root = Path(sys.argv[1]).resolve()
    if root.exists() and any(root.iterdir()):
        sys.exit(f"{root} must be empty")
    fakedata.build_dataset(root)
    only = set(sys.argv[2:])
    for sc in scenarios.SCENARIOS:
        if only and sc[0] not in only:
            continue
        snap = run_legacy(root, *sc)
        snapshot.dump(snap, HERE / "data" / f"{sc[0]}.json")
        print(
            f"{sc[0]:28s} success={snap['status']['success']} entries={len(snap['tree'])}"
        )


if __name__ == "__main__":
    main()
