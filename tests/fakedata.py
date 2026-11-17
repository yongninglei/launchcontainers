"""Fake BIDS dataset + lc_config builder for the golden-snapshot tests.

Every file is empty except the reverse-PE DWI (a tiny 4D NIfTI, because prepare
reads its shape to write mock bval/bvec) and the small JSON/CSV configs. Nothing
here touches real data.

Layout built under ``root``::

    basedir/BIDS/sub-0X/ses-0Y/{anat,dwi,func}
    basedir/BIDS/derivatives/<upstream analyses, both deriv layouts>
    configs/{subseslist.txt, <container>.json, tractparams.csv, fake_annots.zip, fsmask.nii.gz}
    containers/<container>_<version>.sif
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

import yaml

SUBSES = [("01", "01"), ("01", "02"), ("02", "01")]

VERSIONS = {
    "anatrois": "4.6.1-7.3.2",
    "freesurferator": "0.2.1_7.4.1",
    "rtppreproc": "1.2.0-3.0.3",
    "rtp2-preproc": "0.2.1_3.0.4",
    "rtp2-pipeline": "0.2.1_3.0.4",
}


DWI_EXTRA = ("bids", "nibabel", "numpy", "pandas")  # tools/dwi.py require_extra("dwi", ...)


def require_extra(modules=DWI_EXTRA, extra: str = "dwi") -> None:
    """Skip the calling test when a launchcontainers extra is not installed.

    The core install (pyyaml, rich, typer) is supported on its own, so tests of the
    DWI tools / l1_surface prepare skip there instead of erroring.
    """
    import importlib.util

    import pytest

    missing = [m for m in modules if importlib.util.find_spec(m) is None]
    if missing:
        pytest.skip(f"needs the {extra!r} extra ({', '.join(missing)}): uv sync --extra {extra}")


def _touch(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.touch()


def _js(p: Path, d: dict) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(d, indent=2))


def _nii4(p: Path, nvol: int = 3) -> None:
    import nibabel as nib
    import numpy as np

    p.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(np.zeros((2, 2, 2, nvol), dtype=np.int16), np.eye(4))
    nib.save(img, str(p))


def _deriv(bids: Path, cont: str, analysis: str, layout: str) -> Path:
    ver = VERSIONS[cont]
    if layout == "legacy":
        return bids / "derivatives" / f"{cont}_{ver}" / f"analysis-{analysis}"
    return bids / "derivatives" / f"{cont}-{ver}_{analysis}"


def build_dataset(root: Path) -> Path:
    """Create the fake dataset under *root*; return *root*."""
    root = Path(root)
    bids = root / "basedir" / "BIDS"
    _js(bids / "dataset_description.json", {"Name": "fake", "BIDSVersion": "1.8.0"})
    for sub, ses in SUBSES:
        d = bids / f"sub-{sub}" / f"ses-{ses}"
        pre = f"sub-{sub}_ses-{ses}"
        _touch(d / "anat" / f"{pre}_T1w.nii.gz")
        _js(d / "anat" / f"{pre}_T1w.json", {})
        for ext in ("nii.gz", "bval", "bvec"):
            _touch(d / "dwi" / f"{pre}_dir-PA_dwi.{ext}")
        _js(d / "dwi" / f"{pre}_dir-PA_dwi.json", {"PhaseEncodingDirection": "j"})
        # reverse PE without bval/bvec on purpose: prepare must mock them
        _nii4(d / "dwi" / f"{pre}_dir-AP_dwi.nii.gz")
        _js(d / "dwi" / f"{pre}_dir-AP_dwi.json", {"PhaseEncodingDirection": "j-"})
        _touch(d / "func" / f"{pre}_task-fake_run-01_bold.nii.gz")
        _js(
            d / "func" / f"{pre}_task-fake_run-01_bold.json",
            {"RepetitionTime": 1.0, "TaskName": "fake"},
        )

        for layout in ("legacy", "new"):

            def out(c, a, layout=layout, sub=sub, ses=ses):
                return (
                    _deriv(bids, c, a, layout) / f"sub-{sub}" / f"ses-{ses}" / "output"
                )

            o = out("freesurferator", "fakeanat")
            _touch(o / "T1.nii.gz")
            _touch(o / "brain.nii.gz")
            with zipfile.ZipFile(o / "fs.zip", "w") as z:
                for roi in ("L_Ventral", "R_Ventral", "Brain"):
                    z.writestr(f"fs/ROIs/{roi}.nii.gz", b"")
            o2 = out("anatrois", "prefs")
            _touch(o2 / f"anatrois_S{sub}.zip")
            _touch(o2 / f"S{sub}" / "tmp" / "control.dat")
            o3 = out("rtp2-preproc", "fakepre")
            for f in ("t1.nii.gz", "dwi.nii.gz", "dwi.bvals", "dwi.bvecs"):
                _touch(o3 / f)
            o4 = out("rtppreproc", "fakepre")
            for f in ("t1.nii.gz", "dwi.nii.gz", "dwi.bvals", "dwi.bvecs"):
                _touch(o4 / f)

    c = root / "configs"
    c.mkdir(parents=True, exist_ok=True)
    (c / "subseslist.txt").write_text(
        "sub,ses,RUN,anat,dwi,func\n"
        + "".join(f"{s},{e},True,True,True,True\n" for s, e in SUBSES)
    )
    _touch(c / "fake_annots.zip")
    _touch(c / "fsmask.nii.gz")
    (c / "tractparams.csv").write_text(
        "label,slabel,fhemi,roi1,roi2,roi3,roi4,roiexc1,roiexc2\n"
        "L_VOF,LVOF,L,L_Ventral,NO,NO,NO,NO,NO\n"
        "R_VOF,RVOF,R,R_Ventral_AND_Brain,NO,NO,NO,NO,NO\n"
    )
    _js(
        c / "freesurferator.json",
        {"config": {"subject_id": "Sxxx", "reconall_options": "-all"}},
    )
    _js(c / "anatrois.json", {"config": {"subject_id": "Sxxx"}})
    _js(c / "rtp2-preproc.json", {"config": {"pe_dir": "PA", "denoise": True}})
    _js(c / "rtppreproc.json", {"config": {"acqd": "PA"}})
    _js(c / "rtp2-pipeline.json", {"config": {"numberOfNodes": 100}})
    for cont, ver in VERSIONS.items():
        _touch(root / "containers" / f"{cont}_{ver}.sif")
    return root


def container_specific(root: Path) -> dict:
    c = Path(root) / "configs"
    fs_anat = f"freesurferator_{VERSIONS['freesurferator']}"
    return {
        "anatrois": {
            "version": VERSIONS["anatrois"],
            "pre_fs": True,
            "prefs_dir_name": f"anatrois_{VERSIONS['anatrois']}",
            "prefs_analysis_name": "prefs",
            "prefs_zipname": r"^anatrois_S.*\.zip$",
            "annotfile": None,
            "mniroizip": None,
            "use_src_session": "01",
        },
        "freesurferator": {
            "version": VERSIONS["freesurferator"],
            "pre_fs": False,
            "prefs_dir_name": f"anatrois_{VERSIONS['anatrois']}",
            "prefs_analysis_name": "prefs",
            "prefs_zipname": r"^anatrois.*\.zip$",
            "control_points": False,
            "prefs_unzipname": "S.*$",
            "annotfile": str(c / "fake_annots.zip"),
            "mniroizip": None,
            "use_src_session": "01",
        },
        "rtppreproc": {
            "version": VERSIONS["rtppreproc"],
            "precontainer_anat": fs_anat,
            "anat_analysis_name": "fakeanat",
            "separated_shell_files": False,
            "rpe": False,
        },
        "rtp2-preproc": {
            "version": VERSIONS["rtp2-preproc"],
            "precontainer_anat": fs_anat,
            "anat_analysis_name": "fakeanat",
            "separated_shell_files": False,
            "rpe": True,
            "use_qmap": False,
            "qmap_dir_name": "x",
            "qmap_analysis_name": "x",
            "qmap_fname": r"^qmap.*\.zip$",
        },
        "rtp2-pipeline": {
            "version": VERSIONS["rtp2-pipeline"],
            "precontainer_anat": fs_anat,
            "anat_analysis_name": "fakeanat",
            "precontainer_preproc": f"rtp2-preproc_{VERSIONS['rtp2-preproc']}",
            "preproc_analysis_name": "fakepre",
            "tractparams": str(c / "tractparams.csv"),
            "fsmask": None,
            "use_qmap": False,
            "qmap_dir_name": "x",
            "qmap_analysis_name": "x",
            "qmap_fname": r"^qmap.*\.zip$",
        },
    }


def make_config(
    root: Path,
    container: str,
    name: str,
    layout: str = "legacy",
    host: str = "local",
    **overrides,
) -> Path:
    """Write ``configs/lc_<name>.yaml`` for *container*; return its path."""
    root = Path(root)
    cs = container_specific(root)
    cs[container].update(overrides)
    cfg = {
        "general": {
            "basedir": str(root / "basedir"),
            "bidsdir_name": "BIDS",
            "deriv_layout": layout,
            "containerdir": str(root / "containers"),
            "container": container,
            "analysis_name": name,
            "host": host,
            "use_dask": False,
            "force": False,
        },
        "container_specific": cs,
        "host_options": {
            "local": {
                "use_module": False,
                "apptainer": "apptainer/latest",
                "mount_options": [str(root)],
                "manager": "local",
                "launch_mode": "serial",
                "njobs": 2,
                "memory_limit": "4GiB",
                "threads_per_worker": 1,
            },
            "BCBL": {
                "use_module": True,
                "apptainer": "apptainer/latest",
                "mount_options": ["/bcbl", "/tmp"],
                "manager": "sge",
                "job_name": "lc",
                "cores": 6,
                "memory": "32G",
                "queue": "long.q",
                "walltime": "24:00:00",
            },
            "DIPC": {
                "use_module": True,
                "apptainer": "Apptainer/1.2.4",
                "mount_options": ["/scratch"],
                "manager": "slurm",
                "job_name": "lc",
                "cores": 6,
                "memory": "32G",
                "partition": "general",
                "qos": "regular",
                "walltime": "24:00:00",
            },
        },
    }
    p = root / "configs" / f"lc_{name}.yaml"
    p.write_text(yaml.safe_dump(cfg, sort_keys=False))
    return p
