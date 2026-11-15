"""The prepare scenarios every implementation of the DWI containers must reproduce.

(name, container, deriv_layout, host, overrides)
"""

from __future__ import annotations

ZIP_BOTH = r"^anatrois_S0[12]\.zip$"

SCENARIOS = [
    ("anatrois_legacy", "anatrois", "legacy", "local", {}),
    ("anatrois_new", "anatrois", "new", "local", {}),
    ("freesurferator_legacy", "freesurferator", "legacy", "local", {}),
    (
        "freesurferator_new_cp",
        "freesurferator",
        "new",
        "local",
        {
            "pre_fs": True,
            "control_points": True,
            "prefs_zipname": ZIP_BOTH,
            "use_src_session": None,
        },
    ),
    (
        "rtppreproc_legacy",
        "rtppreproc",
        "legacy",
        "local",
        {"precontainer_anat": "freesurferator_0.2.1_7.4.1"},
    ),
    ("rtp2-preproc_legacy", "rtp2-preproc", "legacy", "local", {}),
    ("rtp2-preproc_new", "rtp2-preproc", "new", "local", {}),
    ("rtp2-preproc_norpe", "rtp2-preproc", "legacy", "local", {"rpe": False}),
    ("rtp2-pipeline_legacy", "rtp2-pipeline", "legacy", "local", {}),
    ("rtp2-pipeline_new", "rtp2-pipeline", "new", "local", {}),
    ("rtp2-pipeline_fsmask", "rtp2-pipeline", "legacy", "local", {"fsmask": "FSMASK"}),
    ("rtp2-pipeline_bcbl", "rtp2-pipeline", "legacy", "BCBL", {}),
    ("rtp2-pipeline_dipc", "rtp2-pipeline", "legacy", "DIPC", {}),
]
