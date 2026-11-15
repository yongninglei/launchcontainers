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

import json
import os
import os.path as op
import re
import shutil
import sys
import errno
import uuid
from datetime import datetime
from pathlib import Path

import yaml
from yaml.loader import SafeLoader

from launchcontainers.log_setup import console


def parse_hms(ts: str) -> str:
    """
    Normalise any time string to zero-padded HH:MM:SS.
    Handles ISO datetime, sub-seconds, single-digit hours.
    """
    s = str(ts).strip()
    if "T" in s:
        s = s.split("T")[1]
    s = s.split(".")[0]
    for fmt in ("%H:%M:%S", "%H:%M", "%H%M%S", "%H%M"):
        try:
            return datetime.strptime(s, fmt).strftime("%H:%M:%S")
        except ValueError:
            continue
    return s


def hms_to_sec(ts: str) -> float:
    """HH:MM:SS[.f] → total seconds as float; returns NaN on failure."""
    try:
        h, m, s = str(ts).strip().split(":")
        return int(h) * 3600 + int(m) * 60 + float(s)
    except Exception:
        return float("nan")


def times_match(t1: str, t2: str, max_diff_sec: int = 30) -> bool:
    """Return True if |t1 - t2| <= max_diff_sec."""
    if t1 is None or t2 is None:
        return False
    try:
        dt1 = datetime.strptime(parse_hms(t1), "%H:%M:%S")
        dt2 = datetime.strptime(parse_hms(t2), "%H:%M:%S")
    except ValueError:
        return False
    return abs((dt1 - dt2).total_seconds()) <= max_diff_sec


def read_json_acqtime(json_path: str | Path) -> str:
    """
    Read the AcquisitionTime field from a BIDS JSON sidecar.

    Parameters
    ----------
    json_path : str or Path
        Path to the JSON sidecar file.

    Returns
    -------
    str
        The raw AcquisitionTime string (e.g. ``"10:05:32.500000"``),
        or an empty string if the field is absent or the file cannot be read.
    """
    try:
        with open(json_path) as fh:
            return json.load(fh).get("AcquisitionTime", "")
    except Exception:
        return ""


def parse_subses_list(path: str | Path) -> list[tuple[str, str]]:
    """
    Read a subseslist CSV or TSV file and return (sub, ses) pairs.

    The file must have a header row with at least ``sub`` and ``ses`` columns.
    If a ``RUN`` column is present, only rows where ``RUN`` is True are
    returned (case-insensitive); otherwise all rows are included.
    Blank lines and ``#`` comments are ignored, ``sub-``/``ses-`` prefixes are
    stripped, and numeric labels are zero-padded to two digits.
    A malformed row, a RUN value other than True/False, or a duplicate
    sub/ses is a hard error (``ValueError``): a typo must stop the run.
    TSV files (``.tsv`` extension) are auto-detected; everything else is
    treated as comma-separated.

    Parameters
    ----------
    path : str or Path
        Path to the subseslist file.

    Returns
    -------
    list[tuple[str, str]]
        Ordered list of ``(sub, ses)`` string pairs.
    """
    path = Path(path)
    delimiter = "\t" if path.suffix.lower() == ".tsv" else ","

    def _bare(value: str, prefix: str) -> str:
        v = value.strip()
        if v.startswith(f"{prefix}-"):
            v = v[len(prefix) + 1 :]
        return v.zfill(2) if v.isdigit() else v

    rows = [
        (lineno, line)
        for lineno, line in enumerate(path.read_text().splitlines(), start=1)
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if not rows:
        raise ValueError(f"{path}: subseslist is empty")
    header = [c.strip() for c in rows[0][1].split(delimiter)]
    if "sub" not in header or "ses" not in header:
        raise ValueError(
            f"{path}:{rows[0][0]}: header must contain 'sub' and 'ses', got {header}"
        )
    i_sub, i_ses = header.index("sub"), header.index("ses")
    i_run = header.index("RUN") if "RUN" in header else None

    pairs: list[tuple[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for lineno, line in rows[1:]:
        cols = [c.strip() for c in line.split(delimiter)]
        needed = max(i for i in (i_sub, i_ses, i_run) if i is not None)
        if len(cols) <= needed or not cols[i_sub] or not cols[i_ses]:
            raise ValueError(f"{path}:{lineno}: malformed row {line!r}")
        if i_run is not None:
            run = cols[i_run].lower()
            if run not in ("true", "false"):
                raise ValueError(
                    f"{path}:{lineno}: RUN must be True or False, got {cols[i_run]!r}"
                )
            if run == "false":
                continue
        pair = (_bare(cols[i_sub], "sub"), _bare(cols[i_ses], "ses"))
        if pair in seen:
            raise ValueError(
                f"{path}:{lineno}: duplicate row for sub-{pair[0]} ses-{pair[1]}"
            )
        seen.add(pair)
        pairs.append(pair)
    return pairs


_RUN_RE = re.compile(r"_run-(\d+)")


def substitute_run(name: str, new_run_int: int, zero_pad: int = 2) -> str:
    """Return *name* with its ``_run-NN`` entity replaced.

    Parameters
    ----------
    name : str
        BIDS filename or stem (e.g. ``sub-10_ses-02_task-retFF_run-01_bold``).
    new_run_int : int
        The new run number to substitute in.
    zero_pad : int
        Digit width for the replacement label (``2`` → ``run-01``; ``1`` → ``run-1``).

    Returns
    -------
    str
        The name with the first ``_run-NN`` occurrence replaced.
    """
    return _RUN_RE.sub(f"_run-{new_run_int:0{zero_pad}d}", name, count=1)


def atomic_rename_pairs(
    pairs: list[tuple[Path, Path]],
    dry_run: bool = True,
) -> None:
    """Rename a list of (src, dst) pairs atomically using a two-phase UUID strategy.

    Phase 1: every src is renamed to a unique temp name in the same directory.
    Phase 2: every temp is renamed to its final dst.

    This prevents any file being overwritten — if any dst already exists before
    we start, a RuntimeError is raised and nothing is touched.

    Parameters
    ----------
    pairs : list[tuple[Path, Path]]
        List of (source, destination) path pairs. Pairs where src == dst are
        skipped. Non-existent sources are skipped silently.
    dry_run : bool
        When True, do nothing on disk (pairs are validated but not renamed).
    """
    # Filter out no-ops and missing sources
    ops = [(s, d) for s, d in pairs if s != d and s.exists()]
    if not ops or dry_run:
        return

    # Guard: refuse to overwrite any pre-existing destination that is NOT
    # itself a source in this same batch (those will be safely parked in
    # phase 1 before phase 2 needs the slot — handles swap cases).
    src_set = {s for s, _ in ops}
    conflicts = [d for _, d in ops if d.exists() and d not in src_set]
    if conflicts:
        names = ", ".join(c.name for c in conflicts)
        raise RuntimeError(
            f"Rename aborted — destination(s) already exist: {names}\n"
            "Remove or rename them manually before re-running."
        )

    tag = uuid.uuid4().hex
    # Phase 1: park every source under a unique tmp name
    tmps: list[tuple[Path, Path]] = []
    for src, dst in ops:
        tmp = src.parent / f"._rnm_{tag}_{src.name}"
        src.rename(tmp)
        tmps.append((tmp, dst))

    # Phase 2: rename each tmp to its final destination
    for tmp, dst in tmps:
        tmp.rename(dst)


def reorder_bids_runs(
    files: list[Path],
    run_map: dict[int, int],
    zero_pad: int = 2,
    dry_run: bool = True,
) -> list[tuple[Path, Path]]:
    """Rename BIDS files in-place by substituting run numbers per *run_map*.

    Only files whose ``_run-NN`` label is a key in *run_map* are touched; all
    others are skipped silently.  A two-phase rename (source → UUID temp →
    final) prevents clobbering when source and destination run ranges overlap
    (e.g. renumbering run-02 → run-01 while run-01 still exists).

    Parameters
    ----------
    files : list[Path]
        Candidate files.  May contain any mix of regular files, symlinks, or
        paths that don't exist; only those whose name contains ``_run-NN``
        and whose run integer appears in *run_map* are processed.
    run_map : dict[int, int]
        ``{old_run_int: new_run_int}`` mapping.
    zero_pad : int
        Digit width for the new run label (default ``2`` → ``run-01``).
    dry_run : bool
        When ``True``, log the planned renames and return the list without
        modifying anything on disk.

    Returns
    -------
    list[tuple[Path, Path]]
        ``(old_path, new_path)`` for every rename that was planned (and
        executed when *dry_run* is ``False``).
    """
    moves: list[tuple[Path, Path]] = []
    for f in sorted(files):
        m = _RUN_RE.search(f.name)
        if not m:
            continue
        old_int = int(m.group(1))
        if old_int not in run_map:
            continue
        new_name = substitute_run(f.name, run_map[old_int], zero_pad)
        dst = f.parent / new_name
        if dst == f:
            continue
        console.print(f"  [cyan]RENUMBER[/cyan]  {f.name}  \u2192  {new_name}")
        moves.append((f, dst))

    if dry_run or not moves:
        return moves

    # Phase 1: rename all sources to temp names
    tmps: list[tuple[Path, Path, Path]] = []
    for src, dst in moves:
        tmp = src.parent / f"._rnm_{uuid.uuid4().hex[:8]}_{src.name}"
        src.rename(tmp)
        tmps.append((src, tmp, dst))

    # Phase 2: rename temp names to final destinations
    for _, tmp, dst in tmps:
        tmp.rename(dst)

    return moves


def die(*args):
    """
    Log an error message and terminate the current process.

    Parameters
    ----------
    *args
        Positional message parts forwarded to :meth:`console.print`.
    """
    console.print(*args, style="red")
    sys.exit(1)


def read_yaml(path_to_config_file):
    """
    Read a YAML configuration file.

    Parameters
    ----------
    path_to_config_file : str or path-like
        Path to the YAML file to load.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    with open(path_to_config_file) as v:
        config = yaml.load(v, Loader=SafeLoader)

    return config


def copy_file(src_file, dst_file, force):
    """
    Copy a file to a destination path with optional overwrite behavior.

    Parameters
    ----------
    src_file : str or path-like
        Source file to copy.
    dst_file : str or path-like
        Destination file path.
    force : bool
        If ``True``, overwrite an existing destination file.

    Returns
    -------
    str
        Destination path.

    Raises
    ------
    FileExistsError
        If the source file does not exist.
    """
    console.print("\n" + "=" * 30 + "COPY_FILE_WORKING" + "=" * 30, style="cyan")
    if not os.path.isfile(src_file):
        console.print("\n \u274c Source file does not exist.", style="red")
        raise FileExistsError("the source file is not here")

    try:
        if ((not os.path.isfile(dst_file)) or (force)) or (
            os.path.isfile(dst_file) and force
        ):
            shutil.copy(src_file, dst_file)
            console.print(
                "\n"
                + f"\u2705 {src_file} has been successfully copied to \
                     {os.path.dirname(dst_file)} directory \n"
                + "\U0001f37a REMEMBER TO CHECK/EDIT TO HAVE THE CORRECT PARAMETERS IN THE FILE\n",
                style="cyan",
            )
        elif os.path.isfile(dst_file) and not force:
            console.print(
                "\n" + f"\u274c copy are not operating, the {src_file} already exist",
                style="yellow",
            )

    # If source and destination are the same
    except shutil.SameFileError:
        console.print(
            "\n \u274c Source and destination represent the same file.\n", style="red"
        )

    # If there is any permission issue, skip it
    except PermissionError:
        console.print(
            f"\n \u274c Permission denied: {dst_file}. Skipping...\n", style="yellow"
        )

    # For other errors
    except Exception as e:
        console.print(
            f"\n \u274c Error occurred while copying file: {e}\n", style="red"
        )

    return dst_file


def copy_configs(output_path, force=True):
    """
    Copy packaged example configuration files into a target directory.

    Parameters
    ----------
    output_path : str or path-like
        Directory where the example configuration files should be copied.
    force : bool, default=True
        If ``True``, overwrite existing files in ``output_path``.
    """
    # first, know where the tar file is stored
    import pkg_resources

    config_path = pkg_resources.resource_filename("launchcontainers", "example_configs")

    # second, copy all the files from the source folder to the output_path
    all_cofig_files = os.listdir(config_path)
    for src_fname in all_cofig_files:
        src_file_fullpath = op.join(config_path, src_fname)
        targ_file_fullpath = op.join(output_path, src_fname)
        copy_file(src_file_fullpath, targ_file_fullpath, force)

    return


def force_symlink(file1, file2, force):
    """
    Create or refresh a symbolic link and validate the result.

    Parameters
    ----------
    file1 : str or path-like
        Link target, usually an output from a previous processing step.
    file2 : str or path-like
        Link path to create for the current container input.
    force : bool
        If ``True``, replace an existing link at ``file2``.

    Raises
    ------
    OSError
        If the source is missing or the link cannot be created.
    """
    console.print("\n" + "=" * 30 + "SYMLINK_WORKING" + "=" * 30, style="cyan")
    # If force is set to False (we do not want to overwrite)
    if not force:
        try:
            # Try the command, if the files are correct and the symlink does not exist, create one

            os.symlink(file1, file2)
            console.print(
                "\n"
                + f"\u2705 Created symlink for source file: {file1} and destination file: {file2}\n",
                style="cyan",
            )
        # If raise [erron 2]: file does not exist, print the error and pass
        except OSError as n:
            if n.errno == 2:
                console.print(
                    "\n" + "Input files are missing, please check \n", style="red"
                )
                raise
            # If raise [errno 17] the symlink exist,
            # we don't force and print that we keep the original one
            elif n.errno == errno.EEXIST:
                console.print(
                    "\n" + " Symlink exist, not overwriting, remain old \n",
                    style="yellow",
                )
            else:
                console.print("\n" + "Unknown error, break the program", style="red")
                raise n

    # If we set force to True (we want to overwrite)
    if force:
        console.print(
            "\n"
            + "---force is set to True, we will overwrite the existing symlink if it exist\n",
            style="cyan",
        )
        try:
            # Try the command, if the file are correct and symlink not exist, it will create one
            os.symlink(file1, file2)
            console.print(
                "\n"
                + f"\u2705 Created symlink for source file: {file1} and destination file: {file2}\n",
                style="cyan",
            )
        # If the symlink exists, OSError will be raised
        except OSError as e:
            if e.errno == errno.EEXIST:
                os.remove(file2)
                console.print("\n" + "Overwriting the existing symlink", style="yellow")
                os.symlink(file1, file2)
                console.print(
                    "\n"
                    + f"\u2705 Created symlink for source file: {file1} and destination file: {file2}\n",
                    style="cyan",
                )
            elif e.errno == 2:
                console.print(
                    "\n"
                    + "\u274c Input files are missing, please check that they exist\n",
                    style="red",
                )
                raise e
            else:
                console.print(
                    "\n" + "\u274c ERROR\n" + "We do not know what happened\n",
                    style="red",
                )
                raise e
    check_symlink(file2)
    return


def check_symlink(path: str) -> None:
    """
    Validate that a path is an existing symbolic link target.

    Parameters
    ----------
    path : str
        Link path to inspect.

    Raises
    ------
    FileNotFoundError
        If ``path`` is a symlink but its target does not exist.
    """
    if op.islink(path):
        if op.exists(path):
            console.print(
                f" √ Symlink {path!r} is valid and points to {op.realpath(path)!r}",
                style="cyan",
            )
        else:
            target = os.readlink(path)
            console.print(
                f"X Symlink {path!r} is broken (target {target!r} not found)",
                style="red",
            )
            raise FileNotFoundError(f"Broken symlink: {path!r} → {target!r}")

    else:
        console.print(f" {path!r} is not a symlink", style="cyan")


# Container names whose derivative dirs are created by ``_create_analysis_dir``.
# Sorted longest first so "rtp2-preproc" wins over a hypothetical "rtp2".
_DERIV_CONTAINERS = sorted(
    (
        "anatrois",
        "rtppreproc",
        "rtp-pipeline",
        "freesurferator",
        "rtp2-preproc",
        "rtp2-pipeline",
    ),
    key=len,
    reverse=True,
)


def resolve_deriv_dir(lc_config: dict, dir_name: str, analysis_name: str) -> str:
    """
    Return the derivatives analysis dir of an upstream container run.

    Mirrors ``do_prepare._create_analysis_dir``:

    - ``legacy``: ``derivatives/<container>_<version>/analysis-<analysis>``
    - otherwise:  ``derivatives/<container>-<version>_<analysis>``

    ``dir_name`` is the config value naming the upstream run (e.g.
    ``precontainer_anat: freesurferator_0.2.1_7.4.1``). It is accepted in either
    ``<container>_<version>`` or ``<container>-<version>`` form; for the new
    layout the first form is converted to the second. Names that do not start
    with a known container (e.g. a qmap derivative) are used unchanged.
    """
    basedir = lc_config["general"]["basedir"]
    bidsdir_name = lc_config["general"]["bidsdir_name"]
    deriv_layout = lc_config["general"].get("deriv_layout", "legacy")
    deriv_root = op.join(basedir, bidsdir_name, "derivatives")

    if deriv_layout == "legacy":
        return op.join(deriv_root, dir_name, f"analysis-{analysis_name}")

    for cont in _DERIV_CONTAINERS:
        if dir_name.startswith(cont + "_"):
            dir_name = f"{cont}-{dir_name[len(cont) + 1 :]}"
            break
    return op.join(deriv_root, f"{dir_name}_{analysis_name}")


def is_old_anatrois(name: str) -> bool:
    """True for ``anatrois_4.5.x`` / ``anatrois-4.5.x`` (minor version < 6)."""
    m = re.match(r"anatrois[-_]\d+\.(\d+)", name)
    return bool(m) and int(m.group(1)) < 6
