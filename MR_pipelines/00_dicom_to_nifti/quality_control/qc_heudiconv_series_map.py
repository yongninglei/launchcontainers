#!/usr/bin/env python3
"""QC of one DICOM -> raw_nifti (heudiconv) conversion, per session.

For each sub/ses it joins three sources:

1. ``<raw_nifti>/.heudiconv/<sub>/ses-<ses>/info/dicominfo_ses-<ses>.tsv``
   (every DICOM series heudiconv saw),
2. ``<sub>_ses-<ses>.auto.txt`` (the heuristic's series -> output mapping;
   ``{item}`` = position within the key, i.e. the run number),
3. the converted files and ``scans.tsv`` acq_time in ``<raw_nifti>``,

and optionally the DICOM input tree (``<dicom_dir>/sub-XX/ses-YY``) and the lab
note (``VOTCLOC_subses_list.xlsx``, one sheet per subject).

Writes, in ``--out-dir``:

- ``sub-XX_ses-YY_series_map.csv``: one row per (series, output)
- ``heudiconv_qc_summary.csv``: one row per session with the checks below

Checks (columns of the summary; ``ok`` is True only if all hard checks pass):

- ``n_dicom_dates`` / ``dicom_dates``: scan dates in the DICOM file names (1 expected)
- ``broken_links`` / ``dup_dicom_names``: input tree sanity (0 expected)
- ``edit_differs_from_auto``: someone hand-edited the mapping (False expected)
- ``missing_outputs``: mapped series whose NIfTI does not exist (0 expected)
- ``unexpected_unconverted``: series not converted that are not scouts /
  phase-SBRefs / DWI SBRefs (empty expected; an interrupted run shows up here)
- ``label_mismatch``: scanner label (``fLoc_run03``) vs output run (``run-03``)
- ``run_order_bad``: per kind, run numbers not 1..N in acquisition-time order
- ``orphan_sbref``: task/run with an sbref but no magnitude (or vice versa) —
  the heuristic drops interrupted bold series but keeps their 1-file SBRef
- ``labnote_only`` / ``nifti_only``: runs in the lab note but not converted, and back
- ``max_abs_time_diff_min``: lab-note time_start vs acq_time (hand-written: a few min)

Usage::

    python qc_heudiconv_series_map.py -r <raw_nifti> -s 10,05 -o <out_dir> \
        [-d <dicom_dir>] [-l VOTCLOC_subses_list.xlsx]
    python qc_heudiconv_series_map.py -r <raw_nifti> -f subseslist.txt -o <out_dir>
"""

from __future__ import annotations

import ast
import os
import re
from pathlib import Path
from typing import Optional

import pandas as pd
import typer
from rich.console import Console

app = typer.Typer(add_completion=False)
console = Console()

NOT_CONVERTED_OK = r"Scout|SBRef_Pha|dMRI.*SBRef"
FUNC_RE = re.compile(r"task-(\w+?)_run-(\d+)_(magnitude|phase|sbref|bold)$")
SCANNER_RE = re.compile(r"(fLoc|ret\w\w)_run(\d+)")


def read_subses(
    subses: Optional[str], subseslist: Optional[Path]
) -> list[tuple[str, str]]:
    if subses:
        sub, ses = subses.split(",")[:2]
        return [(sub.zfill(2), ses.zfill(2))]
    rows = []
    for line in Path(subseslist).read_text().splitlines()[1:]:
        if not line.strip():
            continue
        parts = re.split(r"[,\t]", line.strip())
        if len(parts) > 2 and parts[2].strip().lower() == "false":  # RUN column
            continue
        rows.append((parts[0].zfill(2), parts[1].zfill(2)))
    return rows


def series_map(raw: Path, sub: str, ses: str) -> pd.DataFrame:
    info = raw / ".heudiconv" / sub / f"ses-{ses}" / "info"
    di = pd.read_csv(info / f"dicominfo_ses-{ses}.tsv", sep="\t", dtype=str)
    auto = ast.literal_eval((info / f"{sub}_ses-{ses}.auto.txt").read_text())
    assign: dict[str, list[str]] = {}
    for (template, _types, _ann), series in auto.items():
        for i, sid in enumerate(series, start=1):
            assign.setdefault(sid, []).append(
                template.format(subject=sub, session=f"ses-{ses}", item=i)
            )
    scans = pd.read_csv(
        raw / f"sub-{sub}" / f"ses-{ses}" / f"sub-{sub}_ses-{ses}_scans.tsv", sep="\t"
    )
    acq = {
        f.replace(".nii.gz", ""): t
        for f, t in zip(scans["filename"], scans["acq_time"])
    }
    rows = []
    for _, r in di.iterrows():
        for rel in assign.get(r["series_id"], [""]):
            sesrel = rel.split("/", 2)[2] if rel else ""
            rows.append(
                dict(
                    series_number=int(r["series_id"].split("-")[0]),
                    series_id=r["series_id"],
                    series_desc=r["series_description"],
                    files=int(r["series_files"]),
                    dims=f"{r['dim1']}x{r['dim2']}x{r['dim3']}x{r['dim4']}",
                    output=os.path.basename(sesrel),
                    exists=(raw / f"{rel}.nii.gz").exists() if rel else None,
                    acq_time=acq.get(sesrel, ""),
                )
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["series_number", "output"])
        .reset_index(drop=True)
    )


def dicom_checks(dicom_dir: Path, sub: str, ses: str) -> dict:
    d = dicom_dir / f"sub-{sub}" / f"ses-{ses}"
    names, broken, dates = [], 0, {}
    for root, _dirs, files in os.walk(d, followlinks=False):
        for f in files:
            p = Path(root) / f
            names.append(f)
            if p.is_symlink() and not p.exists():
                broken += 1
            m = re.search(r"167004\.(20\d{6})", f)
            if m:
                dates[m.group(1)] = dates.get(m.group(1), 0) + 1
    return dict(
        n_dicom_files=len(names),
        broken_links=broken,
        dup_dicom_names=len(names) - len(set(names)),
        n_dicom_dates=len(dates),
        dicom_dates=";".join(f"{k}:{v}" for k, v in sorted(dates.items())),
    )


def labnote_runs(xlsx: Path, sub: str, ses: str) -> pd.DataFrame:
    ln = pd.read_excel(xlsx, sheet_name=f"sub-{sub}")
    ln["ses"] = ln["ses"].ffill()
    ln = ln[pd.to_numeric(ln["ses"], errors="coerce") == int(ses)]
    return ln[
        ln["protocol_name"].astype(str).str.match(r"(fLoc|ret\w\w)_run-\d+$|fmap_\d+$")
    ]


def session_checks(df: pd.DataFrame, raw: Path, sub: str, ses: str) -> dict:
    conv = df[df["output"] != ""]
    info = raw / ".heudiconv" / sub / f"ses-{ses}" / "info"
    edit = info / f"{sub}_ses-{ses}.edit.txt"
    out = dict(
        n_series=df["series_id"].nunique(),
        n_converted=len(conv),
        edit_differs_from_auto=edit.exists()
        and edit.read_text() != (info / f"{sub}_ses-{ses}.auto.txt").read_text(),
        missing_outputs=int((conv["exists"] != True).sum()),  # noqa: E712
        unexpected_unconverted=";".join(
            df[
                (df["output"] == "")
                & ~df["series_desc"].str.contains(NOT_CONVERTED_OK, regex=True)
            ]["series_id"]
        ),
    )
    mism = []
    for _, r in conv.iterrows():
        m, o = SCANNER_RE.match(r["series_desc"]), FUNC_RE.search(r["output"])
        if m and o and (m.group(1) != o.group(1) or int(m.group(2)) != int(o.group(2))):
            mism.append(f"{r['series_desc']}->{r['output']}")
    out["label_mismatch"] = ";".join(mism)
    kinds = conv[
        conv["output"].str.contains(r"_(magnitude|sbref|epi)$")
        & ~conv["output"].str.contains("votcloc")
    ].copy()
    kinds["kind"] = (
        kinds["output"]
        .str.replace(r"_run-\d+", "", regex=True)
        .str.replace(rf"sub-{sub}_ses-{ses}_", "", regex=False)
    )
    kinds["run"] = kinds["output"].str.extract(r"run-(\d+)")[0].astype(int)
    bad = [
        k
        for k, g in kinds.groupby("kind")
        if not g["acq_time"].is_monotonic_increasing
        or list(g["run"]) != list(range(1, len(g) + 1))
    ]
    out["run_order_bad"] = ";".join(bad)
    func = conv["output"].str.extract(FUNC_RE).dropna()
    func.columns = ["task", "run", "suffix"]
    have = func.groupby(["task", "run"])["suffix"].apply(set)
    out["orphan_sbref"] = ";".join(
        f"{t}_run-{r}:{'+'.join(sorted(s))}"
        for (t, r), s in have.items()
        if {"sbref", "magnitude"} - s
    )
    out["counts"] = ";".join(
        f"{k}={n}" for k, n in kinds.groupby("kind").size().items()
    )
    return out


def labnote_checks(df: pd.DataFrame, xlsx: Path, sub: str, ses: str) -> dict:
    conv = df[df["output"] != ""]
    nii, fm = {}, 0
    for _, r in conv.iterrows():
        o = re.search(r"task-(\w+?)_run-(\d+)_magnitude$", r["output"])
        if o:
            nii[f"{o.group(1)}_run-{o.group(2)}"] = r["acq_time"][11:19]
        elif re.search(r"acq-fMRI_dir-AP_run-\d+_epi$", r["output"]):
            fm += 1
            nii[f"fmap_{fm:02d}"] = r["acq_time"][11:19]
    ln = labnote_runs(xlsx, sub, ses)
    diffs = []
    for _, r in ln.iterrows():
        t, a = r["time_start"], nii.get(r["protocol_name"])
        if a is None or pd.isna(t) or isinstance(t, str) or not hasattr(t, "hour"):
            continue
        diffs.append(
            abs(
                (
                    pd.Timestamp(f"2000-01-01 {a}") - pd.Timestamp(f"2000-01-01 {t}")
                ).total_seconds()
            )
            / 60
        )
    lk = set(ln["protocol_name"])
    return dict(
        labnote_only=";".join(sorted(lk - set(nii))),
        nifti_only=";".join(sorted(set(nii) - lk)),
        n_labnote_times=len(diffs),
        max_abs_time_diff_min=round(max(diffs), 1) if diffs else None,
    )


@app.command()
def main(
    raw_nifti: Path = typer.Option(
        ..., "-r", "--raw-nifti", help="heudiconv output dir (has .heudiconv/)"
    ),
    out_dir: Path = typer.Option(
        ..., "-o", "--out-dir", help="where the CSVs are written"
    ),
    subses: Optional[str] = typer.Option(
        None, "-s", "--subses", help="one session, e.g. 10,05"
    ),
    subseslist: Optional[Path] = typer.Option(
        None, "-f", "--subseslist", help="sub,ses list with header"
    ),
    dicom_dir: Optional[Path] = typer.Option(
        None, "-d", "--dicom-dir", help="heudiconv input root (dicom/)"
    ),
    labnote: Optional[Path] = typer.Option(
        None, "-l", "--labnote", help="VOTCLOC_subses_list.xlsx"
    ),
    max_time_diff: float = typer.Option(
        5.0, help="lab-note vs acq_time tolerance (min)"
    ),
):
    if not subses and not subseslist:
        console.print("[red]ERROR[/red] give -s or -f")
        raise typer.Exit(2)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sub, ses in read_subses(subses, subseslist):
        row = dict(sub=sub, ses=ses)
        try:
            df = series_map(raw_nifti, sub, ses)
        except FileNotFoundError as e:
            console.print(f"[red]ERROR[/red] sub-{sub} ses-{ses}: {e}")
            rows.append({**row, "ok": False, "error": str(e)})
            continue
        df.to_csv(out_dir / f"sub-{sub}_ses-{ses}_series_map.csv", index=False)
        row.update(session_checks(df, raw_nifti, sub, ses))
        if dicom_dir:
            row.update(dicom_checks(dicom_dir, sub, ses))
        if labnote:
            row.update(labnote_checks(df, labnote, sub, ses))
        hard = [
            row["missing_outputs"] == 0,
            not row["unexpected_unconverted"],
            not row["label_mismatch"],
            not row["run_order_bad"],
            not row["orphan_sbref"],
            not row["edit_differs_from_auto"],
        ]
        if dicom_dir:
            hard += [
                row["n_dicom_dates"] == 1,
                row["broken_links"] == 0,
                row["dup_dicom_names"] == 0,
            ]
        if labnote:
            hard += [
                not row["labnote_only"],
                not row["nifti_only"],
                row["max_abs_time_diff_min"] is None
                or row["max_abs_time_diff_min"] <= max_time_diff,
            ]
        row["ok"] = all(hard)
        tag = "[green]OK[/green]" if row["ok"] else "[red]CHECK[/red]"
        console.print(
            f"{tag} sub-{sub} ses-{ses}: {row['n_converted']}/{row['n_series']} series converted; {row['counts']}"
        )
        rows.append(row)
    summary = out_dir / "heudiconv_qc_summary.csv"
    pd.DataFrame(rows).to_csv(summary, index=False)
    console.print(f"[green]saved[/green] {summary}")


if __name__ == "__main__":
    app()
