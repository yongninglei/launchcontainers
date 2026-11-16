from __future__ import annotations

import os
import shutil
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import typer


def process_tracts(sub, ses, analysis_dir):
    """
    Unpack and reorganize tract outputs for one subject/session.

    Parameters
    ----------
    sub : str
        Subject identifier without the ``sub-`` prefix.
    ses : str
        Session identifier without the ``ses-`` prefix.
    analysis_dir : str
        Analysis directory containing per-session output folders.
    """
    base = Path(analysis_dir) / f"sub-{sub}" / f"ses-{ses}" / "output"
    zip_path = base / "tracts.zip"
    csv_path = base / "tracts.csv"
    extract_dir = base / "tracts"

    print(f"\n For sub-{sub} ses-{ses}")
    # 1) Unzip
    if not zip_path.exists():
        print(f"No zip at {zip_path}")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(base)

    # 2) Rename zip, folder, csv
    old_zip = base / "old_tracts.zip"
    zip_path.rename(old_zip)

    old_dir = base / "old_tracts"
    extract_dir.rename(old_dir)

    old_csv_path = base / "tracts.csv"
    csv_path.rename(old_csv_path)

    # 3) Copy selected files
    dest = base / "RTP" / "mrtrix"
    dest.mkdir(parents=True, exist_ok=True)

    for pattern in ("L_Ope*", "L_Tri*", "L_Orb*"):
        for src in old_dir.glob(pattern):
            if src.is_file():
                shutil.copy2(src, dest / src.name)


def find_subseslist(analysis_dir):
    """
    Locate ``subseslist.txt`` somewhere under an analysis directory.

    Parameters
    ----------
    analysis_dir : str or path-like
        Root directory to search.

    Returns
    -------
    str
        Full path to the discovered ``subseslist.txt`` file.
    """
    for dirpath, dirnames, filenames in os.walk(analysis_dir):
        for fname in filenames:
            if fname.lower() == "subseslist.txt":
                return os.path.join(dirpath, fname)
    raise FileNotFoundError(f"No subseslist.txt found under {analysis_dir}")


def main(analysis_dir):
    """
    Reprocess tract outputs in parallel for every runnable DWI session.

    Parameters
    ----------
    analysis_dir : str
        Analysis directory containing the prepared ``subseslist.txt`` file.
    """
    path_to_subses = find_subseslist(analysis_dir)
    df_subSes = pd.read_csv(path_to_subses, sep=",", dtype=str)

    # # serial processing
    # for row in df_subSes.itertuples(index=True, name='Pandas'):
    #     sub = row.sub
    #     ses = row.ses
    #     RUN = row.RUN
    #     dwi = row.dwi
    #     if RUN == 'True' and dwi == 'True':
    #         process_tracts(sub, ses, analysis_dir)

    # parallel processing
    # Keep only the rows you actually want to process
    df_filtered = df_subSes[(df_subSes.RUN == "True") & (df_subSes.dwi == "True")]

    # 2) Build parallel argument lists
    subs = df_filtered["sub"].tolist()
    ses = df_filtered["ses"].tolist()
    # repeat the same analysis_dir for each task
    ana_dirs = [analysis_dir] * len(subs)

    # 3) Dispatch in parallel
    with ThreadPoolExecutor(max_workers=30) as executor:
        # executor.map will call process_tracts(sub, ses, analysis_dir)
        executor.map(process_tracts, subs, ses, ana_dirs)


if __name__ == "__main__":
    typer.run(main)
