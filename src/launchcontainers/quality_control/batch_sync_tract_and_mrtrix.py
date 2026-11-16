import os
import subprocess
import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

"""
This script is because I don't really know how the RTP-pipeline

output the result to which place

so I rsync things in place for checking

"""


def rsync_single_subject(analysis_dir, sub, ses):
    """
    Synchronize tract files for one subject/session using ``rsync``.

    Parameters
    ----------
    analysis_dir : str or path-like
        Analysis directory containing session outputs.
    sub : str
        Subject identifier without the ``sub-`` prefix.
    ses : str
        Session identifier without the ``ses-`` prefix.

    Returns
    -------
    str
        Human-readable status message for the transfer.
    """

    # get the output dir for each session
    subses_outputdir = Path(analysis_dir) / f"sub-{sub}" / f"ses-{ses}" / "output"
    # The directory which stores all the tracts
    file_path_tracts = subses_outputdir / "tracts"
    # the RTP working dir and all the tracts file will be under RTP/mrtrix
    file_path_rtp = subses_outputdir / "RTP" / "mrtrix"
    # Another place that stores the RTP output, it should be a zip file
    # tract_dir = subses_outputdir / "RTP_PIPELINE_ALL_OUTPUT" / "mrtrix"
    # Skip if source doesn't exist
    if not file_path_tracts.exists():
        return f"sub-{sub}_ses-{ses}: No tracts directory"

    # Create destination directory
    file_path_rtp.mkdir(parents=True, exist_ok=True)

    # Run rsync
    cmd = [
        "rsync",
        "-avzP",
        "-h",
        str(file_path_rtp) + "/",
        str(file_path_tracts) + "/",
    ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600)
        if result.returncode == 0:
            return f"sub-{sub}_ses-{ses}: SUCCESS"
        else:
            return f"sub-{sub}_ses-{ses}: FAILED - {result.stderr}"
    except Exception as e:
        return f"sub-{sub}_ses-{ses}: ERROR - {str(e)}"


def find_subseslist(analysis_dir):
    """
    Locate ``subseslist.txt`` somewhere under an analysis directory.
    """
    for dirpath, dirnames, filenames in os.walk(analysis_dir):
        for fname in filenames:
            if fname.lower() == "subseslist.txt":
                return os.path.join(dirpath, fname)
    raise FileNotFoundError(f"No subseslist.txt found under {analysis_dir}")


def batch_rsync_tracts(analysis_dir):
    """
    Run tract synchronization across all runnable DWI sessions in parallel.

    Parameters
    ----------
    analysis_dir : str or path-like
        Analysis directory containing ``subseslist.txt`` and output folders.

    Returns
    -------
    list[str]
        Status messages returned by :func:`rsync_single_subject`.
    """

    # Load subseslist
    path_to_subses = find_subseslist(analysis_dir)
    df_subSes = pd.read_csv(path_to_subses, sep=",", dtype=str)

    # Prepare tasks
    tasks = []
    for row in df_subSes.itertuples(index=False, name="Pandas"):
        if row.RUN == "True" and row.dwi == "True":
            tasks.append((analysis_dir, row.sub, row.ses))

    print(f"Processing {len(tasks)} subjects with 35 workers...")

    # Run parallel rsync
    results = []
    with ThreadPoolExecutor(max_workers=18) as executor:
        futures = {executor.submit(rsync_single_subject, *task): task for task in tasks}

        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            print(result)  # Print progress

    print(f"\nCompleted! Processed {len(results)} subjects")
    return results


if __name__ == "__main__":
    analysis_dir = (
        "/bcbl/home/public/DB/devtrajtract/DATA/MINI/nifti/derivatives/"
        "rtp2-pipeline_0.2.1_3.0.4rc2/analysis-paper_dv-main"
    )
    results = batch_rsync_tracts(analysis_dir)
