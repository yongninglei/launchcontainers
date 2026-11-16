"""Tract processing and validation functions"""

import pandas as pd
from pathlib import Path
import os
from launchcontainers.log_setup import console


def load_tract_params(tractparams_file):
    """
    Load tract names from a tract parameter CSV file.

    Parameters
    ----------
    tractparams_file : str or path-like
        Path to the tract parameter CSV file.

    Returns
    -------
    pandas.Series
        Derived tract names that should be present in the output.
    """
    tractparams_file = Path(tractparams_file)

    if not tractparams_file.exists():
        raise FileNotFoundError(f"tractparams.tsv not found: {tractparams_file}")

    try:
        df = pd.read_csv(tractparams_file, sep=",")
        exclude = ["Ang", "Sup", "IPS0", "IPS1"]

        df = df[~df["nhlabel"].str.contains("|".join(exclude))]

        tract_names = (
            df["shemi"].astype(str)
            + "_"
            + df["nhlabel"].str.replace("-", "_")
            + "_clean"
        )
        console.print(f"Loaded {len(tract_names)} tract names", style="cyan")
        return tract_names

    except Exception as e:
        console.print(f"Error reading tractparams.tsv: {e}", style="red")
        raise


def get_expected_tract_files():
    """
    Return the file suffixes expected for each tract output.

    Returns
    -------
    list[str]
        Suffixes that define a complete tract result.
    """
    return [
        "_fa_bin.nii.gz",
        ".tck",
    ]


def check_tract_completeness(subses_outputdir):
    """
    Check whether each expected tract has all required output files.

    Parameters
    ----------
    subses_outputdir : str or path-like
        Output directory for one subject/session.

    Returns
    -------
    tuple[list[str], list[str], dict]
        Complete tracts, missing tracts, and a per-tract status dictionary.
    """
    subses_outputdir = Path(subses_outputdir)

    tractparams_file = (
        subses_outputdir
        / ".."
        / ".."
        / ".."
        / "tractparams_votc-ifg-ipl_280tract_hemi-both.csv"
    )
    tract_names = load_tract_params(tractparams_file)

    output_summary_csv = subses_outputdir / "RTP_PIPELINE_ALL_OUTPUT.csv"

    if not output_summary_csv.exists():
        return [], tract_names.copy(), {}

    if not subses_outputdir.exists():
        return [], tract_names.copy(), {}

    expected_suffixes = get_expected_tract_files()
    complete_tracts = []
    missing_tracts = []
    tract_status = {}

    for tract_name in tract_names:
        found_suffixes = []
        missing_suffixes = []

        # this is for checking in the folder
        for suffix in expected_suffixes:
            file_path_tracts = subses_outputdir / "tracts" / f"{tract_name}{suffix}"
            if file_path_tracts.exists():  # or file_path_rtp.exists():
                found_suffixes.append(suffix)
            else:
                missing_suffixes.append(suffix)

        # # this is for checking in the csv
        # for suffix in expected_suffixes:
        #     tract_in_csv = f"mrtrix/{tract_name}{suffix}"

        #     if output_summary_df["FileName"].str.contains(tract_in_csv).any():
        #         found_suffixes.append(suffix)
        #     else:
        #         missing_suffixes.append(suffix)

        tract_status[tract_name] = {
            "found": found_suffixes,
            "missing": missing_suffixes,
            "complete": len(found_suffixes) == len(expected_suffixes),
        }

        if len(found_suffixes) == len(expected_suffixes):
            complete_tracts.append(tract_name)
        else:
            missing_tracts.append(tract_name)

    return complete_tracts, missing_tracts, tract_status


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


def check_and_tract_dir(subses_outputdir):
    """
    Check whether tract output directories or archives are present.

    Parameters
    ----------
    subses_outputdir : str or path-like
        Output directory for one subject/session.

    Returns
    -------
    tuple[bool, bool, bool, str]
        Presence of the tract directory, presence of the tract archive,
        whether the state is acceptable, and an optional warning message.
    """
    subses_outputdir = Path(subses_outputdir)
    tract_dir = subses_outputdir / "RTP_PIPELINE_ALL_OUTPUT"
    tract_zip = subses_outputdir / "RTP_PIPELINE_ALL_OUTPUT.zip"

    has_tract_dir = tract_dir.exists() and tract_dir.is_dir()
    has_tract_zip = tract_zip.exists() and tract_zip.is_file()

    if has_tract_dir and has_tract_zip:
        console.print("Both tract/ and tract.zip exist", style="cyan")
        return has_tract_dir, has_tract_zip, True, ""
    elif not has_tract_zip:
        console.print("output.zip doesn't exist, not finishing", style="cyan")


def check_all_sub(analysis_dir):
    """
    Print missing tract summaries for all runnable DWI sessions.

    Parameters
    ----------
    analysis_dir : str or path-like
        Analysis directory containing ``subseslist.txt`` and output folders.
    """

    path_to_subses = find_subseslist(analysis_dir)
    df_subSes = pd.read_csv(path_to_subses, sep=",", dtype=str)
    for row in df_subSes.itertuples(index=True, name="Pandas"):
        sub = row.sub
        ses = row.ses
        RUN = row.RUN
        dwi = row.dwi
        if RUN == "True" and dwi == "True":
            subses_outputdir = os.path.join(
                analysis_dir, f"sub-{sub}", f"ses-{ses}", "output"
            )
            complete_tracts, missing_tracts, tract_status = check_tract_completeness(
                subses_outputdir
            )
            if not len(missing_tracts) == 0:
                print(f"sub-{sub}_ses-{ses} have missing tracts {missing_tracts}")
