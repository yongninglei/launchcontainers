from __future__ import annotations

import os
import shutil
import zipfile
from concurrent.futures import as_completed
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
from launchcontainers.log_setup import console


def find_subseslist(analysis_dir):
    """
    Locate ``subseslist.txt`` somewhere under an analysis directory.
    This is limited to RTP2 pipeline outputs, which should always have this file.
    TODO:
    make other pipelines also stores a subseslist.txt in the same format,
    and then this function can be used for all pipelines
    """
    for dirpath, dirnames, filenames in os.walk(analysis_dir):
        for fname in filenames:
            if fname.lower() == "subseslist.txt":
                return os.path.join(dirpath, fname)
    raise FileNotFoundError(f"No subseslist.txt found under {analysis_dir}")


def flatten_directory(directory):
    """
    Flatten a single nested directory produced by archive extraction.

    Parameters
    ----------
    directory : str or path-like
        Directory that may contain one unnecessary top-level subdirectory.
    """
    directory = Path(directory)
    items = list(directory.iterdir())
    if len(items) == 1 and items[0].is_dir():
        subdir = items[0]
        for item in subdir.iterdir():
            shutil.move(str(item), str(directory / item.name))
        subdir.rmdir()


def unzip_subses_output(subses_outputdir: Path, force):
    """
    Ensure that one session output has an extracted RTP output directory.

    Parameters
    ----------
    subses_outputdir : pathlib.Path
        Output directory for one subject/session.
    force : bool
        If ``True``, re-extract the archive even when an output directory
        already exists.

    Returns
    -------
    tuple[bool, bool, bool, str]
        Presence of the extracted directory, presence of the zip archive,
        extraction success, and an optional warning message.
    """
    subses_outputdir = Path(subses_outputdir)

    tract_dir = subses_outputdir / "RTP_PIPELINE_ALL_OUTPUT"
    tract_zip = subses_outputdir / "RTP_PIPELINE_ALL_OUTPUT.zip"

    has_tract_dir = (
        tract_dir.exists()
        and tract_dir.is_dir()
        and (len(os.listdir(tract_dir)) != 0 and (os.path.exists(tract_dir / "mrtrix")))
    )

    has_tract_zip = tract_zip.exists() and tract_zip.is_file()
    unzip_success = False
    warning_msg = ""

    if has_tract_dir and has_tract_zip and not force:
        console.print(
            "Both RTP_PIPELINE_ALL_OUTPUT/ "
            + "and RTP_PIPELINE_ALL_OUTPUT.zip exist, skipping",
            style="cyan",
        )
        return has_tract_dir, has_tract_zip, True, ""

    elif not has_tract_dir and has_tract_zip or force:
        try:
            console.print(f"Unzipping {tract_zip}", style="cyan")
            if tract_dir.exists():
                shutil.rmtree(tract_dir)
            with zipfile.ZipFile(tract_zip, "r") as zip_ref:
                tract_dir.mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(tract_dir)
                flatten_directory(tract_dir)

            unzip_success = True
            has_tract_dir = True
            console.print(
                "Successfully unzipped RTP_PIPELINE_ALL_OUTPUT.zip", style="cyan"
            )

        except Exception as e:
            warning_msg = f"Failed to unzip RTP_PIPELINE_ALL_OUTPUT.zip: {e}"
            console.print(warning_msg, style="red")

    elif has_tract_dir and has_tract_zip and force:
        try:
            # first remove the dir that are there
            shutil.rmtree(tract_dir)
            console.print(f"Unzipping {tract_zip}", style="cyan")
            with zipfile.ZipFile(tract_zip, "r") as zip_ref:
                tract_dir.mkdir(parents=True, exist_ok=True)
                zip_ref.extractall(tract_dir)
                flatten_directory(tract_dir)

            unzip_success = True
            has_tract_dir = True
            console.print("Successfully unzipped tract.zip", style="cyan")

        except Exception as e:
            warning_msg = f"Failed to unzip tract.zip: {e}"
            console.print(warning_msg, style="red")

    else:
        warning_msg = "Missing files (no tract/ or tract.zip)"
        console.print(warning_msg, style="yellow")

    return has_tract_dir, has_tract_zip, unzip_success, warning_msg


# this is actually used to check tract result, the previous function is for checking and unzipping


def check_subses_output(subses_outputdir):
    """
    Check whether one session output is ready for tract inspection.

    Parameters
    ----------
    subses_outputdir : str or path-like
        Output directory for one subject/session.

    Returns
    -------
    tuple[bool, bool, bool, str]
        Presence of the extracted directory, presence of the zip archive,
        whether additional extraction is required, and an optional warning.
    """
    subses_outputdir = Path(subses_outputdir)
    tract_dir = subses_outputdir / "RTP_PIPELINE_ALL_OUTPUT"
    tract_zip = subses_outputdir / "RTP_PIPELINE_ALL_OUTPUT.zip"

    has_tract_dir = (
        tract_dir.exists()
        and tract_dir.is_dir()
        and (len(os.listdir(tract_dir)) != 0 and (os.path.exists(tract_dir / "mrtrix")))
    )
    has_tract_zip = tract_zip.exists() and tract_zip.is_file()
    unzip_success = False
    warning_msg = ""

    if has_tract_dir and has_tract_zip:
        if len(os.listdir(tract_dir / "mrtrix")) != 0:
            console.print(
                "Both RTP_PIPELINE_ALL_OUTPUT/ "
                + "and RTP_PIPELINE_ALL_OUTPUT.zip exist, skipping",
                style="cyan",
            )
            return has_tract_dir, has_tract_zip, True, ""
        else:
            console.print("has RTP_PIPELINE_ALL_OUTPUT but not complete", style="cyan")
            return (
                False,
                has_tract_zip,
                False,
                "has RTP_PIPELINE_ALL_OUTPUT but not complete",
            )

    elif not has_tract_dir and has_tract_zip:
        console.print(f"need to unzip {tract_zip}", style="cyan")
        unzip_success = False
        has_tract_dir = True

    else:
        warning_msg = (
            f"Missing files: has_tract_dir {has_tract_dir},"
            + f"and has_tract_zip: {has_tract_zip} "
        )
        console.print(warning_msg, style="yellow")

    return has_tract_dir, has_tract_zip, unzip_success, warning_msg


"""
Parallel processing the unzipping
"""

"""
Parallel processing the unzipping
"""


def process_single_subject(analysis_dir, sub, ses, run, dwi, force):
    """
    Unzip tract outputs for one subject/session if the row should run.
    """
    try:
        if run == "True" and dwi == "True":
            subses_outputdir = os.path.join(
                analysis_dir,
                f"sub-{sub}",
                f"ses-{ses}",
                "output",
            )

            # Call your unzip_subses_output function
            result = unzip_subses_output(subses_outputdir, force)

            console.print(f"Processed sub-{sub}_ses-{ses}: Success", style="cyan")
            return {
                "subject": sub,
                "session": ses,
                "success": True,
                "result": result,
                "output_dir": subses_outputdir,
            }
        else:
            console.print(
                f"Skipped sub-{sub}_ses-{ses}: RUN={run}, dwi={dwi}", style="cyan"
            )
            return {
                "subject": sub,
                "session": ses,
                "success": True,
                "skipped": True,
                "reason": f"RUN={run}, dwi={dwi}",
            }

    except Exception as e:
        console.print(f"Error processing sub-{sub}_ses-{ses}: {str(e)}", style="red")
        return {
            "subject": sub,
            "session": ses,
            "success": False,
            "error": str(e),
        }


def unzipping_tracts_parallel(analysis_dir, force, n_workers=65):
    """
    Unzip all runnable tract archives in parallel.

    Parameters
    ----------
    analysis_dir : str or path-like
        Analysis directory containing ``subseslist.txt`` and output folders.
    force : bool
        If ``True``, re-extract existing output directories.
    n_workers : int, default=65
        Maximum number of worker threads.

    Returns
    -------
    list[dict]
        Per-session processing results.
    """

    console.print(
        f"Starting parallel tract unzipping with {n_workers} workers", style="cyan"
    )

    # Load subject-session list
    path_to_subses = find_subseslist(analysis_dir)
    df_subSes = pd.read_csv(path_to_subses, sep=",", dtype=str)

    console.print(f"Found {len(df_subSes)} subject-session entries", style="cyan")

    # Filter for entries that need processing
    valid_entries = df_subSes[
        (df_subSes["RUN"] == "True") & (df_subSes["dwi"] == "True")
    ]
    console.print(f"Processing {len(valid_entries)} valid entries", style="cyan")

    # Prepare tasks
    tasks = []
    force = False
    for row in df_subSes.itertuples(index=False, name="Pandas"):
        tasks.append((analysis_dir, row.sub, row.ses, row.RUN, row.dwi, force))

    # Execute in parallel
    results = []
    failed_subjects = []
    skipped_subjects = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_subject, *task): task for task in tasks
        }

        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            sub, ses = task[1], task[2]  # Extract subject and session

            try:
                result = future.result()
                results.append(result)

                if not result["success"]:
                    failed_subjects.append(f"sub-{sub}_ses-{ses}")
                elif result.get("skipped", False):
                    skipped_subjects.append(f"sub-{sub}_ses-{ses}")

            except Exception as e:
                console.print(
                    f"Future failed for sub-{sub}_ses-{ses}: {str(e)}", style="red"
                )
                failed_subjects.append(f"sub-{sub}_ses-{ses}")
                results.append(
                    {
                        "subject": sub,
                        "session": ses,
                        "success": False,
                        "error": str(e),
                    }
                )

    # Summary statistics
    total_processed = len(results)
    successful = len(
        [r for r in results if r["success"] and not r.get("skipped", False)]
    )
    failed = len(failed_subjects)
    skipped = len(skipped_subjects)

    console.print("Processing complete:", style="cyan")
    console.print(f"  Total entries: {total_processed}", style="cyan")
    console.print(f"  Successfully processed: {successful}", style="cyan")
    console.print(f"  Failed: {failed}", style="cyan")
    console.print(f"  Skipped: {skipped}", style="cyan")

    if failed_subjects:
        console.print(f"Failed subjects: {', '.join(failed_subjects)}", style="yellow")

    # Save results summary
    save_processing_summary(analysis_dir, results)

    return results


def save_processing_summary(analysis_dir, results):
    """
    Write a JSON summary of the tract-unzipping batch run.
    """
    import json
    from datetime import datetime

    summary_file = (
        Path(analysis_dir)
        / f"unzipping_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_processed": len(results),
        "successful": len(
            [r for r in results if r["success"] and not r.get("skipped", False)]
        ),
        "failed": len([r for r in results if not r["success"]]),
        "skipped": len([r for r in results if r.get("skipped", False)]),
        "detailed_results": results,
    }

    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"Summary saved to: {summary_file}", style="cyan")


def check_and_unzip_tract_wrapper(output_dir, sub, ses):
    """
    Wrap :func:`unzip_subses_output` for executor-based parallel processing.
    """
    try:
        result = unzip_subses_output(output_dir, force)
        return {
            "subject": sub,
            "session": ses,
            "success": True,
            "result": result,
            "output_dir": output_dir,
        }
    except Exception as e:
        return {
            "subject": sub,
            "session": ses,
            "success": False,
            "error": str(e),
        }


# Usage example:
if __name__ == "__main__":
    analysis_dir = Path(
        "/bcbl/home/public/DB/devtrajtract/DATA/MINI/nifti/derivatives"
        + "/rtp2-pipeline_0.2.2_3.0.4/analysis-retest_hemi-both_4TOI_nowbt",
    )
    force = False
    # Use ThreadPoolExecutor (recommended for I/O-bound operations like unzipping)
    results = unzipping_tracts_parallel(analysis_dir, force, n_workers=70)


"""
# Optimized version with progress tracking
def unzipping_tracts_with_progress(analysis_dir, n_workers=35):

    #Version with progress tracking using tqdm (optional)

    try:
        from tqdm import tqdm
        use_tqdm = True
    except ImportError:
        use_tqdm = False
        console.print("tqdm not available, no progress bar will be shown", style="yellow")

    console.print(f"Starting parallel tract unzipping with {n_workers} workers", style="cyan")

    path_to_subses = find_subseslist(analysis_dir)
    df_subSes = pd.read_csv(path_to_subses, sep=',', dtype=str)

    # Filter valid entries
    valid_entries = df_subSes[(df_subSes['RUN'] == 'True') & (df_subSes['dwi'] == 'True')]
    console.print(f"Processing {len(valid_entries)} valid entries", style="cyan")

    tasks = []
    for row in valid_entries.itertuples(index=False, name='Pandas'):
        tasks.append((analysis_dir, row.sub, row.ses, row.RUN, row.dwi))

    results = []

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(process_single_subject, *task): task
            for task in tasks
        }

        # Progress bar setup
        if use_tqdm:
            progress_bar = tqdm(total=len(tasks), desc="Unzipping tracts")

        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)

                if use_tqdm:
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'Current': f"sub-{result['subject']}_ses-{result['session']}",
                        'Success': result['success']
                    })

            except Exception as e:
                console.print(f"Task failed: {str(e)}", style="red")
                if use_tqdm:
                    progress_bar.update(1)

        if use_tqdm:
            progress_bar.close()

    # Save results
    save_processing_summary(analysis_dir, results)
    return results

    # Alternative version using ProcessPoolExecutor (for CPU-intensive tasks)
def unzipping_tracts_parallel_processes(analysis_dir, n_workers=35):

    # Version using ProcessPoolExecutor instead of ThreadPoolExecutor
    # Use this if unzip_subses_output is CPU-intensive rather than I/O bound

    from concurrent.futures import ProcessPoolExecutor

    console.print(f"Starting parallel tract unzipping with {n_workers} processes", style="cyan")

    path_to_subses = find_subseslist(analysis_dir)
    df_subSes = pd.read_csv(path_to_subses, sep=',', dtype=str)

    # Prepare tasks
    tasks = []
    for row in df_subSes.itertuples(index=False, name='Pandas'):
        if row.RUN == 'True' and row.dwi == 'True':
            subses_outputdir = os.path.join(
                analysis_dir,
                f'sub-{row.sub}',
                f'ses-{row.ses}',
                'output'
            )
            tasks.append((subses_outputdir, row.sub, row.ses))

    console.print(f"Processing {len(tasks)} valid entries", style="cyan")

    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        # Submit all tasks
        future_to_task = {
            executor.submit(check_and_unzip_tract_wrapper, task[0], task[1], task[2]): task
            for task in tasks
        }

        # Process completed tasks
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
                results.append(result)
                console.print(f"Completed: sub-{task[1]}_ses-{task[2]}", style="cyan")
            except Exception as e:
                console.print(f"Failed: sub-{task[1]}_ses-{task[2]} - {str(e)}", style="red")
                results.append({
                    'subject': task[1],
                    'session': task[2],
                    'success': False,
                    'error': str(e)
                })

    return results

"""
