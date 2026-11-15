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

import os
import os.path as op
import subprocess as sp
import sys
from argparse import Namespace
from datetime import datetime
from os import makedirs
from launchcontainers import utils as do
from launchcontainers.check import check_dwi_pipelines
from launchcontainers.check import general_checks
from launchcontainers.clusters import local
from launchcontainers.clusters import sge
from launchcontainers.clusters import slurm
from launchcontainers.gen_jobscript import gen_launch_cmd
from launchcontainers.log_setup import console


def write_job_script(job_script, script_dir, job_script_fname):
    """
    Write a generated scheduler script to disk and make it executable.

    Parameters
    ----------
    job_script : str
        Full script text to write.
    script_dir : str
        Directory where the script should be created.
    job_script_fname : str
        Output filename for the script.

    Returns
    -------
    str
        Path to the written script file.
    """

    # Create script directory if specified
    if script_dir:
        makedirs(script_dir, exist_ok=True)
        job_script_fpath = op.join(script_dir, f"{job_script_fname}")

    # Write script to file
    with open(job_script_fpath, "w") as f:
        f.write(job_script)

    # Make executable
    os.chmod(job_script_fpath, 0o755)

    return job_script_fpath


def launch_jobs(
    parse_namespace,
    df_subses,
    job_script_dir,
    run_lc,
):
    """
    Generate launch commands and either print or submit them.

    Depending on the launchcontainers configuration this function performs a
    dry run, submits an array job through SLURM or SGE, or runs the commands
    locally in parallel using ``concurrent.futures``.

    Parameters
    ----------
    parse_namespace : argparse.Namespace
        Parsed CLI arguments for run mode.
    df_subses : list[tuple[str, str]]
        Filtered subject/session rows to launch.
    job_script_dir : str
        Directory used to store generated scripts and batch command files.
    run_lc : bool
        If ``True``, submit jobs for execution. If ``False``, only print the
        generated launch information.
    """
    # read LC config yml from analysis dir
    analysis_dir = parse_namespace.workdir
    lc_config_fpath = op.join(analysis_dir, "lc_config.yaml")
    lc_config = do.read_yaml(lc_config_fpath)
    host = lc_config["general"]["host"]
    # get number of jobs from subseslist
    n_jobs = len(df_subses)
    # write commands into a single file to form batch array
    batch_command_fpath = op.join(job_script_dir, "batch_commands.txt")
    commands = gen_launch_cmd(parse_namespace, df_subses, batch_command_fpath)
    # read the first command as example
    with open(batch_command_fpath) as f:
        command = f.readline().strip()

    # DRY RUN mode
    if not run_lc:
        console.print(
            "\n### No launching, here is the launching command", style="bold red"
        )
        if host == "DIPC":
            job_script = slurm.gen_slurm_array_job_script(
                parse_namespace,
                job_script_dir,
                n_jobs,
            )
            console.print(f"\n### SLURM job script is {job_script}", style="bold red")
        elif host == "BCBL":
            job_script = sge.gen_sge_array_job_script(
                parse_namespace,
                job_script_dir,
                n_jobs,
            )
            console.print(f"\n### SGE job script is {job_script}", style="bold red")
        console.print(f"\n### Example launch command is: {command}", style="bold red")

    # RUN mode
    else:
        console.print(
            "\n### Real running, here is the launching command", style="bold red"
        )

        def launch_cmd(cmd):
            result = sp.run(cmd, shell=True, capture_output=True, text=True, timeout=60)
            return result.returncode

        if host == "local":
            jobqueue_config = lc_config["host_options"][host]
            launch_mode = jobqueue_config.get("launch_mode", "serial")
            if launch_mode == "parallel":
                max_workers = jobqueue_config.get("max_workers", None)
                mem_per_job = jobqueue_config.get("mem_per_job", None)
                local.launch_parallel(
                    commands, max_workers=max_workers, mem_per_job=mem_per_job
                )
            else:
                local.launch_serial(commands)

        elif host == "DIPC":
            batch_command = (
                f"""$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" {batch_command_fpath})"""
            )
            job_script = slurm.gen_slurm_array_job_script(
                parse_namespace,
                job_script_dir,
                n_jobs,
            )
            final_script = job_script.replace("your_command_here", batch_command)
            job_script_fname = "src_launch_script.slurm"
            job_script_fpath = write_job_script(
                final_script,
                job_script_dir,
                job_script_fname,
            )
            console.print(
                f"This is the final job script that is being launched:\n{final_script}",
                style="bold red",
            )
            cmd = f"sbatch {job_script_fpath}"
            try:
                return_code = launch_cmd(cmd)
                console.print(
                    f"\n return code of launch is {return_code} \n",
                    style="bold red",
                )
            except sp.TimeoutExpired:
                console.print("Sbatch submission timed out!", style="bold red")
            except Exception as e:
                console.print(f"Error during submission: {e}", style="bold red")

        elif host == "BCBL":
            batch_command = f"""$(sed -n "${{SGE_TASK_ID}}p" {batch_command_fpath})"""
            job_script = sge.gen_sge_array_job_script(
                parse_namespace,
                job_script_dir,
                n_jobs,
            )
            final_script = job_script.replace("your_command_here", batch_command)
            job_script_fname = "src_launch_script.sh"
            job_script_fpath = write_job_script(
                final_script,
                job_script_dir,
                job_script_fname,
            )
            console.print(
                f"This is the final job script that is being launched:\n{final_script}",
                style="bold red",
            )
            cmd = f"qsub {job_script_fpath}"
            try:
                return_code = launch_cmd(cmd)
                console.print(
                    f"\n return code of launch is {return_code} \n",
                    style="bold red",
                )
            except sp.TimeoutExpired:
                console.print("Qsub submission timed out!", style="bold red")
            except Exception as e:
                console.print(f"Error during submission: {e}", style="bold red")

    return


def main(workdir: str, run_lc: bool = False):
    """
    Validate a prepared analysis directory and launch the requested jobs.

    The run workflow reloads the prepared configuration files, validates that
    the analysis directory contains the expected inputs, shows an example input
    folder structure to the user, asks for confirmation, and then calls
    :func:`launch_jobs`.

    Parameters
    ----------
    workdir : str
        Working directory for run mode.
    run_lc : bool, default=False
        Whether to run launchcontainers.
    """
    # 1. setup run mode logger
    # read the yaml to get input info
    analysis_dir = workdir
    run_lc = run_lc

    # read LC config yml from analysis dir
    lc_config_fpath = op.join(analysis_dir, "lc_config.yaml")
    lc_config = do.read_yaml(lc_config_fpath)
    console.print("\n cli.main() reading lc config yaml", style="cyan")
    # Get general information from the config.yaml file
    bidsdir_name = lc_config["general"]["bidsdir_name"]
    container = lc_config["general"]["container"]
    # get stuff from subseslist for future jobs scheduling
    sub_ses_list_path = op.join(analysis_dir, "subseslist.txt")
    df_subses = do.parse_subses_list(sub_ses_list_path)
    num_of_jobs = len(df_subses)
    # 2. do a independent check to see if everything is in place
    parse_namespace = Namespace(workdir=workdir, run_lc=run_lc)
    if container in [
        "anatrois",
        "rtppreproc",
        "rtp-pipeline",
        "freesurferator",
        "rtp2-preproc",
        "rtp2-pipeline",
    ]:
        check_dwi_pipelines.check_dwi_analysis_folder(parse_namespace, container)
        if container in ["rtp2-pipeline", "rtp-pipeline"]:
            # do a second check for the RTP file, if exist, backup
            check_dwi_pipelines.backup_old_rtp2pipeline_log(parse_namespace, df_subses)
    # 3. tree sub-/ses- structure for checking
    sub, ses = df_subses[0]
    console.print("\n### output example subject folder structure \n", style="bold red")
    general_checks.cli_show_folder_struc(analysis_dir, sub, ses)

    # 4. ask for user input about folder structure and example command
    general_checks.print_option_for_review(
        num_of_jobs,
        lc_config,
        container,
        bidsdir_name,
    )

    # 5. generate the job script
    # TODO: for different containers
    # check here if the log dir and singularity home dir is being put under proper place
    # Setup log dir, create command txt under log dir
    job_script_dir = (
        f"{analysis_dir}/job_script_dir_{datetime.now().strftime('%Y-%m-%d_%H-%M')}"
    )
    os.makedirs(job_script_dir, exist_ok=True)
    # 6. generate command to print
    # === Ask user to confirm before launching anything ===
    # Only a real submission (-R) needs confirmation; a dry run changes nothing.
    ans = "y"
    if run_lc:
        try:
            ans = input(
                "You are about to launch jobs, please review the "
                "commandline info. Continue? [y / N]: ",
            )
        except EOFError:
            ans = ""
    if ans.strip().lower() not in ("y", "yes"):
        console.print("Aborted by user.", style="cyan")
        sys.exit(0)

    # 7. launch the work
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    console.print(f"\n##### The launching time is {timestamp}", style="bold red")
    launch_jobs(
        parse_namespace,
        df_subses,
        job_script_dir,
        run_lc,
    )
    timestamp_finish = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    console.print(f"\n##### The finishing time is {timestamp_finish}", style="bold red")
    # when finished launch QC to read the log and check if everything is there
    return


# # #%%
# if __name__ == '__main__':
#     main()
