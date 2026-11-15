# clusters/slurm.py
from __future__ import annotations

import os.path as op
from launchcontainers import utils as do


def gen_slurm_array_job_script(
    parse_namespace,
    log_dir,
    n_jobs,
):
    """
    Build the SLURM array-job script used for batch container launches.

    Parameters
    ----------
    parse_namespace : argparse.Namespace
        Parsed CLI arguments for run mode.
    log_dir : str
        Directory where scheduler stdout/stderr logs should be written.
    n_jobs : int
        Number of array tasks to request.

    Returns:
        str
            Full SLURM batch script text.
    """
    # read LC config yml from analysis dir
    analysis_dir = parse_namespace.workdir
    lc_config_fpath = op.join(analysis_dir, "lc_config.yaml")
    lc_config = do.read_yaml(lc_config_fpath)
    host = lc_config["general"]["host"]
    jobqueue_config = lc_config["host_options"][host]
    # below is the job specific configs
    job_name = jobqueue_config["job_name"]
    cores = jobqueue_config["cores"]
    memory = jobqueue_config["memory"]
    partition = jobqueue_config["partition"]
    # qos is a DIPC specific command, it is defining the queue
    qos = jobqueue_config["qos"]
    walltime = jobqueue_config["walltime"]

    # Generate array job script
    job_name = f"{job_name}_array"
    job_script = f"""#!/bin/bash
#SBATCH --array=1-{n_jobs}
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}_%A_%a.out
#SBATCH --error={log_dir}/{job_name}_%A_%a.err
#SBATCH --time={walltime}
#SBATCH --cpus-per-task={cores}
#SBATCH --ntasks=1
#SBATCH --mem={memory}
#SBATCH --partition={partition}
#SBATCH --qos={qos}

LOG_DIR={log_dir}
echo "Starting array task $SLURM_ARRAY_TASK_ID on $(hostname)"
echo "Job ID: $SLURM_JOB_ID"

# Read the command for this array index

COMMAND="your_command_here"
echo "Executing: $COMMAND"
eval $COMMAND
exitcode=$?

echo "Task $SLURM_ARRAY_TASK_ID completed with exit code $exitcode"

# Output results to a table
echo "$SLURM_ARRAY_TASK_ID $exitcode" \
    >> $LOG_DIR/${{SLURM_JOB_NAME}}_${{SLURM_ARRAY_JOB_ID}}.tsv
echo Finished tasks $SLURM_ARRAY_TASK_ID with exit code $exitcode
exit $exitcode

    """

    return job_script
