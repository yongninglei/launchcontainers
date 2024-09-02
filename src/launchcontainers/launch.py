# -*- coding: utf-8 -*-
"""
MIT License

Copyright (c) 2020-2024 Garikoitz Lerma-Usabiaga
Copyright (c) 2020-2022 Mengxing Liu
Copyright (c) 2022-2024 Leandro Lecca
Copyright (c) 2022-2024 Yongning Lei
Copyright (c) 2023 David Linhardt
Copyright (c) 2023 Iñigo Tellaetxe

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
"""
import os
import os.path as op
import subprocess as sp
from subprocess import Popen
import numpy as np
import logging
import math

# modules in lc

from bids import BIDSLayout
from dask.distributed import progress

# for package mode, the import needs to import launchcontainer module
from launchcontainers.prepare_inputs import dask_scheduler_config as dsq
from launchcontainers.prepare_inputs import prepare as prepare
from launchcontainers.prepare_inputs import utils as do

# for testing mode through , we can use relative import 
# from prepare_inputs import dask_scheduler_config as dsq
# from prepare_inputs import prepare as prepare
# from prepare_inputs import utils as do


logger = logging.getLogger("Launchcontainers")


# %% launchcontainers
def generate_cmd(
    lc_config, sub, ses, analysis_dir, lst_container_specific_configs, run_lc
):
    """Puts together the command to send to the container.

    Args:
        lc_config (str): _description_
        sub (str): _description_
        ses (str): _description_
        analysis_dir (str): _description_
        lst_container_specific_configs (list): _description_
        run_lc (str): _description_

    Raises:
        ValueError: Raised in presence of a faulty config.yaml file, or when the formed command is not recognized.

    Returns:
        _type_: _description_
    """

    # Relevant directories
    # All other relevant directories stem from this one
    basedir = lc_config["general"]["basedir"]

    homedir = os.path.join(basedir, "singularity_home")
    container = lc_config["general"]["container"]
    host = lc_config["general"]["host"]
    containerdir = lc_config["general"]["containerdir"]

    # Information relevant to the host and container
    jobqueue_config = lc_config["host_options"][host]
    version = lc_config["container_specific"][container]["version"]
    use_module = jobqueue_config["use_module"]
    bind_options = jobqueue_config["bind_options"]

    # Location of the Singularity Image File (.sif)
    container_name = os.path.join(containerdir, f"{container}_{version}.sif")
    # Define the directory and the file name to output the log of each subject
    container_logdir = os.path.join(analysis_dir, "sub-" + sub, "ses-" + ses, "output", "log")
    logfilename = f"{container_logdir}/t-{container}-sub-{sub}_ses-{ses}"

    path_to_sub_derivatives = os.path.join(analysis_dir, f"sub-{sub}", f"ses-{ses}")

    bind_cmd = ""
    for bind in bind_options:
        bind_cmd += f"--bind {bind}:{bind} "

    env_cmd = ""
    if host == "local":
        if use_module == True:
            env_cmd = f"module load {jobqueue_config['apptainer']} &&"

    if container in ["anatrois", "rtppreproc", "rtp-pipeline"]:
        logger.info("\n" + "start to generate the DWI PIPELINE command")
        logger.debug(
            f"\n the sub is {sub} \n the ses is {ses} \n the analysis dir is {analysis_dir}"
        )

        cmd = (
            f"{env_cmd} singularity run -e --no-home {bind_cmd}"
            f"--bind {path_to_sub_derivatives}/input:/flywheel/v0/input:ro "
            f"--bind {path_to_sub_derivatives}/output:/flywheel/v0/output "
            f"--bind {path_to_sub_derivatives}/output/log/config.json:/flywheel/v0/config.json "
            f"{container_name} 1>> {logfilename}.o 2>> {logfilename}.e  "
        )

    if container == "freesurferator":
        logger.info("\n" + "FREESURFERATOR command")
        logger.debug(
            f"\n the sub is {sub} \n the ses is {ses} \n the analysis dir is {analysis_dir}"
        )

        cmd = (
            f"{env_cmd} apptainer run --containall --pwd /flywheel/v0 {bind_cmd}"
            f"--bind {path_to_sub_derivatives}/input:/flywheel/v0/input:ro "
            f"--bind {path_to_sub_derivatives}/output:/flywheel/v0/output "
            f"--bind {path_to_sub_derivatives}/work:/flywheel/v0/work "
            f"--bind {path_to_sub_derivatives}/output/log/config.json:/flywheel/v0/config.json "
            f"--env PATH=/opt/freesurfer/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin:/opt/freesurfer/fsfast/bin:/opt/freesurfer/tktools:/opt/freesurfer/mni/bin:/sbin:/bin:/opt/ants/bin "
            f"--env LANG=C.UTF-8 "
            f"--env GPG_KEY=E3FF2839C048B25C084DEBE9B26995E310250568 "
            f"--env PYTHON_VERSION=3.9.15 "
            f"--env PYTHON_PIP_VERSION=22.0.4 "
            f"--env PYTHON_SETUPTOOLS_VERSION=58.1.0 "
            f"--env PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/66030fa03382b4914d4c4d0896961a0bdeeeb274/public/get-pip.py "
            f"--env PYTHON_GET_PIP_SHA256=1e501cf004eac1b7eb1f97266d28f995ae835d30250bec7f8850562703067dc6 "
            f"--env FLYWHEEL=/flywheel/v0 "
            f"--env ANTSPATH=/opt/ants/bin/ "
            f"--env FREESURFER_HOME=/opt/freesurfer "
            f"--env FREESURFER=/opt/freesurfer "
            f"--env DISPLAY=:50.0 "
            f"--env FS_LICENSE=/flywheel/v0/work/license.txt "
            f"--env OS=Linux "
            f"--env FS_OVERRIDE=0 "
            f"--env FSF_OUTPUT_FORMAT=nii.gz "
            f"--env MNI_DIR=/opt/freesurfer/mni "
            f"--env LOCAL_DIR=/opt/freesurfer/local "
            f"--env FSFAST_HOME=/opt/freesurfer/fsfast "
            f"--env MINC_BIN_DIR=/opt/freesurfer/mni/bin "
            f"--env MINC_LIB_DIR=/opt/freesurfer/mni/lib "
            f"--env MNI_DATAPATH=/opt/freesurfer/mni/data "
            f"--env FMRI_ANALYSIS_DIR=/opt/freesurfer/fsfast "
            f"--env PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5 "
            f"--env MNI_PERL5LIB=/opt/freesurfer/mni/lib/perl5/5.8.5 "
            f"--env XAPPLRESDIR=/opt/freesurfer/MCRv97/X11/app-defaults "
            f"--env MCR_CACHE_ROOT=/flywheel/v0/output "
            f"--env MCR_CACHE_DIR=/flywheel/v0/output/.mcrCache9.7 "
            f"--env FSL_OUTPUT_FORMAT=nii.gz "
            f"--env ANTS_VERSION=v2.4.2 "
            f"--env QT_QPA_PLATFORM=xcb "
            f"--env PWD=/flywheel/v0 "
            f"{container_name} "
            f"-c python run.py 1> {logfilename}.o 2> {logfilename}.e  "
        )

    if container == "rtp2-preproc":
        logger.info("\n" + "rtp2-preprc command")
        logger.debug(
            f"\n the sub is {sub} \n the ses is {ses} \n the analysis dir is {analysis_dir}"
        )

        cmd = (
            f"{env_cmd} apptainer run --containall --pwd /flywheel/v0 {bind_cmd}"
            f"--bind {path_to_sub_derivatives}/input:/flywheel/v0/input:ro "
            f"--bind {path_to_sub_derivatives}/output:/flywheel/v0/output "
            # f"--bind {path_to_sub_derivatives}/work:/flywheel/v0/work "
            f"--bind {path_to_sub_derivatives}/output/log/config.json:/flywheel/v0/config.json "
            f"--env FLYWHEEL=/flywheel/v0 "
            f"--env LD_LIBRARY_PATH=/opt/fsl/lib:  "
            f"--env FSLWISH=/opt/fsl/bin/fslwish  "
            f"--env FSLTCLSH=/opt/fsl/bin/fsltclsh  "
            f"--env FSLMULTIFILEQUIT=TRUE "
            f"--env FSLOUTPUTTYPE=NIFTI_GZ  "
            f"--env FSLDIR=/opt/fsl  "
            f"--env FREESURFER_HOME=/opt/freesurfer  "
            f"--env ARTHOME=/opt/art  "
            f"--env ANTSPATH=/opt/ants/bin  "
            f"--env PYTHON_GET_PIP_SHA256=1e501cf004eac1b7eb1f97266d28f995ae835d30250bec7f8850562703067dc6 "
            f"--env PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/66030fa03382b4914d4c4d0896961a0bdeeeb274/public/get-pip.py "
            f"--env PYTHON_PIP_VERSION=22.0.4  "
            f"--env PYTHON_VERSION=3.9.15  "
            f"--env GPG_KEY=E3FF2839C048B25C084DEBE9B26995E310250568  "
            f"--env LANG=C.UTF-8  "
            f"--env PATH=/opt/mrtrix3/bin:/opt/ants/bin:/opt/art/bin:/opt/fsl/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
            f"--env PYTHON_SETUPTOOLS_VERSION=58.1.0 "
            f"--env DISPLAY=:50.0 "
            f"--env QT_QPA_PLATFORM=xcb  "
            f"--env FS_LICENSE=/opt/freesurfer/license.txt  "
            f"--env PWD=/flywheel/v0 "
            f"{container_name} "
            f"-c python run.py 1> {logfilename}.o 2> {logfilename}.e  "
        )
    
    if container == "rtp2-pipeline":
        logger.info("\n" + "rtp2-pipeline command")
        logger.debug(
            f"\n the sub is {sub} \n the ses is {ses} \n the analysis dir is {analysis_dir}"
        )

        cmd = (
            f"{env_cmd} apptainer run --containall --pwd /flywheel/v0 {bind_cmd}"
            f"--bind {path_to_sub_derivatives}/input:/flywheel/v0/input:ro "
            f"--bind {path_to_sub_derivatives}/output:/flywheel/v0/output "
            # f"--bind {path_to_sub_derivatives}/work:/flywheel/v0/work "
            f"--bind {path_to_sub_derivatives}/output/log/config.json:/flywheel/v0/config.json "
            f"--env PATH=/opt/mrtrix3/bin:/opt/ants/bin:/opt/art/bin:/opt/fsl/bin:/usr/local/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin "
            f"--env LANG=C.UTF-8 "
            f"--env GPG_KEY=E3FF2839C048B25C084DEBE9B26995E310250568 "
            f"--env PYTHON_VERSION=3.9.15 "
            f"--env PYTHON_PIP_VERSION=22.0.4 "
            f"--env PYTHON_SETUPTOOLS_VERSION=58.1.0 "
            f"--env PYTHON_GET_PIP_URL=https://github.com/pypa/get-pip/raw/66030fa03382b4914d4c4d0896961a0bdeeeb274/public/get-pip.py "
            f"--env PYTHON_GET_PIP_SHA256=1e501cf004eac1b7eb1f97266d28f995ae835d30250bec7f8850562703067dc6 "
            f"--env ANTSPATH=/opt/ants/bin "
            f"--env ARTHOME=/opt/art "
            f"--env FREESURFER_HOME=/opt/freesurfer "
            f"--env FSLDIR=/opt/fsl "
            f"--env FSLOUTPUTTYPE=NIFTI_GZ "
            f"--env FSLMULTIFILEQUIT=TRUE "
            f"--env FSLTCLSH=/opt/fsl/bin/fsltclsh "
            f"--env FSLWISH=/opt/fsl/bin/fslwish "
            f"--env LD_LIBRARY_PATH=/opt/mcr/v99/runtime/glnxa64:/opt/mcr/v99/bin/glnxa64:/opt/mcr/v99/sys/os/glnxa64:/opt/mcr/v99/extern/bin/glnxa64:/opt/fsl/lib: "
            f"--env FLYWHEEL=/flywheel/v0 "
            f"--env TEMPLATES=/templates "
            f"--env XAPPLRESDIR=/opt/mcr/v99/X11/app-defaults "
            f"--env MCR_CACHE_FOLDER_NAME=/flywheel/v0/output/.mcrCache9.9 "
            f"--env MCR_CACHE_ROOT=/flywheel/v0/output "
            f"--env MRTRIX_TMPFILE_DIR=/flywheel/v0/output/tmp "
            f"--env PWD=/flywheel/v0 "
            f"{container_name} "
            f"-c python run.py 1> {logfilename}.o 2> {logfilename}.e  "
        )
    
    # Check which container we are using, and define the command accordingly
    if container == "fmriprep":
        logger.info("\n" + f"start to generate the FMRIPREP command")

        nthreads = lc_config["container_specific"][container]["nthreads"]
        mem = lc_config["container_specific"][container]["mem"]
        fs_license = lc_config["container_specific"][container]["fs_license"]
        containerdir = lc_config["general"]["containerdir"]
        container_path = os.path.join(
            containerdir,
            f"{container}_{lc_config['container_specific'][container]['version']}.sif",
        )
        precommand = f"mkdir -p {homedir}; " f"unset PYTHONPATH; "
        if "local" == host:
            cmd = (
                precommand + f"singularity run "
                f"-H {homedir} "
                f"-B {basedir}:/base -B {fs_license}:/license "
                f"--cleanenv {container_path} "
                f"-w {analysis_dir} "
                f"/base/BIDS {analysis_dir} participant "
                f"--participant-label sub-{sub} "
                f"--skip-bids-validation "
                f"--output-spaces func fsnative fsaverage T1w MNI152NLin2009cAsym "
                f"--dummy-scans 0 "
                f"--use-syn-sdc "
                f"--fs-license-file /license/license.txt "
                f"--nthreads {nthreads} "
                f"--omp-nthreads {nthreads} "
                f"--stop-on-first-crash "
                f"--mem_mb {(mem*1000)-5000} "
            )
        if host in ["BCBL", "DIPC"]:
            cmd = (
                precommand + f"singularity run "
                f"-H {homedir} "
                f"-B {basedir}:/base -B {fs_license}:/license "
                f"--cleanenv {container_path} "
                f"-w {analysis_dir} "
                f"/base/BIDS {analysis_dir} participant "
                f"--participant-label sub-{sub} "
                f"--skip-bids-validation "
                f"--output-spaces func fsnative fsaverage T1w MNI152NLin2009cAsym "
                f"--dummy-scans 0 "
                f"--use-syn-sdc "
                f"--fs-license-file /license/license.txt "
                f"--nthreads {nthreads} "
                f"--omp-nthreads {nthreads} "
                f"--stop-on-first-crash "
                f"--mem_mb {(mem*1000)-5000} "
            )

    if container in ["prfprepare", "prfreport", "prfanalyze-vista"]:
        config_name = lc_config["container_specific"][container]["config_name"]
        homedir = os.path.join(basedir, "singularity_home")
        container_path = os.path.join(
            containerdir,
            f"{container}_{lc_config['container_specific'][container]['version']}.sif",
        )
        if host in ["BCBL", "DIPC"]:
            cmd = (
                "unset PYTHONPATH; "
                f"singularity run "
                f"-H {homedir} "
                f"-B {basedir}/derivatives/fmriprep:/flywheel/v0/input "
                f"-B {analysis_dir}:/flywheel/v0/output "
                f"-B {basedir}/BIDS:/flywheel/v0/BIDS "
                f"-B {analysis_dir}/{config_name}.json:/flywheel/v0/config.json "
                f"-B {basedir}/license/license.txt:/opt/freesurfer/.license "
                f"--cleanenv {container_path} "
            )
        elif host == "local":
            cmd = (
                "unset PYTHONPATH; "
                f"singularity run "
                f"-H {homedir} "
                f"-B {basedir}/derivatives/fmriprep:/flywheel/v0/input "
                f"-B {analysis_dir}:/flywheel/v0/output "
                f"-B {basedir}/BIDS:/flywheel/v0/BIDS "
                f"-B {analysis_dir}/{config_name}.json:/flywheel/v0/config.json "
                f"-B {basedir}/license/license.txt:/opt/freesurfer/.license "
                f"--cleanenv {container_path} "
            )
    # If after all configuration, we do not have command, raise an error
    if cmd is None:
        logger.error(
            "\n"
            + f"the DWI PIPELINE command is not assigned, please check your config.yaml[general][host] session\n"
        )
        raise ValueError("cmd is not defined, aborting")

    # GLU: I don't think this is right, run is done below, I will make it work just for local but not in here,
    #      it is good that this function just creates the cmd, I would keep it like that
    if run_lc:
       return(sp.run(cmd, shell = True))
    else:
        return cmd
    #     sp.run(cmd, shell=True)
    #return cmd


# %% the launchcontainer
def launchcontainer(
    analysis_dir,
    lc_config,
    sub_ses_list,
    parser_namespace,
    path_to_analysis_container_specific_config
):
    """
    This function launches containers generically in different Docker/Singularity HPCs
    This function is going to assume that all files are where they need to be.

    Args:
        analysis_dir (str): _description_
        lc_config (str): path to launchcontainer config.yaml file
        sub_ses_list (_type_): parsed CSV containing the subject list to be analyzed, and the analysis options
        parser_namespace (argparse.Namespace): command line arguments
    """
    logger.info("\n" + "#####################################################\n")

    # Get the host and jobqueue config info from the config.yaml file
    host = lc_config["general"]["host"]
    jobqueue_config = lc_config["host_options"][host]
    if host == "local":
        launch_mode = jobqueue_config["launch_mode"]
    logger.debug(f"\n,, this is the job_queue config {jobqueue_config}")

    force = lc_config["general"]["force"]
    daskworker_logdir = os.path.join(analysis_dir, "daskworker_log")

    # Count how many jobs we need to launch from  sub_ses_list
    n_jobs = np.sum(sub_ses_list.RUN == "True")

    run_lc = parser_namespace.run_lc

    lc_configs = []
    subs = []
    sess = []
    dir_analysiss = []
    paths_to_analysis_config_json = []
    run_lcs = []
    # PREPARATION mode
    if not run_lc:
        logger.critical(
            f"\nlaunchcontainers.py was run in PREPARATION mode (without option --run_lc)\n"
            f"Please check that: \n"
            f"    (1) launchcontainers.py prepared the input data properly\n"
            f"    (2) the command created for each subject is properly formed\n"
            f"         (you can copy the command for one subject and launch it "
            f"on the prompt before you launch multiple subjects\n"
            f"    (3) Once the check is done, launch the jobs by adding --run_lc to the first command you executed.\n"
        )
        # If the host is not local, print the job script to be launched in the cluster.
        if host != "local" or (host == "local" and launch_mode == "dask_worker"):
            client, cluster = create_cluster_client(jobqueue_config, n_jobs, daskworker_logdir)
            if host != "local":
                logger.critical(
                    f"The cluster job script for this command is:\n"
                    f"{cluster.job_script()}"
                )
            elif host == "local" and launch_mode == "dask_worker":
                logger.critical(
                    f"The cluster job script for this command is:\n"
                    f"{cluster}"
                )
    # Iterate over the provided subject list
    commands = list()
    for row in sub_ses_list.itertuples(index=True, name="Pandas"):
        sub = row.sub
        ses = row.ses
        RUN = row.RUN
        dwi = row.dwi

        if RUN == "True":
            # Append config, subject, session, and path info in corresponding lists
            lc_configs.append(lc_config)
            subs.append(sub)
            sess.append(ses)
            dir_analysiss.append(analysis_dir)
            paths_to_analysis_config_json.append(
                path_to_analysis_container_specific_config[0]
            )
            run_lcs.append(run_lc)

            # This cmd is only for print the command
            command = generate_cmd(
                lc_config,
                sub,
                ses,
                analysis_dir,
                path_to_analysis_container_specific_config,
                False # set to False to print the command
            )
            commands.append(command)
            if not run_lc:
                logger.critical(
                    f"\nCOMMAND for subject-{sub}, and session-{ses}:\n"
                    f"{command}\n\n"
                )

                if not run_lc and lc_config["general"]["container"] == "fmriprep":
                    logger.critical(
                        f"\n"
                        f"fmriprep now can not deal with session specification, "
                        f"so the analysis are running on all sessions of the "
                        f"subject you are specifying"
                    )

    # RUN mode
    if run_lc and host != "local":
        run_dask(
            jobqueue_config, 
            n_jobs, 
            daskworker_logdir, 
            lc_configs, 
            subs, 
            sess, 
            dir_analysiss,
            paths_to_analysis_config_json,
            run_lcs
        )

    if run_lc and host == "local":
        if launch_mode == "parallel":
            k = 0
            njobs = jobqueue_config["njobs"]
            if njobs == "" or njobs is None:
                njobs = 2
            steps = math.ceil(len(commands)/njobs)
            logger.critical(
                f"\nLocally launching {len(commands)} jobs in parallel every {njobs} jobs "
                f"in {steps} steps, check your server's memory, some jobs might fail\n"
            )
            for stp in range(steps):
                if stp == range(steps)[-1] and (k+njobs) <= len(commands):
                    selected_commands = commands[k:len(commands)]
                else:
                    selected_commands = commands[k:k+njobs]
                logger.critical(
                    f"JOBS in step {stp+1}:\n{selected_commands}\n"
                )
                procs = [ Popen(i, shell=True) for i in selected_commands ]
                for p in procs:
                    p.wait()
                k = k+njobs

        elif launch_mode == "dask_worker":
            logger.critical(
                f"\nLocally launching {len(commands)} jobs with dask-worker, "
                f" keep an eye on your server's memory\n"
            )
            run_dask(
                jobqueue_config, 
                n_jobs, 
                daskworker_logdir, 
                lc_configs, 
                subs, 
                sess, 
                dir_analysiss,
                paths_to_analysis_config_json,
                run_lcs
            )
        elif launch_mode == "serial":  # Run this with dask...
            logger.critical(
                f"Locally launching {len(commands)} jobs in series, this might take a lot of time"
            )
            serial_cmd = ""
            for i, cmd in enumerate(commands):
                if i == 0:
                    serial_cmd = cmd
                else:
                    serial_cmd += f" && {cmd}"
            logger.critical(
                f"LAUNCHING SUPER SERIAL {len(commands)} JOBS:\n{serial_cmd}\n"
            )
            sp.run(serial_cmd, shell=True)

    return

def create_cluster_client(jobqueue_config, n_jobs, daskworker_logdir):
    client, cluster = dsq.dask_scheduler(jobqueue_config, n_jobs, daskworker_logdir)
    return client, cluster

def run_dask(
    jobqueue_config, 
    n_jobs, 
    daskworker_logdir, 
    lc_configs, 
    subs, 
    sess, 
    dir_analysiss,
    paths_to_analysis_config_json,
    run_lcs
    ):
    
    client, cluster = create_cluster_client(jobqueue_config, n_jobs, daskworker_logdir)
    logger.info(
        "---this is the cluster and client\n" + f"{client} \n cluster: {cluster} \n"
    )
    print(subs)
    print(sess)
    # Compose the command to run in the cluster
    futures = client.map(
        generate_cmd,
        lc_configs,
        subs,
        sess,
        dir_analysiss,
        paths_to_analysis_config_json,
        run_lcs
    )
    # Record the progress
    # progress(futures)
    # Get the info and report it in the logger
    results = client.gather(futures)
    logger.info(results)
    logger.info("###########")
    # Close the connection with the client and the cluster, and inform about it
    client.close()
    cluster.close()

    logger.critical("\n" + "launchcontainer finished, all the jobs are done")
    #return client, cluster



# %% main()
def main():
    parser_namespace,parse_dict = do.get_parser()
    copy_configs=parser_namespace.copy_configs
    # Check if download_configs argument is provided
    if copy_configs:
        # Ensure the directory exists
        if not os.path.exists(copy_configs):
            os.makedirs(copy_configs)
        launchcontainers_version = do.copy_configs(copy_configs)
        # # Use the mocked version function for testing
        # launchcontainers_version = do.get_mocked_launchcontainers_version()
        
        # if launchcontainers_version is None:
        #     raise ValueError("Unable to determine launchcontainers version.")
        # do.download_configs(launchcontainers_version, download_configs)
    else:
        # Proceed with normal main functionality
        print("Executing main functionality with arguments")
        # Your main function logic here
        # e.g., launch_container(args.other_arg)
    # read ymal and setup the launchcontainer program
        
        lc_config_path = parser_namespace.lc_config
        lc_config = do.read_yaml(lc_config_path)
        
        run_lc = parser_namespace.run_lc
        verbose = parser_namespace.verbose
        debug = parser_namespace.debug
        

        # Get general information from the config.yaml file
        basedir=lc_config["general"]["basedir"]
        bidsdir_name=lc_config["general"]["bidsdir_name"]
        containerdir=lc_config["general"]["containerdir"]
        container=lc_config["general"]["container"]
        analysis_name=lc_config["general"]["analysis_name"]
        host=lc_config["general"]["host"]
        force=lc_config["general"]["force"]
        print_command_only=lc_config["general"]["print_command_only"]
        log_dir=lc_config["general"]["log_dir"]
        log_filename=lc_config["general"]["log_filename"]
        
        version = lc_config["container_specific"][container]["version"] 
        # get stuff from subseslist for future jobs scheduling
        sub_ses_list_path = parser_namespace.sub_ses_list
        sub_ses_list,num_of_true_run = do.read_df(sub_ses_list_path)
        
        
        if log_dir=="analysis_dir":
            log_dir=op.join(basedir,bidsdir_name,'derivatives',f'{container}_{version}',f"analysis-{analysis_name}")

        do.setup_logger(print_command_only,verbose, debug, log_dir, log_filename)
        
        # logger the settings

        if host == "local":
            njobs = lc_config["host_options"][host]["njobs"]
            if njobs == "" or njobs is None:
                njobs = 2
            launch_mode = lc_config["host_options"]["local"]["launch_mode"]
            valid_options = ["serial", "parallel","dask_worker"]
            if launch_mode in valid_options:
                host_str = ( "#####################################################\n"
                    f"Host is:{host} \ncommands will be launched in {launch_mode} mode every {njobs} jobs.\n"
                    f"Serial is safe but it will take longer.\n"
                    f"If you launch in parallel be aware that some of the "
                    f"processes might be killed if the limit (usually memory) "
                    f"of the machine is reached. "
                    "\n#####################################################\n"
                )
            else:
                do.die(
                    f"local:launch_mode {launch_mode} was passed, valid options are {valid_options}"
                )
        else:
            host_str=f" host is {host}"
        logger.critical(
            "\n"
            + "#####################################################\n"
            + f"Successfully read the config file {lc_config_path} \n"
            + f"SubsesList is read, there are {num_of_true_run} jobs needed to be launched"
            + f'Basedir is: {lc_config["general"]["basedir"]} \n'
            + f'Container is: {container}_{lc_config["container_specific"][container]["version"]} \n'
            + f'analysis folder is: {lc_config["general"]["analysis_name"]} \n'
            + f"##################################################### \n"
        )
        logger.critical(
            "\n"+f"{host_str} \n"
        )
        logger.info("Reading the BIDS layout...")
        
        # Prepare file and launch containers
        # First of all prepare the analysis folder: it create you the analysis folder automatically so that you are not messing up with different analysis
        analysis_dir, dict_store_cs_configs = (
            prepare.prepare_analysis_folder(parser_namespace, lc_config)
        )

        layout = BIDSLayout(os.path.join(basedir, bidsdir_name))
        logger.info("finished reading the BIDS layout.")
        path_to_analysis_container_specific_config=dict_store_cs_configs['config_path']
        # Prepare mode
        if container in [
            "anatrois",
            "rtppreproc",
            "rtp-pipeline",
            "freesurferator",
            "rtp2-preproc",
            "rtp2-pipeline"
        ]:  # TODO: define list in another module for reusability accross modules and functions
            logger.debug(f"{container} is in the list")
            prepare.prepare_dwi_input(
                parser_namespace, analysis_dir, lc_config, sub_ses_list, layout, dict_store_cs_configs
            )
        else:
            logger.error(f"{container} is not in the list")


        # Run mode
        launchcontainer(
            analysis_dir,
            lc_config,
            sub_ses_list,
            parser_namespace,
            path_to_analysis_container_specific_config
        )


# #%%
if __name__ == "__main__":
    main()
