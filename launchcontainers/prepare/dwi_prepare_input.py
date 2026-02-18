"""
MIT License

Copyright (c) 2020-2023 Garikoitz Lerma-Usabiaga
Copyright (c) 2020-2022 Mengxing Liu
Copyright (c) 2022-2024 Leandro Lecca
Copyright (c) 2022-2023 Yongning Lei
Copyright (c) 2023 David Linhardt
Copyright (c) 2023 Iñigo Tellaetxe

Permission is hereby granted, free of charge,
to any person obtaining a copy of this software and associated documentation files
(the "Software"), to deal in the Software without restriction,
including without limitation the rights to use, copy, modify, merge, publish,
distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies
or substantial portions of the Software.
"""
from __future__ import annotations

import errno
import json
import logging
import os
import os.path as op
import re
import subprocess as sp
import sys

import nibabel as nib

from launchcontainers.check import check_dwi_pipelines as check
from launchcontainers.utils import read_df, force_symlink
logger = logging.getLogger('Launchcontainers')


def anatrois(dict_store_cs_configs, analysis_dir, lc_config, sub, ses, layout):
    """anatrois function creates symbolic links for the anatrois container

    Args:
        analysis_dir (_type_): directory to analyze
        lc_config (dict): the lc_config dictionary from _read_config
        sub (str): subject name
        ses (str): session name
        layout (_type_): _description_

    Raises:
        FileNotFoundError: _description_
        FileNotFoundError: _description_
    """

    # General level variables:
    basedir = lc_config['general']['basedir']
    container = lc_config['general']['container']
    bidsdir_name = lc_config['general']['bidsdir_name']
    # If force is False, then we don't want to overwrite anything
    # If force is true, and we didn't run_lc(in the prepare mode),
    # we will do the overwrite and so on
    # If force is true and we do run_lc, then we will never overwrite
    force = (lc_config['general']['force'])

    # Container specific:
    prefs_dir_name = lc_config['container_specific'][container]['prefs_dir_name']
    prefs_analysis_name = lc_config['container_specific'][container]['prefs_analysis_name']
    prefs_zipname = lc_config['container_specific'][container]['prefs_zipname']
    # specific for freesurferator
    if container == 'freesurferator':
        control_points = lc_config['container_specific'][container]['control_points']
        if control_points:
            prefs_unzipname = lc_config['container_specific'][container]['prefs_unzipname']
    use_src_session = lc_config['container_specific'][container]['use_src_session']
    # if have retest session and the src session id is specified, then
    # for each src_session_id, we will do the normal procedure
    # for each not src_session_id, we will create symlink to the src_session_id
    if use_src_session is not None:
        src_session_dpath = op.join(
            analysis_dir,
            'sub-' + sub,
            'ses-' + use_src_session,
        )
        dst_session_dpath = op.join(
            analysis_dir,
            'sub-' + sub,
            'ses-' + ses,
        )
    if use_src_session and (not ses == use_src_session):
        logger.info(
            '\n'
            + '### GOing to create symlinks for repeated sessions\n',
        )
        force_symlink(src_session_dpath, dst_session_dpath, force)
        if not os.path.islink(dst_session_dpath):
            logger.warning(f'***Symbolic link missing: {dst_session_dpath}')
    # if not going to use 1 src session for retest, do the normal thing
    else:
        # define input output folder for this container
        dstDir_input = op.join(
            analysis_dir,
            'sub-' + sub,
            'ses-' + ses,
            'input',
        )
        dstDir_output = op.join(
            analysis_dir,
            'sub-' + sub,
            'ses-' + ses,
            'output',
        )
        if container == 'freesurferator':
            dstDir_work = op.join(
                analysis_dir,
                'sub-' + sub,
                'ses-' + ses,
                'work',
            )
            if not op.exists(dstDir_work):
                os.makedirs(dstDir_work)

        if not op.exists(dstDir_input):
            os.makedirs(dstDir_input)
        if not op.exists(dstDir_output):
            os.makedirs(dstDir_output)

        # read json, this json is already written from previous preparasion step
        json_under_analysis_dir = dict_store_cs_configs['config_path']
        config_json_instance = json.load(open(json_under_analysis_dir))
        required_inputfiles = config_json_instance['inputs'].keys()

        # 5 main filed needs to be in anatrois if all specified, so there will be 5 checks
        if 'anat' in required_inputfiles:
            src_path_anat_lst = layout.get(
                subject=sub, session=ses, extension='nii.gz',
                suffix='T1w', return_type='filename',
            )
            if len(src_path_anat_lst) == 0:
                raise FileNotFoundError(
                    f'the T1w.nii.gz you are specifying for sub-{sub}_ses-{ses} '+
                    f'does NOT exist or the folder is not BIDS format, please check',
                )
            else:
                src_path_anat = src_path_anat_lst[0]
            dst_fname_anat = config_json_instance['inputs']['anat']['location']['name']
            dst_path_anat = op.join(dstDir_input, 'anat', dst_fname_anat)

            if not op.exists(op.join(dstDir_input, 'anat')):
                os.makedirs(op.join(dstDir_input, 'anat'))
            force_symlink(src_path_anat, dst_path_anat, force)

        # If we ran freesurfer before:
        if 'pre_fs' in required_inputfiles:
            pre_fs_path = op.join(
                basedir,
                bidsdir_name,
                'derivatives',
                f'{prefs_dir_name}',
                'analysis-' + prefs_analysis_name,
                'sub-' + sub,
                'ses-' + ses,
                'output',
            )
            logger.info(
                '\n'
                + f'---the patter of fs.zip filename we are searching is {prefs_zipname}\n'
                + f'---the directory we are searching for is {pre_fs_path}',
            )
            logger.debug(
                '\n'
                + f'the tpye of patter is {type(prefs_zipname)}',
            )
            zips = []
            for filename in os.listdir(pre_fs_path):
                if filename.endswith('.zip') and re.match(prefs_zipname, filename):
                    zips.append(filename)
                    if 'control_points' in required_inputfiles:
                        if (
                            op.isdir(op.join(pre_fs_path, filename))
                            and re.match(prefs_unzipname, filename)
                        ):
                            src_path_ControlPoints = op.join(
                                pre_fs_path,
                                filename,
                                'tmp',
                                'control.dat',
                            )
                        else:
                            raise FileNotFoundError("Didn't found control_points .zip file")

            if len(zips) == 0:
                logger.error(
                    '\n'
                    + f'There are no files with pattern: {prefs_zipname} in {pre_fs_path}, \
                         we will listed potential zip file for you',
                )
                raise FileNotFoundError('pre_fs_path is empty, no previous analysis was found')
            elif len(zips) == 1:
                src_path_fszip = op.join(pre_fs_path, zips[0])
            else:
                zips_by_time = sorted(zips, key=op.getmtime)
                answer = input(
                    f'Do you want to use the newset fs.zip: \n{zips_by_time[-1]} \n \
                        we get for you? \n input y for yes, n for no',
                )
                if answer in 'y':
                    src_path_fszip = zips_by_time[-1]
                else:
                    logger.error(
                        '\n' + 'An error occurred'
                        + zips_by_time + '\n'  # type: ignore
                        + 'no target preanalysis.zip file exist, \
                            please check the config_lc.yaml file',
                    )
                    sys.exit(1)

            dst_fname_fs = config_json_instance['inputs']['pre_fs']['location']['name']
            dst_path_fszip = op.join(dstDir_input, 'pre_fs', dst_fname_fs)
            if not op.exists(op.join(dstDir_input, 'pre_fs')):
                os.makedirs(op.join(dstDir_input, 'pre_fs'))
            force_symlink(src_path_fszip, dst_path_fszip, force)

            if 'control_points' in required_inputfiles:
                dst_fname_cp = config_json_instance['inputs']['control_points']['location']['name']
                dst_path_cp = op.join(dstDir_input, 'pre_fs', dst_fname_cp)
                if not op.exists(op.join(dstDir_input, 'control_points')):
                    os.makedirs(op.join(dstDir_input, 'control_points'))
                force_symlink(src_path_ControlPoints, dst_path_cp, force)

        if 'annotfile' in required_inputfiles:

            fname_annot = config_json_instance['inputs']['annotfile']['location']['name']
            src_path_annot = op.join(analysis_dir, fname_annot)
            dst_path_annot = op.join(dstDir_input, 'annotfile', fname_annot)

            if not op.exists(op.join(dstDir_input, 'annotfile')):
                os.makedirs(op.join(dstDir_input, 'annotfile'))
            force_symlink(src_path_annot, dst_path_annot, force)

        if 'mniroizip' in required_inputfiles:

            fname_mniroi = config_json_instance['inputs']['mniroizip']['location']['name']
            src_path_mniroi = op.join(analysis_dir, fname_mniroi)
            dst_path_mniroi = op.join(dstDir_input, 'mniroizip', fname_mniroi)

            if not op.exists(op.join(dstDir_input, 'mniroizip')):
                os.makedirs(op.join(dstDir_input, 'mniroizip'))
            force_symlink(src_path_mniroi, dst_path_mniroi, force)

        logger.info(
            '\n'
            + '-----------------The symlink created-----------------------\n',
        )

    return


def rtppreproc(dict_store_cs_configs, analysis_dir, lc_config, sub, ses, layout):
    """
    Parameters
    ----------
    parser_namespace: parser obj
        it contains all the input argument in the parser

    lc_config : dict
        the lc_config dictionary from _read_config
    sub : str
        the subject name looping from df_subSes
    ses : str
        the session name looping from df_subSes.

    Returns
    -------
    none, create symbolic links

    """

    # general level variables:
    basedir = lc_config['general']['basedir']
    container = lc_config['general']['container']
    bidsdir_name = lc_config['general']['bidsdir_name']
    force = (lc_config['general']['force'])

    # container specific:
    precontainer_anat = lc_config['container_specific'][container]['precontainer_anat']
    anat_analysis_name = lc_config['container_specific'][container]['anat_analysis_name']
    dwi_desc = lc_config['container_specific'][container]['dwi_desc']
    rpe = lc_config['container_specific'][container]['rpe']
    separated_shell_files = lc_config['container_specific'][container]['separated_shell_files']

    # define input output folder for this container
    dstDir_input = op.join(
        analysis_dir,
        'sub-' + sub,
        'ses-' + ses,
        'input',
    )
    dstDir_output = op.join(
        analysis_dir,
        'sub-' + sub,
        'ses-' + ses,
        'output',
    )

    if not op.exists(dstDir_input):
        os.makedirs(dstDir_input)
    if not op.exists(dstDir_output):
        os.makedirs(dstDir_output)

    # read json, this json is already written from previous preparation step
    json_under_analysis_dir = dict_store_cs_configs['config_path']
    config_json_instance = json.load(open(json_under_analysis_dir))
    required_inputfiles = config_json_instance['inputs'].keys()

    config_json_data = json.load(open(json_under_analysis_dir))
    if container == 'rtp2-preproc':
        PE_direction = config_json_data['config']['pe_dir']
    if container == 'rtppreproc':
        PE_direction = config_json_data['config']['acqd']
    # get the rpe dir
    if PE_direction == 'PA':
        RPE_direction = 'AP'
    elif PE_direction == 'AP':
        RPE_direction = 'PA'
    precontainer_anat_dir = op.join(
        basedir,
        bidsdir_name,
        'derivatives',
        f'{precontainer_anat}',
        'analysis-' + anat_analysis_name,
        'sub-' + sub,
        'ses-' + ses,
        'output',
    )
    # there are 8 fieids in the rtppreproc, and 9 fields in
    # rtp2-preproc(qmap), so we will loop for the
    # required_inputfiles to prepare them
    # required: ANAT BVAL BVEC DIFF FSMASK
    # optional for rtppreproc RBVL RBVC RDIF
    # optional for rtp2-preproc extra: qmap

    # prepare the required filed
    # the previous container anatrois/freesurferator analysis
    # ANAT
    # T1 file in anatrois output
    src_path_ANAT = op.join(precontainer_anat_dir, 'T1.nii.gz')
    # FSMASK
    # brain mask file in anatrois output
    # if the container is rtppreproc, and if the version of the precontainer_anat
    # is anatrois 4.5.x,
    # we will use brainmask, otherwise, we will use brain.nii.gz
    if ((container != 'rtp2-preproc')
        and (
            precontainer_anat.split('_')[0] == 'anatrois'
            and int(precontainer_anat.split('.')[1]) < 6
    )):
        src_path_FSMASK = op.join(precontainer_anat_dir, 'brainmask.nii.gz')
    else:
        src_path_FSMASK = op.join(precontainer_anat_dir, 'brain.nii.gz')
    # 3 dwi file that needs to be preprocessed, under BIDS/sub/ses/dwi
    if not separated_shell_files:
        # the bval
        src_path_BVAL = layout.get(
            subject=sub, session=ses, extension='bval',
            suffix='dwi', direction=PE_direction, desc=dwi_desc, return_type='filename',
        )[0]
        # the bve
        src_path_BVEC = layout.get(
            subject=sub, session=ses, extension='bvec',
            suffix='dwi', direction=PE_direction, desc=dwi_desc, return_type='filename',
        )[0]
        # the dwi
        src_path_DIFF = layout.get(
            subject=sub, session=ses, extension='nii.gz',
            suffix='dwi', direction=PE_direction, desc=dwi_desc, return_type='filename',
        )[0]

    else:
        # check how many *dir_dwi.nii.gz there are in the BIDS/sub/ses/dwi directory
        diff_files = layout.get(
            subject=sub, session=ses, extension='nii.gz',
            suffix='dwi', direction=PE_direction, return_type='filename',
        )

        dwi_file_with_acq_in_name = [f for f in diff_files if 'acq-' in f]
        # create the file name, it will be a file after concat
        target_dwi_concat = re.sub(r'acq-[^_]+', '', diff_files[0])
        src_path_DIFF = target_dwi_concat
        bval_files = layout.get(
            subject=sub, session=ses, extension='bval',
            suffix='dwi', direction=PE_direction, desc=dwi_desc, return_type='filename',
        )
        bvec_files = layout.get(
            subject=sub, session=ses, extension='bvec',
            suffix='dwi', direction=PE_direction, desc=dwi_desc, return_type='filename',
        )
        target_bvec = re.sub(r'acq-[^_]+', '', bvec_files[0])
        target_bval = re.sub(r'acq-[^_]+', '', bval_files[0])
        src_path_BVEC = target_bvec
        src_path_BVAL = target_bval
        if len(dwi_file_with_acq_in_name) == 0:
            logger.error(
                '\n'
                + 'No files with different acq- to concatenate.\n',
            )
            raise FileNotFoundError(
                "Didn't found the multi shell DWI, check your bids naming of acq- field",
            )
        elif len(dwi_file_with_acq_in_name) == 1:
            logger.error(
                '\n'
                + f'Found only {dwi_file_with_acq_in_name[0]} to concatenate. \
                    There must be at least two files with different acq.\n',
            )
            raise FileNotFoundError("Didn't found 2 multi shell DWI, only found 1")
        else:
            if not op.isfile(target_dwi_concat):
                logger.info(
                    '\n'
                    + f'Concatenating with mrcat of mrtrix3 these files: \
                    {dwi_file_with_acq_in_name} in: {target_dwi_concat} \n',
                )
                dwi_file_with_acq_in_name.sort()
                sp.run(['mrcat', *dwi_file_with_acq_in_name, target_dwi_concat])
                src_path_DIFF = target_dwi_concat
            else:
                logger.info(
                    '\n'
                    + f'The final DWI file is already being prepared: {target_dwi_concat} \n',
                )
            # also get the bvecs and bvals
            bvals_acq = [f for f in bval_files if 'acq-' in f]
            bvecs_acq = [f for f in bvec_files if 'acq-' in f]
            if len(dwi_file_with_acq_in_name) == len(bvals_acq) and not op.isfile(target_bval):
                bvals_acq.sort()
                bval_cmd = "paste -d ' '"
                for bvalF in bvals_acq:
                    bval_cmd = bval_cmd + ' ' + bvalF
                bval_cmd = bval_cmd + ' > ' + target_bval
                sp.run(bval_cmd, shell=True)
                src_path_BVAL = target_bval
            elif len(dwi_file_with_acq_in_name) != len(bvals_acq):
                logger.error(
                    '\n'
                    + f'Missing bval files for {sub} and {ses} ',
                )
            else:
                logger.info(
                    '\n'
                    + f'The final DWI bvals is already being prepared: {target_bval} \n',
                )
            if len(dwi_file_with_acq_in_name) == len(bvecs_acq) and not op.isfile(target_bvec):
                bvecs_acq.sort()
                bvec_cmd = "paste -d ' '"
                for bvecF in bvecs_acq:
                    bvec_cmd = bvec_cmd + ' ' + bvecF
                bvec_cmd = bvec_cmd + ' > ' + target_bvec
                sp.run(bvec_cmd, shell=True)
                src_path_BVEC = target_bvec
            elif len(dwi_file_with_acq_in_name) != len(bvecs_acq):
                logger.error(
                    '\n'
                    + f'Missing bvec files for {sub} and {ses} ',
                )
            else:
                logger.info(
                    '\n'
                    + f'The final DWI bvec is already being prepared: {target_bvec} \n',
                )
    # destination directory under dstDir_input
    if not op.exists(op.join(dstDir_input, 'ANAT')):
        os.makedirs(op.join(dstDir_input, 'ANAT'))
    if not op.exists(op.join(dstDir_input, 'FSMASK')):
        os.makedirs(op.join(dstDir_input, 'FSMASK'))
    if not op.exists(op.join(dstDir_input, 'DIFF')):
        os.makedirs(op.join(dstDir_input, 'DIFF'))
    if not op.exists(op.join(dstDir_input, 'BVAL')):
        os.makedirs(op.join(dstDir_input, 'BVAL'))
    if not op.exists(op.join(dstDir_input, 'BVEC')):
        os.makedirs(op.join(dstDir_input, 'BVEC'))
    # Create the destination paths
    dst_path_ANAT = op.join(dstDir_input, 'ANAT', 'T1.nii.gz')
    if ((container != 'rtp2-preproc')
        and (
        precontainer_anat.split('_')[0] == 'anatrois'
        and int(precontainer_anat.split('.')[1]) < 6
    )):
        dst_path_FSMASK = op.join(dstDir_input, 'FSMASK', 'brainmask.nii.gz')
    else:
        dst_path_FSMASK = op.join(dstDir_input, 'FSMASK', 'brain.nii.gz')
    dst_path_DIFF = op.join(dstDir_input, 'DIFF', 'dwiF.nii.gz')
    dst_path_BVAL = op.join(dstDir_input, 'BVAL', 'dwiF.bval')
    dst_path_BVEC = op.join(dstDir_input, 'BVEC', 'dwiF.bvec')
    # Create the symbolic links
    force_symlink(src_path_ANAT, dst_path_ANAT, force)
    force_symlink(src_path_FSMASK, dst_path_FSMASK, force)
    force_symlink(src_path_DIFF, dst_path_DIFF, force)
    force_symlink(src_path_BVAL, dst_path_BVAL, force)
    force_symlink(src_path_BVEC, dst_path_BVEC, force)
    logger.info(
        '\n'
        + '-----------------The rtppreproc symlinks created\n',
    )
    # check_create_bvec_bval（force) one of the todo here
    if rpe:
        # the reverse direction nii.gz
        src_path_RDIF = layout.get(
            subject=sub, session=ses, extension='nii.gz',
            suffix='dwi', direction=RPE_direction, return_type='filename',
        )[0]

        # the reverse direction bval
        src_path_RBVL_lst = layout.get(
            subject=sub, session=ses, extension='bval', suffix='dwi',
            direction=RPE_direction, return_type='filename',
        )

        if len(src_path_RBVL_lst) == 0:
            src_path_RBVL = src_path_RDIF.replace('dwi.nii.gz', 'dwi.bval')
            logger.warning('\n the bval Reverse file are not find by BIDS, create empty file !!!')
        else:
            src_path_RBVL = layout.get(
                subject=sub, session=ses, extension='bval',
                suffix='dwi', direction=RPE_direction, return_type='filename',
            )[0]

        # the reverse direction bvec
        src_path_RBVC_lst = layout.get(
            subject=sub, session=ses, extension='bvec',
            suffix='dwi', direction=RPE_direction, return_type='filename',
        )
        if len(src_path_RBVC_lst) == 0:
            src_path_RBVC = src_path_RDIF.replace('dwi.nii.gz', 'dwi.bvec')
            logger.warning('\n the bvec Reverse file are not find by BIDS, create empty file !!!')
        else:
            src_path_RBVC = layout.get(
                subject=sub, session=ses, extension='bvec',
                suffix='dwi', direction=RPE_direction, return_type='filename',
            )[0]

        # If bval and bvec do not exist because it is only b0-s, create them
        # (it would be better if dcm2niix would output them but...)
        # build the img matrix according to the shape of nii.gz
        img = nib.load(src_path_RDIF)  # type: ignore
        volumes = img.shape[3]  # type: ignore
        # if one of the bvec and bval are not there, re-write them
        if (not op.isfile(src_path_RBVL)) or (not op.isfile(src_path_RBVC)):
            # Write bval file
            f = open(src_path_RBVL, 'x')
            f.write(volumes * '0 ')
            f.close()
            logger.warning('\n Finish writing the bval Reverse file with all 0 !!!')
            # Write bvec file
            f = open(src_path_RBVC, 'x')
            f.write(volumes * '0 ')
            f.write('\n')
            f.write(volumes * '0 ')
            f.write('\n')
            f.write(volumes * '0 ')
            f.write('\n')
            f.close()
            logger.warning('\n Finish writing the bvec Reverse file with all 0 !!!')

        if not op.exists(op.join(dstDir_input, 'RDIF')):
            os.makedirs(op.join(dstDir_input, 'RDIF'))
        if not op.exists(op.join(dstDir_input, 'RBVL')):
            os.makedirs(op.join(dstDir_input, 'RBVL'))
        if not op.exists(op.join(dstDir_input, 'RBVC')):
            os.makedirs(op.join(dstDir_input, 'RBVC'))

        dst_path_RDIF = op.join(dstDir_input, 'RDIF', 'dwiR.nii.gz')
        dst_path_RBVL = op.join(dstDir_input, 'RBVL', 'dwiR.bval')
        dst_path_RBVC = op.join(dstDir_input, 'RBVC', 'dwiR.bvec')

        force_symlink(src_path_RDIF, dst_path_RDIF, force)
        force_symlink(src_path_RBVL, dst_path_RBVL, force)
        force_symlink(src_path_RBVC, dst_path_RBVC, force)
        logger.info(
            '\n'
            + '---------------The rtppreproc rpe=True symlinks created',
        )

    if 'qmap' in required_inputfiles:
        qmap_fname = lc_config['container_specific'][container]['qmap_fname']
        qmap_dir_name = lc_config['container_specific'][container]['qmap_dir_name']
        qmap_analysis_name = lc_config['container_specific'][container]['qmap_analysis_name']
        qmap_path = op.join(
            basedir,
            bidsdir_name,
            'derivatives',
            f'{qmap_dir_name}',
            'analysis-' + qmap_analysis_name,
            'sub-' + sub,
            'ses-' + ses,
            'output',
        )
        logger.info(
            '\n'
            + f'---the patter of fs.zip filename we are searching is {qmap_fname}\n'
            + f'---the directory we are searching for is {qmap_path}',
        )
        logger.debug(
            '\n'
            + f'the tpye of patter is {type(qmap_fname)}',
        )
        zips = []
        for filename in os.listdir(qmap_path):
            if filename.endswith('.zip') and re.match(qmap_fname, filename):
                zips.append(filename)
        if len(zips) == 0:
            logger.error(
                '\n'
                + f'There are no files with pattern: {qmap_fname} in {qmap_path}, \
                     we will listed potential zip file for you',
            )
            raise FileNotFoundError('qmap_path is empty, no previous analysis was found')
        elif len(zips) == 1:
            src_path_qmap = op.join(qmap_path, zips[0])
        else:
            zips_by_time = sorted(zips, key=op.getmtime)
            answer = input(
                f'Do you want to use the newset fs.zip: \n{zips_by_time[-1]} \n we get for you? \
                      \n input y for yes, n for no',
            )
            if answer in 'y':
                src_path_qmap = zips_by_time[-1]
            else:
                logger.error(
                    '\n' + 'An error occurred'
                    + zips_by_time + '\n'  # type: ignore
                    + 'no target preanalysis.zip file exist, please check the config_lc.yaml file',
                )
                sys.exit(1)

        dst_fname_qmap = config_json_instance['inputs']['qmap']['location']['name']
        dst_path_qmap = op.join(dstDir_input, 'qmap', dst_fname_qmap)
        if not op.exists(op.join(dstDir_input, 'qmap')):
            os.makedirs(op.join(dstDir_input, 'qmap'))
        force_symlink(src_path_qmap, dst_path_qmap, force)

    return


# %%
def rtppipeline(dict_store_cs_configs, analysis_dir, lc_config, sub, ses):
    """"""

    """
    Parameters
    ----------
    lc_config : dict
        the lc_config dictionary from _read_config
    sub : str
        the subject name looping from df_subSes
    ses : str
        the session name looping from df_subSes.
    container_specific_config_path : str

    Returns
    -------
    none, create symbolic links

    """
    # define local variables from config dict
    # input from get_parser
    # general level variables:
    basedir = lc_config['general']['basedir']
    container = lc_config['general']['container']
    bidsdir_name = lc_config['general']['bidsdir_name']
    force = (lc_config['general']['force'])

    # rtppipeline specefic variables
    precontainer_anat = lc_config['container_specific'][container]['precontainer_anat']
    anat_analysis_name = lc_config['container_specific'][container]['anat_analysis_name']
    precontainer_preproc = lc_config['container_specific'][container]['precontainer_preproc']
    preproc_analysis_num = lc_config['container_specific'][container]['preproc_analysis_name']

    # Create input and output directory for this container,
    # the dstDir_output should be empty, the dstDir_input should contains all the symlinks
    dstDir_input = op.join(
        analysis_dir,
        'sub-' + sub,
        'ses-' + ses,
        'input',
    )
    dstDir_output = op.join(
        analysis_dir,
        'sub-' + sub,
        'ses-' + ses,
        'output',
    )

    if not op.exists(dstDir_input):
        os.makedirs(dstDir_input)
    if not op.exists(dstDir_output):
        os.makedirs(dstDir_output)

    # read json, this json is already written from previous preparasion step
    json_under_analysis_dir = dict_store_cs_configs['config_path']
    config_json_instance = json.load(open(json_under_analysis_dir))
    required_inputfiles = config_json_instance['inputs'].keys()

    # required fields are：
    # anatomical bval bvec dwi fs
    # optional fields are :
    # tractparams fsmask qmap_zip
    # things from anatrois/freesurferator:
    # anatomical
    # fs
    # The source directory
    src_dir_fs = op.join(
        basedir,
        bidsdir_name,
        'derivatives',
        f'{precontainer_anat}',
        'analysis-' + anat_analysis_name,
        'sub-' + sub,
        'ses-' + ses,
        'output',
    )
    src_path_fszip = op.join(src_dir_fs, 'fs.zip')

    src_dir_preproc = op.join(
        basedir,
        bidsdir_name,
        'derivatives',
        precontainer_preproc,
        'analysis-' + preproc_analysis_num,
        'sub-' + sub,
        'ses-' + ses,
        'output',
    )

    # The source file
    src_path_anat = op.join(src_dir_preproc, 't1.nii.gz')
    src_path_bval = op.join(src_dir_preproc, 'dwi.bvals')
    src_path_bvec = op.join(src_dir_preproc, 'dwi.bvecs')
    src_path_dwi = op.join(src_dir_preproc, 'dwi.nii.gz')

    if not op.exists(op.join(dstDir_input, 'anatomical')):
        os.makedirs(op.join(dstDir_input, 'anatomical'))
    if not op.exists(op.join(dstDir_input, 'fs')):
        os.makedirs(op.join(dstDir_input, 'fs'))
    if not op.exists(op.join(dstDir_input, 'dwi')):
        os.makedirs(op.join(dstDir_input, 'dwi'))
    if not op.exists(op.join(dstDir_input, 'bval')):
        os.makedirs(op.join(dstDir_input, 'bval'))
    if not op.exists(op.join(dstDir_input, 'bvec')):
        os.makedirs(op.join(dstDir_input, 'bvec'))

    # Create the destination file
    dst_path_anat = op.join(dstDir_input, 'anatomical', 'T1.nii.gz')
    dst_path_fszip = op.join(dstDir_input, 'fs', 'fs.zip')
    dst_path_dwi = op.join(dstDir_input, 'dwi', 'dwi.nii.gz')
    dst_path_bval = op.join(dstDir_input, 'bval', 'dwi.bval')
    dst_path_bvec = op.join(dstDir_input, 'bvec', 'dwi.bvec')

    force_symlink(src_path_anat, dst_path_anat, force)
    force_symlink(src_path_fszip, dst_path_fszip, force)
    force_symlink(src_path_dwi, dst_path_dwi, force)
    force_symlink(src_path_bvec, dst_path_bvec, force)
    force_symlink(src_path_bval, dst_path_bval, force)

    logger.info(
        '\n'
        + '-----------------The required rtp2-pipeline symlinks created\n',
    )
    if 'tractparams' in required_inputfiles:
        fname_tractparams = config_json_instance['inputs']['tractparams']['location']['name']
        src_path_tractparams = op.join(analysis_dir, fname_tractparams)
        dst_path_tractparams = op.join(dstDir_input, 'tractparams', 'tractparams.csv')
        # the tractparams check, at the analysis folder
        tractparam_df, _ = read_df(src_path_tractparams)
        check.check_tractparam(lc_config, sub, ses, tractparam_df)
        if not op.exists(op.join(dstDir_input, 'tractparams')):
            os.makedirs(op.join(dstDir_input, 'tractparams'))
        # Create the symbolic links
        force_symlink(src_path_tractparams, dst_path_tractparams, force)

    if 'fsmask' in required_inputfiles:

        fname_fsmask = config_json_instance['inputs']['fsmask']['location']['name']
        src_path_fsmask = op.join(analysis_dir, fname_fsmask)
        dst_path_fsmask = op.join(dstDir_input, 'fsmask', fname_fsmask)

        if not op.exists(op.join(dstDir_input, 'fsmask')):
            os.makedirs(op.join(dstDir_input, 'fsmask'))
        force_symlink(src_path_fsmask, dst_path_fsmask, force)

    if 'qmap' in required_inputfiles:
        qmap_fname = lc_config['container_specific'][container]['qmap_fname']
        qmap_dir_name = lc_config['container_specific'][container]['qmap_dir_name']
        qmap_analysis_name = lc_config['container_specific'][container]['qmap_analysis_name']
        qmap_path = op.join(
            basedir,
            bidsdir_name,
            'derivatives',
            f'{qmap_dir_name}',
            'analysis-' + qmap_analysis_name,
            'sub-' + sub,
            'ses-' + ses,
            'output',
        )
        logger.info(
            '\n'
            + f'---the patter of fs.zip filename we are searching is {qmap_fname}\n'
            + f'---the directory we are searching for is {qmap_path}',
        )
        logger.debug(
            '\n'
            + f'the tpye of patter is {type(qmap_fname)}',
        )
        zips = []
        for filename in os.listdir(qmap_path):
            if filename.endswith('.zip') and re.match(qmap_fname, filename):
                zips.append(filename)
        if len(zips) == 0:
            logger.error(
                '\n'
                + f'There are no files with pattern: {qmap_fname} in {qmap_path}, \
                     we will listed potential zip file for you',
            )
            raise FileNotFoundError('qmap_path is empty, no previous analysis was found')
        elif len(zips) == 1:
            src_path_qmap = op.join(qmap_path, zips[0])
        else:
            zips_by_time = sorted(zips, key=op.getmtime)
            answer = input(
                f'Do you want to use the newset fs.zip: \n{zips_by_time[-1]} \n we get for you?\
                      \n input y for yes, n for no',
            )
            if answer in 'y':
                src_path_qmap = zips_by_time[-1]
            else:
                logger.error(
                    '\n' + 'An error occurred'
                    + zips_by_time + '\n'  # type: ignore
                    + 'no target preanalysis.zip file exist, please check the config_lc.yaml file',
                )
                sys.exit(1)

        dst_fname_qmap = config_json_instance['inputs']['qmap']['location']['name']
        dst_path_qmap = op.join(dstDir_input, 'qmap', dst_fname_qmap)
        if not op.exists(op.join(dstDir_input, 'qmap')):
            os.makedirs(op.join(dstDir_input, 'qmap'))
        force_symlink(src_path_qmap, dst_path_qmap, force)

    return
