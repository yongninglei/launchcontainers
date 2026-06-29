% MP2RAGE presurfer
function  presurferT1(tbPath,src_dir, output_dir, sub, ses, force)
% MIT License

% Copyright (c) 2024-2025 Yongning Lei

% Permission is hereby granted, free of charge, to any person obtaining a copy of this software
% and associated documentation files (the "Software"), to deal in the Software without restriction,
% including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
% subject to the following conditions:

% The above copyright notice and this permission notice shall be included in all copies or substantial
% portions of the Software.

    %% Below is the one check if all the things are being loaded, but because we were run it from the cli, I don't think it is needed
    %{
    if system('fslroi')==127
       error("didn't load fsl");
    else
        disp('FSL is loaded, continue');
    end

    if system('3dTstat')==127
       error("didn't load afni");
    else
        disp('AFNI is loaded, continue');
    end
    %}
    %% general loading process
    spm12Path = fullfile(tbPath, 'spm12');
    bidsmatlab_path=fullfile(tbPath,'bids-matlab');
    addpath(bidsmatlab_path);
    addpath(spm12Path);
    fmamtPath = fullfile(tbPath, 'freesurfer_mrtrix_afni_matlab_tools'); % tbUse if not installed
    addpath(genpath(fmamtPath));
    presurferpath=fullfile(tbPath,'presurfer');
    addpath(genpath(presurferpath));
    setenv('FSLOUTPUTTYPE', 'NIFTI_GZ')
    %% define function needed var
    if ~exist(output_dir, 'dir')
       mkdir(output_dir)
    end
    %{
    testing bids, first contrast a bids filter
    filters = struct('sub', '01', ...
                     'ses', '01', ...
                     'extension', '.nii.gz', ...
                     'suffix', 'uni');
    Ok it is not useful at all, because, T1_uni is not the thing BIDS can
    query, one way is to change the filename of the T1s..., but is seems
    too much work
    %}

    sub=['sub-' sub];
    disp(sub)
    ses=['ses-' ses];
    disp(ses)

    src_sesP = fullfile(src_dir, sub, ses);
    out_sesP = fullfile(output_dir, sub, ses);
    if ~exist(out_sesP, 'dir')
       mkdir(out_sesP)
    end
    if ~exist(fullfile(out_sesP,'anat'), 'dir')
       mkdir(fullfile(out_sesP,'anat'))
    end

    sprintf('The source dir is %s and the outputdir is %s', src_sesP, out_sesP)

    % get all the files
    % Path to the anat folder
    anat_dir = fullfile(src_dir, sub, ses, 'anat');

    % Detect all T1w.nii.gz files
    UNI_pattern = fullfile(anat_dir, '*_T1_uni.nii.gz');
    UNI_files = dir(UNI_pattern);
    % Get the number of runs
    num_runs = length(UNI_files);
    disp(num_runs)
    runs = arrayfun(@(x) sprintf('%02d', x), 1:num_runs, 'UniformOutput', false);
    sprintf('Number of runs are %s', num_runs)
    sprintf('Force is %s', force)
    for runI=1:length(runs)
        run = ['run-',runs{runI}];
        %% first run presurfer to denoise mp2rage images
        presurfer_result= fullfile(src_sesP, 'anat', 'presurf_MPRAGEise',[sub,'_',ses,'_',run,'_T1_uni_MPRAGEised.nii']);
        T1w_out = fullfile(out_sesP, 'anat', [sub,'_',ses,'_',run,'_T1w.nii']);
        if exist(presurfer_result) && ~force
            disp('There are uncleaned presurfer result, move rename and clean up! ')
            % move, rename, clean up
            system(['cp ', presurfer_result, ' ', T1w_out]);
            pause(2);
            gzip(T1w_out);
            system(['rm ', T1w_out]);
            system(['rm -r ', fullfile(src_sesP, 'anat', 'presurf_MPRAGEise')]);
        end

        if ~exist([T1w_out,'.gz'], 'file') || force
            % define the files
            sprintf('Going to run presurfer to create %s', T1w_out)
            UNI  = fullfile(src_sesP, 'anat', [sub,'_',ses,'_',run,'_T1_uni.nii']);
            INV2 = fullfile(src_sesP, 'anat', [sub,'_',ses,'_',run,'_T1_inv2.nii']);
            try
                % unzip the data
                gunzip([UNI,'.gz'])
                gunzip([INV2,'.gz'])

                % STEP - 0 : (optional) MPRAGEise UNI
                UNI_out = presurf_MPRAGEise(INV2,UNI);

                % move, rename, clean up
                system(['cp ', UNI_out, ' ', T1w_out]);
                pause(2);
                gzip(T1w_out);
                system(['rm ', UNI, ' ', INV2]);
                system(['rm -r ', fullfile(src_sesP, 'anat', 'presurf_MPRAGEise')]);
            catch
                warning("presurfer may not be launched correctly")
            end
        end
        T1w_json = strrep(T1w_out,'.nii','.json');
        if ~exist(T1w_json, 'file') || force
            % define the files
            sprintf('Going to copy UNI.json to create %s', T1w_json)
            UNI  = fullfile(src_sesP, 'anat', [sub,'_',ses,'_',run,'_T1_uni.nii']);
            UNI_json  = strrep(UNI,'.nii','.json');
            try
                % move, rename, clean up
                system(['cp ', UNI_json, ' ', T1w_json]);
                pause(2);
            catch
                warning("T1w.json is NOT being copied correctly")
            end
        end
    end
    % cp the T2w to target place
    % T2w we only have 1 run so runI=1
    runT2='run-01';
    T2w_in = fullfile(src_sesP, 'anat', [sub,'_',ses,'_',runT2,'_T2w.nii.gz']);
    T2w_json_in = strrep(T2w_in,'.nii.gz','.json');
    T2w_out = fullfile(out_sesP, 'anat', [sub,'_',ses,'_',runT2,'_T2w.nii.gz']);
    T2w_json_out = strrep(T2w_out,'.nii.gz','.json');
    if ~exist(T2w_out, 'file') || force
        try
            % move, rename, clean up
            system(['cp ', T2w_in, ' ', T2w_out]);
            pause(2);
        catch
            warning("T2w is NOT being copied correctly")
        end
    end
    if ~exist(T2w_json_out, 'file') || force
        try
            % move, rename, clean up
            system(['cp ', T2w_json_in, ' ', T2w_json_out]);
            pause(2);
        catch
            warning("T2w json is NOT being copied correctly")
        end
    end
end
