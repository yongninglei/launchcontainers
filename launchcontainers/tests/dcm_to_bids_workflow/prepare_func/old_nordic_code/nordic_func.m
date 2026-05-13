% MP2RAGE presurfer
function  nordic_func(src_dir, output_dir, sub, ses, force)
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

    if ~exist(output_dir, 'dir')
       mkdir(output_dir)
    end

    sub=['sub-' sub];
    disp(sub)
    ses=['ses-' ses];
    disp(ses)
    tbPath = fullfile(bvRP,'..');
    spm12Path = fullfile(tbPath, 'spm12');
    bidsmatlab_path=fullfile(tbPath,'bids-matlab');
    addpath(bidsmatlab_path);
    addpath(spm12Path);
    fmamtPath = fullfile(tbPath, 'freesurfer_mrtrix_afni_matlab_tools'); % tbUse if not installed
    addpath(genpath(fmamtPath));
    presurferpath=fullfile(tbPath,'presurfer');
    addpath(genpath(presurferpath));
    setenv('FSLOUTPUTTYPE', 'NIFTI_GZ')

    src_sesP = fullfile(src_dir, sub, ses);
    out_sesP = fullfile(output_dir, sub, ses);
    if ~exist(out_sesP, 'dir')
       mkdir(out_sesP)
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
    runs = arrayfun(@(x) sprintf('%02d', x), 1:num_runs, 'UniformOutput', false);
    sprintf('Number of runs are %s', num_runs)

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
    end
end
