% ADD FSL TO THE PATH BEFORE LAUNCHING MATLAB
% then do
tbUse BCBLViennaSoft;
% this step is to add pressurfer and NORDIC_RAW into the path so that you
% can use it


%if system('fslroi')==127
%    error("didn't load fsl");
%end

%if system('3dTstat')==127
%    error("didn't load afni");
%end
%%%%%%%%%% EDIT THIS %%%%%%%%%%
%clc;
%clear all;
% VIENNA
% baseP = '/ceph/mri.meduniwien.ac.at/projects/physics/fmri/data/bcblvie22/BIDS';

% BCBL
% tbUse BCBLViennaSoft;
baseP = fullfile('/bcbl/home/public/Gari/VOTCLOC','BIDS');
% GENERIC
subs = {'05'}; %{'bt001','bt002'};
sess = {'day3PF'};
tasks={'fLoc'};
run={'01', '02', '03', '04', '05', '06'};
%
nordic_scans_end = 1;
desiged_measurements =120;
force = false;
doNORDIC = true;
dotsnr = false;
doDetrendtsnr = false ; % do I do tsnr based on detrend result or not
%%%%%%%%%%%
tbPath = fullfile(bvRP,'..');
spm12Path = fullfile(tbPath, 'spm12');
bidsmatlab_path=fullfile(tbPath,'bids-matlab');
addpath(bidsmatlab_path);
addpath(spm12Path);
fmamtPath = fullfile(tbPath, 'freesurfer_mrtrix_afni_matlab_tools'); % tbUse if not installed
addpath(genpath(fmamtPath));
addpath(genpath(fullfile(baseP,'..','code')))

nordicpath=fullfile(tbPath,'NORDIC_Raw');
addpath(genpath(nordicpath));

presurferpath=fullfile(tbPath,'presurfer');
addpath(genpath(presurferpath));

setenv('FSLOUTPUTTYPE', 'NIFTI_GZ')

%% nordic + tsnr
% algorithm:
% get all the magnitude files,
% construct a looping struct
% for all the runs, do NORDIC
% after NORDIC, copy files, and rename files

for subI=1:length(subs)
    sub = ['sub-',subs{subI}];
    for sesI=1:length(sess)
        ses = ['ses-',sess{sesI}];
        % loop to get the working session path
        sesP = fullfile(baseP, sub, ses);
        system(['chmod 777 ', fullfile(sesP, 'func')]);

        % get all the mags and phases files in the working directory
        mags= dir(fullfile(sesP, 'func', '*_magnitude.nii.gz'));
        for magI=1:length(mags)
            try
                % define file names
                fn_magn_in  = fullfile(mags(magI).folder, mags(magI).name);
                fn_phase_in = strrep(fn_magn_in, '_magnitude', '_phase');

                if ~exist(strrep(fn_magn_in, '.nii.gz', '_orig.nii.gz'), 'file')
                    info = niftiinfo(fn_magn_in);
                    system(['cp ', fn_magn_in, ' ', strrep(fn_magn_in, '.nii.gz', '_orig.nii.gz')]);
                    system(['cp ', fn_phase_in, ' ', strrep(fn_phase_in, '.nii.gz', '_orig.nii.gz')]);
                    system(['chmod 777 ', fn_phase_in, ' ', fn_magn_in, ' ']);

                    % maintain 1 volumns for nordic and remove the extra
                    if nordic_scans_end > 1
                        system(['fslroi ', fn_magn_in, ' ', fn_magn_in, ' 0 -1 0 -1 0 -1 0 ', num2str(info.ImageSize(end)-(nordic_scans_end-1))]);
                        system(['fslroi ', fn_phase_in, ' ', fn_phase_in, ' 0 -1 0 -1 0 -1 0 ', num2str(info.ImageSize(end)-(nordic_scans_end-1))]);
                    end
                    system(['fslmaths ', fn_magn_in,  ' ', fn_magn_in,  ' -odt float']);
                    system(['fslmaths ', fn_phase_in, ' ', fn_phase_in, ' -odt float']);
                end
            end
        end

        % prepare ARG struct for each run of the the magnitude.nii.gz
        I = 1; %ARG file index
        for magI=1:length(mags)
            % define file names
            fn_magn_in  = fullfile(mags(magI).folder, mags(magI).name);
            fn_phase_in = strrep(fn_magn_in, '_magnitude', '_phase');
            fn_out      = strrep(fn_magn_in, '_magnitude', '_bold');

            if ~(exist(strrep(fn_out, '.nii.gz', 'magn.nii'), 'file') || exist(fn_out,'file')) && doNORDIC

                ARG(I).temporal_phase = 1;
                ARG(I).phase_filter_width = 10;
                ARG(I).noise_volume_last = 1;
                [ARG(I).DIROUT,fn_out_name,~] =fileparts(fn_out);
                ARG(I).DIROUT = [ARG(I).DIROUT, '/'];
                ARG(I).make_complex_nii = 1;
                ARG(I).save_gfactor_map = 1;

                file(I).phase = fn_phase_in;
                file(I).magni = fn_magn_in;
                %file.out has no .gz only nii
                file(I).out   = strrep(fn_out_name, '.nii', '');

                I = I + 1;

            end
        end

        % Do nordic on all functional runs under this session using parfor
        if exist('ARG', 'var')
            for i=1:length(ARG)
                %              try
                sprintf("Processing Nordic on run- 0%s", i);
                NIFTI_NORDIC(file(i).magni, file(i).phase,file(i).out,ARG(i));
                %              end
            end
            clear ARG file
        end

        % the output of NORDIC is a filed ended with _bold.nii.gzmagn.nii
        for magI=1:length(mags)
            %             try
            % define file names

            fn_magn_in  = fullfile(mags(magI).folder, mags(magI).name);
            fn_phase_in = strrep(fn_magn_in, '_magnitude', '_phase');
            fn_out      = strrep(fn_magn_in, '_magnitude', '_bold');
            gfactorFile = strrep(strrep(fn_out, '.nii.gz', '.nii'),[sub '_ses'],['gfactor_' sub '_ses']);

            if exist(gfactorFile, 'file') && doNORDIC
                % clean up
                info = niftiinfo(strrep(fn_out, '.nii.gz', 'magn.nii'));
                % remove the last one
                system(['fslroi ', strrep(fn_out, '.nii.gz', 'magn.nii'), ' ', fn_out, ' 0 -1 0 -1 0 -1 0 ', num2str(info.ImageSize(end)-1)]);
                gzip(gfactorFile);
                % there will be a file called _boldphase.nii, we didn't
                % remove it
                system(['rm ', strrep(fn_out, '.nii.gz', 'magn.nii'), ' ', gfactorFile, ' ' , strrep(fn_out, '.nii.gz', 'phase.nii')]);
                system(['mv ', strrep(gfactorFile, '.nii', '.nii.gz'), ' ', strrep(strrep(strrep(gfactorFile, '.nii', '.nii.gz'), '_bold', '_gfactor'), 'gfactor_', '')]);
            end

            % copy the events.tsv
            if ~exist(strrep(fn_magn_in, '_magnitude.nii.gz', '_bold.json'), 'file')
                system(['cp ', strrep(fn_magn_in, '_magnitude.nii.gz', '_magnitude.json'), ' ', ...
                    strrep(fn_magn_in, '_magnitude.nii.gz', '_bold.json')]);
            end

            if ~doNORDIC
                info = niftiinfo(fn_magn_in);
                system(['cp ',fn_magn_in, ' ',  strrep(fn_magn_in, '_magnitude', '_bold')]);
                system(['chmod 755 ', strrep(fn_magn_in, '_magnitude', '_bold')]);
                system(['fslroi ', strrep(fn_magn_in, '_magnitude', '_bold'), ' ', ...
                    strrep(fn_magn_in, '_magnitude', '_bold'), ' 0 -1 0 -1 0 -1 0 ', num2str(info.ImageSize(end)-nordic_scans_end)]);
            end
            %             end

            % rename sbref
            sbref_mags = dir(fullfile(sesP, 'func', '*_part-mag_sbref.nii.gz'));
            if ~isempty(sbref_mags)
                for sbref_magI = 1:length(sbref_mags)
                    sbref_mag = fullfile(sbref_mags(sbref_magI).folder, sbref_mags(sbref_magI).name);
                    if ~exist(strrep(sbref_mag, '_part-mag_sbref.nii.gz', '_sbref.json'), 'file')
                        system(['cp ', sbref_mag, ' ', strrep(sbref_mag, '_part-mag', '')]);
                        system(['cp ', strrep(sbref_mag, '.nii.gz', '.json'), ' ', ...
                            strrep(sbref_mag, '_part-mag_sbref.nii.gz', '_sbref.json')]);
                    end
                end
            end
        end


        if dotsnr
            bolds = dir(fullfile(sesP, 'func', '*_bold.nii.gz'));
            mags  = dir(fullfile(sesP, 'func', '*_magnitude.nii.gz'));
            bolds(contains({bolds.name}, 'gfactor')) = [];


            parfor nb=1:length(bolds)

                % Define file names
                magFile  = fullfile(mags(nb).folder, mags(nb).name);
                boldFile = fullfile(bolds(nb).folder, bolds(nb).name);


                tsnrFile = strrep(boldFile,'bold','tsnr_postNordic');
                magtsnrFile = strrep(boldFile,'bold','tsnr_preNordic');
                gfactorFile = strrep(boldFile,'bold','gfactor');
                tsnrGfactorFile = strrep(gfactorFile,'gfactor','gfactorSameSpace');

                % pre NORDIC tSNR
                magData = single(niftiread(magFile));
                magHeader = niftiinfo(magFile);
                magtsnrData = mean(magData,4) ./ std(magData,1,4);
                magtsnrData(isnan(magtsnrData)) = 0;
                magHeader.ImageSize = size(magtsnrData);
                magHeader.PixelDimensions=magHeader.PixelDimensions(1:3);
                niftiwrite(magtsnrData, strrep(magtsnrFile, '.nii', ''), magHeader,'compressed',true)

                % post NORDIC tSNR
                boldData = niftiread(boldFile);
                boldHeader = niftiinfo(boldFile);
                tsnrData = mean(boldData,4) ./ std(boldData,1,4);
                boldHeader.ImageSize = size(tsnrData);
                boldHeader.PixelDimensions=boldHeader.PixelDimensions(1:3);
                niftiwrite(tsnrData, strrep(tsnrFile, '.nii', ''),boldHeader,'compressed',true)

                % Write g factor in same space
                gfactorData = niftiread(gfactorFile);
                gHeader = magHeader;
                gHeader.ImageSize=size(gfactorData);
                gHeader.PixelDimensions=gHeader.PixelDimensions(1:3);
                niftiwrite(gfactorData, strrep(tsnrGfactorFile, '.nii', ''), gHeader,'compressed',true)



            end
        end
    end
end
