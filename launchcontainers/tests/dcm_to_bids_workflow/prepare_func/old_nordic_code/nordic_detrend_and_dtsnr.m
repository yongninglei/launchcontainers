

% NORDIC detrend by Cesar, not quit understand it , but we used it for our
% checking
% DETREND NEW ADDED

for subI=1:length(subs)
    sub = ['sub-',subs{subI}];
    for sesI=1:length(sess); for runI=1:length(runs)
        ses = ['ses-',sess{sesI}];
        run = ['run-',runs{runI}];
        sesP = fullfile(baseP, sub, ses);

        if doDetrend

            bolds = dir(fullfile(sesP, 'func', '*bold.nii.gz'));
            mags  = dir(fullfile(sesP, 'func', '*magnitude.nii.gz'));

            for nb =1:length(bolds)
                % define the mag and bolds file needed for detrend
                magFile = fullfile(mags(nb).folder, mags(nb).name);
                magTmean = strrep(magFile,'magnitude','magnitude_Tmean');
                magDetrend = strrep(magFile,'magnitude','magnitude_Detrend');

                boldFile = fullfile(bolds(nb).folder, bolds(nb).name);
                boldTmean = strrep(boldFile,'bold','bold_Tmean');
                boldDetrend = strrep(boldFile,'bold','bold_Detrend');

                if ~exist(magTmean, 'file') && ~exist(magDetrend, 'file') || force
                    system(['3dTstat -mean -prefix ', magTmean, ' ',  magFile, ' -overwrite']);
                    system(['3dDetrend -polort 6 -prefix ', magDetrend, ' ', magFile, ' -overwrite']);
                    system(['3dcalc -a ',magDetrend,' -b ',magTmean,' -expr ''-a+b'' -prefix ',magDetrend, ' -overwrite']);
                else
                    fprintf("The file are already exist, no action")

                end

                if ~exist(boldTmean, 'file') && ~exist(boldDetrend, 'file') || force
                    system(['3dTstat -mean -prefix ', boldTmean, ' ',  boldFile, ' -overwrite']);
                    system(['3dDetrend -polort 6 -prefix ', boldDetrend, ' ', boldFile, ' -overwrite']);
                    system(['3dcalc -a ',boldDetrend,' -b ',boldTmean,' -expr ''-a+b'' -prefix ',boldDetrend, ' -overwrite']);
                else
                    fprintf("\n\nThe boldTean and boldDetrend files are already exist, no action")
                end
            end

        end

        if doDetrendtsnr && doDetrend

            bolds_Detrend = dir(fullfile(sesP, 'func', '*bold_Detrend.nii.gz'));
            mags_Detrend  = dir(fullfile(sesP, 'func', '*magnitude_Detrend.nii.gz'));
            bolds_Detrend(contains({bolds_Detrend.name}, 'gfactor')) = [];

            parfor nb=1:length(bolds_Detrend)
                fprintf("\n\n%s_%s_run-0%i",sub,ses,nb)
                % Define file names
                magFile_Detrend  = fullfile(mags_Detrend(nb).folder, mags_Detrend(nb).name);
                boldFile_Detrend = fullfile(bolds_Detrend(nb).folder, bolds_Detrend(nb).name);


                tsnrFile_Detrend = strrep(boldFile_Detrend,'bold_Detrend','tsnr_Detrend_postNordic');
                magtsnrFile_Detrend = strrep(boldFile_Detrend,'bold_Detrend','tsnr_Detrend_preNordic');
                % I didn't do detrend for gfactor, am I correct?
                gfactorFile_Detrend = strrep(boldFile_Detrend,'bold_Detrend','gfactor');
                tsnrGfactorFile_Detrend = strrep(gfactorFile_Detrend,'gfactor_Detrend','gfactorSameSpace_Detrend');

                % pre NORDIC tSNR
                magData_Detrend = single(niftiread(magFile_Detrend));
                magHeader_Detrend = niftiinfo(magFile_Detrend);
                magtsnrData_Detrend = mean(magData_Detrend,4) ./ std(magData_Detrend,1,4);
                magtsnrData_Detrend(isnan(magtsnrData_Detrend)) = 0;
                magHeader_Detrend.ImageSize = size(magtsnrData_Detrend);
                magHeader_Detrend.PixelDimensions=magHeader_Detrend.PixelDimensions(1:3);
                niftiwrite(magtsnrData_Detrend, strrep(magtsnrFile_Detrend, '.nii', ''), magHeader_Detrend,'compressed',true)

                % post NORDIC tSNR
                boldData_Detrend = niftiread(boldFile_Detrend);
                boldHeader_Detrend = niftiinfo(boldFile_Detrend);
                tsnrData_Detrend = mean(boldData_Detrend,4) ./ std(boldData_Detrend,1,4);
                boldHeader_Detrend.ImageSize = size(tsnrData_Detrend);
                boldHeader_Detrend.PixelDimensions=boldHeader_Detrend.PixelDimensions(1:3);
                niftiwrite(tsnrData_Detrend, strrep(tsnrFile_Detrend, '.nii', ''),boldHeader_Detrend,'compressed',true)

                % Write g factor in same space
                gfactorData_Detrend = niftiread(gfactorFile_Detrend);
                gHeader_Detrend = magHeader_Detrend;
                gHeader_Detrend.ImageSize=size(gfactorData_Detrend);
                gHeader_Detrend.PixelDimensions=gHeader_Detrend.PixelDimensions(1:3);
                niftiwrite(gfactorData_Detrend, strrep(tsnrGfactorFile_Detrend, '.nii', ''), gHeader_Detrend,'compressed',true)

            end
        end
    end;end
end
