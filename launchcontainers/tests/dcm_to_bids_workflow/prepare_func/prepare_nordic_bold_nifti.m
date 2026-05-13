function prepare_nordic_bold_nifti(src_magnitude,nordic_scans_end ,force)
% MIT License

% Copyright (c) 2024-2025 Yongning Lei

% Permission is hereby granted, free of charge, to any person obtaining a copy of this software
% and associated documentation files (the "Software"), to deal in the Software without restriction,
% including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
% subject to the following conditions:

% The above copyright notice and this permission notice shall be included in all copies or substantial
% portions of the Software.

% this function is replacing the step1 in the main function
% it have 2 jobs:
%   1. create backup files
%   2. change dtpye of src files,
%   3. print the changes
    % define name of the backup files
    src_phase= strrep(src_magnitude, '_magnitude', '_phase');
    mag_backup = strrep(src_magnitude, '.nii.gz', '_orig.nii.gz');
    phase_backup = strrep(src_phase, '.nii.gz', '_orig.nii.gz');

    disp('Change the src_mag and src_phase file permission to 777 \n');
    system(['chmod 777 ', src_phase, ' ', src_magnitude, ' ']);

    if ~(exist(mag_backup,'file') && exist(phase_backup,'file'))
        % if the backup files are not there, we will create it anyway
        system(['cp ', src_magnitude, ' ', mag_backup]);
        system(['cp ', src_phase, ' ', phase_backup]);
        disp('** backups for mag and phase created \n');
    elseif (exist(mag_backup,'file') && exist(phase_backup,'file')) && force
        % if the backup files aare there and we ask to force overwritte
        disp('backup files found, overwritting...... \n');
        delete(src_magnitude);
        delete(src_phase);
        system(['cp ', mag_backup, ' ', src_magnitude]);
        system(['cp ', phase_backup, ' ', src_phase]);

        system(['cp ', src_magnitude, ' ', mag_backup]);
        system(['cp ', src_phase, ' ', phase_backup]);
        disp('** backups for mag and phase created \n');
    else
         disp('backup files found, do nothing...... \n');
    end
    % after the above methods, there will always be 1 mag 1 phase 1 backup
    % for each

    % then on the mag and phase we do the operation:
    % 1. change dtype
    % 2. remove the last several noise scans to remain only 1
    mag_info=niftiinfo(src_magnitude);
    fprintf('The dtype of magnitude is %s \n ', mag_info.Datatype);
    mag_backup_info = niftiinfo(mag_backup);
    fprintf('The dtype of magnitude_orig is %s \n ', mag_backup_info.Datatype);
    if strcmp(mag_backup_info.Datatype, 'uint16')
        fprintf('backup file of %s is in original datatype, we are safe \n', src_magnitude)
        if nordic_scans_end > 1
            system(['fslroi ', src_magnitude, ' ', src_magnitude, ' 0 -1 0 -1 0 -1 0 ', num2str(mag_info.ImageSize(end)-(nordic_scans_end-1))]);
            system(['fslroi ', src_phase, ' ', src_phase, ' 0 -1 0 -1 0 -1 0 ', num2str(mag_info.ImageSize(end)-(nordic_scans_end-1))]);
            fprintf('** Extra noise scans of %s removed \n', src_magnitude);
        end
        if ~(strcmp(mag_info.Datatype, 'single') && mag_info.BitsPerPixel==32) || force
            % might have filetype problems, so needs to check here
            system(['fslmaths ', src_magnitude,  ' ', src_magnitude,  ' -odt float']);
            system(['fslmaths ', src_phase, ' ', src_phase, ' -odt float']);
            after_change_mag_info=niftiinfo(src_magnitude);
            fprintf('The dtype of magnitude file %s after fslmaths is %s \n ', src_magnitude,after_change_mag_info.Datatype);
            disp('** Change data format to float of src mag and src phase .nii.gz \n');
        else
            disp('The dtype of mag and phase input is already float, do nothing');
        end

        %{
        Here I add several testing datatype to it
        mag_short=strrep(src_magnitude,'_magnitude','_desc-short_magnitude');
        phase_short=strrep(mag_short,'_magnitude','_phase' );
        mag_uint16=strrep(src_magnitude,'_magnitude','_desc-uint16_magnitude');
        phase_uint16=strrep(mag_uint16,'_magnitude','_phase' );
        mag_int=strrep(src_magnitude,'_magnitude','_desc-int_magnitude');
        phase_int=strrep(mag_int,'_magnitude','_phase' );
%         % copy the src_magniture and rename it to mag short
%         system(['cp ', src_magnitude, ' ', mag_short]);
%         system(['cp ', src_phase, ' ', phase_short]);
%         system(['cp ', src_magnitude, ' ', mag_uint16]);
%         system(['cp ', src_phase, ' ', phase_uint16]);
        % change the dtype to short and uint16
        % in principle, those two things shouldn't make any difference
        system(['fslmaths ', src_magnitude,  ' ', mag_short,  ' -odt short']);
        system(['fslmaths ', src_phase, ' ', phase_short, ' -odt short']);
        system(['fslmaths ', src_magnitude,  ' ', mag_uint16,  ' -odt uint16']);
        system(['fslmaths ', src_phase, ' ', phase_uint16, ' -odt uint16']);
        system(['fslmaths ', src_magnitude,  ' ', mag_int,  ' -odt int']);
        system(['fslmaths ', src_phase, ' ', phase_int, ' -odt int']);

        mag_short_info=niftiinfo(mag_short);
        mag_uint_info=niftiinfo(mag_uint16);
        mag_int_info=niftiinfo(mag_int);
        fprintf('The dtype of magnitude short after fslmaths is %s \n ', mag_short_info.Datatype);
        fprintf('The dtype of magnitude uint16 after fslmaths is %s \n ', mag_uint_info.Datatype);
        fprintf('The dtype of magnitude int after fslmaths is %s \n ', mag_int_info.Datatype);
        disp('** Change data format to short and uint of src mag and src phase .nii.gz \n');
        %}
    else
        disp('WARNINIG! The backup magnitude file is not uint16 dtype, we might have an issue')
    end
end
