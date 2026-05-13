function prepare_nordic_dwi_nifti(src_magnitude, nordic_scans_end, doNORDIC, force)
% MIT License
% Copyright (c) 2024-2025 Yongning Lei
%
% Permission is hereby granted, free of charge, to any person obtaining a copy of this software
% and associated documentation files (the "Software"), to deal in the Software without restriction,
% including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so,
% subject to the following conditions:
%
% The above copyright notice and this permission notice shall be included in all copies or substantial
% portions of the Software.
%
% This function prepares DWI data for NORDIC processing.
%
% In an unprocessed DWI folder we expect:
%   dir-AP: magnitude, phase, bvec, bval, json
%   dir-PA: magnitude, phase, json  (NO bvec/bval)
%
% Logic (4 cases):
%   Case 1: doNORDIC  + orig exist  -> delete working files, restore from orig, then process
%   Case 2: doNORDIC  + no orig     -> create backups, then process (normal first-run)
%   Case 3: !doNORDIC + no orig     -> nothing to do (caller handles copy)
%   Case 4: !doNORDIC + orig exist  -> nothing to do (caller copies orig to target)
%
% Processing steps (Cases 1, 2 only):
%   1. Remove noise scans if present
%   2. Update bvec/bval to match new volume count (AP only)
%   3. Change dtype to float for NORDIC

    %% ========== Define all related file paths ==========
    src_phase = strrep(src_magnitude, '_magnitude', '_phase');
    src_json  = strrep(src_magnitude, '_magnitude.nii.gz', '_magnitude.json');

    % Detect AP vs PA: AP has bvec/bval, PA does not
    src_bvec = strrep(src_magnitude, '_magnitude.nii.gz', '_magnitude.bvec');
    src_bval = strrep(src_magnitude, '_magnitude.nii.gz', '_magnitude.bval');
    is_AP = exist(src_bvec, 'file') && exist(src_bval, 'file');

    if is_AP
        fprintf('[prepare] Processing AP file: %s\n', src_magnitude);
    else
        fprintf('[prepare] Processing PA file: %s (no bvec/bval)\n', src_magnitude);
    end

    % Define backup (_orig) file paths
    mag_backup   = strrep(src_magnitude, '.nii.gz', '_orig.nii.gz');
    phase_backup = strrep(src_phase, '.nii.gz', '_orig.nii.gz');
    json_backup  = strrep(src_json, '.json', '_orig.json');
    if is_AP
        bvec_backup = strrep(src_bvec, '.bvec', '_orig.bvec');
        bval_backup = strrep(src_bval, '.bval', '_orig.bval');
    end

    %% ========== Check if ALL orig (backup) files exist ==========
    if is_AP
        orig_exist = exist(mag_backup, 'file') && exist(phase_backup, 'file') && ...
                     exist(json_backup, 'file') && exist(bvec_backup, 'file') && ...
                     exist(bval_backup, 'file');
    else
        orig_exist = exist(mag_backup, 'file') && exist(phase_backup, 'file') && ...
                     exist(json_backup, 'file');
    end

    fprintf('[prepare] doNORDIC=%d, orig_exist=%d, force=%d, is_AP=%d\n', ...
            doNORDIC, orig_exist, force, is_AP);

    %% ========== Case 3: !doNORDIC + no orig ==========
    if ~doNORDIC && ~orig_exist
        fprintf('[prepare] Case 3: !doNORDIC + no orig. Nothing to prepare.\n');
        return;
    end

    %% ========== Case 4: !doNORDIC + orig exist ==========
    if ~doNORDIC && orig_exist
        fprintf('[prepare] Case 4: !doNORDIC + orig exist. Caller will copy orig to target.\n');
        return;
    end

    %% ========== Case 1: doNORDIC + orig exist ==========
    if doNORDIC && orig_exist
        fprintf('[prepare] Case 1: doNORDIC + orig exist. Restoring from backups...\n');

        safe_delete(src_magnitude);
        safe_delete(src_phase);
        safe_delete(src_json);
        if is_AP
            safe_delete(src_bvec);
            safe_delete(src_bval);
        end

        system(['cp ', mag_backup,   ' ', src_magnitude]);
        system(['cp ', phase_backup, ' ', src_phase]);
        system(['cp ', json_backup,  ' ', src_json]);
        if is_AP
            system(['cp ', bvec_backup, ' ', src_bvec]);
            system(['cp ', bval_backup, ' ', src_bval]);
        end

        fprintf('[prepare] ** All source files restored from orig backups.\n');
    end

    %% ========== Case 2: doNORDIC + no orig ==========
    if doNORDIC && ~orig_exist
        fprintf('[prepare] Case 2: doNORDIC + no orig. Creating backups...\n');

        system(['cp ', src_magnitude, ' ', mag_backup]);
        system(['cp ', src_phase,     ' ', phase_backup]);
        system(['cp ', src_json,      ' ', json_backup]);
        if is_AP
            system(['cp ', src_bvec, ' ', bvec_backup]);
            system(['cp ', src_bval, ' ', bval_backup]);
        end

        fprintf('[prepare] ** Backups created for all source files.\n');
    end

    %% ========== Processing (Cases 1 & 2 reach here) ==========
    perm_files = [src_magnitude, ' ', src_phase];
    if is_AP
        perm_files = [perm_files, ' ', src_bvec, ' ', src_bval];
    end
    system(['chmod 777 ', perm_files]);

    % Validate backup dtype
    mag_backup_info = niftiinfo(mag_backup);
    fprintf('[prepare] Backup magnitude dtype: %s\n', mag_backup_info.Datatype);
    if ~strcmp(mag_backup_info.Datatype, 'uint16')
        warning('[prepare] Backup magnitude is NOT uint16 (%s). Proceeding with caution.', ...
                mag_backup_info.Datatype);
    end

    mag_info = niftiinfo(src_magnitude);
    fprintf('[prepare] Source magnitude dtype: %s, num_volumes: %d\n', ...
            mag_info.Datatype, mag_info.ImageSize(end));

    %% --- Step A: Remove noise scans if present ---
    if nordic_scans_end > 1
        new_volume_count = mag_info.ImageSize(end) - (nordic_scans_end - 1);
        system(['fslroi ', src_magnitude, ' ', src_magnitude, ...
               ' 0 -1 0 -1 0 -1 0 ', num2str(new_volume_count)]);
        system(['fslroi ', src_phase, ' ', src_phase, ...
               ' 0 -1 0 -1 0 -1 0 ', num2str(new_volume_count)]);
        fprintf('[prepare] ** Noise scans removed, keeping %d of %d volumes.\n', ...
                new_volume_count, mag_info.ImageSize(end));

        if is_AP
            update_bvec_bval(src_bvec, src_bval, new_volume_count);
        end

    elseif nordic_scans_end == 1
        fprintf('[prepare] ** Keeping all volumes including 1 noise scan.\n');
    else
        fprintf('[prepare] ** No noise scans to remove (nordic_scans_end=0).\n');
    end

    %% --- Step B: Change datatype to float for NORDIC ---
    mag_info = niftiinfo(src_magnitude);
    if ~(strcmp(mag_info.Datatype, 'single') && mag_info.BitsPerPixel == 32) || force
        system(['fslmaths ', src_magnitude, ' ', src_magnitude, ' -odt float']);
        system(['fslmaths ', src_phase,     ' ', src_phase,     ' -odt float']);

        after_info = niftiinfo(src_magnitude);
        fprintf('[prepare] ** Changed dtype to float: %s -> %s\n', ...
                mag_info.Datatype, after_info.Datatype);
    else
        fprintf('[prepare] dtype already float, skipping conversion.\n');
    end

    fprintf('[prepare] Done preparing: %s\n\n', src_magnitude);
end


%% ========== Helper: safe file deletion ==========
function safe_delete(filepath)
    if exist(filepath, 'file')
        delete(filepath);
        fprintf('  Deleted: %s\n', filepath);
    end
end


%% ========== Helper: update bvec/bval after volume removal ==========
function update_bvec_bval(bvec_file, bval_file, new_volume_count)
    bvec = load(bvec_file);
    if size(bvec, 1) == 3
        if size(bvec, 2) > new_volume_count
            bvec_new = bvec(:, 1:new_volume_count);
            fid = fopen(bvec_file, 'w');
            fprintf(fid, '%.6f ', bvec_new(1, :)); fprintf(fid, '\n');
            fprintf(fid, '%.6f ', bvec_new(2, :)); fprintf(fid, '\n');
            fprintf(fid, '%.6f ', bvec_new(3, :)); fprintf(fid, '\n');
            fclose(fid);
            fprintf('[prepare] ** bvec updated: %d -> %d volumes.\n', size(bvec, 2), new_volume_count);
        end
    else
        warning('[prepare] bvec format unexpected (%d rows), skipping.', size(bvec, 1));
    end

    bval = load(bval_file);
    if size(bval, 1) > 1
        bval = bval';
    end
    if size(bval, 2) > new_volume_count
        bval_new = bval(1:new_volume_count);
        fid = fopen(bval_file, 'w');
        fprintf(fid, '%.1f ', bval_new);
        fprintf(fid, '\n');
        fclose(fid);
        fprintf('[prepare] ** bval updated: %d -> %d volumes.\n', length(bval), new_volume_count);
    end
end
