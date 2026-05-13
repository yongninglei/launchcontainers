function nordic_dwi(tbPath, src_dir, output_dir, sub, ses, nordic_scans_end, doNORDIC, force)
% MIT License
% Copyright (c) 2024-2026 Yongning Lei
% Modified for DWI processing with NORDIC

    % -----------------------------------------------------------------------
    % Session header
    % -----------------------------------------------------------------------
    fprintf('\n');
    fprintf('==========================================================\n');
    fprintf('  nordic_dwi  sub=%s  ses=%s\n', sub, ses);
    fprintf('  doNORDIC=%d  nordic_scans_end=%d  force=%d\n', ...
            doNORDIC, nordic_scans_end, force);
    fprintf('==========================================================\n');

    sub = ['sub-' sub];
    ses = ['ses-' ses];

    % -----------------------------------------------------------------------
    % Paths and toolboxes
    % -----------------------------------------------------------------------
    if ~exist(output_dir, 'dir'); mkdir(output_dir); end
    addpath(fullfile(tbPath, 'bids-matlab'));
    addpath(fullfile(tbPath, 'spm12'));
    addpath(genpath(fullfile(tbPath, 'freesurfer_mrtrix_afni_matlab_tools')));
    addpath(genpath(fullfile(src_dir, '..', 'code')));
    addpath(genpath('/bcbl/home/home_n-z/tlei/soft/launchcontainers/src/launchcontainers/MR_pipelines'));
    addpath(genpath(fullfile(tbPath, 'NORDIC_Raw')));
    setenv('FSLOUTPUTTYPE', 'NIFTI_GZ');

    if doNORDIC
        acq_label = 'nordic';
    else
        acq_label = 'magonly';
    end

    src_sesP = fullfile(src_dir, sub, ses, 'dwi');
    out_sesP = fullfile(output_dir, sub, ses, 'dwi');
    system(['chmod -R 777 ', src_sesP]);
    if ~exist(out_sesP, 'dir'); mkdir(out_sesP); end
    system(['chmod -R 777 ', out_sesP]);

    fprintf('  src : %s\n', src_sesP);
    fprintf('  dst : %s\n', out_sesP);
    fprintf('  acq : acq-%s\n', acq_label);

    % -----------------------------------------------------------------------
    % Discover runs
    % -----------------------------------------------------------------------
    dwimag_pattern = fullfile(src_sesP, '*_magnitude.nii.gz');
    src_mags = dir(dwimag_pattern);
    src_mags(contains({src_mags.name}, '_orig')) = [];
    num_runs = length(src_mags);

    fprintf('\n  Found %d magnitude file(s):\n', num_runs);
    for k = 1:num_runs
        fprintf('    [%d] %s\n', k, src_mags(k).name);
    end

    % -----------------------------------------------------------------------
    %% STEP 1 — prepare mag/phase for NORDIC (backup + trim + dtype)
    % -----------------------------------------------------------------------
    fprintf('\n----------------------------------------------------------\n');
    fprintf('  STEP 1 — prepare mag/phase (backup, trim, dtype convert)\n');
    fprintf('----------------------------------------------------------\n');
    time_start = datetime('now');

    step1_logs = cell(1, num_runs);
    parfor src_magI = 1:num_runs
        mag_in = fullfile(src_mags(src_magI).folder, src_mags(src_magI).name);
        orig   = strrep(mag_in, '.nii.gz', '_orig.nii.gz');
        msgs = {};
        msgs{end+1} = sprintf('  [run %d/%d] %s', src_magI, num_runs, src_mags(src_magI).name);
        msgs{end+1} = sprintf('    IN  : %s', mag_in);
        msgs{end+1} = sprintf('    ORIG: %s  (backup)', orig);
        prepare_nordic_dwi_nifti(mag_in, nordic_scans_end, doNORDIC, force);
        msgs{end+1} = sprintf('    DONE: dtype→float, noise scans trimmed (nordic_scans_end=%d)', nordic_scans_end);
        step1_logs{src_magI} = strjoin(msgs, '\n');
    end
    for k = 1:num_runs; fprintf('%s\n', step1_logs{k}); end

    % -----------------------------------------------------------------------
    %% STEP 2 — build NORDIC ARG/file structs
    % -----------------------------------------------------------------------
    fprintf('\n----------------------------------------------------------\n');
    fprintf('  STEP 2 — build NORDIC input structs\n');
    fprintf('----------------------------------------------------------\n');

    if doNORDIC
        clear ARG
        I = 1;
        src_mags = dir(dwimag_pattern);
        src_mags(contains({src_mags.name}, '_orig')) = [];
        num_runs = length(src_mags);

        for src_magI = 1:num_runs
            fn_magn_in  = fullfile(src_mags(src_magI).folder, src_mags(src_magI).name);
            fn_phase_in = strrep(fn_magn_in, '_magnitude', '_phase');
            fn_out      = fullfile(out_sesP, rename_mag_to_output(src_mags(src_magI).name, acq_label));

            if ~(exist(strrep(fn_out, '.nii.gz', 'magn.nii'), 'file') || exist(fn_out, 'file')) || force
                ARG(I).temporal_phase    = 3;
                ARG(I).phase_filter_width = 3;
                ARG(I).noise_volume_last = 0;
                [ARG(I).DIROUT, fn_out_name, ~] = fileparts(fn_out);
                ARG(I).DIROUT = [ARG(I).DIROUT, '/'];
                if ~exist(ARG(I).DIROUT, 'dir'); mkdir(ARG(I).DIROUT); end
                ARG(I).make_complex_nii = 1;
                ARG(I).save_gfactor_map = 1;
                file(I).phase = fn_phase_in;
                file(I).magni = fn_magn_in;
                file(I).out   = strrep(fn_out_name, '.nii', '');
                fprintf('  [%d] QUEUED for NORDIC\n', src_magI);
                fprintf('    IN  mag  : %s\n', fn_magn_in);
                fprintf('    IN  phase: %s\n', fn_phase_in);
                fprintf('    OUT base : %s\n', fn_out);
                I = I + 1;
            else
                fprintf('  [%d] SKIP NORDIC — output already exists and force=0\n', src_magI);
                fprintf('    %s\n', fn_out);
            end
        end
    else
        fprintf('  doNORDIC=0 — skipping step 2\n');
    end

    % -----------------------------------------------------------------------
    %% STEP 3 — run NORDIC
    % -----------------------------------------------------------------------
    fprintf('\n----------------------------------------------------------\n');
    fprintf('  STEP 3 — run NIFTI_NORDIC\n');
    fprintf('----------------------------------------------------------\n');

    if doNORDIC && exist('ARG', 'var')
        n_nordic = length(ARG);
        fprintf('  Running NORDIC on %d file(s) (parfor)\n', n_nordic);
        step3_logs = cell(1, n_nordic);
        parfor i = 1:n_nordic
            msgs = {};
            msgs{end+1} = sprintf('  [nordic %d/%d]', i, n_nordic);
            msgs{end+1} = sprintf('    IN  mag  : %s', file(i).magni);
            msgs{end+1} = sprintf('    IN  phase: %s', file(i).phase);
            msgs{end+1} = sprintf('    OUT base : %s', file(i).out);
            NIFTI_NORDIC(file(i).magni, file(i).phase, file(i).out, ARG(i));
            msgs{end+1} = sprintf('    DONE: created *magn.nii, *phase.nii, gfactor*.nii');
            step3_logs{i} = strjoin(msgs, '\n');
        end
        for k = 1:n_nordic; fprintf('%s\n', step3_logs{k}); end
        clear ARG file
    else
        fprintf('  No ARG struct — NORDIC skipped (doNORDIC=0 or all outputs exist)\n');
    end

    % -----------------------------------------------------------------------
    %% STEP 4 — wrap up: gzip, clean intermediates, copy JSON/bvec/bval
    % -----------------------------------------------------------------------
    fprintf('\n----------------------------------------------------------\n');
    fprintf('  STEP 4 — wrap up BIDS outputs (gzip, JSON, bvec, bval)\n');
    fprintf('----------------------------------------------------------\n');
    fprintf('  doNORDIC=%d  acq_label=acq-%s\n', doNORDIC, acq_label);

    % Re-read src_mags after NORDIC
    src_mags = dir(dwimag_pattern);
    src_mags(contains({src_mags.name}, '_orig')) = [];
    num_runs = length(src_mags);

    step4_logs = cell(1, num_runs);
    parfor src_magI = 1:num_runs
        fn_magn_in  = fullfile(src_mags(src_magI).folder, src_mags(src_magI).name);
        fn_out      = fullfile(out_sesP, rename_mag_to_output(src_mags(src_magI).name, acq_label));
        gfactorFile = strrep(strrep(fn_out, '.nii.gz', '.nii'), ...
                      [sub '_ses'], ['gfactor_' sub '_ses']);

        src_bvec_local = strrep(fn_magn_in, '_magnitude.nii.gz', '_magnitude.bvec');
        src_bval_local = strrep(fn_magn_in, '_magnitude.nii.gz', '_magnitude.bval');
        bvec_orig      = strrep(src_bvec_local, '.bvec', '_orig.bvec');
        bval_orig      = strrep(src_bval_local, '.bval', '_orig.bval');
        has_bvec_bval  = exist(src_bvec_local, 'file') || exist(bvec_orig, 'file');

        mag_orig_local = strrep(fn_magn_in, '.nii.gz', '_orig.nii.gz');
        has_orig       = exist(mag_orig_local, 'file');

        msgs = {};
        msgs{end+1} = sprintf('  [run %d/%d] %s', src_magI, num_runs, src_mags(src_magI).name);
        msgs{end+1} = sprintf('    IN  mag  : %s', fn_magn_in);
        msgs{end+1} = sprintf('    OUT dwi  : %s', fn_out);

        % --- doNORDIC: finalise NORDIC output ---
        if doNORDIC && exist(gfactorFile, 'file')
            info = niftiinfo(strrep(fn_out, '.nii.gz', 'magn.nii'));
            if nordic_scans_end > 0
                n_vols_out = info.ImageSize(end) - nordic_scans_end;
                system(['fslroi ', strrep(fn_out, '.nii.gz', 'magn.nii'), ' ', fn_out, ...
                       ' 0 -1 0 -1 0 -1 0 ', num2str(n_vols_out)]);
                msgs{end+1} = sprintf('    NORDIC  : fslroi trimmed → %d vols', n_vols_out);
            else
                system(['mv ', strrep(fn_out, '.nii.gz', 'magn.nii'), ' ', ...
                       strrep(fn_out, '.nii.gz', '_temp.nii')]);
                gzip(strrep(fn_out, '.nii.gz', '_temp.nii'));
                system(['mv ', strrep(fn_out, '.nii.gz', '_temp.nii.gz'), ' ', fn_out]);
                system(['rm -f ', strrep(fn_out, '.nii.gz', '_temp.nii')]);
                msgs{end+1} = sprintf('    NORDIC  : magn.nii → %s', fn_out);
            end
            gzip(gfactorFile);
            gfactor_gz  = strrep(gfactorFile, '.nii', '.nii.gz');
            gfactor_dst = strrep(strrep(strrep(gfactor_gz, '_dwi', '_gfactor'), 'gfactor_', ''), ...
                                 '', '');
            system(['rm -f ', gfactorFile, ' ', strrep(fn_out, '.nii.gz', 'phase.nii')]);
            system(['mv ', gfactor_gz, ' ', gfactor_dst]);
            msgs{end+1} = sprintf('    NORDIC  : gfactor → %s', gfactor_dst);
            msgs{end+1} = sprintf('    NORDIC  : cleaned up magn.nii, phase.nii, gfactor.nii');

        % --- no NORDIC: copy magnitude to output ---
        elseif ~doNORDIC
            if ~exist(fn_out, 'file') || force
                if has_orig
                    copy_src = mag_orig_local;
                    msgs{end+1} = sprintf('    NO-NORDIC: orig → dwi  (%s)', mag_orig_local);
                else
                    copy_src = fn_magn_in;
                    msgs{end+1} = sprintf('    NO-NORDIC: mag → dwi  (%s)', fn_magn_in);
                end
                system(['cp ', copy_src, ' ', fn_out]);
                system(['chmod 755 ', fn_out]);
                if nordic_scans_end > 0
                    info = niftiinfo(fn_out);
                    n_vols_out = info.ImageSize(end) - nordic_scans_end;
                    system(['fslroi ', fn_out, ' ', fn_out, ...
                           ' 0 -1 0 -1 0 -1 0 ', num2str(n_vols_out)]);
                    msgs{end+1} = sprintf('    NO-NORDIC: fslroi trimmed → %d vols', n_vols_out);
                end
            else
                msgs{end+1} = sprintf('    NO-NORDIC: output exists, skipped');
            end

        else
            msgs{end+1} = sprintf('    NORDIC output not yet finalised (gfactor not found) — skipped');
        end

        % --- JSON sidecar ---
        dst_json      = strrep(fn_out, '.nii.gz', '.json');
        src_json_local = strrep(fn_magn_in, '_magnitude.nii.gz', '_magnitude.json');
        src_json_orig  = strrep(src_json_local, '.json', '_orig.json');
        if ~exist(dst_json, 'file') || force
            if has_orig && exist(src_json_orig, 'file')
                system(['cp ', src_json_orig, ' ', dst_json]);
                msgs{end+1} = sprintf('    JSON    : orig → %s', dst_json);
            else
                system(['cp ', src_json_local, ' ', dst_json]);
                msgs{end+1} = sprintf('    JSON    : mag → %s', dst_json);
            end
            system(['chmod 755 ', dst_json]);
        else
            msgs{end+1} = sprintf('    JSON    : exists, skipped  (%s)', dst_json);
        end

        % --- bvec / bval (AP only) ---
        if has_bvec_bval
            dst_bvec = strrep(fn_out, '.nii.gz', '.bvec');
            dst_bval = strrep(fn_out, '.nii.gz', '.bval');

            if ~exist(dst_bvec, 'file') || force
                if exist(src_bvec_local, 'file')
                    system(['cp ', src_bvec_local, ' ', dst_bvec]);
                    msgs{end+1} = sprintf('    BVEC    : mag → %s', dst_bvec);
                elseif exist(bvec_orig, 'file')
                    system(['cp ', bvec_orig, ' ', dst_bvec]);
                    msgs{end+1} = sprintf('    BVEC    : orig → %s', dst_bvec);
                else
                    msgs{end+1} = sprintf('    WARNING : bvec not found for %s', src_mags(src_magI).name);
                end
                system(['chmod 755 ', dst_bvec]);
            else
                msgs{end+1} = sprintf('    BVEC    : exists, skipped  (%s)', dst_bvec);
            end

            if ~exist(dst_bval, 'file') || force
                if exist(src_bval_local, 'file')
                    system(['cp ', src_bval_local, ' ', dst_bval]);
                    msgs{end+1} = sprintf('    BVAL    : mag → %s', dst_bval);
                elseif exist(bval_orig, 'file')
                    system(['cp ', bval_orig, ' ', dst_bval]);
                    msgs{end+1} = sprintf('    BVAL    : orig → %s', dst_bval);
                else
                    msgs{end+1} = sprintf('    WARNING : bval not found for %s', src_mags(src_magI).name);
                end
                system(['chmod 755 ', dst_bval]);
            else
                msgs{end+1} = sprintf('    BVAL    : exists, skipped  (%s)', dst_bval);
            end

            % Trim bvec/bval if noise scans were removed (no-NORDIC only)
            if ~doNORDIC && nordic_scans_end > 0 && exist(fn_out, 'file')
                out_info   = niftiinfo(fn_out);
                out_nvols  = out_info.ImageSize(end);
                trim_bvec_bval(dst_bvec, dst_bval, out_nvols);
                msgs{end+1} = sprintf('    BVEC/VAL: trimmed to %d vols', out_nvols);
            end
        else
            msgs{end+1} = sprintf('    BVEC/VAL: no bvec/bval for this file (PA?) — skipped');
        end

        step4_logs{src_magI} = strjoin(msgs, '\n');
    end
    for k = 1:num_runs; fprintf('%s\n', step4_logs{k}); end

    % -----------------------------------------------------------------------
    % Session footer
    % -----------------------------------------------------------------------
    time_end = datetime('now');
    fprintf('\n==========================================================\n');
    fprintf('  DONE  %s  %s\n', sub, ses);
    fprintf('  runs processed : %d\n', num_runs);
    fprintf('  total time     : %s\n', time_end - time_start);
    fprintf('==========================================================\n\n');
end


%% ========== Rename magnitude filename to output filename ==========
% Replaces acq-<anything> with acq-<acq_label> and _magnitude with _dwi
%
% Example:
%   rename_mag_to_output('sub-07_ses-01_acq-votcloc1d5_dir-PA_run-01_magnitude.nii.gz', 'nordic')
%   -> 'sub-07_ses-01_acq-nordic_dir-PA_run-01_dwi.nii.gz'
function out_name = rename_mag_to_output(mag_name, acq_label)
    out_name = regexprep(mag_name, 'acq-[^_]+', ['acq-' acq_label]);
    out_name = strrep(out_name, '_magnitude', '_dwi');
end


%% ========== Trim bvec/bval in output dir to match volume count ==========
function trim_bvec_bval(bvec_file, bval_file, new_volume_count)
    if exist(bvec_file, 'file')
        bvec = load(bvec_file);
        if size(bvec, 1) == 3 && size(bvec, 2) > new_volume_count
            bvec_new = bvec(:, 1:new_volume_count);
            fid = fopen(bvec_file, 'w');
            fprintf(fid, '%.6f ', bvec_new(1, :)); fprintf(fid, '\n');
            fprintf(fid, '%.6f ', bvec_new(2, :)); fprintf(fid, '\n');
            fprintf(fid, '%.6f ', bvec_new(3, :)); fprintf(fid, '\n');
            fclose(fid);
            fprintf('  Output bvec trimmed: %d -> %d volumes\n', size(bvec, 2), new_volume_count);
        end
    end

    if exist(bval_file, 'file')
        bval = load(bval_file);
        if size(bval, 1) > 1; bval = bval'; end
        if size(bval, 2) > new_volume_count
            bval_new = bval(1:new_volume_count);
            fid = fopen(bval_file, 'w');
            fprintf(fid, '%.1f ', bval_new);
            fprintf(fid, '\n');
            fclose(fid);
            fprintf('  Output bval trimmed: %d -> %d volumes\n', length(bval), new_volume_count);
        end
    end
end
