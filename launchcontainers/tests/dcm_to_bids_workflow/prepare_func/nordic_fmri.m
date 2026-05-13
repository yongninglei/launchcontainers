function nordic_fmri(tbPath, src_dir, output_dir, sub, ses, nordic_scans_end, doNORDIC, dotsnr, force)
% MIT License
% Copyright (c) 2024-2025 Yongning Lei
%
% Permission is hereby granted, free of charge, to any person obtaining a
% copy of this software and associated documentation files (the "Software"),
% to deal in the Software without restriction, including without limitation
% the rights to use, copy, modify, merge, publish, distribute, sublicense,
% and/or sell copies of the Software, and to permit persons to whom the
% Software is furnished to do so, subject to the following conditions:
% The above copyright notice and this permission notice shall be included in
% all copies or substantial portions of the Software.

    % -----------------------------------------------------------------------
    % Session header
    % -----------------------------------------------------------------------
    fprintf('\n');
    fprintf('==========================================================\n');
    fprintf('  nordic_fmri  sub=%s  ses=%s\n', sub, ses);
    fprintf('  doNORDIC=%d  dotsnr=%d  nordic_scans_end=%d  force=%d\n', ...
            doNORDIC, dotsnr, nordic_scans_end, force);
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

    src_sesP = fullfile(src_dir, sub, ses, 'func');
    out_sesP = fullfile(output_dir, sub, ses, 'func');
    system(['chmod -R 777 ', src_sesP]);
    if ~exist(out_sesP, 'dir'); mkdir(out_sesP); end
    system(['chmod -R 777 ', out_sesP]);

    fprintf('  src : %s\n', src_sesP);
    fprintf('  dst : %s\n', out_sesP);

    % -----------------------------------------------------------------------
    % Discover runs
    % -----------------------------------------------------------------------
    funcmag_pattern = fullfile(src_sesP, '*_magnitude.nii.gz');
    src_mags = dir(funcmag_pattern);
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
        prepare_nordic_bold_nifti(mag_in, nordic_scans_end, force);
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
    clear ARG
    I = 1;
    src_mags = dir(funcmag_pattern);   % re-read after step 1
    num_runs = length(src_mags);

    for src_magI = 1:num_runs
        fn_magn_in  = fullfile(src_mags(src_magI).folder, src_mags(src_magI).name);
        fn_phase_in = strrep(fn_magn_in, '_magnitude', '_phase');
        fn_out      = fullfile(out_sesP, strrep(src_mags(src_magI).name, '_magnitude', '_bold'));

        if ~(exist(strrep(fn_out, '.nii.gz', 'magn.nii'), 'file') || exist(fn_out, 'file')) && doNORDIC
            ARG(I).temporal_phase    = 1;
            ARG(I).phase_filter_width = 10;
            ARG(I).noise_volume_last = 1;
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
            fprintf('  [%d] SKIP NORDIC — output already exists or doNORDIC=0\n', src_magI);
            fprintf('    %s\n', fn_out);
        end
    end

    % -----------------------------------------------------------------------
    %% STEP 3 — run NORDIC
    % -----------------------------------------------------------------------
    fprintf('\n----------------------------------------------------------\n');
    fprintf('  STEP 3 — run NIFTI_NORDIC\n');
    fprintf('----------------------------------------------------------\n');
    if exist('ARG', 'var')
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
    %% STEP 4 — wrap up: gzip, clean intermediates, copy JSON/sbref
    % -----------------------------------------------------------------------
    fprintf('\n----------------------------------------------------------\n');
    fprintf('  STEP 4 — wrap up BIDS outputs (gzip, JSON, sbref)\n');
    fprintf('----------------------------------------------------------\n');
    fprintf('  doNORDIC=%d  dotsnr=%d\n', doNORDIC, dotsnr);

    step4_logs = cell(1, num_runs);
    parfor src_magI = 1:num_runs
        fn_magn_in   = fullfile(src_mags(src_magI).folder, src_mags(src_magI).name);
        fn_phase_in  = strrep(fn_magn_in, '_magnitude', '_phase');  %#ok<NASGU>
        fn_out       = fullfile(out_sesP, strrep(src_mags(src_magI).name, '_magnitude', '_bold'));
        gfactorFile  = strrep(strrep(fn_out, '.nii.gz', '.nii'), [sub '_ses'], ['gfactor_' sub '_ses']);

        msgs = {};
        msgs{end+1} = sprintf('  [run %d/%d] %s', src_magI, num_runs, src_mags(src_magI).name);
        msgs{end+1} = sprintf('    IN  mag  : %s', fn_magn_in);
        msgs{end+1} = sprintf('    OUT bold : %s', fn_out);

        % --- doNORDIC: finalise NORDIC output ---
        if exist(gfactorFile, 'file') && doNORDIC
            magn_nii = strrep(fn_out, '.nii.gz', 'magn.nii');
            info = niftiinfo(magn_nii);
            n_vols_out = info.ImageSize(end) - 1;
            system(['fslroi ', magn_nii, ' ', fn_out, ' 0 -1 0 -1 0 -1 0 ', num2str(n_vols_out)]);
            msgs{end+1} = sprintf('    NORDIC  : fslroi %s → %s  (vols: %d→%d)', ...
                magn_nii, fn_out, info.ImageSize(end), n_vols_out);
            gzip(gfactorFile);
            gfactor_gz  = strrep(gfactorFile, '.nii', '.nii.gz');
            gfactor_dst = strrep(strrep(strrep(gfactor_gz, '_bold', '_gfactor'), 'gfactor_', ''), ...
                                 [sub '_ses'], [sub '_ses']);  % keep sub/ses intact
            gfactor_dst = strrep(strrep(gfactor_gz, '_bold', '_gfactor'), 'gfactor_', '');
            system(['rm ', magn_nii, ' ', gfactorFile, ' ', strrep(fn_out, '.nii.gz', 'phase.nii')]);
            system(['mv ', gfactor_gz, ' ', gfactor_dst]);
            msgs{end+1} = sprintf('    NORDIC  : gfactor → %s', gfactor_dst);
            msgs{end+1} = sprintf('    NORDIC  : cleaned up magn.nii, phase.nii, gfactor.nii');

        % --- no NORDIC: copy + trim magnitude ---
        elseif ~doNORDIC
            if ~exist(fn_out, 'file') || force
                info = niftiinfo(fn_magn_in);
                n_vols_out = info.ImageSize(end) - nordic_scans_end;
                system(['cp ', fn_magn_in, ' ', fn_out]);
                system(['chmod 755 ', fn_out]);
                system(['fslroi ', fn_out, ' ', fn_out, ' 0 -1 0 -1 0 -1 0 ', num2str(n_vols_out)]);
                msgs{end+1} = sprintf('    NO-NORDIC: cp+fslroi mag→bold  (vols: %d→%d)', ...
                    info.ImageSize(end), n_vols_out);
            else
                msgs{end+1} = sprintf('    NO-NORDIC: bold already exists, skipped');
            end
        else
            msgs{end+1} = sprintf('    NORDIC output not yet finalised (gfactor not found) — skipped');
        end

        % --- JSON sidecar ---
        bold_json_src = strrep(fn_magn_in, '_magnitude.nii.gz', '_magnitude.json');
        bold_json_dst = strrep(fn_out, '_bold.nii.gz', '_bold.json');
        if ~exist(bold_json_dst, 'file') || force
            system(['cp ', bold_json_src, ' ', bold_json_dst]);
            system(['chmod 755 ', bold_json_dst]);
            msgs{end+1} = sprintf('    JSON    : %s → %s', bold_json_src, bold_json_dst);
        else
            msgs{end+1} = sprintf('    JSON    : exists, skipped  (%s)', bold_json_dst);
        end

        % --- sbref ---
        src_sbref      = strrep(fn_magn_in, '_magnitude.nii.gz', '_sbref.nii.gz');
        src_sbref_json = strrep(fn_magn_in, '_magnitude.nii.gz', '_sbref.json');
        dst_sbref      = strrep(fn_out, '_bold.nii.gz', '_sbref.nii.gz');
        dst_sbref_json = strrep(fn_out, '_bold.nii.gz', '_sbref.json');
        if ~(exist(dst_sbref, 'file') && exist(dst_sbref_json, 'file')) || force
            system(['cp ', src_sbref, ' ', dst_sbref]);
            system(['cp ', src_sbref_json, ' ', dst_sbref_json]);
            system(['chmod 755 ', dst_sbref_json]);
            msgs{end+1} = sprintf('    SBREF   : %s → %s', src_sbref, dst_sbref);
            msgs{end+1} = sprintf('    SBREF   : %s → %s', src_sbref_json, dst_sbref_json);
        else
            msgs{end+1} = sprintf('    SBREF   : exists, skipped  (%s)', dst_sbref);
        end

        step4_logs{src_magI} = strjoin(msgs, '\n');
    end
    for k = 1:num_runs; fprintf('%s\n', step4_logs{k}); end

    % -----------------------------------------------------------------------
    %% STEP 5 (optional) — tSNR maps
    % -----------------------------------------------------------------------
    if dotsnr
        fprintf('\n----------------------------------------------------------\n');
        fprintf('  STEP 5 — tSNR maps\n');
        fprintf('----------------------------------------------------------\n');
        bolds = dir(fullfile(out_sesP, '*_bold.nii.gz'));
        bolds(contains({bolds.name}, 'gfactor')) = [];
        src_mags_tsnr = dir(funcmag_pattern);
        n_bolds = length(bolds);
        fprintf('  Computing tSNR for %d bold file(s) (parfor)\n', n_bolds);

        tsnr_logs = cell(1, n_bolds);
        parfor nb = 1:n_bolds
            magFile  = fullfile(src_mags_tsnr(nb).folder, src_mags_tsnr(nb).name);
            boldFile = fullfile(bolds(nb).folder, bolds(nb).name);
            tsnrFile      = strrep(boldFile, 'bold', 'tsnr_postNordic');
            magtsnrFile   = strrep(boldFile, 'bold', 'tsnr_preNordic');
            gfactorFile   = strrep(boldFile, 'bold', 'gfactor');
            tsnrGfactorFile = strrep(gfactorFile, 'gfactor', 'gfactorSameSpace');

            msgs = {};
            msgs{end+1} = sprintf('  [tsnr %d/%d]', nb, n_bolds);
            msgs{end+1} = sprintf('    IN  mag : %s', magFile);
            msgs{end+1} = sprintf('    IN  bold: %s', boldFile);

            magHeader = niftiinfo(magFile);
            magData = single(niftiread(magHeader));
            magtsnrData = mean(magData, 4) ./ std(magData, 1, 4);
            magtsnrData(isnan(magtsnrData)) = 0;
            magHeader.ImageSize = size(magtsnrData);
            magHeader.PixelDimensions = magHeader.PixelDimensions(1:3);
            magHeader.Datatype = 'single';
            niftiwrite(magtsnrData, strrep(magtsnrFile, '.nii', ''), magHeader, 'compressed', true);
            msgs{end+1} = sprintf('    OUT tsnr_preNordic : %s', magtsnrFile);

            boldHeader = niftiinfo(boldFile);
            boldData = single(niftiread(boldHeader));
            tsnrData = mean(boldData, 4) ./ std(boldData, 1, 4);
            boldHeader.ImageSize = size(tsnrData);
            boldHeader.PixelDimensions = boldHeader.PixelDimensions(1:3);
            boldHeader.Datatype = 'single';
            niftiwrite(tsnrData, strrep(tsnrFile, '.nii', ''), boldHeader, 'compressed', true);
            msgs{end+1} = sprintf('    OUT tsnr_postNordic: %s', tsnrFile);

            gHeader = niftiinfo(gfactorFile);
            gfactorData = single(niftiread(gHeader));
            gHeader.ImageSize = size(gfactorData);
            gHeader.PixelDimensions = gHeader.PixelDimensions(1:3);
            gHeader.Datatype = 'single';
            niftiwrite(gfactorData, strrep(tsnrGfactorFile, '.nii', ''), gHeader, 'compressed', true);
            msgs{end+1} = sprintf('    OUT gfactorSameSpace: %s', tsnrGfactorFile);

            tsnr_logs{nb} = strjoin(msgs, '\n');
        end
        for k = 1:n_bolds; fprintf('%s\n', tsnr_logs{k}); end
    end

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
