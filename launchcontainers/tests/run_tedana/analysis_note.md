# ME-fMRI tedana pipeline — analysis notes

Last updated: 2026-05-20

---

## Why tedana after fMRIprep

fMRIprep with `--me-output-echos` already computes an **optimal combination (OC)**
of the echoes — but that OC is just a T2\*-weighted average with no denoising.
tedana takes those same preprocessed echoes and adds **ICA-based denoising**:

```
fMRIprep echoes → T2* fit → OC → [stop here = fMRIprep output]
                                ↓
                              ICA → classify components (TE-dep vs TE-indep)
                                ↓
                   desc-optcomDenoised  ← what we use for GLM
```

The `hemi-{LR}_space-fsnative_bold.func.gii` in fMRIprep derivatives is the OC
**before** ICA denoising. We do not use it for GLM.

---

## Is tedana's OC the same as fMRIprep's?

Nearly identical, not bit-for-bit equal.

Both use the same formula:

```
w_e  =  TE_e × exp(−TE_e / T2*)
OC   =  Σ_e [ w_e × echo_e ] / Σ_e [ w_e ]
```

Both use `fittype=curvefit` and the same input echoes, so the T2\* maps will be
very close. Small numerical differences can arise from masking and solver
implementation. The OC itself is not the deliverable — the ICA-denoised OC is.

**What determines the OC result:**
1. Echo times (TEs) — fixed by acquisition
2. T2\* fit method — `curvefit` in both fMRIprep and tedana (our setting)
3. Input echoes — same fMRIprep-preprocessed files in both cases
4. Brain mask — tedana adaptive dropout mask (see note below)

---

## Current tedana settings (run_tedana.py)

| Parameter | Value | Why |
|---|---|---|
| `fittype` | `curvefit` | matches fMRIprep's ME fitting; more accurate T2\* than loglin |
| `tedpca` | `aic` (default) | tedana default; more liberal than mdl, appropriate for clean NORDIC data |
| `tree` | `tedana_orig` (default) | most tested and documented decision tree |
| `n_robust_runs` | 30 (default) | number of ICA restarts for stability |
| `ica_method` | `fastica` (default) | standard FastICA from sklearn |
| `mask` | `None` — tedana adaptive | fmriprep T1w mask is on a different grid (T1w space, axis-aligned) than the echo files (native BOLD space, oblique affine) — cannot be used directly |
| `masktype` | `["dropout"]` | tedana builds mask from signal dropout across echoes, which operates in the correct native BOLD space |

**Decision tree options in tedana 26.0.3:**
- `tedana_orig` — our choice; reimplementation of Kundu 2012 MEICA algorithm
- `meica` — closer to original MEICA code; accepts more components, some high-variance
- `minimal` — simple kappa/rho thresholds only; still experimental

Going with `tedana_orig` is the safest and most validated choice.

---

## Full pipeline

```
Step 0  NORDIC (done before BIDS)
        └─ thermal denoising on raw k-space data

Step 1  fMRIprep  (done)
        --me-output-echos --me-t2s-fit-method curvefit
        └─ outputs per-echo T1w niftis:
           sub-{sub}_ses-{ses}_task-{task}_acq-ME_run-{run}_echo-{N}_desc-preproc_bold.nii.gz

Step 2  run_tedana.py
        └─ inputs : fmriprep echo niftis + T1w brain mask
           outputs: sub-{sub}_..._acq-ME_run-{run}_desc-optcomDenoised_bold.nii.gz  (T1w)
                    sub-{sub}_..._acq-ME_run-{run}_desc-tedana_mixing.tsv
                    sub-{sub}_..._acq-ME_run-{run}_desc-tedana_metrics.tsv
                    sub-{sub}_..._acq-ME_run-{run}_desc-tedana_AROMAnoiseICs.csv
                    (+ HTML report)

Step 3  project_to_spaces.py
        └─ inputs : tedana T1w output + fmriprep transforms + midthickness surfaces
           MNI  : antsApplyTransforms with from-T1w_to-MNI152NLin2009cAsym_xfm.h5
           fsnative: nilearn vol_to_surf on fmriprep midthickness gifti
           outputs: sub-{sub}_..._space-MNI152NLin2009cAsym_desc-optcomDenoised_bold.nii.gz
                    sub-{sub}_..._hemi-{LR}_space-fsnative_desc-optcomDenoised_bold.func.gii

Step 4  GLM  (existing glm_surface_*.py scripts)
        └─ BOLD input : hemi-{LR}_space-fsnative_desc-optcomDenoised_bold.func.gii
           Confounds  : motion params + framewise displacement only
                        (skip aCompCor / WM / CSF — ICA already removed these)
           Events     : BIDS events.tsv (unchanged)
```

---

## Confound strategy change vs single-echo GLM

| Confound | Single-echo GLM | After ME-ICA denoising |
|---|---|---|
| Motion 6/24 params | ✓ include | ✓ include (motion not fully captured by ICA) |
| Framewise displacement | ✓ for scrubbing | ✓ for scrubbing |
| aCompCor / WM / CSF | often included | **not needed** — ICA removes these |
| Global signal | sometimes | **not needed** |
| Cosine high-pass | ✓ include | ✓ include |
| tedana noise components | — | **not needed** — already subtracted from BOLD |

---

## Launch commands

```bash
# Step 2 — tedana (40 cores, 3 runs/session)
bash launch_tedana_local.sh -n pilot01 -c 40 -r 3 -s pilot02,01

# Step 3 — project to spaces
micromamba run -n lc python project_to_spaces.py \
    -b /bcbl/home/public/Gari/IRAKEINU/BIDS \
    -fp .../fmriprep-25.1.4_IRpilot \
    -tedana tedana-26.0.3 \
    -n pilot01 -s pilot02,01 \
    --tasks BfLocVideo \
    --spaces MNI,fsnative
```

---

## Key references

- Dupré et al. 2021, eLife — "TE-dependent analysis of multi-echo fMRI with tedana"
- Kundu et al. 2012, NeuroImage — original MEICA paper (basis for tedana_orig tree)
- tedana docs: https://tedana.readthedocs.io
  - Decision trees: https://tedana.readthedocs.io/en/stable/approach.html#component-selection
  - fittype comparison: https://tedana.readthedocs.io/en/stable/approach.html#t2-s0-estimation
