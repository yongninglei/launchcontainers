# GLM Strategy — Technical Notes

Quick reference for running and understanding the surface/volumetric GLM pipeline
for fMRI data (VOTCLOC / IRAKEINU projects).

---

## 1. Parallelism

The pipeline has **two levels** of parallelism controlled by a single `--n-workers` argument.

### Level 1 — Strategy-level (thread pool)

`glm_surface_check_model_strategy.py` wraps each confound strategy in a
`ThreadPoolExecutor`. When multiple strategies are requested, they run concurrently —
one thread per strategy, up to `n_workers` threads total.

### Level 2 — Vertex-level (GLM jobs)

Inside each strategy, `run_glm(Y, X, n_jobs=n_glm_jobs)` parallelises the OLS fitting
across groups of vertices using joblib.

### Budget formula

```
n_glm_jobs = max(1, n_workers // len(strategy_names))
```

| `--n-workers` | Strategies running | `n_glm_jobs` (per strategy) |
|---|---|---|
| 20 | 1 | **20** — all workers go to vertex GLM |
| 20 | 4 | 5 |
| 20 | 20 | 1 |

**Practical rule**: if you run a single strategy (the common case), pass `--n-workers`
equal to the number of CPU cores you want to dedicate — all of them flow to the
vertex-level GLM.

### Where to set `--n-workers`

In the launch scripts:

```bash
# launch_glm_local_VOTCLOC.sh
PY_CMD="python ${PYTHON_SCRIPT} \
    ...
    --n-workers 20"   # ← edit this number
```

For the FIR pipeline (`run_allses_glm_FIR.py`) there is only vertex-level parallelism;
set it with `--n-glm-jobs`:

```bash
# launch_allses_glm_FIR_local.sh
N_GLM_JOBS="$(( $(nproc) - 2 ))"   # ← auto-detect cores, leave 2 free
```

---

## 2. Confound strategies

### YAML file

```
glm_strategy/strategy.yaml
```

Each top-level key is a strategy name. Example entry:

```yaml
basic_MC:
  backend: nilearn
  include_non_steady_state: true
  motion_regressors: true
  wm_csf: true
  ...
```

### Selecting strategies at runtime

| Goal | How |
|---|---|
| Run **one** strategy | `STRATEGY="basic_MC"` (set in the launch `.sh`) |
| Run **all** strategies in the YAML | `STRATEGY=""` (leave empty) |
| Run **multiple** specific strategies | `--strategy strat1,strat2` on the command line |

The strategy name becomes part of the output directory:
`analysis-<analysis_name>_<strategy_name>/`.

### Adding a new strategy

1. Add a new key block to `strategy.yaml`.
2. Run the launch script with `STRATEGY="<new_key>"` to test it first.

---

## 3. Contrast definitions

### YAML file

```
glm_strategy/contrast_VOTCLOC_short.yaml   # VOTCLOC project (in use)
glm_strategy/contrast_VOTCLOC_all.yaml     # extended VOTCLOC set
glm_strategy/contrast_IRAKEINU_new.yaml    # IRAKEINU project
```

### Format

```yaml
RWvsCS:
  positive: ["RW"]
  negative: ["CS"]
```

The GLM builds the contrast vector by assigning `+1 / n_positive` to each positive
condition regressor and `-1 / n_negative` to each negative condition regressor.

### Per-condition vs-null contrasts (for PSC)

Contrasts with an empty `negative` list compare one condition against implicit
baseline (rest periods). These are what produce interpretable PSC values:

```yaml
RWvsNull:
  positive: ["RW"]
  negative: []      # implicit baseline
```

### Pointing the launcher at a contrast file

```bash
# launch_glm_local_VOTCLOC.sh
CONTRAST="/path/to/contrast_VOTCLOC_short.yaml"   # ← change this line
```

---

## 4. Percent Signal Change (PSC)

### Why betas are not raw percent signal change

Before the GLM is fit, each vertex timeseries is **z-scored**:

```python
data_std = scipy.stats.zscore(data_remove_first, axis=1)
```

So `stat-effect` betas (β_z) are in z-score units, not BOLD % units.

### Formula

```
PSC = β_z × (σ_BOLD / μ_BOLD) × 100
```

- `σ_BOLD` — standard deviation of the raw (pre-z-score) BOLD timeseries at that vertex
- `μ_BOLD` — mean of the raw BOLD timeseries at that vertex
- `β_z / σ_raw = β_raw`, then `β_raw / μ_BOLD × 100 = PSC`

### Implementation

In `glm_surface_check_model_strategy.py`:

1. **`prepare_glm_input`** — accumulates raw (unscaled, trimmed) BOLD across all runs
   into `conc_data_raw` and returns it alongside the z-scored data.
2. **`glm_l1`** — receives `conc_data_raw`, computes `_mean_bold` and `_std_bold`
   once before the contrast loop, then for each contrast:
   - Surface (GIFTI): saves `stat-PSC` alongside `stat-effect`.
   - Volumetric (NIfTI / T1w): PSC is **not yet implemented** in the `else` branch.

### Output files

```
sub-XX_ses-YY_task-fLoc_hemi-L_space-fsnative_contrast-RWvsNull_stat-PSC_statmap.func.gii
sub-XX_ses-YY_task-fLoc_hemi-R_space-fsnative_contrast-RWvsNull_stat-PSC_statmap.func.gii
```

One file per hemisphere, per contrast, per strategy.

### Running PSC

PSC is computed automatically whenever the GLM runs — no extra flag needed.
Just re-run the launch script to generate `stat-PSC` maps alongside the usual
`stat-effect`, `stat-t`, `stat-z`, etc.

```bash
bash launch_glm_local_VOTCLOC.sh -n <analysis_name> -f subseslist_HMC.txt
```

### Saving per-regressor betas / residuals

Disabled by default. Pass `--save-betas` to the launch script to also write:
- per-regressor beta maps
- model residuals (whitened)
- fitted timeseries

```bash
bash launch_glm_local_VOTCLOC.sh -n <analysis_name> -s 03,05 --save-betas
```

---

## 5. FIR (Finite Impulse Response) pipeline

The FIR pipeline estimates the full hemodynamic response shape for each condition
without assuming a canonical HRF.

### Scripts

| Script | Role |
|---|---|
| `run_allses_glm_FIR.py` | Fit FIR GLM concatenating all sessions of one subject |
| `fir_curve_metrics.py` | Derive per-vertex metrics from the AllStim FIR curve |
| `launch_allses_glm_FIR_local.sh` | Launcher that runs both in sequence |

### Design

Two design matrices are fit simultaneously to the concatenated data:

1. **`cond`** — one FIR regressor set per named condition (e.g. RW, CS, FF, SC, face, bodylimb).
   Each regressor set has `n_delays` columns, one per TR delay after stimulus onset.
2. **`allstim`** — one FIR set for all stimuli pooled into `AllStim`.

### Key parameters (edit in `launch_allses_glm_FIR_local.sh`)

| Variable | Meaning | Default |
|---|---|---|
| `CONDITIONS` | Comma-separated condition labels matching the events TSV | `RW,CS,FF,SC,bodylimb,face` |
| `N_DELAYS` | Number of FIR delays (frames) | `10` |
| `N_GLM_JOBS` | Vertex-level parallelism | `nproc - 2` |
| `DEFAULT_SESSIONS` | Sessions to concatenate per subject | `01..09` |

### Outputs (per hemisphere, under `allses/`)

| File pattern | Description |
|---|---|
| `desc-FIR<cond>_delays.func.gii` | `n_delays`-frame GIFTI; frame = FIR delay |
| `desc-AllStimFIR0-18s_timeseries.func.gii` | AllStim FIR curve stack |
| `desc-AllStimdelay<d>_statmap.func.gii` | Single-frame map per delay |

### Curve metrics (from `fir_curve_metrics.py`)

Derived automatically after FIR if exit code is 0:

| Output `desc-` | Meaning |
|---|---|
| `FIRpeakamp` | Signed peak amplitude of the AllStim curve |
| `FIRtimetopeak` | Time-to-peak (s), sub-TR via cubic spline |
| `FIRauc` | Area under the curve |
| `FIRfwhm` | Full-width-at-half-max (s) |
| `FIRfitR2` | R² of best-fitting canonical gamma HRF |
| `FIRfitamp` | Amplitude scale of the best-fit HRF |
| `FIRfitcandidate` | Nominal peak time (s) of the winning HRF candidate |

### Running FIR

```bash
# Single subject
bash launch_allses_glm_FIR_local.sh -o FIR_v1 -s 03

# Batch from file
bash launch_allses_glm_FIR_local.sh -o FIR_v1 -f subseslist_HMC.txt

# Skip curve metrics step
bash launch_allses_glm_FIR_local.sh -o FIR_v1 -s 03 --no-metrics
```

---

## Quick-reference: what to edit for common tasks

| Task | File | What to change |
|---|---|---|
| Change strategy | `launch_glm_local_VOTCLOC.sh` | `STRATEGY="..."` |
| Add/modify strategy | `strategy.yaml` | Add/edit key block |
| Add/modify contrast | `contrast_VOTCLOC_short.yaml` | Add/edit contrast entry |
| Change space (T1w / fsnative / MNI) | `launch_glm_local_VOTCLOC.sh` | `SPACE="..."` or pass `-p <space>` |
| Change number of parallel workers | `launch_glm_local_VOTCLOC.sh` | `--n-workers <N>` |
| Change which subjects run | `subseslist_HMC.txt` | Set `Run` column to `True`/`False` |
| Add FIR conditions | `launch_allses_glm_FIR_local.sh` | `CONDITIONS="..."` |
