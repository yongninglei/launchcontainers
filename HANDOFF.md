# HANDOFF — open work for the next person or agent

Read this before starting. Update it when you finish or hand off a task, and
delete an entry once it is done. Repo conventions are in `Claude.md`.

---

## sub-02 ses-07…10 DWI heudiconv conversion (VOTCLOC) — 2026-11-04

**Goal:** convert only the DWI DICOMs of sub-02 ses-07, 08, 09, 10 to BIDS NIfTI
with heudiconv on IPS (SGE).

| | |
|---|---|
| Data | `/bcbl/home/public/Gari/VOTCLOC/main_exp/dicom/sub-02/ses-XX` (symlinks to the lab DICOMs) |
| Output | `/bcbl/home/public/Gari/VOTCLOC/main_exp/raw_nifti` |
| Criteria | `MR_pipelines/00_dicom_to_nifti/heuristic/heuristic_dwi_VOTCLOC.py`: `dMRI`/`1d5iso` series, no SBRef; PA = 6 files, AP = 105 files; magnitude + phase |
| Work list | `/bcbl/home/public/Gari/VOTCLOC/main_exp/code/subseslist_heudiconv_sub02_dwi.txt` |

### Done
- Wrote the DWI-only heuristic. In the lab folders, all four sessions have
  exactly one PA (6 files) and one AP (105 files) series, so it matches them.
- `run_ips/qsub_heurdiconv.sh`: added optional `-H <heuristic>` and
  `-o <output dir under basedir>` flags (the defaults are unchanged), and
  `script_dir` now uses the `/bcbl/home/...` path.
- `src_heudiconv_step{1,2}.sh`: `--bind /export` is added only when `/export`
  exists on the host.
- Created `dicom/sub-02/ses-09` and `ses-10` with
  `prepare/03_create_dicom_symlink.py`, pointing to
  `/bcbl/data/MRI/VOTCLOC_22324/DATA/images/sub-02_ses-{09,10}`.

### Left to do
1. **Run the conversion** (it has not been run yet):
   ```bash
   bash MR_pipelines/00_dicom_to_nifti/run_ips/qsub_heurdiconv.sh -t step2 \
     -f /bcbl/home/public/Gari/VOTCLOC/main_exp/code/subseslist_heudiconv_sub02_dwi.txt \
     -H heuristic_dwi_VOTCLOC.py -o raw_nifti
   ```
   Then check `raw_nifti/sub-02/ses-XX/dwi/` and the logs in
   `raw_nifti/log_heudiconv/subseslist_heudiconv_sub02_dwi_<date>/step2/`.
2. **ses-07 and ses-08 links still point to `/export/home/tlei/lab/...`**, which
   works only on hosts that have `/export` (IPS is fine). Optionally repoint them
   to `/bcbl/data/MRI/...` with `03_create_dicom_symlink.py --force`.
3. **The ses-08 source folder needs confirming.** The lab has both
   `sub-02_ses-08` and `sub-02_ses-08-mar-24`, and both contain DWI. The link
   uses `-mar-24`. For ses-07, `sub-02_ses-07re` is correct, because
   `sub-02_ses-07` has no DWI.

---

## Host-path check over SSH — 2026-11-04

`Claude.md` → "Host paths" says that `/export/home/tlei` (ips, lmc02,
cajal03/04) and `/home/tlei` (brodmann) are the same home, and that
`/export/home/tlei/lab/MRI` is the same folder as `/bcbl/data/MRI`. This has
**not been verified** on the cluster yet.

- From brodmann, key-based SSH to cajal03, cajal04 and lmc02 is refused, and
  `ips` does not answer on port 22 (`ips-0-3` does).
- **To do:** run `ssh-copy-id -i ~/.ssh/id_ed25519.pub tlei@{cajal03,lmc02,ips-0-3}`
  and add `Host cajal03` / `Host ips` (→ `ips-0-3`) entries to `~/.ssh/config`.
  Then run on one host:
  ```bash
  ls -d /bcbl/data/MRI/VOTCLOC_22324 && readlink -f /export/home/tlei/lab/MRI
  ```
  and mark the Claude.md section as verified.
