# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlocCam-ImageProcessing is a Python/ImageJ pipeline for sizing suspended particles (flocs, mud, sand) captured by the FlocARAZI in-situ imaging instrument. The workflow runs sequentially: Python organizes raw images into batches → ImageJ extracts particle measurements → Python filters, sizes, and plots the particle size distributions (PSDs).

## Running the Pipeline

Scripts must be run from within their containing directory (they use relative paths to `ij.jar`, macros, and models):

```bash
cd Code/0_Standard_Processing
python 1_ImageProcess_v04.py        # Step 1: organize images + run ImageJ
python 2_ParticleDataProcess_v08.py # Step 2: filter particles + compute PSDs
python 3_ParticleDataVisualization_v02.py # Step 3: plot individual PSDs
```

Profile processing (vertical water-column casts) uses Jupyter notebooks in `Code/1_Profile_Processing/`, run in order (Step 1 → Step 2 → Step 3A/3B → Step 4). Post-processing settling-velocity analysis lives in `Code/2_Distribution_Analysis/`.

## Path Configuration

All scripts read input/output paths from a CSV rather than hardcoded paths:

- **`Code/0_Standard_Processing/0_Paths.csv`** — first row is the `ImagePath` (folder containing raw image subfolders); subsequent `ProcessPath` rows specify which subfolders to analyze
- **`Code/1_Profile_Processing/0_CastPath.csv`** — path to the profile data directory

Set `where = "local"` in any script to use `os.getcwd()` instead of the CSV.

## Code Architecture

### Standard Processing (`Code/0_Standard_Processing/`)

**Step 1 — `1_ImageProcess_v04.py`**
- Groups raw images into numbered subdirectories (e.g., `001/`, `002/`) of `dir_size` images each
- With `overlap = 1` (default) each subdirectory also gets a copy of the first image of the next batch, so the difference macro yields `dir_size` datasets per batch instead of `dir_size-1` — the last image of a batch is no longer left unpaired. That copy is excluded from `filemod_list.csv` and from `rmsrc` deletion, since the same image is recorded and deleted when it starts the next batch. The final batch gets no overlap image if no further images remain
- Records source/destination/modification-time in `filemod_list.csv` (used later for timestamp extraction)
- Invokes ImageJ headlessly: `java -jar ij.jar -batch <macro>`
- Key user parameters at top: `ts` (time series vs. single PSD), `dir_size`, `image_type`, `subdirp` (recurse subdirs), `rmsrc` (delete originals)
- `group_time` and `group_time_unit` record how much real time one group of `dir_size` images represents. They are written to `0_group_info.csv` in the image directory and read back by Step 2, which uses them for the time index of `dstats_*.csv` and the time axis of the PDF plots. Step 2 falls back to 1 min per group if the file is absent (data grouped before this existed)
- `check_group_time()` cross-checks the declared `group_time` against the median spacing of the image modification times and prints a warning if they disagree by more than `group_time_tol` (default 0.2, set to 0 to disable). It is advisory only — the declared value is always what gets written and used, since file mtimes do not always reflect capture time. Skipped entirely when `ts = 0`, where every image forms a single PSD and there is no grouping interval to check

**ImageJ macros (`ImageJ-macros/`)**
- Implement a *difference-of-images* technique: subtracts consecutive image pairs to isolate particles from background
- Applies Triangle threshold → binary → erode → fill holes → `Analyze Particles`
- Outputs one `.txt` file per image pair containing particle measurements (area, perimeter, bounding box, ellipse fit, etc.)
- `_suppressoutput` variant is the default; `_watershed` variant adds watershed segmentation for touching particles

**Step 2 — `2_ParticleDataProcess_v08.py`** (current version)
- Reads ImageJ `.txt` output files; applies three filters: focus (MaxGreyValue threshold), edge proximity, and minimum area
- Converts pixel measurements to microns using `muperpix`, set together with `img_sz_x`/`img_sz_y` in a per-camera block at the top: updated FlocARAZI (4096×3000, 0.662 µm/px, 10× lens) is active; Osborn FlocARAZI (4000×3000, 0.925, 5× lens) and the old lab cameras are commented out
- Computes d16/d50/d84 from either volume-weighted (`vdist=1`) or d^w-weighted distributions
- Rows of `dstats_*.csv` are indexed by elapsed time, from `0_group_info.csv` (see Step 1); the index name carries the unit, e.g. `time [min]`
- Outputs to `0_analysis_output/`: one CSV per image directory, `d_mu.csv` (all diameters), `dstats_by_volume.csv`, PDF plots
- `2_ParticleDataProcess_v07.py` is kept as the previous version. It additionally ran every image through a scikit-learn streak classifier and could drop streak-flagged particles; v08 removes that entirely, so every image passing the three filters contributes to the PSD. v07 also crashes on the second image directory (`Readme.txt` opened with mode `'x'` inside the per-directory loop); v08 writes no `Readme.txt` at all, since its only purpose was reporting streak counts

**Step 3 — `3_ParticleDataVisualization_v02.py`**
- Reads `d_mu.csv` and plots a single PSD or averaged range of PSDs
- Optionally bins onto LISST size classes from `../LISST_bins/LISST_bins_sphere.csv`

### Profile Processing (`Code/1_Profile_Processing/`)

Jupyter notebooks for FlocARAZI vertical profile deployments. Syncs image timestamps with CastAway CTD time series, assigns depth/temperature/salinity to each particle, and produces depth-binned PSDs and concentration proxies. The cast directory is read from `0_CastPath.csv` (column `profile_path`).

**Step 1 — `1_ProfileProcStep1_extract_CTD_and_image_info_v02.ipynb`**
- Reads the raw CTD time series, the CastAway-processed CTD profile, and the image folder; applies an `hrC`/`minC`/`scC` clock correction to align CTD time to the camera stamp
- Writes into the cast directory: `CTD-timeseries.csv`, `CTD-profile.csv`, `ImageTime.csv`, `Depth.csv` (total flow depth), and `Data-Breakpoints.csv` (the set of time-window groupings for the cast)
- Breakpoints are the time windows the cast is grouped by (e.g. `surface`, `mid-depth`, `bottom`, `profile`). They are currently entered manually as `names` / `startS` / `endS` arrays in seconds

**`breakpoint_detection.py`** — standalone helper for finding breakpoints automatically instead of by hand. `detect_constant_depth_periods(ctd_df, ...)` Savitzky-Golay–smooths `Depth [m]`, flags spans whose depth change over `window_size` samples stays under `depth_threshold`, drops spans shorter than `min_duration`, and returns `(breakpoints, names, depth_smooth)` with names `breakpoint_1…N`. Expects a CTD dataframe with `Depth [m]` and `Time (Seconds)` columns. Note: no notebook imports it yet — Step 1 still uses the manual arrays.

**Step 2 — `2_ProfileProcStep2_Build_MasterParticleData_v01.ipynb`**
- Joins the Step 1 files against the Step 1 image processing output (`0_analysis_output/001.csv` by default) to tag every particle with date, time, depth, salinity, and temperature
- CTD records are averaged to one row per second before the join
- Writes `0_analysis_output/particle_profile_data.csv` — the master particle dataset for Steps 3A/3B

**Steps 3A/3B** — two alternative ways to reduce `particle_profile_data.csv`:
- **3A (`..Process_GroupedData_v02.ipynb`)** groups by the Step 1 breakpoints (or treats the whole cast as one group)
- **3B (`..Process_ProfileData_v01.ipynb`)** groups by depth bins or particle-count bins
- Both share the Step 2 sizing parameters (`muperpix`, `darea`, `vdist`, `useLISST`) and write `ProcData_*` files into the cast directory: `ProcData_0_GroupSummary.csv` / `ProcData_0_ProfileSummary.csv`, `ProcData_1_timeseries-<location>.csv`, and `ProcData_2_PDF-/CDF-<location>.csv`

**Step 4 — `4_CompilingResults_v01.ipynb`**
- Points at a directory holding many processed casts (one level above the individual cast folders), globs each cast's `ProcData_0_GroupSummary.csv` and `Depth.csv`, and concatenates them into `AllCasts_Group_Summaries.csv` with `Cast` and `Flow Depth [m]` columns prepended
- Plots D16/D50/D84 versus time by location; `UTCdiff` shifts computer time to UTC

### Distribution Analysis (`Code/2_Distribution_Analysis/`)

`Settling_Velocity_SingleDist.ipynb` takes finished PSDs and computes fractal settling velocities. Inputs are `datadir` plus a `resultsfolder`/`station` label and a `distlocation` array naming the distributions to load; parameters are temperature `T`, salinity `S`, primary particle size `dp`, fractal dimension `nf`, and sediment density `rhos`. Outputs `<station>_sizestats_ws_calcs.csv` with d16/d50/d84, the volume-weighted average `ws_avg_mm_s`, and the parameters used.

### Supporting files
- `ij.jar` — ImageJ distribution bundled with the repo (required for headless execution)
- `models/streak_remove_1.pickle` — scikit-learn streak classifier trained on Rio de la Plata data; used only by the legacy `2_ParticleDataProcess_v07.py`
- `LISST_bins/` — three CSV files defining LISST-equivalent size bin edges (sphere, random, random-ext)

## Key Domain Conventions

- Particle diameter is derived from area by default (`darea=1`): `d = sqrt(4*A/π)` in µm; set `darea=0` to use ellipse minor axis instead
- Volume-weighted PSDs are standard (`vdist=1`); volume units are µL (1 µm³ = 1e-9 µL)
- Images named with embedded timestamps (e.g., `SWP1-01092021134459-13.Bmp`); the timestamp field index in the filename split must be adjusted per dataset in the profile notebooks
- `filemod_list.csv` in each image directory links original filenames → renamed copies → modification times; it also signals whether a directory has already been processed
