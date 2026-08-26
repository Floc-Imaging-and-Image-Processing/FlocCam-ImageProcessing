# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FlocCam-ImageProcessing is a Python/ImageJ pipeline for sizing suspended particles (flocs, mud, sand) captured by the FlocARAZI in-situ imaging instrument. The workflow runs sequentially: Python organizes raw images into batches → ImageJ extracts particle measurements → Python filters, sizes, and plots the particle size distributions (PSDs).

## Running the Pipeline

Scripts must be run from within their containing directory (they use relative paths to `ij.jar`, macros, and models):

```bash
cd Code/0_Standard_Processing
python 1_ImageProcess_v04.py        # Step 1: organize images + run ImageJ
python 2_ParticleDataProcess_v07.py # Step 2: filter particles + compute PSDs
python 3_ParticleDataVisualization_v02.py # Step 3: plot individual PSDs
```

Profile processing (vertical water-column casts) uses Jupyter notebooks in `Code/1_Profile_Processing/`, run in order (Step 1 → Step 2 → Step 3A/3B → Step 4).

## Path Configuration

All scripts read input/output paths from a CSV rather than hardcoded paths:

- **`Code/0_Standard_Processing/0_Paths.csv`** — first row is the `ImagePath` (folder containing raw image subfolders); subsequent `ProcessPath` rows specify which subfolders to analyze
- **`Code/1_Profile_Processing/0_CastPath.csv`** — path to the profile data directory

Set `where = "local"` in any script to use `os.getcwd()` instead of the CSV.

## Code Architecture

### Standard Processing (`Code/0_Standard_Processing/`)

**Step 1 — `1_ImageProcess_v04.py`**
- Groups raw images into numbered subdirectories (e.g., `001/`, `002/`) of `dir_size` images each
- Records source/destination/modification-time in `filemod_list.csv` (used later for timestamp extraction)
- Invokes ImageJ headlessly: `java -jar ij.jar -batch <macro>`
- Key user parameters at top: `ts` (time series vs. single PSD), `dir_size`, `image_type`, `subdirp` (recurse subdirs), `rmsrc` (delete originals)

**ImageJ macros (`ImageJ-macros/`)**
- Implement a *difference-of-images* technique: subtracts consecutive image pairs to isolate particles from background
- Applies Triangle threshold → binary → erode → fill holes → `Analyze Particles`
- Outputs one `.txt` file per image pair containing particle measurements (area, perimeter, bounding box, ellipse fit, etc.)
- `_suppressoutput` variant is the default; `_watershed` variant adds watershed segmentation for touching particles

**Step 2 — `2_ParticleDataProcess_v07.py`**
- Reads ImageJ `.txt` output files; applies three filters: focus (MaxGreyValue threshold), edge proximity, and minimum area
- Streak detection via a pre-trained scikit-learn classifier (`models/streak_remove_1.pickle`) — features are angle std, major/minor axis ratio, fraction in focus, area/perimeter ratio
- Converts pixel measurements to microns using `muperpix` (0.925 µm/px for FlocARAZI, 1.28 for lab cam)
- Computes d16/d50/d84 from either volume-weighted (`vdist=1`) or d^w-weighted distributions
- Outputs to `0_analysis_output/`: per-directory CSVs, `d_mu.csv` (all diameters), `dstats_by_volume.csv`, streak summary, PDF plots

**Step 3 — `3_ParticleDataVisualization_v02.py`**
- Reads `d_mu.csv` and plots a single PSD or averaged range of PSDs
- Optionally bins onto LISST size classes from `../LISST_bins/LISST_bins_sphere.csv`

### Profile Processing (`Code/1_Profile_Processing/`)

Jupyter notebooks for FlocARAZI vertical profile deployments. Syncs image timestamps with CastAway CTD time series, assigns depth/temperature/salinity to each particle, and produces depth-binned PSDs and concentration proxies.

### Supporting files
- `ij.jar` — ImageJ distribution bundled with the repo (required for headless execution)
- `models/streak_remove_1.pickle` — scikit-learn model trained on Rio de la Plata streak data
- `LISST_bins/` — three CSV files defining LISST-equivalent size bin edges (sphere, random, random-ext)

## Key Domain Conventions

- Particle diameter is derived from area by default (`darea=1`): `d = sqrt(4*A/π)` in µm; set `darea=0` to use ellipse minor axis instead
- Volume-weighted PSDs are standard (`vdist=1`); volume units are µL (1 µm³ = 1e-9 µL)
- Images named with embedded timestamps (e.g., `SWP1-01092021134459-13.Bmp`); the timestamp field index in the filename split must be adjusted per dataset in the profile notebooks
- `filemod_list.csv` in each image directory links original filenames → renamed copies → modification times; it also signals whether a directory has already been processed
