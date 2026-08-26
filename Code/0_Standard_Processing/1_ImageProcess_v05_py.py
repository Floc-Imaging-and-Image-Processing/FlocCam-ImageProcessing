# ............................................................................
# Organizes files and runs pure-Python particle detection (no ImageJ required).
# Parallel version of 1_ImageProcess_v04.py — replaces ImageJ macro with
# flocam_imgproc.py (scikit-image based difference-of-images pipeline).
# ............................................................................

# user inputs
# ............................................................................
where = "0_Paths.csv"  # use "local" to use os.getcwd() instead of the CSV

ts = False  # False = single PSD per folder; True = PSD time series (requires dir_size)
dir_size = 60  # images per PSD batch; ignored when ts = False
image_type = ".jpg"

watershed = False  # True = watershed segmentation for touching particles

save_overlays = True  # True = also save diagnostic images (particle outlines,
# difference, edges) into an "overlays" subfolder of each batch directory so you
# can check the detection. Python equivalent of the non-suppressed ImageJ macro.

rmsrc = False  # True = delete source images after copying to batch subdirectories
subdirp = False  # True = process all sub-directories; False = process only the local directory
# ............................................................................

import os
from datetime import datetime
from glob import glob
from os.path import exists
from shutil import copyfile
import shutil

import pandas as pd

import flocam_imgproc_fast


def process_folder(
    path_main, ts, dir_size, image_type, rmsrc, watershed, save_overlays
):
    """Organise images into numbered batch subdirectories and run particle detection."""
    startTime = datetime.now()
    print("working on", path_main)

    # if a previous run exists, delete its subdirectories and filemod_list.csv
    # so the folder is clean before re-processing
    filemod_csv = os.path.join(path_main, "filemod_list.csv")
    if exists(filemod_csv):
        for d in glob(path_main + "/*/"):
            shutil.rmtree(d)
        os.remove(filemod_csv)

    # collect all images in the folder; skip if none are found
    sorted_files = sorted(glob(path_main + "/*" + image_type))
    if not sorted_files:
        print("  no images found, skipping")
        return

    # for a single PSD (ts=False), put all images in one batch;
    # for a time series (ts=True), use the user-specified dir_size
    if not ts:
        dir_size = len(sorted_files)

    # record original filenames and file-system modification times;
    # mod times are used later to reconstruct image timestamps
    file_names = [os.path.basename(f) for f in sorted_files]
    modtime = [os.stat(f).st_mtime for f in sorted_files]

    # copy images into numbered subdirectories (001/, 002/, …), renaming them
    # to zero-padded integers so they are processed in the correct order
    src_list, dst_list = [], []
    pad = len(str(dir_size))
    n_dirs = len(sorted_files) // dir_size

    for f in range(n_dirs):
        print(f"Moving files to directory {f + 1} of {n_dirs}")
        psd_path = os.path.join(path_main, str(f + 1).zfill(3))
        os.mkdir(psd_path)

        for g in range(dir_size):
            src = sorted_files[f * dir_size + g]
            dst = os.path.join(psd_path, str(g + 1).zfill(pad) + image_type)
            copyfile(src, dst)
            src_list.append(src)
            dst_list.append(dst)
            if rmsrc:
                os.remove(src)

    # save a CSV linking original filenames → renamed copies → mod times;
    # the downstream scripts use this to recover timestamps from filenames
    pd.DataFrame(
        list(zip(file_names, modtime, src_list, dst_list)),
        columns=["file_Name", "T_mod", "Src", "Dst"],
    ).to_csv(filemod_csv, index=False)

    # run the pure-Python difference-of-images particle detection on each
    # batch subdirectory; writes one .txt measurement file per image pair
    for subfolder in sorted(glob(path_main + "/*/")):
        print("Processing subfolder:", subfolder)
        flocam_imgproc_fast.process_directory(
            subfolder,
            image_ext=image_type,
            watershed=watershed,
            save_overlays=save_overlays,
        )

    print(
        "run time:",
        datetime.now() - startTime,
        f"({len(sorted_files)} images in folder)",
    )


# resolve paths: master_path is the root folder containing the image data
if where == "local":
    master_path = os.getcwd()
else:
    paths = pd.read_csv(where)
    master_path = paths.iloc[0, 1]

# process each image subfolder under master_path (subdirp=True),
# or treat master_path itself as the single folder to process (subdirp=False)
if subdirp:
    for folder in sorted(glob(master_path + "/*/")):
        process_folder(
            folder, ts, dir_size, image_type, rmsrc, watershed, save_overlays
        )
else:
    process_folder(
        master_path, ts, dir_size, image_type, rmsrc, watershed, save_overlays
    )
