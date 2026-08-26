# ............................................................................
# Organizes files and then runs an ImageJ macro to obtain particle size information
# ............................................................................

import os
import subprocess
from datetime import datetime
from glob import glob
from os.path import exists
from shutil import copyfile
import shutil

import pandas as pd

# user inputs
# ............................................................................
where = "0_Paths.csv"  # use "local" to use os.getcwd() instead of the CSV

ts = False  # False = single PSD per folder; True = PSD time series (requires dir_size)
dir_size = 60  # images per PSD batch; ignored when ts = False
image_type = ".jpg"  # enter the file extension for your images

# IJcode = "ImageJ-macros/ImageJ_code_diff_v02.txt"
IJcode = "ImageJ-macros/ImageJ_code_diff_v02_suppressoutput.txt"
# IJcode = "ImageJ-macros/ImageJ_code_diff_watershed_v02.txt"

rmsrc = False  # True = delete source images after copying to batch subdirectories
subdirp = False  # True = process all sub-directories; False = process only the local directory
# ............................................................................


def process_folder(path_main, ts, dir_size, image_type, rmsrc, CodePath, IJcode):
    """Organise images into numbered batch subdirectories and run ImageJ."""
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
    # to zero-padded integers so ImageJ processes them in the correct order
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

    # run ImageJ headlessly on the batch subdirectories;
    # cwd=path_main sets ImageJ's startup directory so getDirectory("startup")
    # in the macro resolves to path_main
    subprocess.run(
        [
            "java",
            "-jar",
            os.path.join(CodePath, "ij.jar"),
            "-batch",
            os.path.join(CodePath, IJcode),
        ],
        cwd=path_main,
    )

    print(
        "run time:",
        datetime.now() - startTime,
        f"({len(sorted_files)} images in folder)",
    )


# resolve paths: CodePath is the script directory (where ij.jar lives);
# master_path is the root folder containing the image data
CodePath = os.getcwd()

if where == "local":
    master_path = CodePath
else:
    paths = pd.read_csv(where)
    master_path = paths.iloc[0, 1]

# process each image subfolder under master_path (subdirp=True),
# or treat master_path itself as the single folder to process (subdirp=False)
if subdirp:
    for folder in sorted(glob(master_path + "/*/")):
        process_folder(folder, ts, dir_size, image_type, rmsrc, CodePath, IJcode)
else:
    process_folder(master_path, ts, dir_size, image_type, rmsrc, CodePath, IJcode)
