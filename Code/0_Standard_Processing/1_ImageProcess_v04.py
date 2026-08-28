# ............................................................................
# Organizes files and then runs an ImageJ macro to obtain particle size information
# ............................................................................

# user inputs
# ............................................................................
where = "0_Paths.csv"  # manage paths with the 0_Paths.csv file if you don't run the file locally

ts = 1  # enter 1 if you want PSD time series output. Enter zero for a single PSD. If "1" then a dir_size must be specified
dir_size = 5  # number of images to include per distribution, ignored if ts = 0

group_time = 30  # length of real time that one group of dir_size images represents
group_time_unit = "s"  # unit for group_time, e.g. "min", "s", "hr". Both values are
# written to 0_group_info.csv and picked up by 2_ParticleDataProcess_v08.py to set the
# time column of its output csv files and the time axis of its plots

start_time = 0  # real time already elapsed when the first image was taken, in the
# same unit as group_time. With start_time = 56 and group_time = 30, the first group is
# reported at 86, the second at 116, and so on. Step 2 uses it for the time column of its
# output csv files and the time axis of its plots. Leave at 0 for time measured from the
# first image

group_time_tol = 0.2  # group_time above is taken as correct, but it is checked against
# the image modification times and a warning is printed if the two disagree by more than
# this fraction. Set to 0 to turn the check off
image_type = ".jpg"  # enter the file extention for your images

# IJcode = "ImageJ-macros/ImageJ_code_diff_v02.txt"
IJcode = "ImageJ-macros/ImageJ_code_diff_v02_suppressoutput.txt"
# IJcode = "ImageJ-macros/ImageJ_code_diff_watershed_v02.txt"

java_cmd = "java"  # java used to run ImageJ. The macOS system java 8 disables its JIT
# partway through a run ("CodeCache is full"), which still gives correct output but runs
# interpreted and slow. Point this at a modern arm64 JDK to avoid that, for example
# java_cmd = "/opt/homebrew/opt/openjdk/bin/java". Flags can be appended, e.g. "java -Xmx4g"

overlap = 1  # copy the first image of the next batch into each batch as one extra
# frame, so the last image of the batch also gets differenced. Enter 0 for the
# original behaviour, where a batch of dir_size images yields dir_size-1 datasets.
# The copied image still starts the next batch as its own first image.

rmsrc = 0  # remove source images variable. enter 0 for "no" and "1" for yes
subdirp = (
    0  # process all sub directories (subdirp = 1) or just the local one (subdirp = 0)
)
# ............................................................................

# imports

import numpy as np
import os
from os.path import exists
from glob import glob
from shutil import copyfile
import shutil
import pandas as pd
import time

time1 = time.time()
from datetime import datetime

# how many seconds each accepted group_time_unit represents

seconds_per_unit = {
    "s": 1.0,
    "sec": 1.0,
    "secs": 1.0,
    "second": 1.0,
    "seconds": 1.0,
    "min": 60.0,
    "mins": 60.0,
    "minute": 60.0,
    "minutes": 60.0,
    "hr": 3600.0,
    "hrs": 3600.0,
    "hour": 3600.0,
    "hours": 3600.0,
}


def check_group_time(modtime_list, dir_size, group_time, group_time_unit, tol, ts):
    """Sanity check the declared group_time against the image modification times.

    The declared value stays authoritative and is what gets written to
    0_group_info.csv. This only prints a warning, so that a group_time left over
    from a previous dataset does not silently mislabel the step 2 time axis.
    """

    # with ts = 0 all the images form a single PSD, so there is no grouping
    # interval to check and the comparison would be meaningless

    if ts == 0:
        return

    if tol <= 0:
        return

    unit = str(group_time_unit).lower()
    if unit not in seconds_per_unit:
        print("  group_time check skipped: unrecognised unit", group_time_unit)
        return

    if len(modtime_list) < 2:
        return

    # median spacing is used so that a gap in the images does not skew the estimate

    dt = np.median(np.diff(np.sort(np.array(modtime_list))))
    if dt <= 0:
        print("  group_time check skipped: image timestamps are not increasing")
        return

    measured = dt * dir_size / seconds_per_unit[unit]

    if abs(measured - group_time) > tol * group_time:
        print(
            "  WARNING: group_time is set to",
            group_time,
            group_time_unit,
            "but the image timestamps put",
            dir_size,
            "images at",
            round(measured, 3),
            group_time_unit + ".",
            "Update group_time, or ignore this if the file times are unreliable.",
        )


if where == "local":
    master_path = os.getcwd()
    CodePath = master_path
else:
    paths = pd.read_csv(where)
    master_path = str(paths.iloc[0, 1]).strip()  # strip guards against a stray
    # space after the comma in 0_Paths.csv, which would otherwise be part of the path
    CodePath = os.getcwd()

# for processing all folders in directory ------------------------------------------

if subdirp == 1:
    folders = sorted(glob(master_path + "/*/"))

    for i in range(0, len(folders)):
        startTime = datetime.now()

        os.chdir(folders[i])
        print("working on", folders[i])

        path_main = folders[i]

        # get rid of previous processing files... will delete all folders in working directory

        if exists("filemod_list.csv") == True:
            profolders = glob(path_main + "/*/")
            os.remove("filemod_list.csv")
            for i in range(0, len(profolders)):
                shutil.rmtree(profolders[i])

        # time series or no?

        if ts == 0:
            dir_size = len(sorted(glob(path_main + "/*" + image_type)))

        # find files and re-organize them into directories with each directory containing the images to aggregate into a single PSD

        sorted_files = sorted(glob(path_main + "/*" + image_type))
        file_names = [os.path.basename(x) for x in sorted_files]

        modtime = np.zeros(len(sorted_files))
        for i in range(len(sorted_files)):
            modtime[i] = os.stat(sorted_files[i]).st_mtime
        modtime_list = modtime.tolist()

        check_group_time(
            modtime_list, dir_size, group_time, group_time_unit, group_time_tol, ts
        )

        src_list = []
        dst_list = []

        pad = len(
            str(dir_size + overlap)
        )  # padding length for renaming and sorting of images within each directory

        digits_in_folder_name = 3

        for f in np.arange(np.floor(len(sorted_files) / dir_size)):
            print(
                "Moving files to directory",
                int(f + 1),
                "of",
                int(np.floor(len(sorted_files) / dir_size)),
            )
            psd_path = path_main + "/" + str(int(f + 1)).zfill(digits_in_folder_name)
            os.mkdir(psd_path)

            # the difference macro pairs consecutive images, so a batch of N images
            # yields only N-1 datasets. Copy in the first image of the next batch as
            # an extra frame so the last image of this batch is differenced too. That
            # copy is not recorded in filemod_list.csv and is not deleted by rmsrc --
            # it gets both when it starts the next batch as its own first image.

            n_copy = dir_size
            if overlap == 1 and int(f) * dir_size + dir_size < len(sorted_files):
                n_copy = dir_size + 1

            for g in np.arange(n_copy):
                src = os.path.join(path_main, sorted_files[int(f) * dir_size + g])
                dst = os.path.join(psd_path, (str(int(g + 1)).zfill(pad) + image_type))
                copyfile(src, dst)

                if g < dir_size:
                    src_list.append(src)
                    dst_list.append(dst)

                    if rmsrc == 1:
                        os.remove(src)

        # save dataframe of source, destination and modified time

        files_mod_df = pd.DataFrame(
            list(zip(file_names, modtime_list, src_list, dst_list)),
            columns=["file_Name", "T_mod", "Src", "Dst"],
        )
        file_modlist_csv = path_main + "/filemod_list.csv"
        files_mod_df.to_csv(file_modlist_csv, index=False)

        # save the grouping time base so that step 2 can label its output

        group_info_df = pd.DataFrame(
            [[dir_size, group_time, group_time_unit, start_time]],
            columns=["dir_size", "group_time", "group_time_unit", "start_time"],
        )
        group_info_df.to_csv(path_main + "/0_group_info.csv", index=False)

        # run ImageJ

        str1 = (
            java_cmd
            + ' -jar "'
            + CodePath
            + '/ij.jar" -batch "'
            + CodePath
            + "/"
            + IJcode
            + '"'
        )
        os.system(str1)

        # go back to the main folder
        print(
            "run time:",
            datetime.now() - startTime,
            "(",
            len(sorted_files),
            " images in folder)",
        )

        os.chdir(master_path)

    os.chdir(CodePath)

# for only the files in the current directory ------------------------------------------

if subdirp == 0:
    startTime = datetime.now()

    os.chdir(master_path)

    path_main = master_path

    # get rid of previous processing files... will delete all folders in working directory

    if exists("filemod_list.csv") == True:
        profolders = glob(path_main + "/*/")
        os.remove("filemod_list.csv")
        for i in range(0, len(profolders)):
            shutil.rmtree(profolders[i])

    # time series or no?

    if ts == 0:
        dir_size = len(sorted(glob(path_main + "/*" + image_type)))

    # find files and re-organize them into directories with each directory containing the images to aggregate into a single PSD

    sorted_files = sorted(glob(path_main + "/*" + image_type))
    file_names = [os.path.basename(x) for x in sorted_files]

    modtime = np.zeros(len(sorted_files))
    for i in range(len(sorted_files)):
        modtime[i] = os.stat(sorted_files[i]).st_mtime
    modtime_list = modtime.tolist()

    check_group_time(
        modtime_list, dir_size, group_time, group_time_unit, group_time_tol, ts
    )

    src_list = []
    dst_list = []

    pad = len(
        str(dir_size + overlap)
    )  # padding length for renaming and sorting of images within each directory

    digits_in_folder_name = 3

    for f in np.arange(np.floor(len(sorted_files) / dir_size)):
        print(
            "Moving files to directory",
            int(f + 1),
            "of",
            int(np.floor(len(sorted_files) / dir_size)),
        )
        psd_path = path_main + "/" + str(int(f + 1)).zfill(digits_in_folder_name)
        os.mkdir(psd_path)

        # the difference macro pairs consecutive images, so a batch of N images
        # yields only N-1 datasets. Copy in the first image of the next batch as an
        # extra frame so the last image of this batch is differenced too. That copy
        # is not recorded in filemod_list.csv and is not deleted by rmsrc -- it gets
        # both when it starts the next batch as its own first image.

        n_copy = dir_size
        if overlap == 1 and int(f) * dir_size + dir_size < len(sorted_files):
            n_copy = dir_size + 1

        for g in np.arange(n_copy):
            src = os.path.join(path_main, sorted_files[int(f) * dir_size + g])
            dst = os.path.join(psd_path, (str(int(g + 1)).zfill(pad) + image_type))
            copyfile(src, dst)

            if g < dir_size:
                src_list.append(src)
                dst_list.append(dst)

                if rmsrc == 1:
                    os.remove(src)

    # save dataframe of source, destination and modified time

    files_mod_df = pd.DataFrame(
        list(zip(file_names, modtime_list, src_list, dst_list)),
        columns=["file_Name", "T_mod", "Src", "Dst"],
    )
    file_modlist_csv = path_main + "/filemod_list.csv"
    files_mod_df.to_csv(file_modlist_csv, index=False)

    # save the grouping time base so that step 2 can label its output

    group_info_df = pd.DataFrame(
        [[dir_size, group_time, group_time_unit, start_time]],
        columns=["dir_size", "group_time", "group_time_unit", "start_time"],
    )
    group_info_df.to_csv(path_main + "/0_group_info.csv", index=False)

    # run ImageJ

    str1 = (
        java_cmd
        + ' -jar "'
        + CodePath
        + '/ij.jar" -batch "'
        + CodePath
        + "/"
        + IJcode
        + '"'
    )
    os.system(str1)

    print(
        "run time:",
        datetime.now() - startTime,
        "(",
        len(sorted_files),
        " images in folder)",
    )
    os.chdir(CodePath)
