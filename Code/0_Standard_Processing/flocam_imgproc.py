"""
flocam_imgproc.py — Pure-Python image processing for FlocCam particle analysis.

Replaces the ImageJ macro pipeline (ImageJ_code_diff_v02_suppressoutput.txt).
Processing chain per image pair:
  1. edges  = Sobel magnitude of img1, clipped to uint8 0-255
  2. diff   = clip(img1 - img2, 0, 255)   [brightfield: dark particles on bright background]
  3. binary = diff > threshold_triangle(diff) → erode (cross) → fill holes
              [optional watershed for touching particles]
  4. Measure labeled regions; intensities redirected to edges (matches ImageJ redirect=edges)
  5. Filter: area >= min_area_px, circularity in [min_circ, max_circ], exclude border
  6. Write tab-separated .txt compatible with 2_ParticleDataProcess_v07.py

Subtraction direction: diff = img1 - img2 produces a large positive value wherever a dark
particle appears in img2 (img1 bright background minus img2 dark particle). Swap to
img2 - img1 for dark-field cameras where particles are brighter than the background.

Dependencies: numpy, pandas, scipy, scikit-image
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import cv2
from skimage.filters import threshold_triangle
from skimage.measure import label, regionprops_table
from skimage.morphology import binary_erosion, diamond

# 3x3 cross structuring element — matches ImageJ binary Erode (4-connected neighbourhood)
_CROSS = diamond(1)


def compute_edges(img: np.ndarray) -> np.ndarray:
    """
    Sobel edge magnitude matching ImageJ 'Find Edges'.
    Applies unscaled Sobel kernels to the float image, clips result to uint8 0-255.
    A perfect step edge (0→255) produces magnitude ~1020 before clipping,
    so any sharp edge saturates at 255, matching ImageJ behaviour.
    """
    f = img.astype(np.float64)
    kx = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float64)
    ky = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float64)
    gx = ndi.convolve(f, kx, mode="mirror")
    gy = ndi.convolve(f, ky, mode="mirror")
    mag = np.sqrt(gx**2 + gy**2)
    return np.clip(mag, 0, 255).astype(np.uint8)


def make_binary(
    img1: np.ndarray, img2: np.ndarray, watershed: bool = False
) -> np.ndarray:
    """
    Build a binary particle mask from two consecutive grayscale images.
    Returns a bool array where True = particle foreground.

    Steps match the ImageJ macro:
      diff = img2 - img1 (clipped) → Triangle threshold → erode → fill holes
    """
    # brightfield: subtract img2 from img1 so dark particles in img2 produce
    # large positive values (bright background minus dark particle)
    diff = np.clip(img1.astype(np.int16) - img2.astype(np.int16), 0, 255).astype(
        np.uint8
    )

    try:
        thresh = threshold_triangle(diff)
    except Exception:
        return np.zeros(diff.shape, dtype=bool)

    binary = diff > thresh
    binary = binary_erosion(binary, footprint=_CROSS)
    binary = ndi.binary_fill_holes(binary)

    if watershed:
        from skimage.segmentation import watershed as ws

        dist = ndi.distance_transform_edt(binary)
        local_max = (ndi.maximum_filter(dist, size=3) == dist) & binary
        markers, _ = ndi.label(local_max)
        binary = ws(-dist, markers, mask=binary) > 0

    return binary


def measure_particles(
    binary: np.ndarray,
    edges: np.ndarray,
    min_area_px: int = 16,
    min_circ: float = 0.075,
    max_circ: float = 0.99,
    exclude_border: bool = True,
) -> pd.DataFrame:
    """
    Label connected regions in binary and compute shape + intensity measurements.

    Intensities are taken from the edges image (matching ImageJ redirect=edges).
    The Max column is used by the downstream focus filter in
    2_ParticleDataProcess_v07.py (focus = 100).

    Returns a DataFrame with 17 columns:
      Area, Mean, StdDev, Min, Max, Perim., BX, BY, Width, Height,
      Major, Minor, Angle, Circ., AR, Round, Solidity

    Uses regionprops_table + scipy labeled statistics so all property computation
    and filtering is vectorized — no Python loop over individual particles.
    """
    labeled = label(binary)
    if labeled.max() == 0:
        return pd.DataFrame(
            columns=[
                "Area",
                "Mean",
                "StdDev",
                "Min",
                "Max",
                "Perim.",
                "BX",
                "BY",
                "Width",
                "Height",
                "Major",
                "Minor",
                "Angle",
                "Circ.",
                "AR",
                "Round",
                "Solidity",
            ]
        )

    h, w = binary.shape
    edges_f = edges.astype(np.float64)

    # compute all shape and basic intensity properties at once (C-level loop)
    t = pd.DataFrame(
        regionprops_table(
            labeled,
            intensity_image=edges_f,
            properties=[
                "label",
                "area",
                "perimeter",
                "bbox",
                "axis_major_length",
                "axis_minor_length",
                "orientation",
                "solidity",
                "mean_intensity",
                "min_intensity",
                "max_intensity",
            ],
        )
    )

    # std dev is not in regionprops_table; scipy computes it per-label in C
    t["std_intensity"] = ndi.standard_deviation(
        edges_f, labeled, index=t["label"].values
    )

    # derived quantities computed as numpy array operations (no Python loop)
    # guard against zero denominators with np.where rather than per-element checks
    perim_safe = np.where(t["perimeter"] > 0, t["perimeter"], np.inf)
    major_safe = np.where(t["axis_major_length"] > 0, t["axis_major_length"], np.inf)
    minor_safe = np.where(t["axis_minor_length"] > 0, t["axis_minor_length"], np.inf)

    t["circ"] = 4.0 * np.pi * t["area"] / perim_safe**2
    t["AR"] = t["axis_major_length"] / minor_safe
    t["Round"] = 4.0 * t["area"] / (np.pi * major_safe**2)

    # scikit-image orientation: radians from x-axis, range [-π/2, π/2]
    # ImageJ Angle: degrees, range [0, 180] from horizontal
    t["Angle"] = (90.0 - np.degrees(t["orientation"])) % 180.0

    # bbox columns from regionprops_table:
    # bbox-0=min_row (BY), bbox-1=min_col (BX), bbox-2=max_row, bbox-3=max_col
    t["BX"] = t["bbox-1"]
    t["BY"] = t["bbox-0"]
    t["Width"] = t["bbox-3"] - t["bbox-1"]
    t["Height"] = t["bbox-2"] - t["bbox-0"]

    # apply all filters as vectorized boolean masks
    mask = (
        (t["area"] >= min_area_px)
        & (t["circ"] >= min_circ)
        & (t["circ"] <= max_circ)
        & (t["axis_major_length"] > 0)
        & (t["axis_minor_length"] > 0)
    )
    if exclude_border:
        mask &= (t["BX"] > 0) & (t["BY"] > 0) & (t["bbox-3"] < w) & (t["bbox-2"] < h)
    t = t[mask]

    return pd.DataFrame(
        {
            "Area": t["area"].astype(float),
            "Mean": t["mean_intensity"],
            "StdDev": t["std_intensity"],
            "Min": t["min_intensity"],
            "Max": t["max_intensity"],
            "Perim.": t["perimeter"],
            "BX": t["BX"].astype(int),
            "BY": t["BY"].astype(int),
            "Width": t["Width"].astype(int),
            "Height": t["Height"].astype(int),
            "Major": t["axis_major_length"],
            "Minor": t["axis_minor_length"],
            "Angle": t["Angle"],
            "Circ.": t["circ"],
            "AR": t["AR"],
            "Round": t["Round"],
            "Solidity": t["solidity"],
        }
    )


def process_image_pair(
    img1_path: str,
    img2_path: str,
    output_txt: str,
    watershed: bool = False,
    min_area_px: int = 16,
    min_circ: float = 0.075,
    max_circ: float = 0.99,
) -> int:
    """
    Full pipeline for one image pair → measurement .txt file.
    Returns number of particles detected.
    """
    # cv2.IMREAD_GRAYSCALE returns uint8 directly — faster than skimage imread for BMPs
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    edges = compute_edges(img1)
    binary = make_binary(img1, img2, watershed=watershed)
    df = measure_particles(
        binary, edges, min_area_px=min_area_px, min_circ=min_circ, max_circ=max_circ
    )

    df.to_csv(output_txt, sep="\t", index=False, float_format="%.3f")
    return len(df)


def process_directory(
    dir_path: str,
    image_ext: str = ".Bmp",
    watershed: bool = False,
    min_area_px: int = 16,
    min_circ: float = 0.075,
    max_circ: float = 0.99,
    delete_processed: bool = False,
) -> None:
    """
    Process all sequential image pairs in dir_path.
    Writes one .txt file per pair named after img1 (without extension).
    Matches the per-directory processing of the ImageJ macro.

    If delete_processed is True, removes img1 after each pair is processed,
    matching the ImageJ macro's File.delete behaviour to save disk space.
    """
    images = sorted(glob(os.path.join(dir_path, "*" + image_ext)))
    n_pairs = len(images) - 1

    if n_pairs <= 0:
        print(f"  No image pairs found in {dir_path}")
        return

    for j in range(n_pairs):
        img1_path = images[j]
        img2_path = images[j + 1]
        output_txt = os.path.splitext(img1_path)[0] + ".txt"

        n = process_image_pair(
            img1_path,
            img2_path,
            output_txt,
            watershed=watershed,
            min_area_px=min_area_px,
            min_circ=min_circ,
            max_circ=max_circ,
        )
        print(f"  {os.path.basename(img1_path)}: {n} particles")

        if delete_processed:
            os.remove(img1_path)
