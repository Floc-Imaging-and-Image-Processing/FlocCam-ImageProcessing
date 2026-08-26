"""
flocam_imgproc_fast.py — OpenCV/numpy-accelerated particle detection for FlocCam images.

Same processing chain and output schema as flocam_imgproc.py, but replaces the
skimage.measure.regionprops_table measurement step (which builds a Python object
per detected blob) with cv2 contour ops + numpy bincount. Full-image morphology
(erosion) and edge detection (Sobel) are also switched to cv2's SIMD-optimized
implementations. On 12MP FlocCam images with thousands of raw candidate blobs
per frame, this is ~5-10x faster than flocam_imgproc.py with equivalent output.

Numeric notes vs. flocam_imgproc.py:
  - Area, Mean/StdDev/Min/Max intensity, Major/Minor axis length, and Angle are
    mathematically equivalent (same underlying moment/inertia-tensor formulas).
  - Perimeter uses cv2.arcLength (polygon perimeter along the pixel boundary)
    instead of skimage's Crofton-style approximation, so Perimeter, Circ.,
    Round, and Solidity will differ slightly (see flocam_imgproc_notes.md for
    a numeric comparison against the original).

Processing chain per image pair:
  1. edges  = Sobel magnitude of img1, clipped to uint8 0-255
  2. diff   = clip(img1 - img2, 0, 255)   [brightfield: dark particles on bright background]
  3. binary = diff > threshold_triangle(diff) → erode (cross) → fill holes
              [optional watershed for touching particles]
  4. Measure labeled regions; intensities redirected to edges (matches ImageJ redirect=edges)
  5. Filter: area >= min_area_px, circularity in [min_circ, max_circ], exclude border
  6. Write tab-separated .txt compatible with 2_ParticleDataProcess_v07.py

Dependencies: numpy, pandas, scipy, scikit-image (threshold_triangle only), opencv-python
"""

import os
from glob import glob

import numpy as np
import pandas as pd
import scipy.ndimage as ndi
import cv2
from skimage.filters import threshold_triangle

# 3x3 cross structuring element — matches ImageJ binary Erode (4-connected neighbourhood)
_CROSS = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

# 8-connectivity structure for scipy labeling — matches skimage.measure.label's
# default "full connectivity" for 2D arrays
_STRUCTURE8 = np.ones((3, 3), dtype=int)


def compute_edges(img: np.ndarray) -> np.ndarray:
    """
    Sobel edge magnitude matching ImageJ 'Find Edges', computed via cv2's
    SIMD-optimized Sobel instead of scipy.ndimage.convolve. cv2's default
    border handling (BORDER_REFLECT_101) matches scipy's mode="mirror", and
    the sign of the Sobel kernels doesn't matter since we take the magnitude.
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    return np.clip(mag, 0, 255).astype(np.uint8)


def make_binary(
    img1: np.ndarray, img2: np.ndarray, watershed: bool = False
) -> np.ndarray:
    """
    Build a binary particle mask from two consecutive grayscale images.
    Returns a bool array where True = particle foreground.

    Steps match the ImageJ macro:
      diff = img1 - img2 (clipped) → Triangle threshold → erode → fill holes
    """
    diff = np.clip(img1.astype(np.int16) - img2.astype(np.int16), 0, 255).astype(
        np.uint8
    )

    try:
        thresh = threshold_triangle(diff)
    except Exception:
        return np.zeros(diff.shape, dtype=bool)

    binary_u8 = (diff > thresh).astype(np.uint8)
    # borderValue=0 matches scipy/skimage's default border_value=0 (pixels
    # outside the array are treated as background, so foreground erodes at edges)
    eroded = cv2.erode(binary_u8, _CROSS, borderType=cv2.BORDER_CONSTANT, borderValue=0)

    # fill holes via cv2.floodFill from the border instead of
    # scipy.ndimage.binary_fill_holes — pixel-identical, ~15x faster on these
    # image sizes since it's a single flood fill instead of iterated dilation
    h, w = eroded.shape
    inv = ((1 - eroded) * 255).astype(np.uint8)
    flooded = inv.copy()
    ff_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    cv2.floodFill(flooded, ff_mask, (0, 0), 0)
    holes = (inv > 0) & (flooded > 0)
    binary = eroded.astype(bool) | holes

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
    return_contours: bool = False,
):
    """
    Label connected regions in binary and compute shape + intensity measurements.

    Intensities are taken from the edges image (matching ImageJ redirect=edges).

    Returns a DataFrame with the same 17 columns as flocam_imgproc.measure_particles:
      Area, Mean, StdDev, Min, Max, Perim., BX, BY, Width, Height,
      Major, Minor, Angle, Circ., AR, Round, Solidity

    If return_contours is True, returns (df, contours) where contours is a list
    of cv2 contours (in full-image pixel coordinates) for the particles that
    passed the filters, in the same row order as df — used to draw the
    diagnostic overlay images.

    Speed: area/intensity stats are computed with np.bincount over the whole
    label image at once (no per-region Python loop). Only perimeter, axis
    lengths, angle, and solidity need per-region cv2 contour calls, since
    those require the region's boundary — but cv2.findContours/moments are
    compiled C++ calls on small cropped arrays, ~10x faster than skimage's
    per-region RegionProperties object model.
    """
    empty_cols = [
        "Area", "Mean", "StdDev", "Min", "Max", "Perim.", "BX", "BY",
        "Width", "Height", "Major", "Minor", "Angle", "Circ.", "AR",
        "Round", "Solidity",
    ]

    labeled, num_labels = ndi.label(binary, structure=_STRUCTURE8)
    if num_labels == 0:
        empty_df = pd.DataFrame(columns=empty_cols)
        return (empty_df, []) if return_contours else empty_df

    h, w = binary.shape
    edges_f = edges.astype(np.float64)
    flat_labels = labeled.ravel()

    # area + intensity stats for every label at once via bincount (vectorized,
    # no per-region Python loop — this replaces skimage's per-region "Mean"/
    # "StdDev" computation, which was one of the slowest parts of the pipeline)
    counts = np.bincount(flat_labels, minlength=num_labels + 1).astype(np.float64)
    area = counts[1:]

    flat_edges = edges_f.ravel()
    sum_I = np.bincount(flat_labels, weights=flat_edges, minlength=num_labels + 1)[1:]
    sum_I2 = np.bincount(flat_labels, weights=flat_edges**2, minlength=num_labels + 1)[1:]
    mean_I = sum_I / area
    var_I = np.maximum(sum_I2 / area - mean_I**2, 0.0)
    std_I = np.sqrt(var_I)

    slices = ndi.find_objects(labeled)

    perim = np.zeros(num_labels)
    major = np.zeros(num_labels)
    minor = np.zeros(num_labels)
    angle = np.zeros(num_labels)
    solidity = np.ones(num_labels)
    bx = np.zeros(num_labels, dtype=int)
    by = np.zeros(num_labels, dtype=int)
    width = np.zeros(num_labels, dtype=int)
    height = np.zeros(num_labels, dtype=int)
    min_I = np.zeros(num_labels)
    max_I = np.zeros(num_labels)

    # per-region contours in full-image coordinates, kept only when the caller
    # wants overlays (drawing the detected particle outlines back onto the image)
    contours_full = [None] * num_labels if return_contours else None

    for idx in range(num_labels):
        lbl = idx + 1
        sl = slices[idx]
        y0, x0 = sl[0].start, sl[1].start
        by[idx], bx[idx] = y0, x0
        height[idx] = sl[0].stop - y0
        width[idx] = sl[1].stop - x0

        sub_bool = labeled[sl] == lbl
        sub = sub_bool.astype(np.uint8)

        # min/max of the region's edge intensities — computed on the small
        # cropped sub-array here rather than via scipy.ndimage.minimum/maximum
        # over the whole label image, which internally does a full argsort
        # per call and is disastrously slow with thousands of labels
        region_vals = edges_f[sl][sub_bool]
        min_I[idx] = region_vals.min()
        max_I[idx] = region_vals.max()

        cnts, _ = cv2.findContours(sub, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not cnts:
            continue
        cnt = cnts[0] if len(cnts) == 1 else max(cnts, key=cv2.contourArea)

        # offset the sub-array contour by the bbox origin so it lands in the
        # correct place on the full image (cv2 points are (x, y) = (col, row))
        if return_contours:
            contours_full[idx] = cnt + np.array([[x0, y0]], dtype=cnt.dtype)

        perim[idx] = cv2.arcLength(cnt, True)

        m = cv2.moments(sub, binaryImage=True)
        m00 = m["m00"]
        if m00 <= 0:
            continue

        # central moments from cv2 (x/y convention) feed the same inertia-tensor
        # eigenvalue formula skimage.regionprops uses (row/col convention) — the
        # eigenvalues (hence Major/Minor) are identical either way since they only
        # depend on the (symmetric) matrix's trace/determinant, not axis labeling
        mu20, mu02, mu11 = m["mu20"], m["mu02"], m["mu11"]
        a = mu20 / m00
        c = mu02 / m00
        b = -mu11 / m00
        common = np.sqrt(max(((a - c) / 2.0) ** 2 + b**2, 0.0))
        lam1 = (a + c) / 2.0 + common
        lam2 = max((a + c) / 2.0 - common, 0.0)
        major[idx] = 4.0 * np.sqrt(lam1)
        minor[idx] = 4.0 * np.sqrt(lam2)

        # orientation formula equivalent to skimage regionprops.orientation,
        # re-derived directly in cv2's x/y moment convention; converted to
        # ImageJ's 0-180 degrees-from-horizontal convention below
        orient = 0.5 * np.arctan2(-2.0 * b, c - a)
        angle[idx] = (90.0 - np.degrees(orient)) % 180.0

        hull = cv2.convexHull(cnt)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity[idx] = min(area[idx] / hull_area, 1.0)

    perim_safe = np.where(perim > 0, perim, np.inf)
    major_safe = np.where(major > 0, major, np.inf)
    minor_safe = np.where(minor > 0, minor, np.inf)

    circ = 4.0 * np.pi * area / perim_safe**2
    AR = major / minor_safe
    Round = 4.0 * area / (np.pi * major_safe**2)

    mask = (
        (area >= min_area_px)
        & (circ >= min_circ)
        & (circ <= max_circ)
        & (major > 0)
        & (minor > 0)
    )
    if exclude_border:
        mask &= (bx > 0) & (by > 0) & ((bx + width) < w) & ((by + height) < h)

    df = pd.DataFrame(
        {
            "Area": area[mask],
            "Mean": mean_I[mask],
            "StdDev": std_I[mask],
            "Min": min_I[mask],
            "Max": max_I[mask],
            "Perim.": perim[mask],
            "BX": bx[mask],
            "BY": by[mask],
            "Width": width[mask],
            "Height": height[mask],
            "Major": major[mask],
            "Minor": minor[mask],
            "Angle": angle[mask],
            "Circ.": circ[mask],
            "AR": AR[mask],
            "Round": Round[mask],
            "Solidity": solidity[mask],
        }
    )

    if return_contours:
        kept = [
            contours_full[i]
            for i in range(num_labels)
            if mask[i] and contours_full[i] is not None
        ]
        return df, kept

    return df


def _save_overlays(
    img1: np.ndarray,
    img2: np.ndarray,
    edges: np.ndarray,
    contours: list,
    overlay_dir: str,
    base_name: str,
) -> None:
    """
    Write diagnostic images for one image pair, mirroring the non-suppressed
    ImageJ macro (ImageJ_code_diff_v02.txt). For checking that the detection
    is working, three images are saved into overlay_dir:

      <base>_outlines.jpg : original grayscale img1 with detected particle
                            outlines drawn in green
      <base>_diff.jpg     : the difference image that gets thresholded, with the
                            same outlines — shows what the detector actually sees
      <base>_edges.jpg    : the Sobel edge image used for the focus (Max) filter

    They are written to a separate overlay_dir (not the batch folder) so they
    are never picked up as input frames by the *image_ext glob on a re-run.
    """
    os.makedirs(overlay_dir, exist_ok=True)

    # outlines on the original grayscale image
    outlines = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(outlines, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(overlay_dir, base_name + "_outlines.jpg"), outlines)

    # outlines on the difference image (same subtraction make_binary thresholds)
    diff = np.clip(img1.astype(np.int16) - img2.astype(np.int16), 0, 255).astype(
        np.uint8
    )
    diff_bgr = cv2.cvtColor(diff, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(diff_bgr, contours, -1, (0, 255, 0), 1)
    cv2.imwrite(os.path.join(overlay_dir, base_name + "_diff.jpg"), diff_bgr)

    # the edge image (grayscale)
    cv2.imwrite(os.path.join(overlay_dir, base_name + "_edges.jpg"), edges)


def process_image_pair(
    img1_path: str,
    img2_path: str,
    output_txt: str,
    watershed: bool = False,
    min_area_px: int = 16,
    min_circ: float = 0.075,
    max_circ: float = 0.99,
    save_overlays: bool = False,
    overlay_dir: str = None,
) -> int:
    """
    Full pipeline for one image pair → measurement .txt file.
    Returns number of particles detected.

    If save_overlays is True, also writes diagnostic overlay images (outlines,
    difference, edges) into overlay_dir — see _save_overlays. This is the
    Python equivalent of choosing the non-suppressed ImageJ macro
    (ImageJ_code_diff_v02.txt instead of ..._suppressoutput.txt).
    """
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    edges = compute_edges(img1)
    binary = make_binary(img1, img2, watershed=watershed)
    result = measure_particles(
        binary,
        edges,
        min_area_px=min_area_px,
        min_circ=min_circ,
        max_circ=max_circ,
        return_contours=save_overlays,
    )

    if save_overlays:
        df, contours = result
        if overlay_dir is None:
            overlay_dir = os.path.join(os.path.dirname(img1_path), "overlays")
        base_name = os.path.splitext(os.path.basename(img1_path))[0]
        _save_overlays(img1, img2, edges, contours, overlay_dir, base_name)
    else:
        df = result

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
    save_overlays: bool = False,
) -> None:
    """
    Process all sequential image pairs in dir_path.
    Writes one .txt file per pair named after img1 (without extension).

    If delete_processed is True, removes img1 after each pair is processed,
    matching the ImageJ macro's File.delete behaviour to save disk space.

    If save_overlays is True, also writes diagnostic overlay images into an
    "overlays" subfolder of dir_path (see _save_overlays). The overlays go in a
    subfolder so they are never re-globbed as input frames.
    """
    images = sorted(glob(os.path.join(dir_path, "*" + image_ext)))
    n_pairs = len(images) - 1

    if n_pairs <= 0:
        print(f"  No image pairs found in {dir_path}")
        return

    overlay_dir = os.path.join(dir_path, "overlays") if save_overlays else None

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
            save_overlays=save_overlays,
            overlay_dir=overlay_dir,
        )
        print(f"  {os.path.basename(img1_path)}: {n} particles")

        if delete_processed:
            os.remove(img1_path)
