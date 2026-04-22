"""
Breakpoint Detection Module

This module provides functions to automatically detect periods of relatively constant depth
in CTD (Conductivity-Temperature-Depth) time series data using smoothing and slope analysis.
"""

import numpy as np
from scipy.signal import savgol_filter


def detect_constant_depth_periods(
    ctd_df,
    depth_threshold=0.1,
    min_duration=10,
    window_size=5,
    smooth_window=11,
    poly_order=3,
):
    """
    Automatically detect periods where depth is relatively constant.
    Uses smoothing to reduce noise before analyzing slope.

    Parameters:
    -----------
    ctd_df : pandas DataFrame
        CTD dataframe with 'Depth [m]' and 'Time (Seconds)' columns
    depth_threshold : float
        Maximum depth change (in meters) over window_size to consider "flat" (default: 0.1 m)
    min_duration : int
        Minimum duration (in seconds) for a period to be considered a breakpoint (default: 10 s)
    window_size : int
        Window size for calculating depth change rate (default: 5 data points)
    smooth_window : int
        Window size for Savitzky-Golay smoothing filter (must be odd, default: 11)
    poly_order : int
        Polynomial order for Savitzky-Golay filter (default: 3)

    Returns:
    --------
    breakpoints : list of tuples
        List of (start_time, end_time) for each constant depth period
    names : list of strings
        Suggested names for each breakpoint
    depth_smooth : numpy array
        Smoothed depth data used for analysis
    """

    depth = ctd_df["Depth [m]"].values
    time_sec = ctd_df["Time (Seconds)"].values

    # Smooth the depth data using Savitzky-Golay filter
    # This preserves features while reducing noise
    if len(depth) >= smooth_window:
        depth_smooth = savgol_filter(
            depth, window_length=smooth_window, polyorder=poly_order
        )
    else:
        depth_smooth = depth

    # Calculate depth change over the window on smoothed data
    depth_change = np.zeros(len(depth_smooth))
    for i in range(window_size, len(depth_smooth)):
        depth_change[i] = depth_smooth[i] - depth_smooth[i - window_size]

    # Identify "flat" regions where abs(depth_change) <= threshold
    is_flat = np.abs(depth_change) <= depth_threshold

    # Find transitions (start and end of flat regions)
    transitions = np.diff(is_flat.astype(int))
    starts = np.where(transitions == 1)[0]  # 0->1: start of flat region
    ends = np.where(transitions == -1)[0]  # 1->0: end of flat region

    # Handle edge cases: if series starts flat, add 0 as start
    if len(is_flat) > 0 and is_flat[0] and (len(starts) == 0 or starts[0] > 0):
        starts = np.insert(starts, 0, 0)

    # If series ends flat, add last index as end
    if (
        len(is_flat) > 0
        and is_flat[-1]
        and (len(ends) == 0 or ends[-1] < len(is_flat) - 1)
    ):
        ends = np.append(ends, len(is_flat) - 1)

    # Match starts and ends, filtering by minimum duration
    breakpoints = []
    for start in starts:
        # Find the next end after this start
        next_ends = ends[ends > start]
        if len(next_ends) > 0:
            end = next_ends[0]
            duration = time_sec[end] - time_sec[start]
            if duration >= min_duration:
                breakpoints.append((time_sec[start], time_sec[end]))

    # Generate suggested names
    names = [f"breakpoint_{i + 1}" for i in range(len(breakpoints))]

    return breakpoints, names, depth_smooth
