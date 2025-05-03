#%%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.utils import *
from src.pipeline import *
import numpy as np
from typing import Optional
import scipy
from matplotlib.backends.backend_pdf import PdfPages
import os
from pathlib import Path
import neurokit2 as nk
import bids_explorer.architecture.architecture as arch
import scipy.stats
import re

def plot_results(
    data_dict: dict,
    index_feature: int,
    title: str,
    time: Optional[np.ndarray] = [None, None],
    saving_filename: Optional[os.PathLike] = None
):
    raw = data_dict["feature"][index_feature,:]
    masked = raw.copy().astype(float)
    masked[~data_dict["mask"]] = np.nan
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(data_dict["time"], raw, label = "Raw")
    ax.plot(data_dict["time"], masked, label = "Masked")
    if all([t is not None for t in time]):
        ax.set_xlim(time[0], time[-1])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("A.U")
    ax.spines[["top", "right"]].set_visible(False)
    plt.show()
    if saving_filename is not None:
        fig.savefig(saving_filename)
    return fig

def moving_average_smoothing(signal, window_size=5):
    """
    Apply a simple moving average filter to smooth a signal.
    
    Args:
        signal (np.ndarray): The signal to smooth
        window_size (int): The size of the moving average window
        
    Returns:
        np.ndarray: The smoothed signal
    """
    # Create the smoothing kernel
    kernel = np.ones(window_size) / window_size
    
    # Apply convolution for smoothing, handling NaN values
    smoothed = np.copy(signal)
    valid_mask = ~np.isnan(signal)
    
    # Replace NaNs with zeros for convolution
    signal_for_conv = np.copy(signal)
    signal_for_conv[~valid_mask] = 0
    
    # Apply convolution
    smoothed_signal = np.convolve(signal_for_conv, kernel, mode='same')
    
    # Apply convolution to the mask (to get proper normalization)
    valid_mask_float = valid_mask.astype(float)
    mask_conv = np.convolve(valid_mask_float, kernel, mode='same')
    
    # Normalize by the number of valid points in each window
    # Only where we have at least one valid point
    norm_mask = mask_conv > 0
    smoothed[norm_mask] = smoothed_signal[norm_mask] / mask_conv[norm_mask]
    smoothed[~norm_mask] = np.nan
    
    return smoothed

def interpolate_masked_data(signal, time, max_gap=1.0, method='linear'):
    """
    Interpolate NaN values in a signal.
    
    Args:
        signal (np.ndarray): The signal with NaN values to interpolate
        time (np.ndarray): The corresponding time vector
        max_gap (float): Maximum gap in seconds to interpolate (not used in this version)
        method (str): Interpolation method ('linear', 'cubic', 'nearest')
        
    Returns:
        np.ndarray: The interpolated signal
    """
    # Create a copy of the signal
    interpolated = np.copy(signal)
    
    # Get indices of valid data points
    valid_indices = np.where(~np.isnan(signal))[0]
    
    if len(valid_indices) < 2:
        # Not enough points for interpolation
        return interpolated
    
    # Create an interpolation function using only valid points
    interp_func = scipy.interpolate.interp1d(
        time[valid_indices], 
        signal[valid_indices],
        kind=method,
        bounds_error=False,
        fill_value=np.nan
    )
    
    interpolated = interp_func(time)
    
    return interpolated

def pchip_interpolate_masked_data(signal, time):
    """
    Interpolate NaN values in a signal using PCHIP interpolation.
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves monotonicity
    and is less prone to overshooting than cubic spline.
    
    Args:
        signal (np.ndarray): The signal with NaN values to interpolate
        time (np.ndarray): The corresponding time vector
        max_gap (float): Maximum gap in seconds to interpolate (not used in this version)
        
    Returns:
        np.ndarray: The interpolated signal
    """
    # Create a copy of the signal
    interpolated = np.copy(signal)
    
    # Get indices of valid data points
    valid_indices = np.where(~np.isnan(signal))[0]
    
    if len(valid_indices) < 2:
        # Not enough points for interpolation
        return interpolated
    
    try:
        # Create a PCHIP interpolation function using only valid points
        from scipy.interpolate import PchipInterpolator
        pchip_func = PchipInterpolator(
            time[valid_indices], 
            signal[valid_indices],
            extrapolate=False
        )
        
        # Apply interpolation to the entire time range
        # PchipInterpolator already handles bounds properly
        interp_values = pchip_func(time)
        
        # Only update non-NaN values from the interpolation
        valid_interp = ~np.isnan(interp_values)
        interpolated[valid_interp] = interp_values[valid_interp]
        
    except Exception as e:
        # Fall back to linear interpolation if PCHIP fails
        print(f"PCHIP interpolation failed: {e}. Falling back to linear interpolation.")
        interp_func = scipy.interpolate.interp1d(
            time[valid_indices], 
            signal[valid_indices],
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        interpolated = interp_func(time)
    
    return interpolated

def create_adaptive_threshold_mask(
    signal: np.ndarray,
    time: np.ndarray,
    fs: float,
    window_size: float = 2.0,
    step_size: float = 0.5,
    p_threshold: float = 0.05,
    buffer: float = 0.2,
    min_valid_ratio: float = 0.5
) -> np.ndarray:
    """
    Creates an adaptive threshold mask for detecting outliers in a signal.
    
    For each sliding window:
    1. Computes the distribution of values
    2. Fits a normal distribution
    3. Marks values with p < threshold as outliers
    4. Adds buffer around outliers
    """
    mask = np.ones_like(signal, dtype=bool)
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    buffer_samples = int(buffer * fs)
    valid_data = ~np.isnan(signal)
    
    for start in range(0, len(signal) - window_samples + 1, step_samples):
        end = start + window_samples
        window_data = signal[start:end]
        window_valid = valid_data[start:end]
        
        if np.sum(window_valid) < window_samples * min_valid_ratio:
            continue
        
        valid_values = window_data[window_valid]
        
        try:
            mu, std = scipy.stats.norm.fit(valid_values)
            if std < 1e-6:
                continue
            
            z_critical = scipy.stats.norm.ppf(1 - p_threshold/2)
            lower_bound = mu - z_critical * std
            upper_bound = mu + z_critical * std
            
            outliers = np.logical_or(
                window_data < lower_bound,
                window_data > upper_bound
            )
            
            outlier_indices = np.where(outliers)[0] + start
            
            for idx in outlier_indices:
                buffer_start = max(0, idx - buffer_samples)
                buffer_end = min(len(mask), idx + buffer_samples + 1)
                mask[buffer_start:buffer_end] = False
                
        except Exception:
            continue
    
    return mask

def create_sliding_zscore_mask(
    signal: np.ndarray,
    fs: float,
    window_size: float = 2.0,
    step_size: float = 0.5,
    z_threshold: float = 2,
    buffer: float = 0.2
) -> np.ndarray:
    """
    Creates a mask by detecting outliers based on z-scores in a sliding window.
    
    For each sliding window:
    1. Computes z-scores of values in the window
    2. Marks values with |z-score| > threshold as outliers
    3. Adds buffer around outliers
    
    Args:
        signal (np.ndarray): The input signal
        fs (float): Sampling frequency in Hz
        window_size (float): Size of sliding window in seconds
        step_size (float): Step size for window sliding in seconds
        z_threshold (float): Z-score threshold for outlier detection
        buffer (float): Buffer time in seconds to mark around outliers
        
    Returns:
        np.ndarray: Boolean mask where True indicates valid data points
    """
    mask = np.ones_like(signal, dtype=bool)
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    buffer_samples = int(buffer * fs)
    
    # Handle NaN values in the signal
    valid_data = ~np.isnan(signal)
    
    for start in range(0, len(signal) - window_samples + 1, step_samples):
        end = start + window_samples
        window_data = signal[start:end]
        window_valid = valid_data[start:end]
        
        # Skip windows with insufficient valid data
        if np.sum(window_valid) < window_samples * 0.5:  # At least 50% valid data
            continue
        
        # Extract valid values in this window
        valid_values = window_data[window_valid]
        
        try:
            # Compute z-scores within this window
            mean = np.mean(valid_values)
            std = np.std(valid_values)
            
            # Avoid division by zero
            if std < 1e-6:
                continue
                
            z_scores = np.abs((window_data - mean) / std)
            
            # Identify outliers
            outliers = z_scores > z_threshold
            outlier_indices = np.where(outliers)[0] + start
            
            # Mark outliers and buffer region in the mask
            for idx in outlier_indices:
                buffer_start = max(0, idx - buffer_samples)
                buffer_end = min(len(mask), idx + buffer_samples + 1)
                mask[buffer_start:buffer_end] = False
                
        except Exception as e:
            print(f"Error in window {start}-{end}: {e}")
            continue
    
    return mask

def plot_comparison(signals_dict, time, xlim=None, title="Signal Comparison", figsize=(12, 6)):
    """Plot multiple signals for comparison"""
    fig, ax = plt.subplots(figsize=figsize)
    
    for label, signal in signals_dict.items():
        if label == "Original Signal":
            ax.plot(time, signal, label=label, color = "gray", alpha = 0.5)
        else:
            ax.plot(time, signal, label=label, color = "tab:orange")
    
    if xlim is not None:
        ax.set_xlim(xlim)
    
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Pupil Dilation (A.U)")
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.show()
    
    return fig

def plot_processing_stages(original, stages_dict, time, xlim=None, figsize=(14, 14)):
    """Plot multiple processing stages vertically for comparison"""
    num_stages = len(stages_dict)
    fig, axes = plt.subplots(num_stages, 1, figsize=figsize, sharex=True)
    
    # Plot each processing stage
    for i, (stage_name, signal) in enumerate(stages_dict.items()):
        axes[i].plot(time, original, 'gray', alpha=0.6, label="Original signal")
        axes[i].plot(time, signal, label=stage_name)
        if xlim is not None:
            axes[i].set_xlim(xlim)
        axes[i].set_ylabel("Amplitude")
        axes[i].set_title(f"After {stage_name}")
        axes[i].legend()
        axes[i].spines[["top", "right"]].set_visible(False)
    
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()
    
    return fig

def long_flat_segment_detection(mask, fs, min_duration=3.0):
    """
    Detects long segments of consecutive False values in a mask.
    Any segment with consecutive False values longer than min_duration will remain False,
    while shorter segments will be converted to True.
    
    Args:
        mask (np.ndarray): Boolean mask (True for valid data, False for invalid data)
        fs (float): Sampling frequency in Hz
        min_duration (float): Minimum duration in seconds for a segment to remain masked as False
        
    Returns:
        np.ndarray: Modified boolean mask with short segments of False values converted to True
    """
    # Create a copy of the input mask to avoid modifying the original
    result_mask = mask.copy()
    
    # Calculate the minimum number of samples for a segment to be considered "long"
    min_samples = int(min_duration * fs)
    
    # Check if the mask is entirely True
    if np.all(mask):
        return result_mask
    
    # Handle edge cases by padding the mask with True at both ends
    padded_mask = np.concatenate(([True], mask, [True]))
    
    # Find transitions in the padded mask
    transitions = np.diff(padded_mask.astype(int))
    
    # Find starts of False segments (True to False transitions)
    starts = np.where(transitions == -1)[0]
    
    # Find ends of False segments (False to True transitions)
    ends = np.where(transitions == 1)[0]
    
    # Make sure we have matching pairs
    assert len(starts) == len(ends), "Unequal number of segment starts and ends"
    
    # Iterate through all segments
    for i in range(len(starts)):
        # Adjust indices to match the original mask (accounting for padding)
        start = starts[i]
        end = ends[i]
        
        # Calculate segment length in samples
        segment_length = end - start
        
        # Print debug info for this segment
        print(f"Segment {i}: Start={start}, End={end}, Length={segment_length}, Min={min_samples}")
        
        # If segment is shorter than or equal to min_duration, convert to True
        if segment_length <= min_samples:
            # Adjust indices to account for padding at the start
            result_mask[start:end] = True
    
    return result_mask

#%% Load and prepare data
#architecture = arch.BidsArchitecture(root = "/Users/samuel/Desktop/PHYSIO_BIDS")

#random_subject = np.random.choice(architecture.subjects)
#architecture.select(subject = random_subject, acquisition = "eyelink", inplace = True)
#random_session = np.random.choice(architecture.sessions)
#architecture.select(session = random_session, inplace = True)
#random_task = np.random.choice(architecture.tasks)
#architecture.select(task = random_task, inplace = True)
#files_eyetracking = architecture.select(acquisition = "eyelink", extension = ".gz")
filename_eyetracking_samples = "/Users/samuel/Desktop/PHYSIO_BIDS/sub-02/ses-misc/func/sub-02_ses-misc_task-Rest_run-1_acq-eyelink_recording-samples_physio.tsv.gz"
filename_eyetracking_events = "/Users/samuel/Desktop/PHYSIO_BIDS/sub-02/ses-misc/func/sub-02_ses-misc_task-Rest_run-1_acq-eyelink_recording-events_physio.tsv.gz"
#if files_eyetracking.database.empty:
#    print("No Eyetracking file found")
#    import sys
#    sys.exit()

#filename_eyetracking_samples = [
#    str(file) for file in files_eyetracking.database["filename"].to_list() if "samples" in str(file)
#][0]
#filename_eyetracking_events = [
#    str(file) for file in files_eyetracking.database["filename"].to_list() if "events" in str(file)
#][0]

df_eyetracking_samples = pd.read_csv(filename_eyetracking_samples, sep = "\t", header = None)
df_eyetracking_events = pd.read_csv(filename_eyetracking_events, sep = "\t", header = None)

events = extract_eyelink_events(df_eyetracking_events)
eyetracking_dict = extract_and_process_eyetracking(df_eyetracking_samples)
eyetracking_dict = crop_eyetracking(eyetracking_dict, events)

#%% Normalize and initialize signal
time = eyetracking_dict[0]["time"]
original_signal = eyetracking_dict[0]["feature"][2,:] - np.min(eyetracking_dict[0]["feature"][2,:])
fs = 250 

#%% 1. Flat signal detection
flat_mask = detect_flat_signal(
    original_signal,
    fs = fs,
    flat_duration_threshold=0.05,
    buffer = 0.2,
)
after_flat = original_signal.copy()
flat_mask = np.logical_and(flat_mask, original_signal > 10)
after_flat[~flat_mask] = np.nan

start_time = np.random.choice(time)
zoom_range = (start_time, start_time + 20)

plot_comparison(
    {"Original Signal": original_signal, "After flat signal removal": after_flat},
    time,
    title="Flat Signal Detection",
    xlim=zoom_range
)

#%% 2. High gradient detection
gradient_mask = detect_high_gradient(
    original_signal, 
    fs = fs, 
    threshold = 30, 
    buffer = 0.25
)
combined_mask = np.logical_and(flat_mask, gradient_mask)
after_gradient = original_signal.copy()
after_gradient[~combined_mask] = np.nan

plot_comparison(
    {"Original Signal": original_signal, "After gradient detection": after_gradient},
    time,
    title="High Gradient Detection",
    xlim=zoom_range
)
#%% 3. Sliding z-score detection
zscore_mask = create_sliding_zscore_mask(
    signal=original_signal,
    fs=fs,
    window_size=2.0,
    step_size=1.0,
    z_threshold=2.0,
    buffer=0.1
)
after_zscore = original_signal.copy()
combined_mask = np.logical_and(combined_mask, zscore_mask)
after_zscore[~combined_mask] = np.nan
plot_comparison(
    {"Original Signal": original_signal, "After z-score detection": after_zscore},
    time,
    title="Z-Score Detection",
    xlim=zoom_range
)

#%% 4. Adaptive threshold detection
#adaptive_mask = create_adaptive_threshold_mask(
#    signal=original_signal,
#    time=time,
#    fs=fs,
#    window_size=3.0,
#    step_size=0.25,
#    p_threshold=0.05,
#     buffer=0.1,
#     min_valid_ratio=0.3
# )
# combined_mask = np.logical_and(combined_mask, adaptive_mask)
# adaptive_cleaned = original_signal.copy()
# adaptive_cleaned[~combined_mask] = np.nan
# 
# # Visualize all processing stages
# plot_processing_stages(
#     original_signal,
#     {
#         "Flat Signal Detection": after_flat,
#         "High Gradient Detection": after_gradient,
#         "Adaptive Threshold": adaptive_cleaned
#     },
#     time,
#     xlim=zoom_range
# )

#%% 4. Apply different interpolation methods and separate smoothing

interpolation_methods = {
    "PCHIP": lambda s, t: pchip_interpolate_masked_data(s, t)
}

interpolated_results = {}

for name, method_func in interpolation_methods.items():
    interpolated = method_func(after_gradient, time)
    interpolated_results[name] = interpolated

smoothed_results = {}
window_size_samples = int(0.1 * fs) 
for name, signal in interpolated_results.items():
    smoothed = moving_average_smoothing(after_zscore.copy(), window_size_samples)
    smoothed_interp = method_func(smoothed, time)
    smoothed_results[name] = smoothed_interp

final_mask = long_flat_segment_detection(flat_mask, fs, min_duration = 2.0)
interpolated_results["PCHIP"][~final_mask] = np.nan
smoothed_results["PCHIP"][~final_mask] = np.nan


plot_processing_stages(
    original_signal,
    {
        "After Flat Signal Removal": after_flat,
        "After Gradient Detection": after_gradient,
        "After Z-Score Detection": after_zscore,
        "PCHIP Interpolation": interpolated_results["PCHIP"]
    },
    time,
    xlim=zoom_range
)


after_flat_smoothed = moving_average_smoothing(after_flat.copy(), window_size_samples)
after_gradient_smoothed = moving_average_smoothing(after_gradient.copy(), window_size_samples)
after_zscore_smoothed = moving_average_smoothing(after_zscore.copy(), window_size_samples)

plot_processing_stages(
    original_signal,
    {
        "After Flat Signal Removal": after_flat_smoothed,
        "After Gradient Detection": after_gradient_smoothed,
        "After Z-Score Detection": after_zscore_smoothed,
        "PCHIP (Smoothed)": smoothed_results["PCHIP"]
    },
    time,
    xlim=zoom_range
)


comparison_dict = {
    "After Flat Signal Removal": after_flat,
    "After Gradient Detection": after_gradient,
    "After Z-Score Detection": after_zscore,
}

for name, signal in interpolated_results.items():
    comparison_dict[f"{name} (Raw)"] = signal

for name, signal in smoothed_results.items():
    comparison_dict[f"{name} (Smoothed)"] = signal

#%% 7. Save final results
best_method = "PCHIP"
use_smoothed = True


if use_smoothed:
    final_processed = smoothed_results[best_method]
    processing_description = f"{best_method} interpolation with smoothing"
else:
    final_processed = interpolated_results[best_method]
    processing_description = f"{best_method} interpolation without smoothing"

plot_comparison(
    {"Original Signal": original_signal, f"Final ({processing_description})": final_processed},
    time,
    title=f"Original vs Processed Signal"
)


final_result = {
    "time": time,
    "original_signal": original_signal,
    "processed_signal": final_processed,
    "sampling_rate": fs,
    "processing_info": {
        "flat_signal_threshold": 0.05,
        "gradient_threshold": 30,
        #"adaptive_p_threshold": 0.01,
        "smoothing_window": 0.2 if use_smoothed else None,
        "interpolation_method": best_method,
        "smoothing_applied": use_smoothed
    }
}

# with open("processed_eyetracking_data.pkl", "wb") as f:
#     pickle.dump(final_result, f)

# %%
