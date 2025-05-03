"""Second version of ECG detection and cleaning prototyping.

The purpose of this lab is to implement a more recently published R peak
detection method.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from src.modalities import read_biopac
from src.utils import *
import pandas as pd
import bids_explorer.architecture.architecture as arch
import pickle
from scipy.stats import norm
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec

def detect_r_peaks(signal, fs, window_size_sec=0.25, threshold_factor=0.6, buffer_size_sec=0.1):
    """
    Detect R-peaks in an ECG signal using a Fast Parabolic Fitting algorithm.
    (Felix et al., 2023)

    Parameters:
    - signal: 1D numpy array of ECG signal values.
    - fs: Sampling frequency in Hz.
    - window_size_sec: Size of the moving window in seconds.
    - threshold_factor: Factor to determine the threshold for peak detection.

    Returns:
    - r_peaks: List of indices where R-peaks are detected.
    """
    window_size = int(window_size_sec * fs)
    r_peaks = []
    M_mean = np.mean(signal)
    candidate_found = False
    best_candidate_idx = None
    best_candidate_height = None

    for i in range(len(signal) - window_size):
        window = signal[i:i + window_size]
        local_max = np.max(window)
        local_max_idx = i + np.argmax(window)

        threshold = M_mean * threshold_factor

        if local_max > threshold:
            if not candidate_found:
                candidate_found = True
                best_candidate_idx = local_max_idx
                best_candidate_height = local_max
            else:
                if local_max > best_candidate_height:
                    best_candidate_idx = local_max_idx
                    best_candidate_height = local_max
        else:
            if candidate_found:
                r_peaks.append(best_candidate_idx)
                M_mean = 0.125 * best_candidate_height + 0.875 * M_mean
                candidate_found = False
                best_candidate_idx = None
                best_candidate_height = None
    
    r_peaks = np.array(r_peaks)
    r_peaks = r_peaks[np.where(np.diff(r_peaks) > buffer_size_sec * fs)[0]]
    return r_peaks

#%%
filtered_signal = filter_signal(sig, 
                                fs=fs, 
                                lowcut=1, 
                                highcut=10)
#smoothed_signal = moving_average_smoothing(filtered_signal, 250)

filtered_r_peaks = detect_r_peaks(
    abs(filtered_signal), 
    fs, 
    buffer_size_sec=0.25, 
    threshold_factor=0.7)


def plot_signal_segment(signal, filtered_signal, time, r_peaks, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    random_start_idx = np.random.randint(0, len(time) - int(10 * fs))
    end_idx = random_start_idx + int(10 * fs)
    time_segment = time[random_start_idx:end_idx]
    
    ax.plot(time_segment, signal[random_start_idx:end_idx], color="gray", label='Raw ECG Signal', alpha=0.5)
    ax.plot(time_segment, filtered_signal[random_start_idx:end_idx], color="tab:green", linewidth = 2.5, label='Filtered ECG Signal')
    
    r_peaks_in_segment = r_peaks[(r_peaks >= random_start_idx) & (r_peaks < end_idx)]
    
    if len(r_peaks_in_segment) > 0:
        ax.plot(time[r_peaks_in_segment], filtered_signal[r_peaks_in_segment], 'ro', label='Detected R-peaks')
    
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Amplitude (A.U)')
    ax.set_title('Comparison of Raw and Filtered ECG Signals with Detected R-peaks')
    ax.legend()
    ax.spines[["top","right"]].set_visible(False)
    ax.set_xlim(random_start_idx/fs, end_idx/fs)
    return ax

# %%
def count_bpm(r_peaks, signal, fs,window_size_sec=10):
    """
    Count the beats per minute (BPM) in an ECG signal using a sliding window approach.
    """
    window_size = int(window_size_sec * fs)
    mask = np.zeros(len(signal), dtype=int)
    mask[r_peaks] = 1
    nb_r_peaks = np.sum(np.lib.stride_tricks.sliding_window_view(mask, window_size), axis=1)
    bpm = 60 * nb_r_peaks / window_size_sec
    windowed_bpm = np.lib.stride_tricks.sliding_window_view(bpm, int(window_size/2))
    smoothed_bpm = np.mean(windowed_bpm, axis=1)
    smoothed_bpm = np.concatenate((smoothed_bpm, np.ones(int(window_size + window_size/2)-2)*smoothed_bpm[-1]))
    return smoothed_bpm

# %%
bpm = count_bpm(filtered_r_peaks, filtered_signal, fs)
def plot_bpm(bpm, time, ax = None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))
    
    min_len = min(len(time), len(bpm))
    
    ax.plot(time[:min_len], bpm[:min_len], color="black", label='BPM')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Beats per minute')
    ax.set_xlim(min(time), max(time))
    ax.set_ylim(0, 120)
    txt = f"BPM:\nMean: {np.mean(bpm):.2f}\nStd: {np.std(bpm):.2f}\nMin: {np.min(bpm):.2f}\nMax: {np.max(bpm):.2f}\nMedian: {np.median(bpm):.2f}"

    ax.text(0.01, 0.05, txt, ha='left', va='bottom', transform=ax.transAxes, bbox=dict(boxstyle='round', facecolor='gray', alpha=0.2), fontfamily = "monospace")
    ax.spines[["top","right"]].set_visible(False)
    return ax

plot_bpm(bpm, t)
# %%

def create_r_peak_epochs(signal, r_peaks, fs, pre_time=0.5, post_time=0.5):
    """
    Create epochs of signal around each R peak.
    
    Parameters:
    - signal: 1D numpy array of ECG signal values
    - r_peaks: Array of indices where R-peaks are detected
    - fs: Sampling frequency in Hz
    - pre_time: Time before the R peak in seconds (default: 0.5)
    - post_time: Time after the R peak in seconds (default: 0.5)
    
    Returns:
    - epochs: 2D numpy array where each row is an epoch around an R peak
    - times: 1D array of time values for each epoch (in seconds relative to R peak)
    """
    pre_samples = int(pre_time * fs)
    post_samples = int(post_time * fs)
    epoch_length = pre_samples + post_samples + 1  # +1 to include the R peak itself
    
    times = np.linspace(-pre_time, post_time, epoch_length)
    epochs = np.zeros((len(r_peaks), epoch_length))
    valid_epochs = 0
    for i, peak_idx in enumerate(r_peaks):
        if peak_idx >= pre_samples and peak_idx + post_samples < len(signal):
            start_idx = peak_idx - pre_samples
            end_idx = peak_idx + post_samples + 1
            epochs[valid_epochs] = signal[start_idx:end_idx]
            valid_epochs += 1
    
    return epochs[:valid_epochs], times

epochs, epoch_times = create_r_peak_epochs(
    filtered_signal, 
    filtered_r_peaks, 
    fs, 
    pre_time = 0.3, 
    post_time = 0.5)
# %% Correct the R-peaks
corrected_r_peaks = correct_R_peaks(filtered_signal, filtered_r_peaks)
corrected_epochs, corrected_epoch_times = create_r_peak_epochs(
    filtered_signal, 
    corrected_r_peaks, 
    fs, 
    pre_time = 0.3, 
    post_time = 0.5)
# %%
def plot_epochs(epochs, epoch_times, title = "ECG Signal Epochs Around R Peaks", ax = None):
    """
    Plot the epochs of the ECG signal around the R peaks.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(epoch_times, epochs.T, linewidth = 0.5, color = "black", alpha = 0.1)
    ax.plot(epoch_times, np.mean(epochs, axis=0), color = "red", label = "Mean")
    ax.fill_between(epoch_times, np.mean(epochs, axis=0) - np.std(epochs, axis=0), np.mean(epochs, axis=0) + np.std(epochs, axis=0), color = "orange", alpha = 0.2, label = "Std")
    ax.axvline(x=0, color='g', linestyle='--', label='R Peak', alpha = 0.6)
    ax.set_xlabel('Time relative to R peak (s)')
    ax.set_xlim(min(epoch_times), max(epoch_times))
    ax.set_ylabel('Amplitude (A.U)')
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    return ax

# %%
fig = plt.figure(figsize=(12, 12))
gs = plt.GridSpec(3, 2, figure=fig)

ax1 = fig.add_subplot(gs[0, :])
ax2 = fig.add_subplot(gs[1, :])
ax3 = fig.add_subplot(gs[2, 0])
ax4 = fig.add_subplot(gs[2, 1])

plot_signal_segment(sig, filtered_signal, t, filtered_r_peaks, ax=ax1)
plot_bpm(bpm, t, ax=ax2)
plot_epochs(epochs, epoch_times, title="ECG Signal Epochs Around R Peaks", ax=ax3)
plot_epochs(corrected_epochs, corrected_epoch_times, title="Corrected ECG Signal Epochs Around R Peaks", ax=ax4)

plt.tight_layout()
plt.show()
# %% TRYING TO FIX FALSE DETECTIONS
from scipy.optimize import curve_fit 
with open("/Volumes/LaCie/processed/sub-10/ses-02Beauty/ecg/sub-10_ses-02Beauty_task-PassiveLowVid_run-1_ecg.pkl", "rb") as f:
    ecg = pickle.load(f)

peaks_idx = np.where(ecg["features"][-1])[0]
peaks_amplitudes = ecg["features"][0][peaks_idx]
median_amplitude = np.median(peaks_amplitudes)
if median_amplitude < 0:
    peaks_amplitudes = -peaks_amplitudes
    ecg["features"][0] = -ecg["features"][0]

fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(ecg["time"], ecg["features"][0])
ax.plot(ecg["time"][peaks_idx], ecg["features"][0][peaks_idx], 'ro')
ax.set_xlim(0,120)
ax.set_title("ECG Signal with Detected R-peaks")
ax.set_xlabel("Time (s)")
ax.spines[["top","right"]].set_visible(False)
ax.set_ylabel("Amplitude (A.U)")
plt.show()

fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(ecg["time"], ecg["features"][0])
ax.plot(ecg["time"][peaks_idx], ecg["features"][0][peaks_idx], 'ro')
ax.set_title("ECG Signal with Detected R-peaks (entire signal)")
ax.set_xlabel("Time (s)")
ax.set_ylabel("Amplitude (A.U)")
ax.spines[["top","right"]].set_visible(False)
plt.show()
#%%
# Calculate median and IQR to identify main cluster

Q1 = np.percentile(peaks_amplitudes, 25)
Q3 = np.percentile(peaks_amplitudes, 75)
IQR = Q3 - Q1

# Define the main cluster using 1.5 * IQR rule
lower_bound = Q1 - 1.525 * IQR
upper_bound = Q3 + 2 * IQR
#%%
# Select only values within the main cluster
main_cluster_mask = (peaks_amplitudes >= lower_bound) & (peaks_amplitudes <= upper_bound)
main_cluster_values = peaks_amplitudes[main_cluster_mask]

# Fit Gaussian distribution to the main cluster
mu, std = norm.fit(main_cluster_values)

# Create figure
fig, ax = plt.subplots(figsize=(10, 6))

# Plot histogram of all peak amplitudes
n, bins, patches = ax.hist(peaks_amplitudes, bins=50, density=True, alpha=0.6, color='gray', label='All Peak Amplitudes')

# Plot histogram of main cluster values
ax.hist(main_cluster_values, bins=bins, density=True, alpha=0.6, color='blue', label='Main Cluster')

# Plot fitted Gaussian
xmin, xmax = plt.xlim()
x = np.linspace(xmin, xmax, 100)
p = norm.pdf(x, mu, std)
ax.plot(x, p, 'k', linewidth=2, label=f'Fitted Gaussian (μ={mu:.2f}, σ={std:.2f})')

# Calculate and plot threshold for p < 0.01
threshold = norm.ppf(0.001, mu, std)
ax.axvline(x=threshold, color='r', linestyle='--', label=f'p < 0.01 threshold ({threshold:.2f})')

# Plot median and cluster boundaries
ax.axvline(x=median_amplitude, color='g', linestyle='--', label=f'Median ({median_amplitude:.2f})')
ax.axvline(x=lower_bound, color='purple', linestyle=':', label=f'Lower bound ({lower_bound:.2f})')
ax.axvline(x=upper_bound, color='purple', linestyle=':', label=f'Upper bound ({upper_bound:.2f})')

# Add labels and title
ax.set_xlabel('Peak Amplitude')
ax.set_ylabel('Density')
ax.set_title('Distribution of Peak Amplitudes with Main Cluster Gaussian Fit')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Print statistics
print(f"Total number of peaks: {len(peaks_amplitudes)}")
print(f"Number of peaks in main cluster: {len(main_cluster_values)}")
print(f"Median amplitude: {median_amplitude:.2f}")
print(f"Main cluster - Mean: {mu:.2f}, Std: {std:.2f}")
print(f"Threshold for p < 0.01: {threshold:.2f}")
print(f"Number of peaks below threshold: {np.sum(peaks_amplitudes < threshold)}")

# %%
peaks = scipy.signal.find_peaks(
    ecg["features"][0], 
    distance = 0.5 * fs,
    height = (lower_bound, upper_bound)
    )
# %%
fig, ax = plt.subplots(figsize=(20, 6))
ax.plot(ecg["time"], ecg["features"][0])
ax.plot(ecg["time"][peaks[0]], ecg["features"][0][peaks[0]], 'ro')
ax.set_xlim(0,120)
ax.spines[["top","right"]].set_visible(False)
ax.set_title("Re-adjusted R-peaks")
plt.show()

# %%
# Create figure with custom layout
fig = plt.figure(figsize=(12, 10))
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])

# Main plot spanning all columns in first row
ax_main = fig.add_subplot(gs[0, :])
ax_main.plot(ecg["time"], ecg["features"][0])
ax_main.plot(ecg["time"][peaks[0]], ecg["features"][0][peaks[0]], 'ro')

# Create three zoom subplots
zoom_duration = 20  # seconds
zoom_samples = int(zoom_duration * fs)

# Choose three different starting points for zooms (evenly spaced)
total_duration = ecg["time"][-1] - ecg["time"][0]
zoom_starts = [total_duration * i/4 for i in range(1, 4)]

axes_zoom = []
for i, start_time in enumerate(zoom_starts):
    # Create subplot
    ax_zoom = fig.add_subplot(gs[1, i])
    axes_zoom.append(ax_zoom)
    
    # Find corresponding indices
    start_idx = np.searchsorted(ecg["time"], start_time)
    end_idx = start_idx + zoom_samples
    
    # Plot zoomed section
    ax_zoom.plot(ecg["time"][start_idx:end_idx], ecg["features"][0][start_idx:end_idx])
    mask = (ecg["time"][peaks[0]] >= ecg["time"][start_idx]) & (ecg["time"][peaks[0]] <= ecg["time"][end_idx])
    ax_zoom.plot(ecg["time"][peaks[0][mask]], ecg["features"][0][peaks[0][mask]], 'ro')
    
    # Add rectangle in main plot
    rect = Rectangle((ecg["time"][start_idx], ax_main.get_ylim()[0]),
                    zoom_duration,
                    ax_main.get_ylim()[1] - ax_main.get_ylim()[0],
                    fill=False, color=f'C{i}', alpha=0.5)
    ax_main.add_patch(rect)
    
    # Add connector
    con = ConnectionPatch(
        xyA=(ecg["time"][start_idx], ax_main.get_ylim()[0]),  # main plot point
        xyB=(ecg["time"][start_idx], ax_zoom.get_ylim()[1]),  # zoom plot point
        coordsA="data",
        coordsB="data",
        axesA=ax_main,
        axesB=ax_zoom,
        color=f'C{i}',
        alpha=0.5
    )
    fig.add_artist(con)
    
    # Formatting
    ax_zoom.set_title(f'Segment {i+1}')
    ax_zoom.set_xlabel('Time (s)')
    if i == 0:
        ax_zoom.set_ylabel('Amplitude (A.U)')

# Main plot formatting
ax_main.set_title('Full ECG Signal with Detected Peaks')
ax_main.set_xlabel('Time (s)')
ax_main.set_ylabel('Amplitude (A.U)')

plt.tight_layout()
plt.show()
# %%
data = read_random_biopac_file()
ecg = data["ecg"]
ttl_idx = np.where(ecg.ttl)[0]

# Apply adaptive gradient removal
artifact = filter_signal(ecg.data, fs=ecg.fs, lowcut=15, highcut=None)
cleaned_signal_adaptive = ecg.data - artifact
filtered_signal = filter_signal(ecg.data, fs=ecg.fs, lowcut=1, highcut=10)
# Plot the results
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10))

# Plot a segment around multiple TTLs
segment_start = ttl_idx[20] - int(ecg.fs)  # 1 second before 20th TTL
segment_end = ttl_idx[25] + int(ecg.fs)    # 1 second after 25th TTL
t_segment = np.arange(segment_end - segment_start) / ecg.fs

ax1.plot(t_segment, ecg.data[segment_start:segment_end], 
         label='Original', alpha=0.7)
ax1.plot(t_segment, cleaned_signal_adaptive[segment_start:segment_end],
         label='Adaptively Cleaned', alpha=0.7)
ax1.plot(t_segment, filtered_signal[segment_start:segment_end],
         label='Filtered', color = "red")
ax1.set_title('Original vs Adaptively Cleaned Signal (Multiple TTL Windows)')
ax1.set_xlabel('Time (s)')
ax1.set_ylabel('Amplitude')
ax1.legend()

# Plot zoomed in to a single TTL
example_ttl = ttl_idx[22]  # Use the 22nd TTL as example
window_samples = int(0.2 * ecg.fs)  # Smaller window for detail
t_window = np.arange(-window_samples, window_samples+1) / ecg.fs

ax2.plot(t_window, 
         data["ecg"].data[example_ttl-window_samples:example_ttl+window_samples+1],
         label='Original', alpha=0.7)
ax2.plot(t_window, 
         cleaned_signal_adaptive[example_ttl-window_samples:example_ttl+window_samples+1],
         label='Adaptively Cleaned', alpha=0.7)
ax2.set_title('Original vs Adaptively Cleaned Signal (Single TTL Window)')
ax2.set_xlabel('Time relative to TTL (s)')
ax2.set_ylabel('Amplitude')
ax2.legend()

plt.tight_layout()
plt.show()
# %%

def adaptive_gradient_removal(
    signal, 
    ttl_indices, 
    fs, 
    window_size=10, 
    pre_time=0.5, 
    post_time=0.5
    ):
    """
    Perform adaptive average subtraction of gradient artifacts using a sliding window.
    
    Parameters:
    -----------
    signal : array-like
        The ECG signal
    ttl_indices : array-like
        Indices of TTL triggers
    fs : float
        Sampling frequency in Hz
    window_size : int
        Number of TTLs to use for template computation
    pre_time : float
        Time before TTL in seconds
    post_time : float
        Time after TTL in seconds
        
    Returns:
    --------
    cleaned_signal : array-like
        Signal with gradient artifacts removed adaptively
    """
    cleaned_signal = signal.copy()
    pre_samples = int(pre_time * fs)
    post_samples = int(post_time * fs)
    epoch_length = pre_samples + post_samples + 1
    
    # For each TTL
    for i in range(len(ttl_indices)):
        # Calculate window boundaries
        half_window = window_size // 2
        
        # Handle start of recording
        if i < half_window:
            window_start = 0
            window_end = min(window_size, len(ttl_indices))
        # Handle end of recording
        elif i >= len(ttl_indices) - half_window:
            window_start = max(0, len(ttl_indices) - window_size)
            window_end = len(ttl_indices)
        # Normal case
        else:
            window_start = i - half_window
            window_end = i + half_window + 1
            
        # Get TTLs for this window
        window_ttls = ttl_indices[window_start:window_end]
        
        # Create epochs for template calculation
        epochs = np.zeros((len(window_ttls), epoch_length))
        valid_epochs = 0
        
        for j, ttl_idx in enumerate(window_ttls):
            if ttl_idx >= pre_samples and ttl_idx + post_samples < len(signal):
                start_idx = ttl_idx - pre_samples
                end_idx = ttl_idx + post_samples + 1
                epochs[valid_epochs] = signal[start_idx:end_idx]
                valid_epochs += 1
        
        # Compute template from valid epochs
        if valid_epochs > 0:
            template = np.mean(epochs[:valid_epochs], axis=0)
            
            # Subtract template from current TTL's epoch
            current_ttl = ttl_indices[i]
            if current_ttl >= pre_samples and current_ttl + post_samples < len(signal):
                start_idx = current_ttl - pre_samples
                end_idx = current_ttl + post_samples + 1
                cleaned_signal[start_idx:end_idx] -= template
    
    return cleaned_signal

# %%
data = read_random_biopac_file()
ecg = data["ecg"]
ecg.process()
ecg.plot()
#%%
fig, ax = plt.subplots(figsize=(20, 12), nrows = 4)
ax[0].plot(ecg.time, ecg.data, alpha = 0.7, color = "gray", label = "original")
ax[0].plot(ecg.time, ecg.cleaned_signal, label = "cleaned")
ax[0].plot(ecg.time[ecg.r_peaks], ecg.cleaned_signal[ecg.r_peaks], "ro", label = "Detected R-peaks")
ax[0].legend()
ax[0].set_ylabel("Amplitude (mV)")
ax[0].spines[["top","right"]].set_visible(False)
ax[1].plot(ecg.time, ecg.hrv, color = "gray", linestyle = "--", label = "Interpolated Interbeat intervals")
ax[1].plot(ecg.time[ecg.r_peaks], ecg.rr_intervals, "go", label = "Interbeat intervals")
ax[1].set_ylabel("Interval (s)")
ax[1].spines[["top","right"]].set_visible(False)
ax[1].set_ylim(-1, 2)
ax[2].plot(ecg.time, ecg.data, alpha = 0.7, color = "gray", label = "original")
ax[2].plot(ecg.time, ecg.cleaned_signal, label = "cleaned")
ax[2].plot(ecg.time[ecg.r_peaks], ecg.cleaned_signal[ecg.r_peaks], "ro", label = "Detected R-peaks")
ax[2].legend()
ax[2].set_ylabel("Amplitude (mV)")
ax[2].spines[["top","right"]].set_visible(False)
ax[2].set_ylim(
    min(ecg.data[0:20000])-min(ecg.data[0:20000])*0.1, 
    max(ecg.data[0:20000])+max(ecg.data[0:20000])*0.1)
ax[2].set_xlim(0, 10)
ax[3].plot(ecg.time, ecg.hrv, color = "gray", linestyle = "--", label = "interpolated HRV")
ax[3].plot(ecg.time[ecg.r_peaks], ecg.rr_intervals, "go", label = "RR intervals")
ax[3].set_xlabel("Time (s)")
ax[3].set_ylabel("Interval (s)")
ax[3].set_ylim(-1, 2)
ax[3].spines[["top","right"]].set_visible(False)
ax[3].set_xlim(0, 10)
ax[3].legend()
plt.suptitle("ECG signal and interbeat intervals before cleaning")
plt.show()

Q1 = np.percentile(ecg.rr_intervals, 25)
Q3 = np.percentile(ecg.rr_intervals, 75)
IQR = Q3 - Q1
lower_bound = Q1 - 2 * IQR
upper_bound = Q3 + 2 * IQR
mask = (ecg.rr_intervals >= lower_bound) & (ecg.rr_intervals <= upper_bound)
rr_intervals = ecg.rr_intervals.copy()
rr_intervals[~mask] = np.interp(
    ecg.time[ecg.r_peaks][~mask], 
    ecg.time[ecg.r_peaks][mask], 
    rr_intervals[mask]
    )
interpolant = scipy.interpolate.interp1d(
    ecg.time[ecg.r_peaks],
    rr_intervals,
    kind="linear",
    fill_value="extrapolate"
)
interpolated_rr_intervals = interpolant(ecg.time)


fig, ax = plt.subplots(figsize=(20, 12), nrows = 4)
ax[0].plot(ecg.time, ecg.data, alpha = 0.7, color = "gray", label = "original")
ax[0].plot(ecg.time, ecg.cleaned_signal, label = "cleaned")
ax[0].plot(ecg.time[ecg.r_peaks], ecg.cleaned_signal[ecg.r_peaks], "ro", label = "Detected R-peaks")
ax[0].legend()
ax[0].set_ylabel("Amplitude (mV)")
ax[0].spines[["top","right"]].set_visible(False)
ax[1].plot(ecg.time, interpolated_rr_intervals, color = "gray", linestyle = "--", label = "Interpolated Interbeat intervals")
ax[1].plot(ecg.time[ecg.r_peaks], rr_intervals, "go", label = "Interbeat intervals")
ax[1].set_ylabel("Interval (s)")
ax[1].spines[["top","right"]].set_visible(False)
ax[1].set_ylim(-1, 2)
ax[2].plot(ecg.time, ecg.data, alpha = 0.7, color = "gray", label = "original")
ax[2].plot(ecg.time, ecg.cleaned_signal, label = "cleaned")
ax[2].plot(ecg.time[ecg.r_peaks], ecg.cleaned_signal[ecg.r_peaks], "ro", label = "Detected R-peaks")
ax[2].legend()
ax[2].set_ylabel("Amplitude (mV)")
ax[2].spines[["top","right"]].set_visible(False)
ax[2].set_ylim(
    min(ecg.data[0:20000])-min(ecg.data[0:20000])*0.1, 
    max(ecg.data[0:20000])+max(ecg.data[0:20000])*0.1)
ax[2].set_xlim(0, 10)
ax[3].plot(ecg.time, interpolated_rr_intervals, color = "gray", linestyle = "--", label = "interpolated HRV")
ax[3].plot(ecg.time[ecg.r_peaks], rr_intervals, "go", label = "RR intervals")
ax[3].set_xlabel("Time (s)")
ax[3].set_ylabel("Interval (s)")
ax[3].spines[["top","right"]].set_visible(False)
ax[3].set_ylim(-1, 2)
ax[3].set_xlim(0, 10)
ax[3].legend()
plt.suptitle("ECG signal and interbeat intervals after cleaning")
plt.show()
# %%
# TODO: Reject recording that have a std lower than a typical ECG. Adding STD
# in report and flagging bad recordings.

import numpy as np
import matplotlib.pyplot as plt
from src.modalities import read_biopac
from src.utils import *
import pandas as pd
import bids_explorer.architecture.architecture as arch
import pickle
import src.modalities as modalities
from scipy.stats import norm
from matplotlib.patches import Rectangle
from matplotlib.patches import ConnectionPatch
import matplotlib.gridspec as gridspec
data = modalities.get_random_biopac_file()
ecg = data["ecg"]
ecg.process()
ax =ecg.plot()

# Generate simulated ECG template using neurokit2
# %%
