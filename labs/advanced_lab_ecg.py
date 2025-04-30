"""Second version of ECG detection and cleaning prototyping.

The purpose of this lab is to implement a more recently published R peak
detection method.
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
from src.utils import *
import pandas as pd
import bids_explorer.architecture.architecture as arch

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

architecture = arch.BidsArchitecture(root = "/Users/samuel/Desktop/PHYSIO_BIDS")

random_subject = np.random.choice(architecture.subjects)
architecture.select(subject = random_subject, acquisition = "biopac", inplace = True)
random_session = np.random.choice(architecture.sessions)
architecture.select(session = random_session, inplace = True)
random_task = np.random.choice(architecture.tasks)
architecture.select(task = random_task, inplace = True)
files_biopac = architecture.select(acquisition = "biopac", extension = ".gz")

df = pd.read_csv(files_biopac.database["filename"].values[0], compression="gzip", sep="\t", header=None)
sig = df[4].values
t = df[0].values
fs = 1/(t[1]-t[0])

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
# %%
