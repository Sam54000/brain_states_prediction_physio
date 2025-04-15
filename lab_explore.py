#%%
import bids_explorer.architecture.architecture as arch
import scipy.stats
architecture = arch.BidsArchitecture(root = "/Users/samuel/Downloads/PHYSIO_BIDS")


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
import os
from ecgdetectors import Detectors
from pathlib import Path
from scipy import signal
#%%

df = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-02/ses-01Baby/func/sub-02_ses-01Baby_task-ActiveHighVid_run-1_acq-eyelink_recording-samples_physio.tsv.gz", sep = "\t", header = None)
def mask_pupil_dilation(pupil_data: np.ndarray, threshold: int = 50):
    derivative = np.diff(pupil_data, prepend = pupil_data[0])
    mask_edges = abs(derivative) > threshold
    mask_missing_data = derivative == 0
    return np.logical_or(mask_edges, mask_missing_data)

def zscore_clean(pupil_data: np.ndarray, threshold: int = 3):
    zscore = scipy.stats.zscore(pupil_data)
    return abs(zscore) > threshold

derivative_mask = mask_pupil_dilation(df[3].values)
zscore_mask = zscore_clean(df[3].values)
mask = np.logical_or(derivative_mask, zscore_mask)
df.loc[mask,3] = np.nan
plt.plot(df[0].values, df[3].values)


# %%
df = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-02/ses-01Baby/func/sub-02_ses-01Baby_task-ActiveHighVid_run-1_acq-biopac_recording-samples_physio.tsv.gz", sep = "\t", header = None)
# %%
df = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-02/ses-01Baby/func/sub-02_ses-01Baby_task-ActiveHighVid_run-1_acq-eyelink_recording-events_physio.tsv.gz", header = None, sep = "\t")
# %%
def extract_eyelink_events(
    eyelink_filename: os.PathLike
    ) -> pd.DataFrame:
    """Extract the events of the experiment for the eye tracking data.
    
    Args:
        eyelink_filename (os.PathLike): The filename should contain 
            'acq-eyelink_recording-events'.

    Returns:
        pd.DataFrame
    """
    dataframe = pd.read_csv(eyelink_filename, header=None, sep = "\t")
    message_df = dataframe.loc[dataframe[0] == "MSG"]
    message_df.reset_index(inplace = True)
    scanner_start_idx = message_df[message_df[2] == "SCANNER_START"].index[0]
    cropped_df = message_df.iloc[scanner_start_idx:]
    cropped_df.rename(columns={1:"time_ms",2: "event_name"}, inplace=True)
    return cropped_df[["time_ms","event_name"]]

def filter_signal(data: np.ndarray, fs: float, lowcut: float = 0.5, highcut: float = 40.0, order: int = 5) -> np.ndarray:
    """
    Apply a bandpass filter to ECG signal data using second-order sections.
    
    Args:
        ecg_data (np.ndarray): Raw ECG signal as 1-D numpy array
        fs (float): Sampling frequency in Hz
        lowcut (float, optional): Low cutoff frequency in Hz. Defaults to 0.5 Hz.
        highcut (float, optional): High cutoff frequency in Hz. Defaults to 40.0 Hz.
        order (int, optional): Filter order. Defaults to 5.
        
    Returns:
        np.ndarray: Filtered ECG signal
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    
    sos = signal.butter(order, [low, high], btype='band', output='sos')
    filtered = signal.sosfiltfilt(sos, data)
    
    return filtered

# This function is still under implementation and doesn't work yet
def correct_R_peak(signal: np.ndarray, peaks: np.ndarray) -> np.ndarray:
    """
    Fine-tune the positions of detected R peaks to the true local maxima.
    
    Args:
        signal (np.ndarray): The ECG signal array
        peaks (np.ndarray): Indices of initially detected R peaks
        
    Returns:
        np.ndarray: Corrected R peak positions
    """
    corrected = []
    for array_idx, peak in enumerate(peaks):
        if array_idx == 0:
            previous_peak = peaks[array_idx]
            next_peak = peaks[array_idx+1]
        
        elif array_idx == peaks.shape[0] - 1:
            previous_peak = peaks[array_idx - 1]
            next_peak = peaks[array_idx]
        
        else:
            previous_peak = peaks[array_idx - 1]
            next_peak = peaks[array_idx + 1]
        
        print(f"previous peak: {previous_peak}\nnext peak: {next_peak}")
        
        corrected.append(np.argmax(signal[previous_peak:next_peak]))
    return peaks

def detect_R_peak(ecg_signal: np.ndarray, fs: float) -> np.ndarray:
    """Automatically detect and readjust R peaks of the ECG signal.

    Args:
        ecg_signal (np.ndarray): Should be a filtered ECG signal to remove
            fMRI artifacts
        fs (float): Sampling frequency
    
    Returns:
        np.ndarray: The index when the peaks occur
    """
    detector = Detectors(fs)
    r_peaks_init = np.array(detector.hamilton_detector(ecg_signal))
    r_peaks = correct_R_peak(ecg_signal, r_peaks_init)
    
    return r_peaks

# %%
def extract_eyetracking_data(eyelink_filename: os.PathLike) -> dict:
    """Extract the raw data of the eyetracking system and convert into a dict.

    Args:
        eyelink_filename (os.PathLike): The filename that should contain:
            'acq-eyelink_recording-samples'.

    Returns:
        dict: The formatted dictionary that should contains the following keys:
            'time', 'feature', 'labels', 'feature_info', 'mask'. For now the
            mask is set to True because I need to see with Nicole how to
            process the eyetracking data better.
    """
    raw_dataframe = pd.read_csv(eyelink_filename, header = None, sep="\t")
    