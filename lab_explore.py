#%%
import bids_explorer.architecture.architecture as arch
import scipy.stats
architecture = arch.BidsArchitecture(root = "/Users/samuel/Downloads/PHYSIO_BIDS")


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import kurtosis, entropy
from scipy.signal import welch
import os
from ecgdetectors import Detectors
from pathlib import Path
from scipy import signal
#%%

df = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-02/ses-01Baby/func/sub-02_ses-01Baby_task-ActiveHighVid_run-1_acq-eyelink_recording-samples_physio.tsv.gz", sep = "\t", header = None)
#%%
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
    cropped_df.reset_index(inplace = True)
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

    
    if lowcut is None:
        kwargs = {
            "btype": "low",
            "Wn": highcut/ nyquist
        }
    elif highcut is None:
        kwargs = {
            "btype": "high",
            "Wn": lowcut/nyquist
        }
    elif lowcut is not None and lowcut > 0 and highcut is not None:
        kwargs = {
            "btype": "band",
            "Wn": [lowcut / nyquist, highcut / nyquist]
        }
        
    else:
        raise ValueError(
            "lowcut and highcut must be None or non zero float got instead:\n"/
            f"\tlowcut type: {type(lowcut)}\n\thighcut type: {type(highcut)}"
        )
    sos = signal.butter(order, output='sos', **kwargs)
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
    # I need to skip this function for now. Like wtf eyetracking events and
    # eyetracking samples are not even synchronized... How can we do anything
    # with this BS.
def extract_and_process_eda(data: pd.DataFrame) -> dict:
    """Extract and process electrodermal activity.
    
    The process involve filtering the data with a low-pass filter at 1Hz as 
    EDA is a very slow activity.

    Args:
        data (pandas.DataFrame): The biopac data

    Returns:
        dict
    """
    eda = filter_signal(data[1].values, fs = 1000,lowcut=None, highcut=1)
    return {"time": data[0].values,
            "feature": eda[np.newaxis,:],
            "labels": ["eda"],
            "mask": np.ones_like(data[0].values,dtype=bool),
            "feature_info": "Electrodermal Activity low-pass filtered at 1Hz"
    }


def detect_motion_artifacts(ppg_signal, fs, window_size=5, step_size=0.25, kurtosis_threshold=3, entropy_threshold=1):
    """
    Detects motion artifacts in a PPG signal using kurtosis and Shannon entropy.
    

    Args:
        - ppg_signal: 1D numpy array of the PPG signal.
        - fs: Sampling frequency of the signal in Hz.
        - window_size: Size of the sliding window in seconds.
        - step_size: Step size for the sliding window in seconds.
        - kurtosis_threshold: Threshold for kurtosis to detect artifacts.
        - entropy_threshold: Threshold for entropy to detect artifacts.

    Returns:
        numpy.ndarray: 1D numpy array of the same length as ppg_signal, 
        with True indicating clean segments and False indicating artifacts.
    
    Reference:
        .. _Statistical approach for the detection of motion/noise artifacts in Photoplethysmogram:
        https://pubmed.ncbi.nlm.nih.gov/22255454/
    """
    signal_length = len(ppg_signal)
    mask = np.ones(signal_length)

    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)

    for start in range(0, signal_length - window_samples + 1, step_samples):
        end = start + window_samples
        segment = ppg_signal[start:end]

        seg_kurtosis = kurtosis(segment)

        # Compute power spectral density
        f, Pxx = welch(segment, fs=fs)
        Pxx_norm = Pxx / np.sum(Pxx)
        seg_entropy = entropy(Pxx_norm)

        # Detect artifacts based on thresholds
        if seg_kurtosis > kurtosis_threshold or seg_entropy < entropy_threshold:
            mask[start:end] = 0

    return mask

def extract_and_process_ppg(data: pd.DataFrame) -> dict:
    ppg_signal = filter_signal(
        data[2].values,
        fs = 1000,
        lowcut = None,
        highcut = 10
    )
    time = data[0].values
    mask = detect_motion_artifacts(
        ppg_signal=data[2].values,
        fs = 1000,
        kurtosis = 3, 
        entropy = 0.5
        )
    return {
        "time": time,
        "feature": ppg_signal[np.newaxis,:],
        "labels": ["ppg"],
        "maks": mask.astype(bool),
        "feature_info": "Photoplethysmography"
    }
    
def plot_ppg_with_artifacts(ppg_signal, mask, fs):
    time_axis = np.arange(len(ppg_signal)) / fs

    fig, ax = plt.subplots(figsize=(15, 5))

    ax.plot(
        time_axis, 
        ppg_signal,
        color='blue', 
        label='PPG Signal'
        )
    ax.scatter(
        time_axis[~mask.astype(bool)], 
        np.ones_like(ppg_signal[~mask.astype(bool)])*0.04,
        color='red', 
        marker="o",
        label='Bad Segment'
        )
    ax.set_title('PPG Signal with Detected Motion Artifacts')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('PPG Amplitude')
    ax.set_xlim(300,400)


    ax.legend()
    plt.tight_layout()
    plt.show()
    
#filtered = filter_signal(df[2].values, 1000, lowcut=None, highcut = 10)
mask = detect_motion_artifacts(
    df[2].values, 
    1000, 
    kurtosis_threshold=3,
    entropy_threshold=0.5
    )
plot_ppg_with_artifacts(df[2].values, mask, 1000)

    

# %%
