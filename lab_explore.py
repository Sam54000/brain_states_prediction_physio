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

def bandpass_filter(data: np.ndarray, 
                   sampling_rate: float, 
                   lowcut: float, 
                   highcut: float, 
                   order: int = 5) -> np.ndarray:
    """
    Apply a Butterworth band-pass filter to a time-series data.
    
    Args:
        data (np.ndarray): 1D array of time-series data
        sampling_rate (float): Sampling rate of the data in Hz
        lowcut (float): Lower frequency bound of the filter in Hz
        highcut (float): Higher frequency bound of the filter in Hz
        order (int, optional): Order of the filter. Defaults to 5.
        
    Returns:
        np.ndarray: Filtered data
    """
    # Normalize the frequencies by the Nyquist frequency
    nyquist = sampling_rate / 2.0
    low = lowcut / nyquist
    high = highcut / nyquist
    
    # Design the Butterworth filter
    b, a = signal.butter(order, [low, high], btype='band')
    
    # Apply the filter
    filtered_data = signal.filtfilt(b, a, data)
    
    return filtered_data

# Example usage:
# sampling_rate = 1000  # Hz (1000 samples per second)
# filtered_data = bandpass_filter(data, sampling_rate, lowcut=0.1, highcut=5.0)

# %%
