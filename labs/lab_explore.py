#%%
import bids_explorer.architecture.architecture as arch
import scipy.stats
architecture = arch.BidsArchitecture(root = "/Users/samuel/Downloads/PHYSIO_BIDS")


# %%
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional
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
#filtered = filter_signal(df[2].values, 1000, lowcut=None, highcut = 10)
mask = detect_motion_artifacts(
    df[2].values, 
    1000, 
    kurtosis_threshold=3,
    entropy_threshold=0.5
    )
plot_ppg_with_artifacts(df[2].values, mask, 1000)


# TODO:
# [ ] Check if the time in the eyetracking samples is the same as the time in
# [ ] Write ECG exctractor function
# [X] Write EDA exctractor function
# [X] Write PPG exctractor function
# [ ] Write RESP extractor function
# [X] Write a function to extract the events from the eyelink file
# [ ] Write a function to extract the samples from the eyelink file
