#%%
import bids_explorer.architecture.architecture as arch
import scipy.stats
architecture = arch.BidsArchitecture(root = "/Users/samuel/Downloads/PHYSIO_BIDS")


# %%
import pandas as pd
import matplotlib.pyplot as plt
from src import utils
import numpy as np
from typing import Optional
import scipy
from scipy.stats import kurtosis, entropy
from scipy.signal import welch
import os
from ecgdetectors import Detectors
from pathlib import Path
from scipy import signal
from viz import plotting
from labs import lab_explore
#%%

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
df_samples = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-01/ses-03Beauty/func/sub-01_ses-03Beauty_task-ActiveHighVid_run-1_acq-eyelink_recording-samples_physio.tsv.gz", sep = "\t", header = None)
df_events = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-01/ses-03Beauty/func/sub-01_ses-03Beauty_task-ActiveHighVid_run-1_acq-eyelink_recording-events_physio.tsv.gz", header = None, sep = "\t")
# %%
events = utils.extract_eyelink_events(df_events)
#%%
i = 3
for i in range(5):
    fig, ax = plt.subplots()
    ax.plot(df_samples[0], [0]*df_samples[0].values.shape[0], "bo", label = "samples from samples file")
    ax.axvline(events["time_ms"].values[i],color = "red", label = "event from events file")
    ax.text(events["time_ms"].values[i]+0.1, 0.04, events["event_name"].values[i], color = "red") 
    ax.set_xlim(events["time_ms"].values[i]-30,events["time_ms"].values[i]+30)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    ax.set_xlabel("Time (ms)")
    ax.legend()
    
    plt.savefig(Path().cwd().parent / f"figs/event_displacement_example_{i+1}.png")
#%%
# TODO:
# [ ] Check if the time in the eyetracking samples is the same as the time in
# [ ] Write ECG exctractor function
# [X] Write EDA exctractor function
# [X] Write PPG exctractor function
# [ ] Write RESP extractor function
# [X] Write a function to extract the events from the eyelink file
# [ ] Write a function to extract the samples from the eyelink file
