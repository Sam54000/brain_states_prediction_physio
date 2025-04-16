#%%
import bids_explorer.architecture.architecture as arch
import scipy.stats
architecture = arch.BidsArchitecture(root = "/Users/samuel/Downloads/PHYSIO_BIDS")


# %%
import pandas as pd
import matplotlib.pyplot as plt
from src.utils import *
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
import neurokit2 as nk

df_biopac = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-04/ses-07Baby/func/sub-04_ses-07Baby_task-ActiveHighVid_run-1_acq-biopac_recording-samples_physio.tsv.gz", sep = "\t", header = None)
df_eyetracking_samples = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-04/ses-07Baby/func/sub-04_ses-07Baby_task-ActiveHighVid_run-1_acq-eyelink_recording-samples_physio.tsv.gz", sep = "\t", header = None)
df_eyetracking_events = pd.read_csv("/Users/samuel/Desktop/PHYSIO_BIDS/sub-04/ses-07Baby/func/sub-04_ses-07Baby_task-ActiveHighVid_run-1_acq-eyelink_recording-events_physio.tsv.gz", header = None, sep = "\t")

#%% EYETRACKING
eyetracking_dict = extract_and_process_eyetracking(df_eyetracking_samples)
#%% CROP EYETRACKING
eyetracking_dict = crop_eyetracking(eyetracking_dict, df_eyetracking_events)
#%% EDA
eda_dict = extract_and_process_eda(df_biopac)
#%% PPG
ppg_dict = extract_and_process_ppg(df_biopac)
#%% RSP
rsp_dict = extract_and_process_rsp(df_biopac)
#%% ECG
ecg_dict = extract_and_process_ecg(df_biopac)
#%% PLOT
def plot_results(data_dict: dict, title: str, time: Optional[np.ndarray] = None):
    raw = data_dict["feature"][0,:]
    masked = raw.copy().astype(float)
    masked[~data_dict["mask"]] = np.nan
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(data_dict["time"], raw, label = "Raw")
    ax.plot(data_dict["time"], masked, label = "Masked")
    ax.set_xlim(time[0], time[-1])
    ax.legend()
    ax.set_title(title)
    plt.show()
    
#%%
# TODO:
# [ ] Check if the time in the eyetracking samples is the same as the time in
# [ ] Write ECG exctractor function
# [X] Write EDA exctractor function
# [X] Write PPG exctractor function
# [X] Write RESP extractor function
# [X] Write a function to extract the events from the eyelink file
# [ ] Write a function to extract the samples from the eyelink file
