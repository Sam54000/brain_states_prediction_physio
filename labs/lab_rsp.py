#%%
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
rsp = data["rsp"]
rsp.process()
rsp.data = rsp.data - np.mean(rsp.data)
epochs, times = rsp.epochs(pre_time=1.7, post_time=1.7)
fig, ax = plt.subplots(figsize = (10, 10))
ax.plot(epochs.T, color = "black", alpha = 0.1, linewidth = 0.5)
Q1 = np.quantile(rsp.data, 0.25)
Q3 = np.quantile(rsp.data, 0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
higher_bound = Q3 + 1.5 * IQR
fig, ax = plt.subplots(figsize = (10, 10))
ax.hist(rsp.data, bins = 250)
ax.axvline(lower_bound, linestyle = "--", color = "red")
ax.axvline(higher_bound, linestyle = "--", color = "red")
random_start = np.random.choice(rsp.time)
fig, ax = plt.subplots(nrows = 2, ncols = 1)
ax[0].plot(rsp.time, rsp.data)
ax[0].axhline(lower_bound, linestyle = "--", color = "red")
ax[0].axhline(higher_bound, linestyle = "--", color = "red")
ax[1].plot(rsp.time, rsp.data)
ax[1].axhline(lower_bound, linestyle = "--", color = "red")
ax[1].axhline(higher_bound, linestyle = "--", color = "red")
ax[1].set_xlim(random_start, random_start + 120)

fig, ax = plt.subplots(nrows = 2, ncols = 1)
ax[0].plot(rsp.time, rsp.data)
ax[0].axhline(lower_bound, linestyle = "--", color = "red")
ax[0].axhline(higher_bound, linestyle = "--", color = "red")
ax[0].plot(rsp.time[~rsp.masks], rsp.data[~rsp.masks], "rx")
ax[1].plot(rsp.time, rsp.data)
ax[1].axhline(lower_bound, linestyle = "--", color = "red")
ax[1].axhline(higher_bound, linestyle = "--", color = "red")
ax[1].plot(rsp.time[~rsp.masks], rsp.data[~rsp.masks], "rx")
ax[1].set_xlim(random_start, random_start + 120)


# %%
plt.plot(rsp.time, rsp.data)
plt.xlim(380, 430)

# %%
import src.utils as utils
sig = utils.filter_signal(rsp.data, 
                          rsp.fs, 
                          lowcut = 0.1, 
                          highcut = None, 
                          order = 4
                          )
sig = utils.wavelet_denoise(sig, wavelet = "db4", level = 3, threshold_type = "soft")
sig = utils.moving_average_smoothing(sig[:-1], window_size = 250)

masks = utils.detect_flat_signal(
    sig, 
    rsp.fs, 
    threshold = 0.001, 
    min_duration=0.75,
)

plt.plot(rsp.time, rsp.data)
plt.plot(rsp.time, sig)
#plt.plot(rsp.time[~masks], rsp.data[~masks], "rx")
plt.xlim(380, 430)
#plt.xlim(500, 550)
#plt.ylim(-8,-7)

# %%

import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
from sklearn.preprocessing import MinMaxScaler
from scipy.signal import decimate
data = modalities.get_random_biopac_file()
rsp = data["rsp"]
downsample_factor = 40
rsp.data = decimate(rsp.data, downsample_factor)
rsp.fs = rsp.fs / downsample_factor
rsp.time = np.arange(0, len(rsp.data)) / rsp.fs
rsp.process()
signal = rsp.data

signal[60*int(rsp.fs):65*int(rsp.fs)] = np.random.uniform(low = -1, high = 1, size = 5*int(rsp.fs))
signal[150*int(rsp.fs):155*int(rsp.fs)] += np.random.normal(0, 2, 5*int(rsp.fs))  # noise burst
signal[400*int(rsp.fs):405*int(rsp.fs)] = np.clip(signal[400*int(rsp.fs):405*int(rsp.fs)] * 5, -0.5, 0.5)  # clipping

# Normalize
scaler = MinMaxScaler()
signal_scaled = scaler.fit_transform(signal.reshape(-1, 1)).flatten()
epochs, times = rsp.epochs(pre_time=2.5, post_time=2.5)
mean_epoch = np.mean(epochs, axis = 0)
mean_epoch_scaled = scaler.fit_transform(mean_epoch.reshape(-1, 1)).flatten()

# Template (normal breathing pattern)
template = mean_epoch_scaled

# Sliding window DTW
window_size = len(template)
step = 10
dtw_scores = []
for i in range(0, len(signal_scaled) - window_size, step):
    window = signal_scaled[i:i + window_size]
    distance, _ = fastdtw(template, window)
    dtw_scores.append(distance)
interpolant = scipy.interpolate.interp1d(
    np.arange(0, len(signal_scaled), step),
    dtw_scores,
    kind = "cubic"
)
dtw_scores = interpolant(np.arange(len(signal_scaled)))
# Plot
fig, ax = plt.subplots(figsize = (10, 4))
ax1 = ax.twinx()
ax.plot(dtw_scores)
ax.set_title("DTW Distance (lower is better match to template)")
ax.set_xlabel("Sliding Window Index")
ax.set_ylabel("DTW Distance")
ax1.plot(signal, color = "red")
#ax.set_xlim(20000,22000)
plt.show()
# %%
