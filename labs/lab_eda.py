"""Lab to develop better cleaning for EDA signals.

"""
#%%
import pandas as pd
import numpy as np
from typing import Optional
import numpy.lib.stride_tricks as sliding_window
import pywt
import src.utils as utils
import matplotlib
import src.modalities as modalities
from PyEMD import EMD, Visualisation
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
import scipy
import src.utils as utils
import neurokit2 as nk
from neurokit2.misc.report import create_report
from neurokit2.signal import signal_sanitize
from neurokit2.eda.eda_clean import eda_clean
from neurokit2.eda.eda_methods import eda_methods
from neurokit2.eda.eda_peaks import eda_peaks
from neurokit2.eda.eda_phasic import eda_phasic
from neurokit2.eda.eda_plot import eda_plot
from scipy.signal import savgol_filter

#%%

def wavelet_denoise(signal, wavelet="db4", level=6, threshold_type="soft"):
    coeffs = pywt.wavedec(signal, wavelet, mode="per")
    
    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [pywt.threshold(c, value=uthresh, mode=threshold_type) for c in coeffs[1:]]

    return pywt.waverec(coeffs, wavelet, mode="per")

def dual_wavelet_denoise(signal, wavelet="db4"):
    denoise_fast = wavelet_denoise(
        signal, 
        wavelet=wavelet, 
        level=6, 
        threshold_type="soft"
    )
    denoise_slow = wavelet_denoise(
        signal, 
        wavelet=wavelet, 
        level=8, threshold_type="soft")
    averaged = (denoise_fast + denoise_slow) / 2.0
    
    return averaged


def smooth_signal(signal, window_length=5000, polyorder=3):
    if window_length >= len(signal):
        window_length = len(signal) - 1 if len(signal) % 2 == 0 else len(signal)
    if window_length % 2 == 0:
        window_length += 1

    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)


def perform_emd(signal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Empirical Mode Decomposition on a 1D signal.
    
    Args:
        signal (np.ndarray): Input 1D signal to decompose
        
    Returns:
        tuple[np.ndarray, np.ndarray]: A tuple containing:
            - imfs: Array of Intrinsic Mode Functions
            - residual: The residual trend of the signal
    """
    emd = EMD()
    
    imfs = emd(signal)
    viz = Visualisation(emd)
    viz.plot_imfs(imfs)
    residual = signal - np.sum(imfs, axis=0)
    
    return imfs, residual

#%%
data = modalities.get_random_biopac_file()
eda = data["eda"]
def eda_process(
    eda_signal, sampling_rate=1000, method="neurokit", report=None, **kwargs
):
    """**Process Electrodermal Activity (EDA)**

    Convenience function that automatically processes electrodermal activity (EDA) signal.

    Parameters
    ----------
    eda_signal : Union[list, np.array, pd.Series]
        The raw EDA signal.
    sampling_rate : int
        The sampling frequency of ``"eda_signal"`` (in Hz, i.e., samples/second).
    method : str
        The processing pipeline to apply. Can be one of ``"biosppy"`` or ``"neurokit"`` (default).
    report : str
        The filename of a report containing description and figures of processing
        (e.g. ``"myreport.html"``). Needs to be supplied if a report file
        should be generated. Defaults to ``None``. Can also be ``"text"`` to
        just print the text in the console without saving anything.
    **kwargs
        Other arguments to be passed to specific methods. For more information,
        see :func:`.rsp_methods`.

    Returns
    -------
    signals : DataFrame
        A DataFrame of same length as ``"eda_signal"`` containing the following
        columns:

        .. codebookadd::
            EDA_Raw|The raw signal.
            EDA_Clean|The cleaned signal.
            EDA_Tonic|The tonic component of the signal, or the Tonic Skin Conductance Level (SCL).
            EDA_Phasic|The phasic component of the signal, or the Phasic Skin Conductance Response (SCR).
            SCR_Onsets|The samples at which the onsets of the peaks occur, marked as "1" in a list of zeros.
            SCR_Peaks|The samples at which the peaks occur, marked as "1" in a list of zeros.
            SCR_Height|The SCR amplitude of the signal including the Tonic component. Note that cumulative \
                effects of close-occurring SCRs might lead to an underestimation of the amplitude.
            SCR_Amplitude|The SCR amplitude of the signal excluding the Tonic component.
            SCR_RiseTime|The time taken for SCR onset to reach peak amplitude within the SCR.
            SCR_Recovery|The samples at which SCR peaks recover (decline) to half amplitude, marked  as "1" \
                in a list of zeros.

    info : dict
        A dictionary containing the information of each SCR peak (see :func:`eda_findpeaks`),
        as well as the signals' sampling rate.

    See Also
    --------
    eda_simulate, eda_clean, eda_phasic, eda_findpeaks, eda_plot

    Examples
    --------
    .. ipython:: python

      import neurokit2 as nk

      eda_signal = nk.eda_simulate(duration=30, scr_number=5, drift=0.1, noise=0)
      signals, info = nk.eda_process(eda_signal, sampling_rate=1000)

      @savefig p_eda_process.png scale=100%
      nk.eda_plot(signals, info)
      @suppress
      plt.close()

    """
    eda_signal = wavelet_denoise(eda_signal, wavelet="db4", level=10, threshold_type="soft")
    eda_signal = smooth_signal(eda_signal, window_length=10000)
    eda_signal = signal_sanitize(eda_signal)
    methods = eda_methods(sampling_rate=sampling_rate, method=method, **kwargs)
    eda_cleaned = eda_clean(
        eda_signal,
        sampling_rate=sampling_rate,
        method=methods["method_cleaning"],
        **methods["kwargs_cleaning"],
    )
    if methods["method_phasic"] is None or methods["method_phasic"].lower() == "none":
        eda_decomposed = pd.DataFrame({"EDA_Phasic": eda_cleaned})
    else:
        eda_decomposed = eda_phasic(
            eda_cleaned,
            sampling_rate=sampling_rate,
            method=methods["method_phasic"],
            **methods["kwargs_phasic"],
        )

    peak_signal, info = eda_peaks(
        eda_decomposed["EDA_Phasic"].values,
        sampling_rate=sampling_rate,
        amplitude_min=0.1,
        **methods["kwargs_peaks"],
    )
    info["sampling_rate"] = sampling_rate  

    signals = pd.DataFrame({"EDA_Raw": eda_signal, "EDA_Clean": eda_cleaned})

    signals = pd.concat([signals, eda_decomposed, peak_signal], axis=1)

    if report is not None:
        if ".html" in str(report):
            fig = eda_plot(signals, info, static=False)
        else:
            fig = None
        create_report(file=report, signals=signals, info=methods, fig=fig)

    return signals, info

signals, info = eda_process(eda.data, 1000)

weights = np.zeros(signals["EDA_Phasic"].shape[0])
peaks1 = scipy.signal.find_peaks(
    signals["EDA_Phasic"],
    prominence=0.05,
    distance=5000,
    wlen=10000,
)

peaks2 = scipy.signal.find_peaks(
    signals["EDA_Phasic"],
    prominence=0.05,
    distance=2500,
    wlen=5000,
)

peaks3 = scipy.signal.find_peaks(
    signals["EDA_Phasic"],
    prominence=0.1,
    distance=2500,
)

peaks4 = scipy.signal.find_peaks(
    signals["EDA_Phasic"],
    prominence=0.05,
    distance=2500,
)

for peak in [peaks1, peaks2, peaks3, peaks4]:
    for detected_peak in peak[0]:
        weights[detected_peak] += 1
weights += signals["SCR_Peaks"].values
mask = weights == 5
final_peaks = signals["SCR_Peaks"].values.copy()
final_peaks[~mask] = 0
idx_weight = np.where(final_peaks > 0)[0]
idx = np.where(signals["SCR_Peaks"] > 0)[0]

time = np.arange(signals["EDA_Clean"].shape[0])/info['sampling_rate']
fig, ax = plt.subplots()
ax.plot(time, signals["EDA_Clean"], color = "black")
ax.plot(time[idx], signals["EDA_Clean"].values[idx], "og", label="Detection Without Weighting")
ax.plot(time[idx_weight], signals["EDA_Clean"].values[idx_weight], "or", label="Detection With Weighting")
ax.spines[["top", "right"]].set_visible(False)
ax.set_xlabel("Time (s)")
ax.set_ylabel("EDA (A.U)")
ax.legend()
plt.show()

# %%
fig, ax = plt.subplots()
time = np.arange(signals["EDA_Phasic"].values.shape[0])/info['sampling_rate']
ax.plot(time, signals["EDA_Phasic"])
ax.set_xlim(0,time[-1])
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.set_xlabel("Time (s)")
ax.set_ylabel("EDA Phasic Component (A.U)")
x1, x2, y1, y2 = 200, 240, -0.018, 0.018  
axins = ax.inset_axes(
    [0.5, 0.45, 0.47, 0.47],
    xlim=(x1, x2), ylim=(y1, y2),yticklabels=[])
axins.plot(time,signals["EDA_Phasic"])
axins.spines[["top", "right", "left", "bottom"]].set_color("tab:orange")
axins.set_xticklabels(["","210 s","220 s","230 s",""])
axins.tick_params(axis="x",direction="in", pad=-15, colors="tab:orange")
axins.tick_params(axis="y",direction="in", pad=-15, colors="tab:orange")
ax.indicate_inset_zoom(axins, edgecolor="tab:orange", alpha=1)

plt.show()

# %%
def plot_with_zoom(
    signal,
    fs,
    zoom_time: Optional[tuple] = None,
    time_bounds: Optional[tuple] = None
):
    fig, ax = plt.subplots()
    time = np.arange(signal.shape[0])/fs
    ax.plot(time, signal)
    ax.set_xlim(0,time[-1])
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("EDA (A.U)")
    if time_bounds is None:
        x1 = np.random.choice(time)
        x2 = zoom_time if zoom_time is not None else x1 + 40
    else:
        x1, x2, = time_bounds
    
    signal_box = signal[np.where(time == x1)[0][0]:np.where(time == x2)[0][0]]
    y1 = signal_box.min() - signal_box.min() * 0.01
    y2 = signal_box.max() + signal_box.max() * 0.01
        
    axins = ax.inset_axes(
        [0.5, 0.5, 0.45, 0.45],
        xlim=(x1, x2), ylim=(y1, y2), xticklabels=[])
    axins.plot(time,signal)
    axins.spines[["top", "right", "left", "bottom"]].set_color("tab:orange")
    axins.tick_params(axis="x",direction="in", colors="tab:orange")
    axins.tick_params(axis="y",direction="in", colors="tab:orange")


    rect = matplotlib.patches.Rectangle(
        (x1, y1),
        x2 - x1,
        y2 - y1,
        linewidth=1.5,
        edgecolor="tab:orange",
        facecolor="none",
        zorder=10
    )

    ax.add_patch(rect)
    corners = [
        ((x2, y1), (x1, y1)),
        ((x2, y2), (x1, y2)),
    ]

    for (xyA, xyB) in corners:
        con = ConnectionPatch(
            xyA=xyA, coordsA=ax.transData,
            xyB=xyB, coordsB=axins.transData,
            color="tab:orange", linewidth=1, alpha=1
        )
        ax.add_artist(con)
    
    return fig, ax

# %% DENOISING INVESTIGATION
data = modalities.get_random_biopac_file()
eda = data["eda"]
sign = eda.data
fig, ax = plot_with_zoom(sign, time_bounds=(200,240))
ax.set_title("EDA RAW Signal")

w8 = wavelet_denoise(sign, wavelet="db4", level=10, threshold_type="soft")
smooth10w8 = smooth_signal(w8, window_length=10000)
fig, ax = plot_with_zoom(smooth10w8, time_bounds=(200,240))
ax.set_title("Smoothed Wavelet Denoised Signal (DB4, level=10, window_length=10000)")

#%%
import numpy as np
from scipy.ndimage import uniform_filter1d

def flat_mask(signal, fs, threshold=1e-3, min_duration=3.0):
    """
    Returns a boolean mask where True indicates valid (non-flat) signal.
    
    Parameters:
    - signal: 1D numpy array
    - fs: sampling frequency in Hz
    - threshold: max derivative to be considered flat
    - min_duration: minimum flat segment duration in seconds
    
    Returns:
    - Boolean mask (same length as signal): True = good, False = flat
    """
    
    #smoothed = savgol_filter(signal, window_length=11, polyorder=2, mode='interp')
    derivative = np.abs(np.diff(signal, prepend=signal[0]))
    is_flat = derivative < threshold

    changes = np.diff(is_flat.astype(int), prepend=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    if is_flat[-1]:
        ends = np.append(ends, len(is_flat)//2)

    mask = np.ones(len(signal), dtype=bool)
    min_samples = int(min_duration * fs)
    for start, end in zip(starts, ends):
        if end - start >= min_samples:
            mask[start:end] = False

    return mask


eda_signal = nk.eda_simulate(duration=800, scr_number=4, noise=0)
noise = np.random.normal(0, 0.1, eda_signal.shape[0])
eda_signal += noise
for i in range(np.random.randint(1,5)):
    start = np.random.randint(0, eda_signal.shape[0] - 10000)
    stop = start + np.random.randint(1000, 50000)
    val_start = np.random.normal(-6,2)
    value = np.random.normal(val_start,0.01,stop-start)
    eda_signal[start:stop] = value

time = np.arange(eda_signal.shape[0])/1000
eda = modalities.EDA(eda_signal, time=time, ttl=np.zeros(eda_signal.shape[0]))
eda.process()
mask = flat_mask(
    eda.cleaned_signal["EDA_Raw"],
    fs=1000,
    threshold=1e-2,
    min_duration=3.0
)
changes = np.diff(mask.astype(int), prepend=0)
stops = np.where(changes == 1)[0]
starts = np.where(changes == -1)[0]

fig, ax = plot_with_zoom(eda.cleaned_signal["EDA_Raw"], fs=1000, time_bounds=(200,240))
ax.plot(time[~mask], eda.cleaned_signal["EDA_Raw"][~mask], "rx")
ax.set_title("EDA RAW Signal with Noise")



# %%
data = modalities.get_random_biopac_file()
eda = data["eda"]
fig, ax = plt.subplots()
ax.plot(eda.time, eda.data, color="gray", alpha=0.4)
eda.process()
print(f"EDA data shape: {eda.data.shape}")
print(f"Masks shape: {eda.masks.shape}")
print(f"Peaks weights shape: {eda.cleaned_signal['peaks_weights'].shape}")
print(f"Raw signal shape: {eda.cleaned_signal['EDA_Raw'].shape}")
print(f"Clean signal shape: {eda.cleaned_signal['EDA_Clean'].shape}")
ax.plot(eda.time[eda.masks], eda.cleaned_signal["EDA_Clean"][eda.masks], color="black")
ax.plot(eda.time[eda.cleaned_signal["peaks_weights"] == 1], eda.cleaned_signal["EDA_Clean"][eda.cleaned_signal["peaks_weights"] == 1], "ro")
ax.set_title("EDA RAW Signal with Noise")
# %%
