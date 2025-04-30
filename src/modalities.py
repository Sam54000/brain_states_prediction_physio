# %%
import numpy as np
import pandas as pd
from scipy.stats import kurtosis, entropy
import scipy
from scipy.signal import welch
from typing import Optional
from src.utils import (
    catch_error,
    wavelet_denoise,
    smooth_signal,
    correct_shapes,
    detect_flat_signal,
    detect_high_gradient,
    filter_signal,
    moving_average_smoothing,
    pchip_interpolate_masked_data,
    create_sliding_zscore_mask,
    long_flat_segment_detection,
)
from numpy.lib.stride_tricks import sliding_window_view
from dataclasses import dataclass
import matplotlib.pyplot as plt
import neurokit2 as nk
import bids_explorer.architecture.architecture as arch
import pickle


def plot_signal_segment(signal, cleaned_signal, time, r_peaks, fs, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 4))

    random_start_idx = np.random.randint(0, len(time) - int(10 * fs))
    end_idx = random_start_idx + int(10 * fs)
    time_segment = time[random_start_idx:end_idx]

    ax.plot(
        time_segment,
        signal[random_start_idx:end_idx],
        color="gray",
        label="Raw Signal",
        alpha=0.5,
    )

    ax.plot(
        time_segment,
        cleaned_signal[random_start_idx:end_idx],
        color="tab:green",
        linewidth=2.5,
        label="Cleaned Signal",
    )

    r_peaks_in_segment = r_peaks[(r_peaks >= random_start_idx) & (r_peaks < end_idx)]

    if len(r_peaks_in_segment) > 0:
        ax.plot(
            time[r_peaks_in_segment],
            cleaned_signal[r_peaks_in_segment],
            "ro",
            label="Detected R-peaks",
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (A.U)")
    ax.set_title("Comparison of Raw and Cleaned Signal with Detected R-peaks")
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(random_start_idx / fs, end_idx / fs)
    return ax


def read_biopac(filename: str):
    dataframe = pd.read_csv(filename, compression="gzip", sep="\t", header=None)
    rising_edges = np.diff(dataframe[5].values, prepend=0)
    rising_edges[rising_edges <= 0] = 0
    rising_edges[rising_edges > 0] = 1

    data = {
        "time": dataframe[0].values,
        "ttl": rising_edges.astype(bool),
        "ttl_idx": np.where(rising_edges.astype(bool))[0],
        "ttl_times": dataframe[0].values[np.where(rising_edges.astype(bool))[0]],
        "eda": EDA(dataframe[1].values, dataframe[0].values, dataframe[5].values),
        "ppg": PPG(dataframe[2].values, dataframe[0].values, dataframe[5].values),
        "rsp": RSP(dataframe[3].values, dataframe[0].values, dataframe[5].values),
        "ecg": ECG(dataframe[4].values, dataframe[0].values, dataframe[5].values),
    }
    return data


def read_eyetracking(samples_filename: str):
    sample_dataframe = pd.read_csv(
        samples_filename, compression="gzip", sep="\t", header=None
    )

    for col in [1, 2, 3]:
        sample_dataframe[col] = sample_dataframe[col].apply(catch_error)
    events_dataframe = pd.read_csv(
        str(samples_filename).replace("samples", "events"),
        compression="gzip",
        sep="\t",
        header=None,
    )

    time_samples = sample_dataframe[0] / 1000

    message_df = events_dataframe.loc[events_dataframe[0] == "MSG"]
    message_df.reset_index(inplace=True)
    scanner_start_idx = message_df[message_df[2] == "SCANNER_START"].index[0]
    cropped_df = message_df.iloc[scanner_start_idx:]
    cropped_df.rename(columns={1: "time", 2: "event_name"}, inplace=True)
    cropped_df.reset_index(inplace=True)
    cropped_df["time"] = cropped_df["time"].astype(int) / 1000
    events_time = cropped_df["time"].values
    events_name = cropped_df["event_name"].values
    idx_start = np.argmin(abs(time_samples - events_time[0]))
    idx_stop = np.argmin(abs(time_samples - events_time[-2]))
    sample_dataframe = sample_dataframe.iloc[idx_start:idx_stop]
    time_samples = sample_dataframe[0].values / 1000
    x = sample_dataframe[1].values
    y = sample_dataframe[2].values
    pupil = sample_dataframe[3].values

    time_samples = time_samples - time_samples[0]
    events_time = events_time - events_time[0]

    data = {
        "eyetracking": EyeTracking(
            x=x,
            y=y,
            pupil=pupil,
            events_time=events_time,
            events_name=events_name,
            time=time_samples,
        )
    }
    return data


def calculate_quality(mask: np.ndarray):
    """Calculate the quality of the signal based on the mask.

    Args:
        mask (np.ndarray): The mask of the signal.

    Returns:
        float: The quality of the signal.
    """
    return np.sum(mask) / len(mask)


@dataclass
class BaseModality:
    data: np.ndarray
    time: np.ndarray
    ttl: np.ndarray

    def __post_init__(self):
        self.fs = 1 / (self.time[1] - self.time[0])


@dataclass
class EyeTracking:
    pupil: np.ndarray
    x: np.ndarray
    y: np.ndarray
    events_time: np.ndarray
    events_name: np.ndarray
    time: np.ndarray

    def adapt_ttl(self, ttl_times: np.ndarray) -> "EyeTracking":
        adjusted_ttl_times = np.zeros_like(self.time)
        for ttl_time in ttl_times:
            idx = np.argmin(abs(self.time - ttl_time))
            adjusted_ttl_times[idx] = 1
        self.ttl = adjusted_ttl_times
        return self

    def process(self) -> "EyeTracking":
        fs = 1 / (self.time[1] - self.time[0])
        original_signal = self.pupil - np.min(self.pupil)

        flat_mask = detect_flat_signal(
            original_signal,
            fs=fs,
            threshold=1e-3,
            min_duration=1.0
        )
        flat_masks = np.logical_and(flat_mask, original_signal > 10)
        gradient_mask = detect_high_gradient(
            original_signal, fs=fs, threshold=30, buffer=0.25
        )
        combined_masks = np.logical_and(flat_masks, gradient_mask)
        zscore_mask = create_sliding_zscore_mask(
            signal=original_signal,
            fs=fs,
            window_size=2.0,
            step_size=1.0,
            z_threshold=2.0,
            buffer=0.1,
        )
        combined_masks = np.logical_and(combined_masks, zscore_mask)
        after_zscore = original_signal.copy()
        after_zscore[~combined_masks] = np.nan
        window_size_samples = int(0.1 * fs)
        smoothed = moving_average_smoothing(after_zscore.copy(), window_size_samples)
        smoothed_interp = pchip_interpolate_masked_data(smoothed, self.time)
        self.cleaned_pupil = smoothed_interp
        final_masks = long_flat_segment_detection(combined_masks, fs, min_duration=2.0)
        self.masks = np.stack([combined_masks, final_masks], axis=0)
        self.info = "Eyetracking data after post-processing"
        self.first_derivative = np.diff(
            self.cleaned_pupil, prepend=self.cleaned_pupil[0]
        )
        self.second_derivative = np.diff(
            self.cleaned_pupil, n=2, prepend=self.first_derivative[:2]
        )
        self.quality = {
            "masks_based": calculate_quality(self.masks[0]),
            "masks_based_long_flat": calculate_quality(self.masks[1]),
        }
        return self

    def plot(self, title: str = "Eyetracking data") -> plt.Figure:
        """Plot multiple processing stages vertically for comparison"""
        fig = plt.figure(figsize=(12, 7))
        gs = fig.add_gridspec(3, 1, height_ratios=[0.1, 1, 1])
        ax_top = fig.add_subplot(gs[0])
        ax_top.set_visible(False)
        ax1 = fig.add_subplot(gs[1])
        ax2 = fig.add_subplot(gs[2])

        cleaned_sig_masked = self.cleaned_pupil.copy()
        cleaned_sig_masked[~self.masks[1]] = np.nan
        random_time = np.random.choice(self.time)

        ax1.plot(self.time, self.pupil, label="Original signal", color="gray")
        ax1.plot(
            self.time, cleaned_sig_masked, label="Cleaned signal", color="tab:green"
        )
        ax1.axvspan(
            random_time,
            random_time + 20,
            color="orange",
            alpha=0.4,
            label="Segment for Enlarged View",
        )
        ax1.set_xlim(0, self.time[-1])
        ax1.spines[["top", "right"]].set_visible(False)

        ax2.plot(self.time, self.pupil, label="Original signal", color="gray")
        ax2.plot(
            self.time,
            cleaned_sig_masked,
            label="Cleaned signal",
            color="tab:green",
            linewidth=2,
        )
        ax2.spines[["top", "right"]].set_visible(False)
        ax2.set_xlim(random_time, random_time + 20)
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Pupil Size (A.U)")

        before_interpolation = (
            calculate_quality(np.logical_and(self.masks[0], self.masks[1])) * 100
        )
        after_interpolation = calculate_quality(self.masks[1]) * 100

        txt = [
            f"Good Before Interpolation: {before_interpolation:.2f}%",
            f"Good After Interpolation: {after_interpolation:.2f}%",
        ]
        txt = "\n".join(txt)

        handles, labels = ax1.get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.1, 0.80))

        fig.text(
            x=0.6,
            y=0.80,
            s=txt,
            ha="left",
            va="top",
            bbox=dict(
                facecolor="white",
                edgecolor="black",
            ),
            fontfamily="monospace",
        )

        fig.suptitle(title, y=0.80)
        plt.tight_layout()
        plt.subplots_adjust(top=0.80)

        return fig

    def save(self, saving_path: str) -> None:
        if not hasattr(self, "ttl"):
            raise AttributeError(
                "TTL is not defined, please run `adapt_ttl`"
                / "method first adapt the TTL to the eyetracking time space"
            )

        if not hasattr(self, "cleaned_pupil"):
            raise AttributeError(
                "cleaned_pupil is not defined, please run" / "`process` method first"
            )

        saving_dict = {
            "time": self.time,
            "features": np.stack(
                [
                    self.x,
                    self.y,
                    self.cleaned_pupil,
                    self.first_derivative,
                    self.second_derivative,
                ],
                axis=0,
            ),
            "labels": ["x", "y", "pupil", "first_derivative", "second_derivative"],
            "quality": self.quality,
            "masks": self.masks,
            "info": self.info,
            "process": [
                "Mask Flat Signal",
                "Mask High Gradient",
                "Mask Z-Score (First Mask)",
                "Moving Average Smoothing",
                "Interpolate Missing Data with PCHIP",
                "Mask Long Flat Segments (Second Mask)",
            ],
            "ttl": self.ttl,
        }

        with open(saving_path, "wb") as f:
            pickle.dump(saving_dict, f)

        events = {"time": self.events_time, "event": self.events_name}
        events_filename = str(saving_path).replace("eyetracking.pkl", "events.pkl")
        with open(events_filename, "wb") as f:
            pickle.dump(events, f)


@dataclass
class EDA(BaseModality):
    def __post_init__(self):
        super().__post_init__()

    def _generate_peaks_weights(self) -> np.ndarray:
        weights = np.zeros(self.cleaned_signal["EDA_Phasic"].shape[0])
        args = {
            "prominence": [0.05, 0.05, 0.1, 0.05],
            "distance": [5000, 2500, 2500, 2500],
            "wlen": [10000, 5000, None, None],
        }

        for i in range(len(args["prominence"])):
            peaks = scipy.signal.find_peaks(
                self.cleaned_signal["EDA_Phasic"],
                prominence=args["prominence"][i],
                distance=args["distance"][i],
                wlen=args["wlen"][i],
            )
            for detected_peak in peaks[0]:
                weights[detected_peak] += 1
        weights += self.cleaned_signal["SCR_Peaks"].values
        weights /= 5
        return weights


    def process(self) -> "EDA":
        self.masks = detect_flat_signal(
            self.data, 
            fs=self.fs, 
            threshold=1e-3,
            min_duration=3.0
        )

        self.cleaned_signal = filter_signal(
            self.data, fs=self.fs, lowcut=None, highcut=5, order=4
        )
        self.cleaned_signal = wavelet_denoise(
            self.cleaned_signal, 
            wavelet="db4",
            level=10,
            threshold_type="hard"
            )
        self.cleaned_signal = smooth_signal(
            self.cleaned_signal, 
            window_length=5000
            )

        self.cleaned_signal, self._info = nk.eda_process(
            self.cleaned_signal, sampling_rate=self.fs
        )
        self.cleaned_signal["peaks_weights"] = self._generate_peaks_weights()
        self.info = "EDA data after post-processing"
        self.quality = {
            "masks_based": calculate_quality(self.masks),
        }
        self = correct_shapes(self, ["masks", "cleaned_signal", "data"])
        return self

    def plot(self, title: str = "EDA Signal") -> plt.Figure:
        nk.eda_plot(self.cleaned_signal, self._info)
        fig = plt.gcf()
        fig.set_size_inches(10, 12, forward=True)
        fig.suptitle(title)
        return fig

    def save(self, saving_path: str) -> None:
        temp_dictionary = self.cleaned_signal.to_dict(orient="list")
        self.features = np.stack(list(temp_dictionary.values()), axis=0)
        self.labels = list(temp_dictionary.keys())
        saving_dict = {
            "time": self.time,
            "features": self.features,
            "labels": self.labels,
            "quality": {"masks_based": calculate_quality(self.masks)},
            "masks": self.masks,
            "info": self.info,
            "process": ["Mask Flat Signal", "Process with NeuroKit2"],
            "ttl": self.ttl,
        }

        with open(saving_path, "wb") as f:
            pickle.dump(saving_dict, f)


@dataclass
class PPG(BaseModality):
    def __post_init__(self):
        super().__post_init__()

    def _detect_motion_artifacts(
        self,
        window_size: float = 5,
        step_size: float = 0.25,
        kurtosis_threshold: float = 3,
        entropy_threshold: float = 1,
    ) -> np.ndarray:
        """
        Detects motion artifacts in a PPG signal using kurtosis and Shannon entropy.

        Args:
            window_size (float, optional): Size of the sliding window in seconds.
                Defaults to 5.
            step_size (float, optional): Step size for the sliding window in seconds.
                Defaults to 0.25.
            kurtosis_threshold (float, optional): Threshold for kurtosis to detect
                artifacts. Defaults to 3.
            entropy_threshold (float, optional): Threshold for entropy to detect
                artifacts. Defaults to 1.

        Returns:
            numpy.ndarray: 1D numpy array of the same length as signal,
            with True indicating clean segments and False indicating artifacts.

        Reference:
            .. _Statistical approach for the detection of motion/noise artifacts in Photoplethysmogram:
            https://pubmed.ncbi.nlm.nih.gov/22255454/
        """
        signal_length = len(self.data)
        mask = np.ones(signal_length, dtype=bool)

        window_samples = int(window_size * self.fs)
        step_samples = int(step_size * self.fs)

        for start in range(0, signal_length - window_samples + 1, step_samples):
            end = start + window_samples
            segment = self.data[start:end]

            seg_kurtosis = kurtosis(segment)
            f, Pxx = welch(segment, fs=self.fs)
            Pxx_norm = Pxx / np.sum(Pxx)
            seg_entropy = entropy(Pxx_norm)
            if seg_kurtosis > kurtosis_threshold or seg_entropy < entropy_threshold:
                mask[start:end] = False

        return mask

    def process(self) -> "PPG":
        self.masks = detect_flat_signal(
            self.data, 
            fs=self.fs, 
            threshold=1e-3,
            min_duration=1.0
        )

        self.masks = np.logical_and(
            self.masks,
            self._detect_motion_artifacts(
                signal=self.data,  # For some reason the detector
                # works better on
                # non filtered data
                fs=self.fs,
                kurtosis_threshold=3,
                entropy_threshold=0.5,
            ),
        )

        self.cleaned_signal = filter_signal(
            self.data, fs=self.fs, lowcut=None, highcut=8, order=4
        )

        self.cleaned_signal, self._info = nk.ppg_process(
            self.cleaned_signal, sampling_rate=self.fs
        )

        self.info = "PPG data after post-processing"
        self.quality = {
            "nk2": self.cleaned_signal["PPG_Quality"].mean(),
            "masks_based": calculate_quality(self.masks),
        }

        return self

    def plot(self, title: str = "PPG Signal") -> plt.Figure:
        nk.ppg_plot(self.cleaned_signal, self._info)
        fig = plt.gcf()
        fig.set_size_inches(12, 10, forward=True)
        fig.suptitle(title)
        return fig

    def save(self, saving_path: str) -> None:
        temp_dictionary = self.cleaned_signal.to_dict(orient="list")
        self.features = np.stack(list(temp_dictionary.values()), axis=0)
        self.labels = list(temp_dictionary.keys())
        saving_dict = {
            "time": self.time,
            "features": self.features,
            "labels": self.labels,
            "quality": self.quality,
            "masks": self.masks,
            "info": self.info,
            "process": [
                "Mask Flat Signal",
                "Mask Motion Artifacts",
                "Filter Signal",
                "Process with NeuroKit2",
            ],
            "ttl": self.ttl,
        }

        with open(saving_path, "wb") as f:
            pickle.dump(saving_dict, f)


@dataclass
class RSP(BaseModality):
    def __post_init__(self):
        super().__post_init__()

    def process(self) -> "RSP":
        self.cleaned_signal = filter_signal(
            self.data, fs=self.fs, lowcut=0.1, highcut=4, order=4
        )
        self.cleaned_signal, self._info = nk.rsp_process(
            self.cleaned_signal, sampling_rate=self.fs
        )

        self.masks = detect_flat_signal(
            signal=self.data, 
            fs=self.fs, 
            threshold=1e-3,
            min_duration=2.0
        )

        self.quality = {
            "masks_based": calculate_quality(self.masks),
        }

        return self

    def plot(self, title: str = "RSP Signal") -> plt.Figure:
        nk.rsp_plot(self.cleaned_signal, self._info)
        fig = plt.gcf()
        fig.set_size_inches(10, 12, forward=True)
        fig.suptitle(title)
        return fig

    def save(self, saving_path: str) -> None:
        temp_dictionary = self.cleaned_signal.to_dict(orient="list")
        self.features = np.stack(list(temp_dictionary.values()), axis=0)
        self.labels = list(temp_dictionary.keys())
        saving_dict = {
            "time": self.time,
            "features": self.features,
            "labels": self.labels,
            "quality": self.quality,
            "masks": self.masks,
            "info": "Respiration Signal after post-processing",
            "process": ["Mask Flat Signal", "Process with NeuroKit2"],
            "ttl": self.ttl,
        }

        with open(saving_path, "wb") as f:
            pickle.dump(saving_dict, f)


@dataclass
class ECG(BaseModality):
    def __post_init__(self):
        self.fs = 1 / (self.time[1] - self.time[0])

    def _detect_r_peaks(
        self,
        window_size_sec: float = 0.25,
        threshold_factor: float = 0.6,
        buffer_size_sec: float = 0.1,
    ) -> "ECG":
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
        absolute_signal = abs(self.cleaned_signal)
        window_size = int(window_size_sec * self.fs)
        r_peaks = []
        M_mean = np.mean(absolute_signal)
        candidate_found = False
        best_candidate_idx = None
        best_candidate_height = None

        for i in range(len(absolute_signal) - window_size):
            window = absolute_signal[i : i + window_size]
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
        self.r_peaks = r_peaks[
            np.where(np.diff(r_peaks) > buffer_size_sec * self.fs)[0]
        ]

        return self

    def _adjust_r_peaks(self):
        """
        Correct the R-peaks in an ECG signal.
        """
        signal = abs(self.cleaned_signal)
        num_peak = self.r_peaks.shape[0]
        peaks_corrected_list = list()
        for index in range(num_peak):
            i = self.r_peaks[index]
            cnt = i
            if cnt - 1 < 0:
                break
            if signal[cnt] < signal[cnt - 1]:
                while signal[cnt] < signal[cnt - 1]:
                    cnt -= 1
                    if cnt < 0:
                        break
            elif signal[cnt] < signal[cnt + 1]:
                while signal[cnt] < signal[cnt + 1]:
                    cnt += 1
                    if cnt < 0:
                        break
            peaks_corrected_list.append(cnt)
        peaks_corrected = np.asarray(peaks_corrected_list)
        self.corrected_r_peaks = peaks_corrected
        return self

    def _count_bpm(self, window_size_sec: float = 10) -> "ECG":
        """
        Count the beats per minute (BPM) in an ECG signal using a sliding window approach.

        Args:
            window_size_sec: The size of the window in seconds.

        Returns:
            self: The instance of the class with the BPM data.
        """
        window_size = int(window_size_sec * self.fs)
        count_mask = np.zeros(len(self.cleaned_signal), dtype=int)
        count_mask[self.corrected_r_peaks] = 1
        nb_r_peaks = np.sum(sliding_window_view(count_mask, window_size), axis=1)
        bpm = 60 * nb_r_peaks / window_size_sec
        windowed_bpm = sliding_window_view(bpm, int(window_size / 2))
        smoothed_bpm = np.mean(windowed_bpm, axis=1)
        smoothed_bpm = np.concatenate(
            (
                smoothed_bpm,
                np.ones(int(window_size + window_size / 2) - 2) * smoothed_bpm[-1],
            )
        )
        self.bpm = smoothed_bpm
        return self

    def _plot_bpm(self, ax: plt.Axes = None) -> plt.Axes:
        if ax is None:
            fig, ax = plt.subplots(figsize=(12, 4))
        min_len = min(len(self.time), len(self.bpm))
        ax.plot(self.time[:min_len], self.bpm[:min_len], color="black", label="BPM")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Beats per minute")
        ax.set_xlim(min(self.time), max(self.time))
        ax.set_ylim(0, 120)
        txt = [
            f"Mean BPM: {np.mean(self.bpm):.2f}",
            f"Std BPM: {np.std(self.bpm):.2f}",
            f"Min BPM: {np.min(self.bpm):.2f}",
            f"Max BPM: {np.max(self.bpm):.2f}",
            f"Median BPM: {np.median(self.bpm):.2f}",
            f"Signal Quality: {calculate_quality(self.masks) * 100:.2f}%",
        ]
        txt = "\n".join(txt)

        ax.text(
            0.01,
            0.05,
            txt,
            ha="left",
            va="bottom",
            transform=ax.transAxes,
            bbox=dict(
                boxstyle="round", facecolor="white", edgecolor="black", alpha=0.3
            ),
            fontfamily="monospace",
        )

        ax.spines[["top", "right"]].set_visible(False)
        return ax

    def _create_r_peak_epochs(
        self, r_peaks: np.ndarray, pre_time: float = 0.5, post_time: float = 0.5
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Create epochs of signal around each R peak.

        Parameters:
        - r_peaks: Array of indices where R-peaks are detected
        - pre_time: Time before the R peak in seconds (default: 0.5)
        - post_time: Time after the R peak in seconds (default: 0.5)

        Returns:
        - epochs: 2D numpy array where each row is an epoch around an R peak
        - times: 1D array of time values for each epoch (in seconds relative to R peak)
        """
        pre_samples = int(pre_time * self.fs)
        post_samples = int(post_time * self.fs)
        epoch_length = pre_samples + post_samples + 1
        times = np.linspace(-pre_time, post_time, epoch_length)
        epochs = np.zeros((len(r_peaks), epoch_length))
        valid_epochs = 0

        for i, peak_idx in enumerate(r_peaks):
            conditions = [
                peak_idx >= pre_samples,
                peak_idx + post_samples < len(self.cleaned_signal),
            ]

            if all(conditions):
                start_idx = peak_idx - pre_samples
                end_idx = peak_idx + post_samples + 1
                epochs[valid_epochs] = self.cleaned_signal[start_idx:end_idx]
                valid_epochs += 1

        return epochs[:valid_epochs], times

    def _plot_epochs(
        self,
        epochs: np.ndarray,
        epoch_times: np.ndarray,
        title: str = "ECG Signal Epochs Around R Peaks",
        ax: Optional[plt.Axes] = None,
    ) -> plt.Axes:
        """
        Plot the epochs of the ECG signal around the R peaks.

        Args:
            epochs: 2D numpy array of ECG signal epochs
            epoch_times: 1D numpy array of time values for each epoch
            title: str = "ECG Signal Epochs Around R Peaks"
            ax: plt.Axes = None

        Returns:
            ax: plt.Axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(epoch_times, epochs.T, linewidth=0.5, color="black", alpha=0.1)
        ax.plot(epoch_times, np.mean(epochs, axis=0), color="red", label="Mean")
        ax.fill_between(
            epoch_times,
            np.mean(epochs, axis=0) - np.std(epochs, axis=0),
            np.mean(epochs, axis=0) + np.std(epochs, axis=0),
            color="orange",
            alpha=0.2,
            label="Std",
        )
        ax.axvline(x=0, color="g", linestyle="--", label="R Peak", alpha=0.6)
        ax.set_xlabel("Time relative to R peak (s)")
        ax.set_xlim(min(epoch_times), max(epoch_times))
        ax.set_ylabel("Amplitude (A.U)")
        ax.set_title(title)
        ax.legend()
        ax.grid(True)
        return ax

    def process(self) -> "ECG":
        self.cleaned_signal = filter_signal(self.data, fs=self.fs, lowcut=1, highcut=10)

        self._detect_r_peaks(buffer_size_sec=0.25, threshold_factor=0.7)
        self._adjust_r_peaks()
        self._count_bpm()

        self.masks = detect_flat_signal(
            self.data,
            fs=self.fs,
            threshold=1e-3,
            min_duration=1.0
        )

        self.masks = np.logical_and(self.masks, self.bpm > 50)

        self.info = "ECG data after post-processing"
        self.quality = {
            "masks_based": calculate_quality(self.masks),
        }
        return self

    def plot(self, title: str = "ECG Signal") -> plt.Figure:
        epochs, epoch_times = self._create_r_peak_epochs(
            self.r_peaks, pre_time=0.3, post_time=0.5
        )
        corrected_epochs, corrected_epoch_times = self._create_r_peak_epochs(
            self.corrected_r_peaks, pre_time=0.3, post_time=0.5
        )

        fig = plt.figure(figsize=(12, 12))
        gs = plt.GridSpec(3, 2, figure=fig)

        ax1 = fig.add_subplot(gs[0, :])
        ax2 = fig.add_subplot(gs[1, :])
        ax3 = fig.add_subplot(gs[2, 0])
        ax4 = fig.add_subplot(gs[2, 1])

        plot_signal_segment(
            self.data, self.cleaned_signal, self.time, self.r_peaks, self.fs, ax=ax1
        )
        self._plot_bpm(ax=ax2)

        self._plot_epochs(
            epochs, epoch_times, title="ECG Signal Epochs Around R Peaks", ax=ax3
        )

        self._plot_epochs(
            corrected_epochs,
            corrected_epoch_times,
            title="Corrected ECG Signal Epochs Around R Peaks",
            ax=ax4,
        )

        fig.suptitle(title)
        plt.tight_layout()

        return fig

    def save(self, saving_path: str) -> None:
        peaks_to_length = np.zeros_like(self.cleaned_signal)
        for peak in self.corrected_r_peaks:
            peaks_to_length[peak] = 1
        saving_dict = {
            "time": self.time,
            "features": np.stack(
                [self.cleaned_signal, self.bpm, peaks_to_length], axis=0
            ),
            "labels": ["cleaned_signal", "bpm", "peaks_to_length"],
            "quality": {"masks_based": calculate_quality(self.masks)},
            "masks": self.masks,
            "info": self.info,
            "process": [
                "Mask Flat Signal",
                "Filter Signal",
                "Detect R-Peaks",
                "Adjust R-Peaks",
                "Count BPM",
                "Create Epochs",
                "Plot Epochs",
            ],
            "ttl": self.ttl,
        }

        with open(saving_path, "wb") as f:
            pickle.dump(saving_dict, f)


# %% TEST EYETRACKING
def test_eyetracking():
    architecture = arch.BidsArchitecture(root="/Users/samuel/Desktop/PHYSIO_BIDS")

    random_subject = np.random.choice(architecture.subjects)
    architecture.select(subject=random_subject, acquisition="eyelink", inplace=True)
    random_session = np.random.choice(architecture.sessions)
    architecture.select(session=random_session, inplace=True)
    random_task = np.random.choice(architecture.tasks)
    architecture.select(task=random_task, inplace=True)
    files_eyetracking = architecture.select(acquisition="eyelink", extension=".gz")
    filename_eyetracking_samples = [
        str(file)
        for file in files_eyetracking.database["filename"].to_list()
        if "samples" in str(file)
    ][0]
    data = read_eyetracking(filename_eyetracking_samples)
    data["eyetracking"].process().plot()


# %% TEST
def get_random_biopac_file(root: str = "/Users/samuel/Desktop/PHYSIO_BIDS"):
    architecture = arch.BidsArchitecture(
        root=root, acquisition="biopac"
    )
    random_subject = np.random.choice(architecture.subjects)
    architecture.select(subject=random_subject, inplace=True)
    random_session = np.random.choice(architecture.sessions)
    architecture.select(session=random_session, inplace=True)
    random_task = np.random.choice(architecture.tasks)
    architecture.select(task=random_task, inplace=True)
    files_biopac = architecture.select(extension=".gz")
    filename_biopac = [
        str(file)
        for file in files_biopac.database["filename"].to_list()
        if "samples" in str(file)
    ][0]
    return read_biopac(filename_biopac)


def test_ecg(data: dict):
    data["ecg"].process().plot()


def test_eda(data: dict):
    data["eda"].process().plot()


def test_ppg(data: dict):
    data["ppg"].process().plot()


def test_rsp(data: dict):
    data["rsp"].process().plot()


# %%
if __name__ == "__main__":
    data = get_random_biopac_file()
    test_rsp(data)
    test_ecg(data)
    test_eyetracking()
    test_ppg(data)
# %%
# TODO
# [X] Transform TTL for eyetracking
# [X] Quality measurement (with masks)
# [X] Rename "data" into "raw_signal" to be consistent with "cleaned_signal"
# [ ] Think about creating a ECGepochs class that will contain the epochs and the times and we can plot
