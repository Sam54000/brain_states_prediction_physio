import pywt
import pandas as pd
import cv2
import numpy as np
from typing import Optional
import scipy
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt
import os
import neurokit2 as nk


def filter_signal(
    signal: np.ndarray,
    fs: float,
    order: int = 5,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None,
) -> np.ndarray:
    """
    Apply a bandpass filter to ECG signal data using second-order sections.

    Args:
        signal (np.ndarray): Raw signal as 1-D numpy array
        fs (float): Sampling frequency in Hz
        lowcut (float, optional): Low cutoff frequency in Hz. Defaults to 0.5 Hz.
        highcut (float, optional): High cutoff frequency in Hz. Defaults to 40.0 Hz.
        order (int, optional): Filter order. Defaults to 5.

    Returns:
        np.ndarray: Filtered ECG signal
    """
    nyquist = 0.5 * fs

    if lowcut is None:
        kwargs = {"btype": "low", "Wn": highcut / nyquist}

    elif highcut is None:
        kwargs = {"btype": "high", "Wn": lowcut / nyquist}

    elif lowcut is not None and lowcut > 0 and highcut is not None:
        kwargs = {"btype": "band", "Wn": [lowcut / nyquist, highcut / nyquist]}

    else:
        raise ValueError(
            "lowcut and highcut must be None or non zero float got instead:\n"
            f"\tlowcut type: {type(lowcut)}\n"
            f"\thighcut type: {type(highcut)}"
        )
    sos = scipy.signal.butter(order, output="sos", **kwargs)
    filtered = scipy.signal.sosfiltfilt(sos, signal)

    return filtered


def catch_error(x):
    try:
        if isinstance(x, str):
            return float(x.strip())
        else:
            return x
    except Exception:
        return np.nan


def detect_high_amplitude_artifacts(
    signal: np.ndarray,
    fs: float,
    window_size: float = 0.2,  # Shorter window to better capture transients
    step_size: float = 0.05,  # Smaller step for better resolution
    amplitude_threshold_multiplier: float = 2.0,
) -> np.ndarray:
    """
    Detects transient high frequency artifacts in physiological signals recorded in MRI.

    This function works by:
    1. Applying a high-pass filter to isolate high frequency components
    2. Computing the local amplitude (envelope) of the high frequency components
    3. Using a sliding window to detect segments where the amplitude exceeds a local threshold
    4. Only marking segments that exceed a minimum duration as artifacts

    Args:
        signal (np.ndarray): 1D numpy array of the physiological signal
        fs (float): Sampling frequency of the signal in Hz
        window_size (float, optional): Size of the sliding window in seconds. Defaults to 0.2.
        step_size (float, optional): Step size for the sliding window in seconds. Defaults to 0.05.
        highpass_cutoff (float, optional): Cutoff frequency for high-pass filter in Hz. Defaults to 10.0.
        amplitude_threshold_multiplier (float, optional): Multiplier for standard deviation to set
            the amplitude threshold. Defaults to 2.0.
        buffer (float, optional): Buffer in seconds to extend the mask. Defaults to 0.1.
        min_duration (float, optional): Minimum duration in seconds for an artifact to be detected.
            Defaults to 0.1.
    Returns:
        numpy.ndarray: 1D numpy array of the same length as signal, with True indicating
        clean segments and False indicating artifacts.
    """

    signal = filter_signal(signal=signal, fs=fs, lowcut=1, highcut=None)

    signal -= np.mean(signal)

    signal_length = len(signal)
    mask = np.ones(signal_length, dtype=bool)

    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    mask = np.ones(signal_length, dtype=bool)

    for start in range(0, signal_length - window_samples + 1, step_samples):
        end = start + window_samples
        segment = signal[start:end]

        Q1 = np.percentile(segment, 25)
        Q3 = np.percentile(segment, 75)
        IQR = Q3 - Q1

        high_bound = Q3 + amplitude_threshold_multiplier * IQR
        low_bound = Q1 - amplitude_threshold_multiplier * IQR

        mask[start:end] = (segment < high_bound) | (segment > low_bound)

    mask = scipy.signal.convolve(mask, np.ones(100), mode="same") > 0

    return mask


def long_flat_segment_detection(mask, fs, min_duration=3.0):
    """
    Detects long segments of consecutive False values in a mask.
    Any segment with consecutive False values longer than min_duration will remain False,
    while shorter segments will be converted to True.

    Args:
        mask (np.ndarray): Boolean mask (True for valid data, False for invalid data)
        fs (float): Sampling frequency in Hz
        min_duration (float): Minimum duration in seconds for a segment to remain masked as False

    Returns:
        np.ndarray: Modified boolean mask with short segments of False values converted to True
    """

    result_mask = mask.copy()

    min_samples = int(min_duration * fs)

    if np.all(mask):
        return result_mask

    padded_mask = np.concatenate(([True], mask, [True]))

    transitions = np.diff(padded_mask.astype(int))

    starts = np.where(transitions == -1)[0]

    ends = np.where(transitions == 1)[0]

    assert len(starts) == len(ends), "Unequal number of segment starts and ends"

    for i in range(len(starts)):
        start = starts[i]
        end = ends[i]
        segment_length = end - start
        if segment_length <= min_samples:
            result_mask[start:end] = True

    return result_mask


def detect_high_gradient(
    signal: np.ndarray, fs: float, threshold: float = 10, buffer: float = 0.5
) -> np.ndarray:
    """
    Detects segments of high gradient in a signal.
    """
    derivative = np.diff(signal, prepend=np.random.choice(signal))
    mask = np.abs(derivative) < threshold
    buffer_samples = int(buffer * fs)
    for i in np.where(~mask)[0]:
        if mask[i]:
            mask[i - buffer_samples : i + buffer_samples] = False
    return mask


def zscore_mask(signal: np.ndarray, threshold: float = 3) -> np.ndarray:
    """
    Detects segments of high z-score in a signal.
    """
    zscore = scipy.stats.zscore(signal)
    mask = zscore < threshold
    return mask


def detect_flat_signal(
    signal: np.ndarray,
    fs: float,
    threshold: float = 1e-2,
    min_duration: float = 3.0,
    buffer: float = 0.2,
) -> np.ndarray:
    """
    Detects segments of flat signal by counting consecutive zeros in the derivative.

    Args:
        signal (np.ndarray): 1D numpy array of the physiological signal
        fs (float): Sampling frequency of the signal in Hz
        flat_duration_threshold (float, optional): Minimum duration in seconds for
            a flat segment to be marked. Defaults to 1.0.

    Returns:
        numpy.ndarray: 1D numpy array of the same length as signal, with True indicating
        normal segments and False indicating flat segments.
    """
    derivative = np.abs(np.diff(signal, prepend=signal[0]))
    is_flat = derivative < threshold

    changes = np.diff(is_flat.astype(int), prepend=0)
    starts = np.flatnonzero(changes == 1)
    ends = np.flatnonzero(changes == -1)

    if is_flat[-1]:
        ends = np.append(ends, len(is_flat))

    mask = np.ones(len(signal), dtype=bool)
    min_samples = int(min_duration * fs)
    for start, end in zip(starts, ends):
        if end - start >= min_samples:
            mask[start:end] = False
    mask = scipy.signal.convolve(mask, np.ones(int(buffer * fs)), mode="same") == 1

    return mask


def correct_shapes(obj, attributes: list[str]) -> list[np.ndarray]:
    """
    Correct the shapes of the data to be the same as the signal.
    """
    sizes = []
    for attribute in attributes:
        sizes.append(getattr(obj, attribute).shape[0])
    min_size = np.min(sizes)
    for attribute in attributes:
        setattr(obj, attribute, getattr(obj, attribute)[:min_size])
    return obj


def test_artifact_detectors(save_path: Optional[os.PathLike] = None):
    """
    Test the artifact detection functions with synthetic data containing known artifacts.

    This function:
    1. Generates synthetic RSP and ECG-like signals with controlled artifacts using neurokit2
    2. Applies our artifact detection algorithms
    3. Visualizes the results with the detected artifacts highlighted

    Args:
        save_path (os.PathLike, optional): Path to save the visualization plots.
            If None, plots will be displayed but not saved.

    Returns:
        tuple: Two dictionaries containing the test results for RSP and ECG.
    """
    fs = 1000
    duration = 10
    t = np.arange(0, duration, 1 / fs)
    len(t)
    np.random.seed(42)
    ecg_signal = nk.ecg_simulate(
        duration=duration,
        sampling_rate=fs,
        heart_rate=80,
        random_state=42,
        noise=0.5,
    )

    rsp_signal = nk.rsp_simulate(
        duration=duration,
        sampling_rate=fs,
        respiratory_rate=15,
        noise=0.5,
    )

    hf_start_rsp = int(1 * fs)
    hf_end_rsp = int(2 * fs)
    hf_artifact_rsp = 0.5 * np.sin(2 * np.pi * 20 * t[hf_start_rsp:hf_end_rsp])
    rsp_signal[hf_start_rsp:hf_end_rsp] += hf_artifact_rsp

    flat_start_rsp = int(5 * fs)
    flat_end_rsp = int(6.5 * fs)
    rsp_signal[flat_start_rsp:flat_end_rsp] = rsp_signal[flat_start_rsp]

    hf_start_ecg = int(3 * fs)
    hf_end_ecg = int(3.5 * fs)
    hf_artifact_ecg = 0.5 * np.sin(2 * np.pi * 50 * t[hf_start_ecg:hf_end_ecg])
    ecg_signal[hf_start_ecg:hf_end_ecg] += hf_artifact_ecg

    flat_start_ecg = int(7 * fs)
    flat_end_ecg = int(8 * fs)
    ecg_signal[flat_start_ecg:flat_end_ecg] = ecg_signal[flat_start_ecg]

    time = np.arange(0, len(rsp_signal)) / fs
    pd.DataFrame({0: time, 3: rsp_signal})

    pd.DataFrame({0: time, 4: ecg_signal})

    hf_rsp_mask = detect_high_amplitude_artifacts(
        signal=rsp_signal,
        fs=fs,
        window_size=1.0,
        step_size=0.25,
        amplitude_threshold_multiplier=1.0,
    )

    flat_rsp_mask = detect_flat_signal(
        signal=rsp_signal, fs=fs, flat_duration_threshold=1.0
    )

    combined_rsp_mask = hf_rsp_mask & flat_rsp_mask

    hf_ecg_mask = detect_high_amplitude_artifacts(
        signal=ecg_signal,
        fs=fs,
        window_size=0.5,
        step_size=0.1,
        amplitude_threshold_multiplier=2.5,
    )

    flat_ecg_mask = detect_flat_signal(
        signal=ecg_signal, fs=fs, flat_duration_threshold=0.5
    )

    combined_ecg_mask = hf_ecg_mask & flat_ecg_mask

    fig, axes = plt.subplots(4, 1, figsize=(15, 10), sharex=True)

    axes[0].plot(t, rsp_signal)
    axes[0].set_title("Synthetic RSP Signal with Artifacts")
    axes[0].set_ylabel("Amplitude")

    axes[1].plot(
        t, np.logical_not(hf_rsp_mask) * 1.0, "r-", label="High-freq artifacts"
    )
    axes[1].plot(t, np.logical_not(flat_rsp_mask) * 0.8, "g-", label="Flat segments")
    axes[1].plot(
        t, np.logical_not(combined_rsp_mask) * 0.6, "b-", label="Combined mask"
    )
    axes[1].set_title("RSP Artifact Detection")
    axes[1].set_ylabel("Artifact Present")
    axes[1].legend()
    axes[1].set_ylim(-0.1, 1.1)

    axes[2].plot(t, ecg_signal)
    axes[2].set_title("Synthetic ECG Signal with Artifacts")
    axes[2].set_ylabel("Amplitude")

    axes[3].plot(
        t, np.logical_not(hf_ecg_mask) * 1.0, "r-", label="High-freq artifacts"
    )
    axes[3].plot(t, np.logical_not(flat_ecg_mask) * 0.8, "g-", label="Flat segments")
    axes[3].plot(
        t, np.logical_not(combined_ecg_mask) * 0.6, "b-", label="Combined mask"
    )
    axes[3].set_title("ECG Artifact Detection")
    axes[3].set_xlabel("Time (s)")
    axes[3].set_ylabel("Artifact Present")
    axes[3].legend()
    axes[3].set_ylim(-0.1, 1.1)

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)

    fig2, axes2 = plt.subplots(2, 1, figsize=(15, 8), sharex=True)

    axes2[0].plot(t, rsp_signal, "b-", label="Artifacted RSP")
    axes2[0].axvspan(1, 2, color="r", alpha=0.2, label="High-freq artifact")
    axes2[0].axvspan(5, 6.5, color="y", alpha=0.2, label="Flat segment")
    axes2[0].set_title("Comparison of Clean vs Artifacted RSP Signal")
    axes2[0].set_ylabel("Amplitude")
    axes2[0].legend()

    axes2[1].plot(t, ecg_signal, "b-", label="Artifacted ECG")
    axes2[1].axvspan(3, 3.5, color="r", alpha=0.2, label="High-freq artifact")
    axes2[1].axvspan(7, 8, color="y", alpha=0.2, label="Flat segment")
    axes2[1].set_title("Comparison of Clean vs Artifacted ECG Signal")
    axes2[1].set_xlabel("Time (s)")
    axes2[1].set_ylabel("Amplitude")
    axes2[1].legend()

    plt.tight_layout()

    if save_path is not None:
        comparison_path = str(save_path).replace(".png", "_comparison.png")
        plt.savefig(comparison_path)

    rsp_results = {
        "signal": rsp_signal,
        "time": t,
        "high_freq_mask": hf_rsp_mask,
        "flat_mask": flat_rsp_mask,
        "combined_mask": combined_rsp_mask,
        "known_artifacts": {"high_freq": (1.0, 2.0), "flat": (5.0, 6.5)},
    }

    ecg_results = {
        "signal": ecg_signal,
        "time": t,
        "high_freq_mask": hf_ecg_mask,
        "flat_mask": flat_ecg_mask,
        "combined_mask": combined_ecg_mask,
        "known_artifacts": {"high_freq": (3.0, 3.5), "flat": (7.0, 8.0)},
    }

    return rsp_results, ecg_results


def pchip_interpolate_masked_data(signal, time):
    """
    Interpolate NaN values in a signal using PCHIP interpolation.
    PCHIP (Piecewise Cubic Hermite Interpolating Polynomial) preserves monotonicity
    and is less prone to overshooting than cubic spline.

    Args:
        signal (np.ndarray): The signal with NaN values to interpolate
        time (np.ndarray): The corresponding time vector
        max_gap (float): Maximum gap in seconds to interpolate (not used in this version)

    Returns:
        np.ndarray: The interpolated signal
    """

    interpolated = np.copy(signal)

    valid_indices = np.where(~np.isnan(signal))[0]

    if len(valid_indices) < 2:
        return interpolated

    try:
        from scipy.interpolate import PchipInterpolator

        pchip_func = PchipInterpolator(
            time[valid_indices], signal[valid_indices], extrapolate=False
        )

        interp_values = pchip_func(time)

        valid_interp = ~np.isnan(interp_values)
        interpolated[valid_interp] = interp_values[valid_interp]

    except Exception as e:
        print(f"PCHIP interpolation failed: {e}. Falling back to linear interpolation.")
        interp_func = scipy.interpolate.interp1d(
            time[valid_indices],
            signal[valid_indices],
            kind="linear",
            bounds_error=False,
            fill_value=np.nan,
        )
        interpolated = interp_func(time)

    return interpolated


def moving_average_smoothing(signal, window_size=5):
    """
    Apply a simple moving average filter to smooth a signal.

    Args:
        signal (np.ndarray): The signal to smooth
        window_size (int): The size of the moving average window

    Returns:
        np.ndarray: The smoothed signal
    """

    kernel = np.ones(window_size) / window_size

    smoothed = np.copy(signal)
    valid_mask = ~np.isnan(signal)

    signal_for_conv = np.copy(signal)
    signal_for_conv[~valid_mask] = 0

    smoothed_signal = np.convolve(signal_for_conv, kernel, mode="same")

    valid_mask_float = valid_mask.astype(float)
    mask_conv = np.convolve(valid_mask_float, kernel, mode="same")

    norm_mask = mask_conv > 0
    smoothed[norm_mask] = smoothed_signal[norm_mask] / mask_conv[norm_mask]
    smoothed[~norm_mask] = np.nan

    return smoothed


def interpolate_masked_data(signal, time, max_gap=1.0, method="linear"):
    """
    Interpolate NaN values in a signal.

    Args:
        signal (np.ndarray): The signal with NaN values to interpolate
        time (np.ndarray): The corresponding time vector
        max_gap (float): Maximum gap in seconds to interpolate (not used in this version)
        method (str): Interpolation method ('linear', 'cubic', 'nearest')

    Returns:
        np.ndarray: The interpolated signal
    """

    interpolated = np.copy(signal)

    valid_indices = np.where(~np.isnan(signal))[0]

    if len(valid_indices) < 2:
        return interpolated

    interp_func = scipy.interpolate.interp1d(
        time[valid_indices],
        signal[valid_indices],
        kind=method,
        bounds_error=False,
        fill_value=np.nan,
    )

    interpolated = interp_func(time)

    return interpolated


def create_sliding_zscore_mask(
    signal: np.ndarray,
    fs: float,
    window_size: float = 2.0,
    step_size: float = 0.5,
    z_threshold: float = 2,
    buffer: float = 0.2,
) -> np.ndarray:
    """
    Creates a mask by detecting outliers based on z-scores in a sliding window.

    For each sliding window:
    1. Computes z-scores of values in the window
    2. Marks values with |z-score| > threshold as outliers
    3. Adds buffer around outliers

    Args:
        signal (np.ndarray): The input signal
        fs (float): Sampling frequency in Hz
        window_size (float): Size of sliding window in seconds
        step_size (float): Step size for window sliding in seconds
        z_threshold (float): Z-score threshold for outlier detection
        buffer (float): Buffer time in seconds to mark around outliers

    Returns:
        np.ndarray: Boolean mask where True indicates valid data points
    """
    mask = np.ones_like(signal, dtype=bool)
    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)
    buffer_samples = int(buffer * fs)

    valid_data = ~np.isnan(signal)

    for start in range(0, len(signal) - window_samples + 1, step_samples):
        end = start + window_samples
        window_data = signal[start:end]
        window_valid = valid_data[start:end]

        if np.sum(window_valid) < window_samples * 0.5:
            continue

        valid_values = window_data[window_valid]

        try:
            mean = np.mean(valid_values)
            std = np.std(valid_values)

            if std < 1e-6:
                continue

            z_scores = np.abs((window_data - mean) / std)

            outliers = z_scores > z_threshold
            outlier_indices = np.where(outliers)[0] + start

            for idx in outlier_indices:
                buffer_start = max(0, idx - buffer_samples)
                buffer_end = min(len(mask), idx + buffer_samples + 1)
                mask[buffer_start:buffer_end] = False

        except Exception as e:
            print(f"Error in window {start}-{end}: {e}")
            continue

    return mask


def wavelet_denoise(signal, wavelet="db4", level=6, threshold_type="soft"):
    coeffs = pywt.wavedec(signal, wavelet, mode="per")

    sigma = np.median(np.abs(coeffs[-level])) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(len(signal)))
    coeffs[1:] = [
        pywt.threshold(c, value=uthresh, mode=threshold_type) for c in coeffs[1:]
    ]

    return pywt.waverec(coeffs, wavelet, mode="per")


def smooth_signal(signal, window_length=5000, polyorder=3):
    if window_length >= len(signal):
        window_length = len(signal) - 1 if len(signal) % 2 == 0 else len(signal)
    if window_length % 2 == 0:
        window_length += 1

    return savgol_filter(signal, window_length=window_length, polyorder=polyorder)


def perform_emd(
    signal: np.ndarray,
    fs: float,
    max_imfs: int = 8,
    noise_std: float = 0.1,
    ensemble_size: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Perform Ensemble Empirical Mode Decomposition (EEMD) on an ECG signal.

    EEMD is an improved version of EMD that reduces mode mixing by adding white noise
    to the signal and performing multiple decompositions.

    Args:
        signal (np.ndarray): 1D numpy array of the ECG signal
        fs (float): Sampling frequency in Hz
        max_imfs (int, optional): Maximum number of Intrinsic Mode Functions (IMFs) to extract.
            Defaults to 8.
        noise_std (float, optional): Standard deviation of the white noise to add.
            Defaults to 0.1.
        ensemble_size (int, optional): Number of ensemble members. Defaults to 100.

    Returns:
        tuple[np.ndarray, np.ndarray]:
            - IMFs: 2D numpy array of shape (n_imfs, n_samples) containing the IMFs
            - residual: 1D numpy array containing the residual trend
    """
    try:
        from PyEMD import EEMD
    except ImportError:
        raise ImportError(
            "PyEMD package is required for EMD. Install it using: pip install EMD-signal"
        )

    # Initialize EEMD
    eemd = EEMD(trials=ensemble_size, noise_width=noise_std, max_imf=max_imfs)

    # Perform the decomposition
    imfs = eemd.eemd(signal)

    # The last component is the residual
    residual = imfs[-1]
    imfs = imfs[:-1]

    return imfs, residual


def plot_emd_results(
    signal: np.ndarray,
    imfs: np.ndarray,
    residual: np.ndarray,
    fs: float,
    save_path: Optional[os.PathLike] = None,
) -> None:
    """
    Plot the results of EMD decomposition.

    Args:
        signal (np.ndarray): Original signal
        imfs (np.ndarray): IMFs from EMD
        residual (np.ndarray): Residual trend
        fs (float): Sampling frequency in Hz
        save_path (os.PathLike, optional): Path to save the plot. If None, plot is displayed.
    """
    time = np.arange(len(signal)) / fs

    n_imfs = len(imfs)
    fig, axes = plt.subplots(n_imfs + 2, 1, figsize=(12, 2 * (n_imfs + 2)), sharex=True)

    # Plot original signal
    axes[0].plot(time, signal)
    axes[0].set_title("Original Signal")
    axes[0].set_ylabel("Amplitude")

    # Plot IMFs
    for i, imf in enumerate(imfs):
        axes[i + 1].plot(time, imf)
        axes[i + 1].set_title(f"IMF {i + 1}")
        axes[i + 1].set_ylabel("Amplitude")

    # Plot residual
    axes[-1].plot(time, residual)
    axes[-1].set_title("Residual")
    axes[-1].set_ylabel("Amplitude")
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path)
    else:
        plt.show()

    plt.close()


def extract_video_luminosity(video_path: os.PathLike) -> np.ndarray:
    """
    Extract the luminosity series from a video.

    It extract the average luminositiy of each frame in the video for
    the entire screen.

    Args:
        video_path (os.PathLike): The path to the video file.

    Returns:
        np.ndarray: The luminosity as a time series..
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise IOError("Error opening video file")

    luminosity_series = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        avg_luminosity = np.mean(gray)
        luminosity_series.append(avg_luminosity)

    cap.release()
    luminosity_series = np.array(luminosity_series)
    return luminosity_series


# %%
