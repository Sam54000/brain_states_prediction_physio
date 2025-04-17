#%%
import pandas as pd
import numpy as np
from typing import Optional
import scipy
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use("Agg")
from scipy.stats import kurtosis, entropy
from scipy.signal import welch
import os
from ecgdetectors import Detectors
from pathlib import Path
import neurokit2 as nk

def extract_eyelink_events(data: pd.DataFrame) -> pd.DataFrame:
    """Extract the events of the experiment for the eye tracking data.
    
    Args:
        data (pd.DataFrame): The dataframe should contain coming from the
            file that should contain 'acq-eyelink_recording-events'.

    Returns:
        pd.DataFrame
    """
    message_df = data.loc[data[0] == "MSG"]
    message_df.reset_index(inplace = True)
    scanner_start_idx = message_df[message_df[2] == "SCANNER_START"].index[0]
    cropped_df = message_df.iloc[scanner_start_idx:]
    cropped_df.rename(columns={1:"time",2: "event_name"}, inplace=True)
    cropped_df.reset_index(inplace = True)
    cropped_df["time"] = cropped_df["time"].astype(int)/1000
    return cropped_df[["time","event_name"]]

def filter_signal(
    signal: np.ndarray,
    fs: float,
    order: int = 5,
    lowcut: Optional[float] = None,
    highcut: Optional[float] = None
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
            "lowcut and highcut must be None or non zero float got instead:\n"\
            f"\tlowcut type: {type(lowcut)}\n"\
            f"\thighcut type: {type(highcut)}"
        )
    sos = scipy.signal.butter(order, output='sos', **kwargs)
    filtered = scipy.signal.sosfiltfilt(sos, signal)
    
    return filtered

def catch_error(x):
    try:
        if isinstance(x, str):
            return float(x.strip())
        else:
            return x
    except:
        return np.nan

def extract_and_process_eyetracking(data: pd.DataFrame) -> dict:
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
    for col in [1,2,3]:
        data[col] = data[col].apply(catch_error)
    first_derivative = np.diff(data[3].values.astype(float), prepend = 0)
    second_derivative = np.diff(data[3].values.astype(float), n = 2, prepend = [0,0])
    mask = np.ones_like(data[0].values, dtype = bool)
    mask[np.abs(first_derivative) > 10] = False
    mask[data[3].values.astype(float) < 100] = False

    return {
        "time": data[0].values.astype(int)/1000,
        "feature": np.stack([
            data[1].values.astype(float),
            data[2].values.astype(float),
            data[3].values.astype(float),
            first_derivative,
            second_derivative], 
           axis = 0),
        "labels": [
            "X",
            "Y",
            "pupil_dilation", 
            "pupil_first_derivative",
            "pupil_second_derivative"],
        "mask": mask,
        "feature_info": "Eyetracking data"
    }

def extract_and_process_eda(
    data: pd.DataFrame,
    plot: bool = False,
    saving_filename: Optional[os.PathLike] = None
    ) -> dict:
    """Extract and process electrodermal activity.
    
    The process involve filtering the data with a low-pass filter at 1Hz as 
    EDA is a very slow activity.

    Args:
        data (pandas.DataFrame): The biopac data

    Returns:
        dict
    """
    signals, info = nk.eda_process(data[1].values, sampling_rate=1000)
    temp_dictionary = signals.to_dict(orient="list")
    feature = np.stack(list(temp_dictionary.values()), axis=0)
    labels = list(temp_dictionary.keys())
    if plot:
        nk.eda_plot(signals, info)
        fig = plt.gcf() 
        fig.set_size_inches(10, 12, forward=True)
        if saving_filename is not None:
            fig.savefig(saving_filename)
    
    return fig, {
        "time": data[0].values,
        "feature": feature,
        "labels": labels,
        "mask": np.ones_like(data[0].values, dtype=bool),
        "feature_info": "Respiratory signal",
    }

def detect_motion_artifacts(
    signal: np.ndarray,
    fs: float,
    window_size: float = 5,
    step_size: float = 0.25,
    kurtosis_threshold: float = 3,
    entropy_threshold: float = 1
    ) -> np.ndarray:
    """
    Detects motion artifacts in a PPG signal using kurtosis and Shannon entropy.
    
    Args:
        signal (np.ndarray): 1D numpy array of the PPG signal.
        fs (float): Sampling frequency of the signal in Hz.
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
    signal_length = len(signal)
    mask = np.ones(signal_length)

    window_samples = int(window_size * fs)
    step_samples = int(step_size * fs)

    for start in range(0, signal_length - window_samples + 1, step_samples):
        end = start + window_samples
        segment = signal[start:end]

        seg_kurtosis = kurtosis(segment)
        f, Pxx = welch(segment, fs=fs)
        Pxx_norm = Pxx / np.sum(Pxx)
        seg_entropy = entropy(Pxx_norm)
        if seg_kurtosis > kurtosis_threshold or seg_entropy < entropy_threshold:
            mask[start:end] = 0

    return mask

def extract_and_process_ppg(
    data: pd.DataFrame,
    plot: bool = False,
    saving_filename: Optional[os.PathLike] = None
    ) -> dict:
    """

    Args:
        data (pd.DataFrame): Should be from biopac samples

    Returns:
        dict:
    """
    signal = filter_signal(
        data[2].values,
        fs = 1000,
        lowcut = None,
        highcut = 10
    )
    time = data[0].values
    signals, info= nk.ppg_process(signal, sampling_rate=1000)
    temp_dictionary = signals.to_dict(orient="list")
    feature = np.stack(list(temp_dictionary.values()), axis=0)
    labels = list(temp_dictionary.keys())
    mask = detect_motion_artifacts(
        signal=data[2].values, # For some reason the detector works better on 
                               # non filtered data         
        fs = 1000,
        kurtosis_threshold = 3, 
        entropy_threshold = 0.5
        )
    if plot:
        nk.ppg_plot(signals, info)
        fig = plt.gcf() 
        fig.set_size_inches(12, 10, forward=True)
        if saving_filename is not None:
            fig.savefig(saving_filename)
    return fig, {
        "time": time,
        "feature": feature,
        "labels": labels,
        "mask": mask.astype(bool),
        "feature_info": "Photoplethysmography",
        "quality": {"mean": signals["PPG_Quality"].mean(), "std": signals["PPG_Quality"].std()}
    }

def extract_and_process_rsp(
    data: pd.DataFrame,
    plot: bool = False,
    saving_filename: Optional[os.PathLike] = None
    ) -> dict:
    """Extract and process respiratory signal.
    
    The function uses the neurokit2 library to process the respiratory signal.
    The feature extracted are:
    
    - RSP_Raw: The raw signal.
    - RSP_Clean: The raw signal.
    - RSP_Peaks: The respiratory peaks (exhalation onsets) marked as “1” in a 
        list of zeros.
    - RSP_Troughs: The respiratory troughs (inhalation onsets) marked as “1” 
        in a list of zeros.
    - RSP_Rate: The breathing rate interpolated between inhalation peaks.
    - RSP_Amplitude: The breathing amplitude interpolated between inhalation   
        peaks.
    - RSP_Phase: The breathing phase, marked by “1” for inspiration and “0” 
        for expiration.
    - RSP_Phase_Completion: The breathing phase completion, expressed in
        percentage (from 0 to 1), representing the stage of the current 
        respiratory phase.
    - RSP_RVT: Respiratory volume per time (RVT).

    Args:
        data (pandas.DataFrame): The biopac data

    Returns:
        dict
    """
    sig = filter_signal(
        data[3].values,
        fs = 1000,
        lowcut = None,
        highcut = 4
    )
    signals, info = nk.rsp_process(sig, sampling_rate=1000)
    temp_dictionary = signals.to_dict(orient="list")
    feature = np.stack(list(temp_dictionary.values()), axis=0)
    labels = list(temp_dictionary.keys())
    if plot:
        nk.rsp_plot(signals, info)
        fig = plt.gcf() 
        fig.set_size_inches(10, 12, forward=True)
        if saving_filename is not None:
            fig.savefig(saving_filename)
    return fig, {
        "time": data[0].values,
        "feature": feature,
        "labels": labels,
        "mask": np.ones_like(data[0].values, dtype=bool),
        "feature_info": "Respiratory signal",
        "quality": {"mean": signals["RSP_Rate"].mean(), "std": signals["RSP_Rate"].std()}
    }

def extract_and_process_ecg(
    data: pd.DataFrame,
    plot: bool = False,
    saving_filename: Optional[os.PathLike] = None
    ) -> dict:
    """Extract and process electrocardiogram signal.
    
    The function uses the neurokit2 library to process the ECG signal.
    The feature extracted are:
    
    - ECG_Raw: The raw signal.
    - ECG_Clean: The cleaned signal.
    - ECG_Rate: Heart rate interpolated between R-peaks.
    - ECG_Quality: The quality of the cleaned signal.
    - ECG_R_Peaks: The R-peaks marked as “1” in a list of zeros.
    - ECG_R_Onsets: The R-onsets marked as “1” in a list of zeros.
    - ECG_R_Offsets: The R-offsets marked as “1” in a list of zeros.
    - ECG_P_Peaks: The P-peaks marked as “1” in a list of zeros.
    - ECG_P_Onsets: The P-onsets marked as “1” in a list of zeros.
    - ECG_P_Offsets: The P-offsets marked as “1” in a list of zeros.
    - ECG_Q_Peaks: The Q-peaks marked as “1” in a list of zeros.
    - ECG_S_Peaks: The S-peaks marked as “1” in a list of zeros.
    - ECG_T_Peaks: The T-peaks marked as “1” in a list of zeros.
    - ECG_T_Onsets: The T-onsets marked as “1” in a list of zeros.
    - ECG_T_Offsets: The T-offsets marked as “1” in a list of zeros.
    - ECG_Phase_Atrial: Cardiac phase, marked by “1” for systole and “0” for
        diastole.
    - ECG_Phase_Completion_Atrial: Cardiac phase (atrial) completion,
        expressed in percentage (from 0 to 1), representing the stage of 
            the current cardiac phase.
    - ECG_Phase_Completion_Ventricular: Cardiac phase (ventricular) completion,
        expressed in percentage (from 0 to 1), representing the stage of the
        current cardiac phase.
    """
    sig = filter_signal(
        data[4].values,
        fs = 1000,
        lowcut = 0.5,
        highcut = 10,
        order = 10
    )
    signals, info = nk.ecg_process(sig, sampling_rate=1000)
    temp_dictionary = signals.to_dict(orient="list")
    feature = np.stack(list(temp_dictionary.values()), axis=0)
    labels = list(temp_dictionary.keys())
    if plot:
        nk.ecg_plot(signals, info)
        fig = plt.gcf() 
        fig.set_size_inches(12, 10, forward=True)
        if saving_filename is not None:
            fig.savefig(saving_filename)
    return fig, {
        "time": data[0].values,
        "feature": feature,
        "labels": labels,
        "mask": np.ones_like(data[0].values, dtype=bool),
        "feature_info": f"Electrocardiogram",
        "quality": {"mean": signals["ECG_Quality"].mean(), "std": signals["ECG_Quality"].std()}
    }

def crop_eyetracking(
    eyetracking_dict: dict,
    eyetracking_events: pd.DataFrame
) -> dict:
    """Crop the eyetracking data to get only data when the scanner is on.

    
    Args:
        eyetracking_dict (dict): The formated eyetracking data.
        eyetracking_events (pd.DataFrame): The events extracted. This should
            be the output from the function `extract_eyeling_events`.

    Returns:
        dict: The croped data
    """
    start_event = eyetracking_events["time"].values[0]
    end_event = eyetracking_events["time"].values[-2]
    idx_start = np.argmin(abs(eyetracking_dict["time"] - start_event))
    idx_stop = np.argmin(abs(eyetracking_dict["time"] - end_event))
    eyetracking_dict["time"] = eyetracking_dict["time"][idx_start:idx_stop]
    eyetracking_dict["feature"] = eyetracking_dict["feature"][:,idx_start:idx_stop]
    eyetracking_dict["mask"] = eyetracking_dict["mask"][idx_start:idx_stop]
    eyetracking_dict["time"] = eyetracking_dict["time"] - eyetracking_dict["time"][0]
    eyetracking_events["time"] = eyetracking_events["time"] - eyetracking_events["time"][0]

    return eyetracking_dict, eyetracking_events.to_dict(orient = "list")