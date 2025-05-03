""" First version of ECG detection and cleaning prototyping."""
#%%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.utils import *
import numpy as np
from typing import Optional
import scipy
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import kurtosis, entropy
from scipy.signal import welch
import os
from ecgdetectors import Detectors
from pathlib import Path
from scipy import signal
from viz import plotting
import neurokit2 as nk
import bids_explorer.architecture.architecture as arch
import scipy.stats
import re

def calculate_global_std(signal):
    """Calculate the global mean and standard deviation of a signal.
    
    Args:
        signal: The input signal
        
    Returns:
        tuple: (mean, std)
    """
    mean = np.mean(signal)
    std = np.std(signal)
    return mean, std

filename = "/Users/samuel/Desktop/PHYSIO_BIDS/sub-01/ses-04Advice/func/sub-01_ses-04Advice_task-ActiveHighVid_run-1_acq-biopac_recording-samples_physio.tsv.gz"
df = pd.read_csv(filename, sep = "\t")

print("DataFrame columns:", df.columns.tolist())
print("DataFrame shape:", df.shape)
print("First few rows:")
print(df.head())

time = df.iloc[:, 0].to_numpy()
print(df.iloc[:, 4].isna().sum())
ecg_signal = df.iloc[:, 4].to_numpy()

fs = 1000
ecg_signal = filter_signal(ecg_signal, fs, lowcut=1, highcut=4)

global_mean, global_std = calculate_global_std(ecg_signal)
print(f"\nGlobal ECG signal statistics: Mean = {global_mean:.4f}, Std = {global_std:.4f}")

def R_correction(signal, peaks):
    num_peak=peaks.shape[0]
    peaks_corrected_list=list()
    for index in range(num_peak):
        i=peaks[index]
        cnt=i
        if cnt-1<0:
            break
        if signal[cnt]<signal[cnt-1]:
            while signal[cnt]<signal[cnt-1]:
                cnt-=1
                if cnt<0:
                    break
        elif signal[cnt]<signal[cnt+1]:
            while signal[cnt]<signal[cnt+1]:
                cnt+=1
                if cnt<0:
                    break
        peaks_corrected_list.append(cnt)
    peaks_corrected=np.asarray(peaks_corrected_list)            
    return peaks_corrected 
# %%

os.makedirs("./ecg_detector_plots", exist_ok=True)
detectors = Detectors(fs)
ecgdetector_methods = [
    'christov_detector',
    'engzee_detector', 
    'hamilton_detector',
    'matched_filter_detector',
    'pan_tompkins_detector', 
    'swt_detector', 
    'two_average_detector', 
    'wqrs_detector'
]

neurokit_methods = [
    'elgendi2010',
    'neurokit',
    # 'neurokit', 'biosppy', 'vg', 'templateconvolution'
]

all_peaks = {}
pdf_path = "./ecg_detector_plots/ecg_detector_comparison_5Hz.pdf"
with PdfPages(pdf_path) as pdf:
    
    print("\nProcessing ecgdetectors methods:")
    for method_name in ecgdetector_methods:
        print(f"Processing {method_name}...")
        
        detector_method = getattr(detectors, method_name)
        
        try:
            r_peaks = detector_method(ecg_signal)
            r_peaks_corrected = R_correction(ecg_signal, np.array(r_peaks))
            all_peaks[f"ecgdetectors_{method_name}"] = r_peaks_corrected
            plt.figure(figsize=(15, 6))
            start_sample = 0
            duration = 20 * fs
            end_sample = start_sample + duration
            signal_segment = ecg_signal[start_sample:end_sample]
            plt.plot(time[start_sample:end_sample], signal_segment, 'k', label='ECG Signal')
            plt.axhline(y=global_mean, color='blue', linestyle='--', alpha=0.5, label='Global Mean')
            plt.axhline(y=global_mean + global_std, color='blue', linestyle='-', alpha=0.3, label='+1 SD')
            plt.axhline(y=global_mean - global_std, color='blue', linestyle='-', alpha=0.3, label='-1 SD')
            plt.fill_between(time[start_sample:end_sample], 
                             np.ones_like(signal_segment) * (global_mean - global_std), 
                             np.ones_like(signal_segment) * (global_mean + global_std), 
                             color='blue', alpha=0.1)
            
            plot_peaks = r_peaks_corrected[(r_peaks_corrected >= start_sample) & (r_peaks_corrected < end_sample)]
            plt.plot(time[plot_peaks], ecg_signal[plot_peaks], 'ro', markersize=8, label='R Peaks', alpha=0.5)
            plt.title(f'R Peak Detection using ecgdetectors.{method_name.replace("_", " ").title()}', fontsize=14)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            pdf.savefig()
            plt.savefig(f"./ecg_detector_plots/ecgdetectors_{method_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            print(f"Error with {method_name}: {str(e)}")
            
    print("\nProcessing neurokit2 methods:")
    for method_name in neurokit_methods:
        print(f"Processing neurokit2 {method_name}...")
        
        try:
            signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs, method=method_name)
            r_peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]
            
            if len(r_peaks) == 0:
                print(f"No peaks detected with neurokit2 {method_name}.")
                continue
            r_peaks_corrected = R_correction(ecg_signal, np.array(r_peaks))
            all_peaks[f"neurokit2_{method_name}"] = r_peaks_corrected
            plt.figure(figsize=(15, 6))
            start_sample = 0
            duration = 20 * fs
            end_sample = start_sample + duration
            signal_segment = ecg_signal[start_sample:end_sample]
            plt.plot(time[start_sample:end_sample], signal_segment, 'k', label='ECG Signal')
            plt.axhline(y=global_mean, color='blue', linestyle='--', alpha=0.5, label='Global Mean')
            plt.axhline(y=global_mean + global_std, color='blue', linestyle='-', alpha=0.3, label='+1 SD')
            plt.axhline(y=global_mean - global_std, color='blue', linestyle='-', alpha=0.3, label='-1 SD')
            plt.fill_between(time[start_sample:end_sample], 
                                np.ones_like(signal_segment) * (global_mean - global_std), 
                                np.ones_like(signal_segment) * (global_mean + global_std), 
                                color='blue', alpha=0.1)
            
            plot_peaks = r_peaks_corrected[(r_peaks_corrected >= start_sample) & (r_peaks_corrected < end_sample)]
            plt.plot(time[plot_peaks], ecg_signal[plot_peaks], 'ro', markersize=8, label='R Peaks', alpha=0.5)
            
            plt.title(f'R Peak Detection using neurokit2.{method_name}', fontsize=14)
            plt.xlabel('Time (s)', fontsize=12)
            plt.ylabel('Amplitude', fontsize=12)
            plt.legend(loc='upper right')
            plt.grid(True, alpha=0.3)
            pdf.savefig()
            plt.savefig(f"./ecg_detector_plots/neurokit2_{method_name}.png", dpi=300, bbox_inches='tight')
            plt.close()
                
        except Exception as e:
            print(f"Error with neurokit2 {method_name}: {str(e)}")
            
print(f"All plots saved to {pdf_path} and individual PNGs saved to ecg_detector_plots/ directory")

# %%
summary_data = []

for method_name in ecgdetector_methods:
    try:
        detector_method = getattr(detectors, method_name)
        r_peaks = detector_method(ecg_signal)
        r_peaks_corrected = R_correction(ecg_signal, np.array(r_peaks))
        num_peaks = len(r_peaks_corrected)

        if num_peaks > 1:
            rr_intervals = np.diff(r_peaks_corrected) / fs 
            mean_hr = 60 / np.mean(rr_intervals)
            hrv_sdnn = np.std(rr_intervals) * 1000  # in ms

        else:
            mean_hr = 0
            hrv_sdnn = 0
            
        summary_data.append({
            'Method': f"ecgdetectors_{method_name}",
            'Peaks Detected': num_peaks,
            'Mean HR (BPM)': mean_hr,
            'HRV SDNN (ms)': hrv_sdnn
        })
    except Exception as e:
        summary_data.append({
            'Method': f"ecgdetectors_{method_name}",
            'Peaks Detected': 'Error',
            'Mean HR (BPM)': 'Error',
            'HRV SDNN (ms)': 'Error'
        })

for method_name in neurokit_methods:
    try:
        signals, info = nk.ecg_process(ecg_signal, sampling_rate=fs, method=method_name)
        r_peaks = np.where(signals["ECG_R_Peaks"] == 1)[0]

        if len(r_peaks) == 0:
            summary_data.append({
                'Method': f"neurokit2_{method_name}",
                'Peaks Detected': 0,
                'Mean HR (BPM)': 0,
                'HRV SDNN (ms)': 0
            })
            continue
            
        r_peaks_corrected = R_correction(ecg_signal, np.array(r_peaks))
        num_peaks = len(r_peaks_corrected)

        if num_peaks > 1:
            rr_intervals = np.diff(r_peaks_corrected) / fs 
            mean_hr = 60 / np.mean(rr_intervals)
            hrv_sdnn = np.std(rr_intervals) * 1000  # in ms

        else:
            mean_hr = 0
            hrv_sdnn = 0
            
        summary_data.append({
            'Method': f"neurokit2_{method_name}",
            'Peaks Detected': num_peaks,
            'Mean HR (BPM)': mean_hr,
            'HRV SDNN (ms)': hrv_sdnn
        })
    except Exception as e:
        summary_data.append({
            'Method': f"neurokit2_{method_name}",
            'Peaks Detected': 'Error',
            'Mean HR (BPM)': 'Error',
            'HRV SDNN (ms)': 'Error'
        })

summary_df = pd.DataFrame(summary_data)
print("\nSummary of all methods:")
print(summary_df)
summary_df.to_csv("./ecg_detector_summary.csv", index=False)
print("Summary saved to ecg_detector_summary.csv")
