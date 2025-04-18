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
def plot_results(
    data_dict: dict,
    index_feature: int,
    title: str,
    time: Optional[np.ndarray] = [None, None],
    saving_filename: Optional[os.PathLike] = None
    ):
    raw = data_dict["feature"][index_feature,:]
    masked = raw.copy().astype(float)
    masked[~data_dict["mask"]] = np.nan
    fig, ax = plt.subplots(figsize = (10, 5))
    ax.plot(data_dict["time"], raw, label = "Raw")
    ax.plot(data_dict["time"], masked, label = "Masked")
    if all([t is not None for t in time]):
        ax.set_xlim(time[0], time[-1])
    ax.legend()
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("A.U")
    ax.spines[["top", "right"]].set_visible(False)
    plt.show()
    if saving_filename is not None:
        fig.savefig(saving_filename)
    return fig

architecture = arch.BidsArchitecture(root = "/Users/samuel/Desktop/PHYSIO_BIDS")
#%%
idx = 1
selection = architecture.select(subject = "01")
tasks = selection.tasks
sessions = selection.sessions
print(tasks[idx])
print(sessions[idx])
sub_selection = selection.select(task = tasks[idx], session = sessions[idx])
#%%
files_biopac = sub_selection.select(acquisition = "biopac", extension = ".gz")
if files_biopac.database.empty:
    print("No Biopac file found")
else:
    df_biopac = pd.read_csv(files_biopac.database["filename"].to_list()[0], sep = "\t", header = None)
    # EDA
    eda_dict = extract_and_process_eda(df_biopac, plot = True)
    # PPG
    ppg_dict = extract_and_process_ppg(df_biopac, plot = True)
    # RSP
    rsp_dict = extract_and_process_rsp(df_biopac, plot = True)
    # ECG
    ecg_dict = extract_and_process_ecg(df_biopac, plot = True)
    plot_results(ppg_dict, 1, "PPG", time = [40, 60])
    plot_results(ecg_dict, 1, "ECG", time = [40, 60])
    plot_results(eda_dict, 1, "EDA", time = [40, 60])
    plot_results(rsp_dict, 1, "RSP", time = [40, 60])
#%%
files_eyetracking = sub_selection.select(acquisition = "eyelink", extension = ".gz")
if files_eyetracking.database.empty:
    print("No Eyetracking file found")
else:
    filename_eyetracking_samples = [
        str(file) for file in files_eyetracking.database["filename"].to_list() if "samples" in str(file)
    ][0]
    filename_eyetracking_events = [
        str(file) for file in files_eyetracking.database["filename"].to_list() if "events" in str(file)
    ][0]

    df_eyetracking_samples = pd.read_csv(filename_eyetracking_samples, sep = "\t", header = None)
    df_eyetracking_events = pd.read_csv(filename_eyetracking_events, sep = "\t", header = None)
#%%
    events = extract_eyelink_events(df_eyetracking_events)
    eyetracking_dict = extract_and_process_eyetracking(df_eyetracking_samples)
    eyetracking_dict = crop_eyetracking(eyetracking_dict, events)
    plot_results(eyetracking_dict, 2, "Eye Tracking", time = [None, None])
# EYETRACKING
#%%
plot_results(ecg_dict, 0, "ECG", time = [40, 50])
plot_results(ecg_dict, 1, "ECG", time = [40, 50])
# PLOT EVERYTHING
# TODO:
# [X] Check if the time in the eyetracking samples is the same as the time in
# [X] Write ECG exctractor function
# [X] Write EDA exctractor function
# [X] Write PPG exctractor function
# [X] Write RESP extractor function
# [X] Write a function to extract the events from the eyelink file
# [X] Write a function to extract the samples from the eyelink file

# %% PIPELINE
saving_location = Path("/Users/samuel/Desktop/PHYSIO_BIDS/derivatives")
saving_location.mkdir(parents = True, exist_ok = True)
subjects = architecture.subjects
report = {
    "subject": [],
    "task": [],
    "session": [],
    "ppg_quality_mean": [],
    "ppg_quality_std": [],
    "rsp_quality_mean": [],
    "rsp_quality_std": [],
    "ecg_quality_mean": [],
    "ecg_quality_std": [],
    "message": []
}
for subject in subjects[:1]:
    selection = architecture.select(subject = subject)
    tasks = selection.tasks
    sessions = selection.sessions
    for task in tasks[:2]:
        for session in sessions[:2]:
            file_parts = [
                f"sub-{subject}",
                f"ses-{session}",
                f"task-{task}",
            ]
            saving_dir = saving_location / file_parts[0] / file_parts[1]
            with PdfPages(saving_dir / f"sub-{subject}_ses-{session}_task-{task}_desc-report.pdf") as pdf:
                saving_base = "_".join(file_parts)
                report["subject"].append(subject)
                report["task"].append(task)
                report["session"].append(session)
                sub_selection = selection.select(task = task, session = session)
                files_biopac = sub_selection.select(acquisition = "biopac", extension = ".gz")
                if files_biopac.database.empty:
                    report["message"].append("No Biopac file found")
                    report["ppg_quality_mean"].append(np.nan)
                    report["ppg_quality_std"].append(np.nan)
                    report["rsp_quality_mean"].append(np.nan)
                    report["rsp_quality_std"].append(np.nan)
                    report["ecg_quality_mean"].append(np.nan)
                    report["ecg_quality_std"].append(np.nan)
                else:
                    biopac = True
                    df_biopac = pd.read_csv(files_biopac.database["filename"].to_list()[0], sep = "\t", header = None)
                    # EDA ================================
                    eda_dir = saving_dir / "eda"
                    eda_filename = eda_dir / "_".join([str(saving_base), "eda.pkl"])
                    eda_filename.parent.mkdir(parents = True, exist_ok = True)
                    eda_fig, eda_dict = extract_and_process_eda(df_biopac, plot = True)
                    pdf.savefig(eda_fig)
                    fig = plot_results(eda_dict, 1, "example EDA 20 sec", time = [40, 60])
                    pdf.savefig(fig)
                    with open(eda_filename, "wb") as f:
                        pickle.dump(eda_dict, f)
                    # PPG ================================
                    ppg_dir = saving_dir / "ppg"
                    ppg_filename = ppg_dir / "_".join([str(saving_base), "ppg.pkl"])
                    ppg_filename.parent.mkdir(parents = True, exist_ok = True)
                    fig, ppg_dict = extract_and_process_ppg(df_biopac, plot = True)
                    pdf.savefig(fig)
                    report["ppg_quality_mean"].append(ppg_dict["quality"]["mean"])
                    report["ppg_quality_std"].append(ppg_dict["quality"]["std"])
                    fig = plot_results(ppg_dict, 1, "example PPG 20 sec", time = [40, 60])
                    pdf.savefig(fig)
                    with open(ppg_filename, "wb") as f:
                        pickle.dump(ppg_dict, f)
                    # RSP ================================
                    rsp_dir = saving_dir / "rsp"
                    rsp_filename = rsp_dir / "_".join([str(saving_base), "rsp.pkl"])
                    rsp_filename.parent.mkdir(parents = True, exist_ok = True)
                    fig, rsp_dict = extract_and_process_rsp(df_biopac, plot = True)
                    pdf.savefig(fig)
                    report["rsp_quality_mean"].append(rsp_dict["quality"]["mean"])
                    report["rsp_quality_std"].append(rsp_dict["quality"]["std"])
                    fig = plot_results(rsp_dict, 1, "example RSP 20 sec", time = [40, 60])
                    pdf.savefig(fig)
                    with open(rsp_filename, "wb") as f:
                        pickle.dump(rsp_dict, f)
                    # ECG ================================
                    ecg_dir = saving_dir / "ecg"
                    ecg_filename = ecg_dir / "_".join([str(saving_base), "ecg.pkl"])
                    ecg_filename.parent.mkdir(parents = True, exist_ok = True)
                    fig, ecg_dict = extract_and_process_ecg(df_biopac, plot = True)
                    pdf.savefig(fig)
                    report["ecg_quality_mean"].append(ecg_dict["quality"]["mean"])
                    report["ecg_quality_std"].append(ecg_dict["quality"]["std"])
                    fig = plot_results(ecg_dict, 1, "example ECG 20 sec", time = [40, 60])
                    pdf.savefig(fig)
                    with open(ecg_filename, "wb") as f:
                        pickle.dump(ecg_dict, f)
                # EYETRACKING ================================
                files_eyetracking = sub_selection.select(acquisition = "eyelink", extension = ".gz")
                if files_eyetracking.database.empty:
                    report["message"].append("No Eyetracking file found")
                else:
                    eyetracking = True
                    filename_eyetracking_samples = [
                        str(file) for file in files_eyetracking.database["filename"].to_list() if "samples" in str(file)
                    ][0]
                    filename_eyetracking_events = [
                        str(file) for file in files_eyetracking.database["filename"].to_list() if "events" in str(file)
                    ][0]

                    df_eyetracking_samples = pd.read_csv(filename_eyetracking_samples, sep = "\t", header = None)
                    df_eyetracking_events = pd.read_csv(filename_eyetracking_events, sep = "\t", header = None)
                    events = extract_eyelink_events(df_eyetracking_events)
                    eyetracking_dir = saving_dir / "eyetracking"
                    eyetracking_filename = eyetracking_dir / "_".join([str(saving_base), "eyetracking.pkl"])
                    eyetracking_filename.parent.mkdir(parents = True, exist_ok = True)
                    eyetracking_dict = extract_and_process_eyetracking(df_eyetracking_samples)
                    eyetracking_dict, events = crop_eyetracking(eyetracking_dict, events)
                    with open(eyetracking_filename, "wb") as f:
                        pickle.dump(eyetracking_dict, f)
                    events_filename = eyetracking_dir / "_".join([str(saving_base), "events.pkl"])
                    with open(events_filename, "wb") as f:
                        pickle.dump(events, f)
                    fig = plot_results(eyetracking_dict, 2, "Pupil size", time = [None, None])
                    pdf.savefig(fig)
                if biopac and eyetracking:
                    report["message"].append("Biopac and Eyetracking found")
            pdf.close()
report_df = pd.DataFrame(report)
report_df.to_csv(saving_location / "report.csv", index = False)

# %%
