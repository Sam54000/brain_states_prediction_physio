import pandas as pd
import matplotlib.pyplot as plt
import pickle
from src.utils import *
import numpy as np
from typing import Optional
import itertools
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
    if saving_filename is not None:
        fig.savefig(saving_filename)
    return fig

if __name__ == "__main__":
    architecture = arch.BidsArchitecture(root = "/Users/samuel/Desktop/PHYSIO_BIDS")
    saving_location = Path("/Users/samuel/Desktop/processed")
    saving_location.mkdir(parents = True, exist_ok = True)
    subjects = architecture.subjects
    for subject in subjects:
        report = {
            "subject": [],
            "task": [],
            "session": [],
            "run": [],
            "ppg": [],
            "rsp": [],
            "ecg": [],
            "eda": [],
            "biopac": [],
            "eyetracking": [],
            "ppg_quality_mean": [],
            "ppg_quality_std": [],
            "rsp_quality_mean": [],
            "rsp_quality_std": [],
            "ecg_quality_mean": [],
            "ecg_quality_std": [],
        }
        selection = architecture.select(
            subject = subject,
            extension = ".gz")
        products = list(itertools.product(
            selection.sessions,
            selection.tasks,
            selection.runs
        ))
        for session, task, run in products:
            file_parts = [
                f"sub-{subject}",
                f"ses-{session}",
                f"task-{task}",
                f"run-{run}"
            ]
            saving_dir = saving_location / file_parts[0] / file_parts[1]
            report["subject"].append(subject)
            report["task"].append(task)
            report["session"].append(session)
            report["run"].append(run)
            pdf_filename = saving_dir / f"sub-{subject}_ses-{session}_task-{task}_desc-report.pdf"
            with PdfPages(pdf_filename) as pdf:
                saving_base = "_".join(file_parts)
                sub_selection = selection.select(
                    task = task,
                    session = session,
                    run = run
                )
                try:
                    files_biopac = sub_selection.select(acquisition = "biopac")
                    if files_biopac.database.empty:
                        biopac = False
                    else:
                        biopac = True
                        df_biopac = pd.read_csv(files_biopac.database["filename"].to_list()[0], sep = "\t", header = None)
                        # EDA ================================
                        eda_dir = saving_dir / "eda"
                        eda_filename = eda_dir / "_".join([str(saving_base), "eda.pkl"])
                        eda_filename.parent.mkdir(parents = True, exist_ok = True)
                        eda_fig, eda_dict = extract_and_process_eda(df_biopac, plot = True)
                        pdf.savefig(eda_fig)
                        plt.close(eda_fig)
                        fig = plot_results(eda_dict, 1, "example EDA 20 sec", time = [40, 60])
                        pdf.savefig(fig)
                        plt.close(fig)
                        report["eda"].append(True)
                        with open(eda_filename, "wb") as f:
                            pickle.dump(eda_dict, f)
                        # PPG ================================
                        ppg_dir = saving_dir / "ppg"
                        ppg_filename = ppg_dir / "_".join([str(saving_base), "ppg.pkl"])
                        ppg_filename.parent.mkdir(parents = True, exist_ok = True)
                        fig, ppg_dict = extract_and_process_ppg(df_biopac, plot = True)
                        pdf.savefig(fig)
                        plt.close(fig)
                        report["ppg_quality_mean"].append(ppg_dict["quality"]["mean"])
                        report["ppg_quality_std"].append(ppg_dict["quality"]["std"])
                        report["ppg"].append(True)
                        fig = plot_results(ppg_dict, 1, "example PPG 20 sec", time = [40, 60])
                        pdf.savefig(fig)
                        plt.close(fig)
                        with open(ppg_filename, "wb") as f:
                            pickle.dump(ppg_dict, f)
                        # RSP ================================
                        rsp_dir = saving_dir / "rsp"
                        rsp_filename = rsp_dir / "_".join([str(saving_base), "rsp.pkl"])
                        rsp_filename.parent.mkdir(parents = True, exist_ok = True)
                        fig, rsp_dict = extract_and_process_rsp(df_biopac, plot = True)
                        pdf.savefig(fig)
                        plt.close(fig)
                        report["rsp_quality_mean"].append(rsp_dict["quality"]["mean"])
                        report["rsp_quality_std"].append(rsp_dict["quality"]["std"])
                        fig = plot_results(rsp_dict, 1, "example RSP 20 sec", time = [40, 60])
                        pdf.savefig(fig)
                        plt.close(fig)
                        report["rsp"].append(True)
                        with open(rsp_filename, "wb") as f:
                            pickle.dump(rsp_dict, f)
                        # ECG ================================
                        try:
                            ecg_dir = saving_dir / "ecg"
                            ecg_filename = ecg_dir / "_".join([str(saving_base), "ecg.pkl"])
                            ecg_filename.parent.mkdir(parents = True, exist_ok = True)
                            fig, ecg_dict = extract_and_process_ecg(df_biopac, plot = True)
                            pdf.savefig(fig)
                            plt.close(fig)
                            report["ecg_quality_mean"].append(ecg_dict["quality"]["mean"])
                            report["ecg_quality_std"].append(ecg_dict["quality"]["std"])
                            fig = plot_results(ecg_dict, 1, "example ECG 20 sec", time = [40, 60])
                            pdf.savefig(fig)
                            plt.close(fig)
                            report["ecg"].append(True)
                            with open(ecg_filename, "wb") as f:
                                pickle.dump(ecg_dict, f)
                        except Exception as e:
                            report["ecg_quality_mean"].append(np.nan)
                            report["ecg_quality_std"].append(np.nan)
                            report["ecg"].append(False)
                except Exception as e:
                    biopac = False

                report["biopac"].append(biopac)
                if not biopac:
                    report["ppg_quality_mean"].append(np.nan)
                    report["ppg_quality_std"].append(np.nan)
                    report["rsp_quality_mean"].append(np.nan)
                    report["rsp_quality_std"].append(np.nan)
                    report["ecg_quality_mean"].append(np.nan)
                    report["ecg_quality_std"].append(np.nan)
                    report["eda"].append(False)
                    report["ppg"].append(False)
                    report["rsp"].append(False)
                    report["ecg"].append(False)
                # EYETRACKING ================================
                try:
                    files_eyetracking = sub_selection.select(acquisition = "eyelink", extension = ".gz")
                    if files_eyetracking.database.empty:
                        report["eyetracking"].append(False)
                    else:
                        report["eyetracking"].append(True)
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
                        plt.close(fig)
                except Exception as e:
                    report["eyetracking"].append(False)
            pdf.close()
        with open(saving_location / f"sub-{subject}_report.pkl", "wb") as f:
            pickle.dump(report, f)
        report_df = pd.DataFrame(report)
        report_df.to_csv(saving_location / f"sub-{subject}_report.csv", index = False)