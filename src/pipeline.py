#%%
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import numpy as np
from src.modalities import (
    read_biopac,
    read_eyetracking,
)
import itertools
from matplotlib.backends.backend_pdf import PdfPages
from pathlib import Path
import bids_explorer.architecture.architecture as arch
import logging

# Configure logging
logging.basicConfig(
    filename='pipeline.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

if __name__ == "__main__":
    architecture = arch.BidsArchitecture(
        root = "/Users/samuel/Desktop/PHYSIO_BIDS"
        )
    saving_location = Path("/Volumes/LaCie/processed_2")
    saving_location.mkdir(parents = True, exist_ok = True)
    subjects = architecture.subjects
    for subject in subjects[:1]:
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
            "ppg_quality": [],
            "rsp_quality": [],
            "ecg_quality": [],
            "ecg_snr": [],
            "eda_quality": [],
            "eyetracking_quality": [],
        }

        selection = architecture.select(
            subject = subject,
            extension = ".gz")

        products = list(itertools.product(
            selection.sessions[:1],
            selection.tasks[:1],
            ["1"]
        ))

        for session, task, run in products:
            file_parts = [
                f"sub-{subject}",
                f"ses-{session}",
                f"task-{task}",
                f"run-{run}"
            ]

            logging.info(f"sub-{subject}_ses-{session}_task-{task}_run-{run}")

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
                    run = "1"
                )

                try:
                    files_biopac = sub_selection.select(acquisition = "biopac")
                    exception_biopac = False
                    if files_biopac.database.empty:
                        biopac = False
                    else:
                        biopac = True
                        biopac_data = read_biopac(files_biopac.database["filename"].to_list()[0])
                        # EDA ================================
                        eda_dir = saving_dir / "eda"
                        eda_filename = eda_dir / "_".join([str(saving_base), "eda.pkl"])
                        eda_filename.parent.mkdir(parents = True, exist_ok = True)
                        biopac_data["eda"].process().save(eda_filename)
                        eda_fig = biopac_data["eda"].plot()
                        pdf.savefig(eda_fig)
                        plt.close(eda_fig)
                        report["eda"].append(True)
                        report["eda_quality"].append(biopac_data["eda"].quality["masks_based"])

                        # PPG ================================
                        ppg_dir = saving_dir / "ppg"
                        ppg_filename = ppg_dir / "_".join([str(saving_base), "ppg.pkl"])
                        ppg_filename.parent.mkdir(parents = True, exist_ok = True)
                        biopac_data["ppg"].process().save(ppg_filename)
                        ppg_fig = biopac_data["ppg"].plot()
                        pdf.savefig(ppg_fig)
                        plt.close(ppg_fig)
                        report["ppg"].append(True)
                        report["ppg_quality"].append(biopac_data["ppg"].quality["masks_based"])

                        # RSP ================================
                        rsp_dir = saving_dir / "rsp"
                        rsp_filename = rsp_dir / "_".join([str(saving_base), "rsp.pkl"])
                        rsp_filename.parent.mkdir(parents = True, exist_ok = True)
                        biopac_data["rsp"].process().save(rsp_filename)
                        rsp_fig = biopac_data["rsp"].plot()
                        pdf.savefig(rsp_fig)
                        plt.close(rsp_fig)
                        report["rsp"].append(True)
                        report["rsp_quality"].append(biopac_data["rsp"].quality["masks_based"])
                        # ECG ================================
                        try:
                            ecg_dir = saving_dir / "ecg"
                            ecg_filename = ecg_dir / "_".join([str(saving_base), "ecg.pkl"])
                            ecg_filename.parent.mkdir(parents = True, exist_ok = True)
                            biopac_data["ecg"].process().save(ecg_filename)
                            ecg_fig = biopac_data["ecg"].plot()
                            pdf.savefig(ecg_fig)
                            plt.close(ecg_fig)
                            report["ecg"].append(True)
                            report["ecg_quality"].append(biopac_data["ecg"].quality["masks_based"])
                            report["ecg_snr"].append(biopac_data["ecg"].snr)
                        except Exception as e:
                            report["ecg"].append(False)
                            report["ecg_quality"].append(np.nan)
                            logging.exception(e)

                except Exception as e:
                    biopac = False
                    logging.exception(e)
                report["biopac"].append(biopac)
                if not biopac:
                    report["ppg_quality"].append(np.nan)
                    report["rsp_quality"].append(np.nan)
                    report["ecg_quality"].append(np.nan)
                    report["eda_quality"].append(np.nan)
                    report["eda"].append(False)
                    report["ppg"].append(False)
                    report["rsp"].append(False)
                    report["ecg"].append(False)

                # EYETRACKING ================================
                try:
                    files_eyetracking = sub_selection.select(acquisition = "eyelink", extension = ".gz")
                    exception_eyetracking = False
                    if files_eyetracking.database.empty:
                        report["eyetracking"].append(False)
                    else:
                        report["eyetracking"].append(True)
                        fname_eye = [
                            str(file) for file in files_eyetracking.database["filename"].to_list() if "samples" in str(file)
                        ][0]
                        eye_data = read_eyetracking(fname_eye)
                        eye_dir = saving_dir / "eyetracking"
                        eye_filename = eye_dir / "_".join([str(saving_base), "eyetracking.pkl"])
                        eye_filename.parent.mkdir(parents = True, exist_ok = True)
                        eye_data["eyetracking"].process().adapt_ttl(biopac_data["ttl_times"])
                        eye_data["eyetracking"].save(eye_filename)
                        eye_fig = eye_data["eyetracking"].plot()
                        report["eyetracking_quality"].append(eye_data["eyetracking"].quality["masks_based_long_flat"])
                        pdf.savefig(eye_fig)
                        plt.close(eye_fig)

                except Exception as e:
                    report["eyetracking"].append(False)
                    report["eyetracking_quality"].append(np.nan)
                    logging.exception(e)
            pdf.close()
        logging.info(f"sub-{subject} Done")
        with open(saving_location / f"sub-{subject}_report.pkl", "wb") as f:
            pickle.dump(report, f)
        
        #report_df = pd.DataFrame(report)
        #report_df.to_csv(saving_location / f"sub-{subject}_report.csv", index = False)
#%%
# TODO
# - Fix the eyetracking quality issue that output booleans
# - Fix the size of eyetracking quality to have the same size as the rest for
# putting in dataframe. 