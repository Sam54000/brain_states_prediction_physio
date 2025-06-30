#%%
import re
import pickle
from pathlib import Path
import pandas as pd
import shutil
import bids_explorer.architecture.architecture as arch
import logging

#%%
brainstates_dir = Path("/data2/Projects/knenning/natview_eeg/scripts/_tiktok_data/_data_extracted_v1/_ts_extracted")
logging.basicConfig(
    filename='transfer_brainstates_files.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

for file in brainstates_dir.iterdir():
    subject = re.search(r"sub-(\w+)(?=\_)", file.name).group(0)
    session = re.search(r"ses-(\w+)(?=\_)", file.name).group(0)
    task = re.search(r"task-(\w+)(?=\_)", file.name).group(0)
    add_info = re.search(r"(?<=rec-standardIC_).*(?=\.)", file.name).group(0)
    add_info = "".join(
        [x.capitalize() if not "Yeo" in x else x for x in add_info.split("_")]
        )
    saving_path = Path(f"/projects/TIKTOK/data/physio_processing/PHYSIO_BIDS_PROCESSED/{subject}/{session}/brainstates/")
    saving_path.mkdir(parents = True, exist_ok = True)
    filename = f"{subject}_{session}_{task}_run-1_rec-standardIC_desc-{add_info}_brainstates.pkl"
    message = (
        f"Processing:\n\tsubject: {subject}\n\tsession: {session}\n\ttask: {task}\n\t"
    )
    logging.info(message)
    brainstates_data = pd.read_csv(file)
    brainstate_dir = {
        "features": brainstates_data[brainstates_data.columns[1:]].values.T,
        "labels": brainstates_data.columns[1:],
        "masks": brainstates_data.values[:,0].astype(bool)
    }
    message = f"Saving to:\n\t{saving_path / filename}\n"
    logging.info(message)

    with open(saving_path / filename, "wb") as f:
        pickle.dump(brainstate_dir, f)