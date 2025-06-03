#%%
from pathlib import Path
import numpy as np
import bids_explorer.architecture.architecture as arch
import sklearn
import pandas as pd
import pickle
import scipy
import logging
import re
import os
import shutil

#%%
logging.basicConfig(
    filename='multimodal_processing.log',
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

architecture = arch.BidsArchitecture(root = "/Volumes/LaCie/processed_2")
architecture.remove(datatype = "multimodal", inplace = True)
eyetracking_selection = architecture.select(
    datatype = "eyetracking", 
    suffix = "eyetracking"
    )
other_selection = architecture.remove(datatype = "eyetracking")

for file_idx, eyetracking_file in eyetracking_selection:
    message = (
        f"Processing:\n\tsubject: {eyetracking_file["subject"]}\n\tsession: {eyetracking_file["session"]}\n\ttask: {eyetracking_file["task"]}\n\t"
    )
    logging.info(message)
    
    multimodal_dict = {}
    
    with open(eyetracking_file["filename"], "rb") as ef:
        eyetracking_data = pickle.load(ef)
        
    et_data_length = eyetracking_data["features"].shape[1]
    multimodal_dict["time"] = eyetracking_data["time"]
    
    multimodal_dict["eyetracking"] = {
        "features": eyetracking_data["features"],
        "labels": eyetracking_data["labels"],
        "masks": eyetracking_data["masks"][1]
    }
    
    saving_folder = Path(f"{eyetracking_file["root"]}/sub-{eyetracking_file["subject"]}/ses-{eyetracking_file["session"]}/multimodal/")
    filename = f"sub-{eyetracking_file['subject']}_ses-{eyetracking_file['session']}_task-{eyetracking_file['task']}_run-{eyetracking_file['run']}_multimodal.pkl"

    if (saving_folder / filename).exists():
        logging.info(f"Multimodal file already exists")
        continue

    saving_folder.mkdir(parents = True, exist_ok = True)

    for modality in other_selection.datatypes:
        
        subselection = other_selection.select(
            subject = eyetracking_file["subject"],
            session = eyetracking_file["session"],
            datatype = modality,
            task = eyetracking_file["task"],
            suffix = modality
        )
        
        if subselection.database.empty:
            logging.info(f"\t{modality}: no file found")
            continue

        for sub_selection_idx, sub_selection in subselection:
            with open(sub_selection["filename"], "rb") as mf:
                modality_data = pickle.load(mf)
            if modality == "brainstates":
                multimodal_dict[sub_selection["description"]] = modality_data
                logging.info(f"\t{sub_selection['description']}: OK")
            else:
                logging.info(f"\t{modality}: OK")
                mod_data_length = modality_data["features"].shape[1]
                max_length = min(et_data_length,mod_data_length)
                
                decimated_features = scipy.signal.decimate(
                    modality_data["features"],
                    4
                )
                decimated_features = modality_data["features"][:,::4]
                decimated_features = decimated_features[:,:max_length]
                mask = modality_data["masks"][::4]
                mask = mask[:max_length]
                mask = mask.astype(bool)
                multimodal_dict["ttl"] = eyetracking_data["ttl"]
                multimodal_dict[modality] = {
                    "features": decimated_features,
                    "labels": modality_data["labels"],
                    "masks": mask
                    }

    logging.info(f"Saving {filename}\n")
    with open(saving_folder / filename, "wb") as f:
        pickle.dump(multimodal_dict, f)

#%%