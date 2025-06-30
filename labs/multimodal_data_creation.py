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

architecture = arch.BidsArchitecture(root = "/projects/TIKTOK/data/physio_processing/PHYSIO_BIDS_PROCESSED/")
architecture.remove(datatype = "multimodal", inplace = True)
eyetracking_selection = architecture.select(
    datatype = "eyetracking", 
    suffix = "eyetracking"
    )
other_selection = architecture.remove(datatype = ["eyetracking", "multimodal"])

#%%
for file_idx, eyetracking_file in eyetracking_selection:
    data_length = []
    bs_data_length = []
    message = (
        f"Processing:\n\tsubject: {eyetracking_file["subject"]}\n\tsession: {eyetracking_file["session"]}\n\ttask: {eyetracking_file["task"]}\n\t"
    )
    logging.info(message)
    
    multimodal_dict = {}
    
    with open(eyetracking_file["filename"], "rb") as ef:
        eyetracking_data = pickle.load(ef)
        
    ttl = eyetracking_data["ttl"].astype(bool)
    multimodal_dict["eyetracking"] = {
        "features": eyetracking_data["features"],
        "labels": eyetracking_data["labels"],
        "masks": eyetracking_data["masks"][1]
    }
    et_data_length = multimodal_dict["eyetracking"]["features"].shape[1]
    data_length.append(et_data_length)
    multimodal_dict["time"] = eyetracking_data["time"][ttl]
    multimodal_dict["ttl"] = ttl
    
    
    saving_folder = Path(f"{eyetracking_file["root"]}/sub-{eyetracking_file["subject"]}/ses-{eyetracking_file["session"]}/multimodal/")
    filename = f"sub-{eyetracking_file['subject']}_ses-{eyetracking_file['session']}_task-{eyetracking_file['task']}_run-{eyetracking_file['run']}_multimodal.pkl"

    #if (saving_folder / filename).exists():
    #    logging.info(f"Multimodal file already exists")
    #    continue

    saving_folder.mkdir(parents = True, exist_ok = True)

    for modality in other_selection.datatypes:
        
        subselection = other_selection.select(
            subject = eyetracking_file["subject"],
            session = eyetracking_file["session"],
            datatype = modality,
            task = eyetracking_file["task"],
            suffix = modality
        )
        
        brainstates = True
        if subselection.database.empty:
            logging.info(f"\t{modality}: no file found")
            if modality == "brainstates":
                brainstates = False
                break
            else:
                continue
        
        for sub_selection_idx, sub_selection in subselection:
            with open(sub_selection["filename"], "rb") as mf:
                modality_data = pickle.load(mf)
            if modality == "brainstates":
                multimodal_dict[sub_selection["description"]] = modality_data
                bs_data_length.append(modality_data["features"].shape[1])
                logging.info(f"\t{sub_selection['description']}: OK")
            else:
                logging.info(f"\t{modality}: OK")
                decimated_features = modality_data["features"][:,::4]
                decimated_mask = modality_data["masks"][::4]
                decimated_mask = decimated_mask.astype(bool)
                mod_data_length = decimated_features.shape[1]
                data_length.append(mod_data_length)

                multimodal_dict[modality] = {
                    "features": decimated_features,
                    "labels": modality_data["labels"],
                    "masks": decimated_mask
                    }
    
    if brainstates:
        max_length = min(data_length)
        bs_max_length = min(bs_data_length)
        overal_max_length = [bs_max_length]
        
        for modality in multimodal_dict.keys():
            if modality in ["ecg", "eda", "ppg", "rsp", "eyetracking"]:
                if modality == "eyetracking" and multimodal_dict["eyetracking"]["features"].shape[1] > max_length:
                    multimodal_dict["ttl"] = multimodal_dict["ttl"][:max_length]
                    multimodal_dict["time"] = multimodal_dict["time"][:max_length]

                multimodal_dict[modality]["features"] = multimodal_dict[modality]["features"][:,:max_length]
                multimodal_dict[modality]["features"] = multimodal_dict[modality]["features"][:,multimodal_dict["ttl"]]
                multimodal_dict[modality]["masks"] = multimodal_dict[modality]["masks"][:max_length]
                multimodal_dict[modality]["masks"] = multimodal_dict[modality]["masks"][multimodal_dict["ttl"]]
                overal_max_length.append(multimodal_dict[modality]["features"].shape[1])

        overal_max_length = min(overal_max_length)
        
        #To get same length as brainstates
        for modality in multimodal_dict.keys():
            if modality not in ["brainstates", "time", "ttl"]:
                multimodal_dict[modality]["features"] = multimodal_dict[modality]["features"][:,:overal_max_length]
                multimodal_dict[modality]["masks"] = multimodal_dict[modality]["masks"][:overal_max_length]

        multimodal_dict["ttl"] = multimodal_dict["ttl"][:overal_max_length]
        multimodal_dict["time"] = multimodal_dict["time"][:overal_max_length]

        logging.info(f"Saving {filename}\n")
        with open(saving_folder / filename, "wb") as f:
            pickle.dump(multimodal_dict, f)

    else:
        logging.info("Not Saving Multimodal file: Brainstates not present")
        continue
