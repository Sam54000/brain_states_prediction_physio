#%%
import re
import pickle
from pathlib import Path
import pandas as pd
import shutil
import bids_explorer.architecture.architecture as arch

#%%
architecture = arch.BidsArchitecture(root = "/projects/TIKTOK/data/physio_processing/PHYSIO_BIDS_PROCESSED/")
multimodal_selection = architecture.select(datatype = "multimodal")
for file_idx, file in multimodal_selection:
    bs_selection = architecture.select(
        subject = file["subject"],
        session = file["session"],
        task = file["task"],
        suffix = "brainstates"
    )
    with open(file["filename"], "rb") as f:
        multimodal_data = pickle.load(f)
    for bs_idx, bs_file in bs_selection:
        with open(bs_file["filename"], "rb") as f:
            bs_data = pickle.load(f)
        multimodal_data[bs_file["description"]] = bs_data
    with open(file["filename"], "wb") as f:
        pickle.dump(multimodal_data, f)

# %%
architecture = arch.BidsArchitecture(root = "/Volumes/LaCie/processed_2",
                                     datatype = "multimodal")
for file_idx, file in architecture:
    destination = Path(f"/Volumes/DUOSTICK/sub-{file["subject"]}/ses-{file["session"]}/multimodal/")
    destination.mkdir(parents = True, exist_ok = True)
    shutil.copy(file["filename"], destination / file["filename"].name)
# %%