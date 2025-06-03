#%%
import re
import pickle
from pathlib import Path
import pandas as pd
import shutil
import bids_explorer.architecture.architecture as arch

#%%
brainstates_dir = Path("/Volumes/LaCie/brainstates/")
for file in brainstates_dir.iterdir():
    subject = re.search(r"sub-(\w+)(?=\_)", file.name).group(0)
    session = re.search(r"ses-(\w+)(?=\_)", file.name).group(0)
    task = re.search(r"task-(\w+)(?=\_)", file.name).group(0)
    add_info = re.search(r"(?<=rec-standardIC_).*(?=\.)", file.name).group(0)
    add_info = "".join(
        [x.capitalize() if not "Yeo" in x else x for x in add_info.split("_")]
        )
    saving_path = Path(f"/Volumes/LaCie/processed_2/{subject}/{session}/brainstates/")
    saving_path.mkdir(parents = True, exist_ok = True)
    filename = f"{subject}_{session}_{task}_run-1_rec-standardIC_desc-{add_info}.pkl"
    print(filename)
    brainstates_data = pd.read_csv(file)
    brainstate_dir = {
        "features": brainstates_data[brainstates_data.columns[1:]].values.T,
        "labels": brainstates_data.columns[1:],
        "masks": brainstates_data.values[:,0].astype(bool)
    }
    with open(saving_path / filename, "wb") as f:
        pickle.dump(brainstate_dir, f)
    
#%%
architecture = arch.BidsArchitecture(root = "/Volumes/LaCie/processed_2")
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