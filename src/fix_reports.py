#%%
import pickle
import numpy as np
import pandas as pd
#%%
for i in range(1,11):
    with open(f"/Volumes/LaCie/processed/sub-{i:02}_report.pkl", "rb") as f:
        report = pickle.load(f)
    temp = np.zeros_like(np.array(report["eyetracking"]))
    qual = 0
    for n, eyetracking in enumerate(report["eyetracking"]):
        if eyetracking:
            temp[n] = report["eyetracking_quality"][qual]
            qual =+ 1
        else:
            temp[n] = np.nan
    
    report["eyetracking_quality"] = temp
    df = pd.DataFrame(report)
    df.to_csv(f"/Volumes/LaCie/processed/sub-{i:02}_report.csv",index=False)

# %%
all_reports = pd.DataFrame()
for nb in range(1,11):
    df = pd.read_csv(f"/Volumes/LaCie/processed/sub-{nb:02}_report.csv")
    all_reports = pd.concat([all_reports,df],axis=0)
# %%
all_reports
# %%
import bids_explorer.architecture.architecture as arch
architecture = arch.BidsArchitecture(
    root = "/Volumes/LaCie/processed",
    datatype="eyetracking",
    suffix="eyetracking",
    extension=".pkl"
)
# %%
report = pd.read_csv("/Users/samuel/01_projects/brain_states_prediction_physio/all_report_2.csv")
#%%
report["eyetracking_quality"] = report["eyetracking_quality"].astype("float64")
report["eyetracking_before_interp"] = 0.0
for idx, file in architecture:
    with open(file["filename"], "rb") as f:
        data = pickle.load(f)
    
    mask = (
        (report["subject"] == int(file["subject"]))
        &(report["task"] == file["task"]) 
        & (report["session"] == file["session"])
    )
    
    report.loc[mask, "eyetracking_quality"] = data["quality"]["masks_based_long_flat"]
    report.loc[mask, "eyetracking_before_interp"] = data["quality"]["masks_based"]
    
    print(file["subject"])
    print(f"\t{file["session"]}")
    print(f"\t\t{file["task"]}")
    
# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

modalities = [
    "ppg_quality", 
    "rsp_quality",
    "ecg_quality",
    "eda_quality",
    "eyetracking_quality",
    ]

for modality in modalities:
    grouped = report.groupby('subject')[modality]
    means = grouped.mean()
    counts = grouped.count()
    stds = grouped.std()
    
    ci95 = 1.96 * (stds / np.sqrt(counts))
    
    plt.figure(figsize=(10, 6))
    plt.bar(means.index, means.values, yerr=ci95, capsize=5)
    plt.xlabel('Subject')
    plt.ylabel('Mean Quality')
    plt.title(f'{modality.replace("_", " ").capitalize()} by Subject')
    plt.ylim(0, 1.1) 
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

# %%
