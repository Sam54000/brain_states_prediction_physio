#%%
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
#%%
modalities = ["ecg", 
                "rsp", 
                "eda", 
                "eyetracking"]
combinations = []
for size_combination in range(2,5):
    combinations += ([list(comb) for comb in itertools.combinations(modalities, size_combination)])

for strings in combinations:
    modality = "".join([s.capitalize() for s in strings])
    #modality = modality.capitalize()
    df = pd.DataFrame()
    for subject_nb in range(1,11):
        try:
            working_dir = Path.cwd()
            print(working_dir)
            filename = working_dir.parent / f"data/sub-{subject_nb:02}_desc-{modality}_predictions.csv"
            print(filename)
            temp = pd.read_csv(filename)
            df = pd.concat([df, temp])
        except Exception as e:
            continue

    brainstate_method = df["brainstate_method"].unique()
    #selection = [brainstate_method[0], "GsNrBpf"]
    selections = [b_state for b_state in brainstate_method if "Yeo" in b_state]
    for selection in selections:
        selected = df[df["brainstate_method"] == selection]
        selected = selected[selected["task"] != "PVT"]

        fig, ax = plt.subplots()
        sns.stripplot(
            selected, 
            x = "brainstate",
            y = "Pearson_R",
            alpha = 0.6,
            color = "gray",
            ax=ax
        )

        sns.barplot(
            selected,
            x = "brainstate",
            y = "Pearson_R",
            ci = 65,
            ax = ax,
            palette="Paired"
        )


        ax.spines[["top", "right"]].set_visible(False)
        ax.set_ylim(-0.2,1)
        ax.set_xticklabels(labels = ax.get_xticklabels(), rotation = 90)
        ax.set_title(f"{selection} Prediction with {modality}")
        saving_folder = Path.cwd().parent / "figs" / modality
        if not saving_folder.exists():
            saving_folder.mkdir(parents=True, exist_ok=True)

        fig.savefig(saving_folder / f"{selection}_prediction.png")

# %%
modalities = ["ecg", 
              "rsp", 
              "eda", 
              "eyetracking"]
comb = []
for size_combination in range(1,5):
    comb += [list(comb) for comb in itertools.combinations(modalities, size_combination)]

df = pd.DataFrame()
for strings in comb:
    if isinstance(strings,list):
        modality = "".join([s.capitalize() for s in strings ])
    else:
        modality = strings.capitalize()
    for subject_nb in range(1,11):
        try:
            working_dir = Path.cwd()
            filename = working_dir.parent / f"data/sub-{subject_nb:02}_desc-{modality}_predictions.csv"
            print(filename)
            temp = pd.read_csv(filename)
            temp["modality"] = modality
            temp["nb_modality"] = len(strings) if isinstance(strings, list) else 1
            df = pd.concat([df, temp])
            
        except Exception as e:
            continue
# %%
brainstate_method = df["brainstate_method"].unique()

for method in brainstate_method:
    fig, ax = plt.subplots(figsize = (12,6) )
    data = df[df["brainstate_method"] == method]
    sns.violinplot(
        data = data,
        x = "modality",
        y = "Pearson_R",
        ax = ax,
        alpha = 0.45,
    )
    sns.lineplot(
        data=data,
        x = "modality",
        y = "Pearson_R",
        ax = ax,
        alpha = 0.25
    )

    ax.set_ylim(-0.2,1)
    ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
    ax.spines[["top","right"]].set_visible(False)
    ax.set_title(method)
    plt.tight_layout()
    fig.savefig(f"evolution_{method}.png")
# %%
