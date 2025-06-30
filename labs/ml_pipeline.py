#%%
import pickle
from pathlib import Path
import numpy as np
import os
import logging
import scipy.stats
import itertools
import sklearn
import pandas as pd
import scipy
from typing import Optional
import bids_explorer.architecture.architecture as arch
import sklearn.linear_model

def concatenate_data(
    architecture: arch.BidsArchitecture, 
    modalities: Optional[list] = ["ecg","eda","rsp","eyetracking"],
    nb_seconds: Optional[int] = 6,
    brainstates_names: Optional[list[str]] = ["CapsNrHpf"]
    ):
    
    desired_index = {
        "ecg":[1, 3],
        "eda": [2,3],
        "rsp": [2,3,4,7,8],
        "eyetracking": [2,3,4],
    }

    all_features_stacked = []
    all_masks_stacked = []
    brainstates = {}
    X = []

    for file_idx, file in architecture:
        
        with open(file["filename"], "rb") as opened_file:
            data = pickle.load(opened_file)
        

        modalities_stack = []
        masks = []
        name_to_exclude = ["ecg","eda","ppg","rsp","eyetracking","time","ttl"]
        brainstates_labels = {}
        conditions = np.concatenate( 
            [np.isin(modalities,list(data.keys())),
            np.isin(brainstates_names, list(data.keys()))]
        )

        if all(conditions):
            for modality in modalities:
                modalities_stack.append(
                    scipy.stats.zscore(
                        data[modality]["features"][desired_index[modality],:],
                        axis = 1,
                    )
                )
                masks.append(data[modality]["masks"][nb_seconds:])


            for brainstate in brainstates_names: 
                bs_dict = data[brainstate]
                
                if brainstates_labels.get(brainstate) is None:
                    brainstates_labels[brainstate] = bs_dict["labels"]

                if brainstates.get(brainstate) is None:
                    brainstates[brainstate] = bs_dict["features"][:,nb_seconds:]
                
                else:
                    brainstates[brainstate] = np.concatenate(
                        [brainstates[brainstate], 
                         bs_dict["features"][:,nb_seconds:]],
                        axis = 1
                    )
                
                masks.append(bs_dict["masks"][nb_seconds:])
                
            modalities_stack = np.concatenate(modalities_stack, axis = 0)
            masks = np.stack(masks)
            masks = np.all(masks,axis = 0)

            for index in range(nb_seconds, bs_dict["features"].shape[1]):
                X.append(modalities_stack[:,index-nb_seconds:index].flatten())

            
            all_features_stacked.append(modalities_stack)
            all_masks_stacked.append(masks)
        else:
            continue
    
    if X == []:
        return None, None, None, None
    else:
        X = np.stack(X)
        masks = np.concatenate(all_masks_stacked)

        return X, brainstates, masks, brainstates_labels

#%%

def train_and_test(
    train_architecture:arch.BidsArchitecture,
    test_architecture:arch.BidsArchitecture,
    brainstates: Optional[list[str]] = ["CapsNrHpf"],
    modalities: Optional[list[str]] = ["ecg","eda","rsp","eyetracking"]
):
    X_train, bs_train, masks_train, labels_train = concatenate_data(
        train_architecture,
        modalities = modalities,
        brainstates_names = brainstates,
        )
    if X_train is None:
        return None

    X_test, bs_test, masks_test, labels_test = concatenate_data(
        test_architecture,
        modalities = modalities,
        brainstates_names = brainstates
        )
    if X_test is None:
        return None


    results = {
        "Pearson_R": [],
        "brainstate_method": [],
        "brainstate": [],
    }
    imputator = sklearn.impute.SimpleImputer(strategy = "median")
    X_train = X_train[masks_train,:]
    X_test = X_test[masks_test,:]
    X_train = imputator.fit_transform(X_train)
    X_test = imputator.fit_transform(X_test)

    for bs in brainstates:
        for state_idx, state_name in enumerate(labels_train[bs]):
            Y_train = bs_train[bs][state_idx,masks_train]
            Y_test = bs_test[bs][state_idx,masks_test]
            model = sklearn.linear_model.Ridge()
            model.fit(X_train,Y_train)
            predicted = model.predict(X_test)
            pearson_R = np.corrcoef(Y_test, predicted)[0,1]
            results["Pearson_R"].append(pearson_R)
            results["brainstate_method"].append(bs)
            results["brainstate"].append(state_name)
    
    return results
            
        
def pipeline(
    architecture: arch.BidsArchitecture,
    brainstates: Optional[list[str]],
    modalities: Optional[list[str]] = ["ecg","eda","rsp","eyetracking"],
    ):

    for subject in architecture.subjects:
        overall_result = {
            "subject": [],
            "task": [],
            "Pearson_R": [],
            "brainstate_method": [],
            "brainstate": []
        }
        for task in architecture.tasks:
            test = architecture.select(subject = subject, task = task)
            train = architecture.remove(subject = subject, task = task)

            if all([not test.database.empty, not train.database.empty]):
                results = train_and_test(
                    train_architecture= train, 
                    test_architecture = test,
                    modalities = modalities,
                    brainstates=brainstates)

                if results is None:
                    continue

                overall_result["subject"] += [subject]*len(results["Pearson_R"])
                overall_result["task"] += [task]*len(results["Pearson_R"])
                overall_result["Pearson_R"] += results["Pearson_R"]
                overall_result["brainstate_method"] += results["brainstate_method"]
                overall_result["brainstate"] += results["brainstate"]
        df = pd.DataFrame(overall_result)
        df.to_csv(f"data/sub-{subject}_desc-{''.join([m.capitalize() for m in modalities])}_predictions.csv")
        

# %%
if __name__ == "__main__":
    architecture = arch.BidsArchitecture(
        root =  "/projects/TIKTOK/data/physio_processing/PHYSIO_BIDS_PROCESSED/",
    )
    bs_architecture = architecture.select(datatype = "brainstates")
    modalities = ["ecg", 
                  "rsp", 
                  "eda", 
                  "eyetracking"]
    combinations = []
    for size_combination in range(1,5):
        combinations += [list(comb) for comb in itertools.combinations(modalities, size_combination)]

    for modalities_combination in combinations:
        pipeline(
            architecture = architecture.select(datatype = "multimodal"),
            brainstates = bs_architecture.descriptions[2:],
            modalities=modalities_combination
        )