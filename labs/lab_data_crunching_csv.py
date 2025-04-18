#%%
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot
#%%
big_df = pd.DataFrame()
for i in range(1,11):
    df = pd.read_csv(f"/Users/samuel/Desktop/processed/sub-{i:02}/sub-{i:02}_report.csv")
    big_df = pd.concat([big_df, df])
# %%
big_df.to_csv("report_all.csv")
# %%
