import pandas as pd 
import numpy as np 
import os 
df = pd.read_csv("earthquakes_filtred.csv")
good_idx = df[df['properties.tsunami']==0]
final_good_idx = good_idx.index

final_random = np.random.choice(final_good_idx, size=2000, replace=False)
bad_idx = df[df['properties.tsunami']==1]

final_bad_idx = bad_idx.index
os.makedirs("../machine_learning/", exist_ok=True)

np.save('../machine_learning/bad_idx.npy', final_bad_idx)
np.save('../machine_learning/good_idx.npy', final_random)