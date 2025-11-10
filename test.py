import pandas as pd 
import numpy as np 

df = pd.read_csv("earthquakes_filtred.csv")
good_idx = df[df['properties.tsunami']==0]
final_good_idx = good_idx.index

final_random = np.random.choice(final_good_idx, size=500, replace=False)
bad_idx = df[df['properties.tsunami']==1]

final_bad_idx = bad_idx.index
np.save('bad_idx.npy', final_bad_idx)
np.save('good_idx.npy', final_random)