#%%
#import libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import svd
#%%

# loading the dataset
df = pd.read_csv(r'C:\Users\Dimitris\Desktop\DTU\_Semester 03\Intro to ML\diamonds.csv')

# Transform dataframe to numpy array
raw_data = df.values

# maybe it makes sense from now to drop the first column, as it contains the number of the row
raw_data = np.delete(raw_data, 0, axis = 1)

print(raw_data.shape)


# selecting X and y 

y = raw_data[: , 1]
X = np.delete(raw_data, 1, 1)


# creating the rest of the variables
N = np.shape(X)[0]
M = np.shape(X)[1]
attributeNames = list(df.columns)

#%%
# permutation = [0, 1, 3, 4, 5, 6, 8, 9, 10, 2, 7]
#idx = np.empty_like(permutation)
#idx[permutation] = np.arange(len(permutation))

cols = range(0, 10) 
X = raw_data[:, cols]

# We can extract the attribute names that came from the header of the csv
attributeNames = np.asarray(df.columns[cols])

#%%

# re-arrange the data

# drop the categorical columns

# standardize them