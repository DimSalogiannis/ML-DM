import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.linalg import svd

#comment
# loading the dataset
df = pd.read_csv(r'C:\Users\Dimitris\Desktop\DTU\_Semester 03\Intro to ML\diamonds.csv')

df.info()

# We can see that there are no empty (NaN) values in our dataframe.


# We extract the attribute names:
attributeNames = np.asarray(df.columns)

cut = np.array(df.cut)
color = np.array(df.color)
clarity = np.array(df.clarity)
df = df.drop([['cut', 'color', 'clarity']], axis=1)

print(df.to_string())

# We can see that
