import pandas as pd
import numpy as np

df = pd.read_csv('C:/Users/Dell/Desktop/Vivek/Pscript/train dataset.csv')
df['split'] = np.random.randn(df.shape[0], 1)

msk = np.random.rand(len(df)) <= 0.7

train = df[msk]
test = df[~msk]
train.to_csv('train.csv', index=False)
print("Train.csv file  created")
test.to_csv('test.csv', index=False)
print("Test.csv file created")
