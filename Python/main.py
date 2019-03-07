import pandas as pd


df = pd.read_csv("winequality-red.csv")
df = df[['one', 'two']]
dataset = df.astype(float).values.tolist()
X = df.values #returns a numpy array
