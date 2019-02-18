import pandas as pd
import numpy as np
import sklearn as sc
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder

df_shakespeare = pd.read_csv('./winequality-red.csv')
df_shakespeare.head()