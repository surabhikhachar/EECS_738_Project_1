
import matplotlib.pyplot as plt
import matplotlib
matplotlib.style.use('ggplot')
import numpy as np
import pandas as pd
from copy import deepcopy
import random

from kmeans import K_Means

df_iris = pd.read_csv('../data/Iris.csv')
df_iris.head()


xAxis      = df_iris['PetalLengthCm']
xAxisLabel = 'Petal Length (cm)'
yAxis      = df_iris['PetalWidthCm']
yAxisLabel = 'Petal Width (cm)'
classVerify = df_iris['Species']

#Feature enginering
if xAxis.min() == xAxis.max() or yAxis.min() == yAxis.max():
    raise Exception('Cannot run K-Means without at-least 2 distinct data points')
xAxis -= xAxis.min()
xAxis /= xAxis.max()
xAxis = 4*xAxis - 2

yAxis -= yAxis.min()
yAxis /= yAxis.max()
yAxis = 4*yAxis - 2

colmap = [name for name, hex in matplotlib.colors.cnames.items()]
random.Random(0).shuffle(colmap)
k = len(set(classVerify))
print("K is %s" % k)
kmeans = K_Means(xAxis, yAxis, classVerify)

classificationColors = []
for i in range(len(xAxis)):
    index = kmeans.predict(xAxis[i],yAxis[i])
    color = colmap[index]
    classificationColors.append(color)

fig = plt.figure(figsize=(5, 5))

plt.scatter(xAxis, yAxis, color=classificationColors, edgecolor='k')
print(kmeans.centroids)
plt.scatter([kmeans.centroids[i][0] for i in range(k)], [kmeans.centroids[i][1] for i in range(k)], color='k')
plt.xlim(-2.5, 2.5)
plt.ylim(-2.5, 2.5)
plt.show()