from scipy.stats import pearsonr

import numpy as np
import pandas as pd
from math import fabs
'''''
from knn import KNN


ldata = np.array(pd.read_csv("Data/iris.data.learning", header=None))
tdata = np.array(pd.read_csv("Data/iris.data.test", header=None))

#testData, testTarget = np.array_split(tdata, [4], axis=1)
b = KNN(2, ldata, "E")
b.predict(tdata)

print(pearsonr(b.data[0],b.testData[0]))
print(b.testData[0])
print(b.data[0])
 
'''''
class dist:

    def manhattanDistance(data, testData):
        sum = 0
        for i in range(4):
            sum = sum + fabs(data[i] - testData[i])

            #print(i)
            #print(fabs(data[i] - testData[i]))
        return sum



