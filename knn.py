import pandas as pd
import numpy as np
from scipy.spatial import distance

class KNN:
    def __init__(self, K, ldata, method):
        self.K = K
        self.ldata = ldata
        self.method = method
        self.data, self.target = np.array_split(self.ldata, [4], axis=1)
        self.counter = 1




    def predict(self, testdata):
        self.testData, self.testTarget = np.array_split(testdata, [4], axis=1)
        #print(self.testData)

        for i in range(self.data.shape[0]):
            for j in range(self.testData.shape[0]):

                self.distances = distance.euclidean(self.data[i],self.testData[j])
                self.counter = self.counter + 1
                print("{0} --- {1} ---- {2} ---- {3}".format(self.counter, self.data[i],self.testData[j],self.distances))


    def score(self, tdata):
        print("b")


