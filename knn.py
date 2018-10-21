import pandas as pd
import numpy as np
from scipy.spatial import distance
from heapq import nsmallest

class KNN:
    def __init__(self, K, ldata, method):
        self.K = K
        self.ldata = ldata
        self.method = method
        self.data, self.target = np.array_split(self.ldata, [4], axis=1)
        self.counter = 0
        #self.a = 10
        self.plus = 0
        self.minus = 0

    def setA(self,a):
        self.a = a





    def predict(self, testdata):
        self.testData, self.testTarget = np.array_split(testdata, [4], axis=1)
        #print(self.testData)


        '''''
        self.testData = sorted(self.testData, key=lambda a_entry: a_entry[0])
        print("----------------------------------------------------")
        print(self.testData)
        
        sortowanie wg 1 kolumny
        
        '''''

        self.distances = []




        ''''
        for i in range(self.data.shape[0]):
            for j in range(self.testData.shape[0]):

                self.distances = distance.euclidean(self.data[i],self.testData[j])
                self.counter = self.counter + 1
                print("{0} --- {1} ---- {2} ---- {3}".format(self.counter, self.data[i],self.testData[j],self.distances))
        '''''



        for i in range(self.data.shape[0]):
            self.counter = self.counter + 1

            self.distances.append(distance.euclidean(self.data[i], self.testData[self.a]))

            #print(self.counter, self.distances)


        #kopiuje całą tabele
        self.ldataLocal = self.ldata
        self.distances =  np.asarray(self.distances)

        #dodaje kolumne w z dystansami w ldataLocal
        self.ldataLocal = np.insert(self.ldataLocal,5,self.distances,axis=1)

        #sortuje cały array wg kolumny 5
        self.ldataLocal = sorted(self.ldataLocal, key=lambda a_entry: a_entry[5])

        #po posortowaniu wybiera K wierszów o najmniejszym dystansie
        kNajblizszychSasiadow = []
        for i in range(self.K):
            kNajblizszychSasiadow.append(self.ldataLocal[i][4])
            #print(kNajblizszychSasiadow[i])


        def most_common(lista):
            return max(set(lista), key=lista.count)

        print("------------------------------------")
        print(most_common(kNajblizszychSasiadow))
        print(self.testTarget[self.a])
        print(most_common(kNajblizszychSasiadow) == self.testTarget[self.a] )
        if most_common(kNajblizszychSasiadow) == self.testTarget[self.a]:
            self.plus = self.plus + 1
        else:
            self.minus = self.minus + 1








    def score(self, tdata):
        print("b")


