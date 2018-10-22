import numpy as np
from scipy.spatial import distance
from additionalDisctances import dist


class KNN:
    def __init__(self, K, ldata, method):
        self.K = K
        self.ldata = ldata
        self.method = method
        self.data, self.target = np.array_split(self.ldata, [4], axis=1)
        self.counter = 0

        self.plus = 0
        self.minus = 0
        self.predictedTarget = []
        # self.predictedTarget = np.empty(self.target.shape[0])

        #print("Method {0}".format(self.method))

    def setIndex(self, a):
        self.index = a

    def predictOne(self, testdata):
        self.testData, self.testTarget = np.array_split(testdata, [4], axis=1)
        # print(self.testData)

        self.distances = []


            #self.counter = self.counter + 1

        if(self.method == "E"):
            for i in range(self.data.shape[0]):
                print("E")
                self.distances.append(distance.euclidean(self.data[i], self.testData[self.index]))
        elif(self.method == "M"):
            for i in range(self.data.shape[0]):
                print("M")
                self.distances.append(dist.manhattanDistance(self.data[i], self.testData[self.index]))
         # print(self.counter, self.distances)
        #print(self.distances)
        # kopiuje całą tabele
        self.ldataLocal = self.ldata
        self.distances = np.asarray(self.distances)

        # dodaje kolumne w z dystansami w ldataLocal
        self.ldataLocal = np.insert(self.ldataLocal, 5, self.distances, axis=1)

        # sortuje cały array wg kolumny 5
        self.ldataLocal = sorted(self.ldataLocal, key=lambda a_entry: a_entry[5])

        # po posortowaniu wybiera K wierszów o najmniejszym dystansie
        kNajblizszychSasiadow = []
        for i in range(self.K):
            kNajblizszychSasiadow.append(self.ldataLocal[i][4])
            # print(kNajblizszychSasiadow[i])

        # z kNajblizszychSasiadow wybiera najczesciej wystepujace
        def most_common(lista):
            return max(set(lista), key=lista.count)

        '''''
        print("------------------------------------")
        print(most_common(kNajblizszychSasiadow))
        print(self.testTarget[self.index])
        print(most_common(kNajblizszychSasiadow) == self.testTarget[self.index] )
        '''''

        # if most_common(kNajblizszychSasiadow) == self.testTarget[self.index]:
        # self.plus = self.plus + 1
        # else:
        # self.minus = self.minus + 1
        # print(self.plus / (self.plus + self.minus) * 100)

        return most_common(kNajblizszychSasiadow)

    def predict(self, testData):
        for i in range(testData.shape[0]):
            # print(i)
            self.setIndex(i)
            # np.append(self.predictedTarget,self.predictOne(testData) )
            self.predictedTarget.append(self.predictOne(testData))
        return self.predictedTarget

    def score(self, tdata, testTarget):
        predictedTarget = self.predict(tdata)
        predictedTarget = np.asarray(predictedTarget)

        #print(predictedTarget.shape)
        #print(self.testTarget.shape)

        # print(predictedTarget[3])
        # print(testTarget[:,0][3])
        for i in range(testTarget.shape[0]):

            print("Predicted: {0}, Correct value: {1}, counter: {2}".format(predictedTarget[i], testTarget[:, 0][i], self.counter))
            self.counter = self.counter + 1
            if predictedTarget[i] == testTarget[:, 0][i]:
                print("true")
                self.plus = self.plus + 1
            else:
                print("false")
                self.minus = self.minus + 1

            print("----------------------------------------------")
        return (self.plus / (self.plus + self.minus) * 100)
