import numpy as np
import pandas as pd

from knn import KNN


def main():
    ldata = np.array(pd.read_csv("Data/iris.data.learning", header=None))
    tdata = np.array(pd.read_csv("Data/iris.data.test", header=None))

    testData, testTarget = np.array_split(tdata, [4], axis=1)

    a = KNN(2, ldata, "E")

    #print(a.predict(tdata))
    print(a.score(testData, testTarget))
    # a.score(tdata,testTarget)


if __name__ == '__main__':
    main()
