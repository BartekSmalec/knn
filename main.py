from knn import KNN
import pandas as pd
import numpy as np



def main():
    ldata = np.array(pd.read_csv("Data/iris.data.learning"))
    tdata = np.array(pd.read_csv("Data/iris.data.test"))
    # print(ldata[:,0])  # print kolumne "0"



    # popraw to a
    a = KNN(2,ldata, "E")
    for i in range (14):
        a.setA(i)
        a.predict(tdata)
    print(a.plus/(a.plus + a.minus)*100)

if __name__ == '__main__':
    main()