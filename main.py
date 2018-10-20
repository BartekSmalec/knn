from knn import KNN
import pandas as pd
import numpy as np



def main():
    ldata = np.array(pd.read_csv("Data/iris.data.learning"))
    tdata = np.array(pd.read_csv("Data/iris.data.test"))
    # print(ldata[:,0])  # print kolumne "0"


    a = KNN(5,ldata, "E")
    a.predict(tdata)

if __name__ == '__main__':
    main()