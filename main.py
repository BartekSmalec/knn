from knn import KNN
import pandas as pd
import numpy as np



def main():

    ldata = np.array(pd.read_csv("Data/iris.data.learning", header=None))
    tdata = np.array(pd.read_csv("Data/iris.data.test", header=None))


    # print(ldata[:,0])  # print kolumne "0"



    # popraw to a
    a = KNN(2,ldata, "E")
    a.predict(tdata)
    #for i in range (tdata.shape[0]):
        #print(i)
        #a.setA(i)
        #a.predict(tdata)
    print(a.plus + a.minus)
    print(a.plus/(a.plus + a.minus)*100)

if __name__ == '__main__':
    main()

    # 5 virgi 7 veris 2 setosa