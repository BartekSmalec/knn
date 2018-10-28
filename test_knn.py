import unittest

import numpy as np
import pandas as pd

from knn import KNN

try:
    ldata = np.array(pd.read_csv("Data/iris.data.learning", header=None))
    tdata = np.array(pd.read_csv("Data/iris.data.test", header=None))
except:
    print("Bad path to file")

testData, testTarget = np.array_split(tdata, [4], axis=1)


class TestKnn(unittest.TestCase):

    def test_type_constructor(self):
        self.assertRaises(TypeError, lambda: KNN(5.0, [0, 0, 0, 0], 1))
        self.assertRaises(TypeError, lambda: KNN(5j, [0, 0, 0, 0], 22.0))
        self.assertRaises(TypeError, lambda: KNN("M", -444, 22j))
        self.assertRaises(TypeError, lambda: KNN("M", True, 22j))

    def test_value_constructor(self):
        self.assertRaises(ValueError, lambda: KNN(-2, ldata, "P"))
        self.assertRaises(ValueError, lambda: KNN(0, ldata, "C"))
        self.assertRaises(ValueError, lambda: KNN(-555, ldata, "L"))

    def test_type_score(self):
        self.assertRaises(TypeError, lambda: KNN.score(testData, testTarget))
        self.assertRaises(TypeError, lambda: KNN.score(55, 22j))
        self.assertRaises(TypeError, lambda: KNN.score("A", True))
        self.assertRaises(TypeError, lambda: KNN.score(False, "B"))

    def test_type_predict(self):
        self.assertRaises(TypeError, lambda: KNN.predict(55))
        self.assertRaises(TypeError, lambda: KNN.predict(22j))
        self.assertRaises(TypeError, lambda: KNN.predict("A"))
        self.assertRaises(TypeError, lambda: KNN.predict(False))
