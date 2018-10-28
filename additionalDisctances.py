from math import fabs

import numpy as np
from scipy.stats import pearsonr


class dist:

    def manhattanDistance(data, testData):
        if type(data) is not np.ndarray or type(testData) is not np.ndarray:
            raise TypeError("Onlyd numpy.ndarray allowed")

        sum = 0
        for i in range(4):
            sum = sum + fabs(data[i] - testData[i])

        return sum

    def pearson(data, testData):
        if type(data) is not np.ndarray or type(testData) is not np.ndarray:
            raise TypeError("Onlyd numpy.ndarray allowed")

        a, b = pearsonr(data, testData)
        return b
