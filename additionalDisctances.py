from math import fabs

from scipy.stats import pearsonr


class dist:

    def manhattanDistance(data, testData):
        sum = 0
        for i in range(4):
            sum = sum + fabs(data[i] - testData[i])

            # print(i)
            # print(fabs(data[i] - testData[i]))
        return sum

    def pearson(data, testData):
        a, b = pearsonr(data, testData)
        print(b)
        return b
