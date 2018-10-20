from utilities import loadTestingData,loadLearningData
from knn import KNN


def main():
    a = KNN(5,"iris.data.learning", "M")

if __name__ == '__main__':
    main()