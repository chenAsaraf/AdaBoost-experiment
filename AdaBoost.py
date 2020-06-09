import math
import numpy as np
import random
from Rectangles import Rectangle


def read_data(path):
    file = np.loadtxt(path)
    file = np.where(file == 2, -1, file)
    x1 = np.array(file[:, 0])
    x2 = np.array(file[:, 2])
    y = np.array(file[:, 1])
    data = np.vstack((x1, x2, y)).T
    return data


def splitTestTrain(data):
    np.random.shuffle(data)
    train = data[: 65, :]
    test = data[65:, :]
    return train, test


def splitToFeaturesLabels(data):
    X = np.array(data[:, 0:2])
    Y = np.array(data[:, -1])
    return X, Y


def adaBoost(numberOfModels):
    # Initialization of utility variables
    path = 'C:/Users/owner/Desktop/dataset.txt'
    data = read_data(path)
    dataPointsNumber = data.shape[0]

    weights = np.empty(dataPointsNumber)
    weights.fill(1 / dataPointsNumber)

    # final results:
    hypothesis = []

    for i in range(numberOfModels):
        train, test = splitTestTrain(data)
        # print(train)
        trainFeatures, trainLabels = splitToFeaturesLabels(train)
        # print(trainFeatures , trainLabels)
        bestRectangle, minError = Rectangle(trainFeatures, trainLabels, weights)
        print("minError: ", minError)
        new_weights = float(0.5 * np.log((1.0 - minError) / minError))
        print("new_weights: ", new_weights)
        print("------")
        for point in range(trainFeatures.shape[0]):
            if bestRectangle.h(trainFeatures[point]) != trainLabels[point]:
                # if update_weights(bestRectangle,features[point]) != labels[point]:
                weights[point] = weights[point] * np.exp(-new_weights)
            else:
                weights[point] = weights[point] * np.exp(new_weights)
        weights /= np.sum(weights)
    hypothesis.append(bestRectangle)


if __name__ == '__main__':
    r = 8
    for t in range(1, r):
        for i in range(10):
            adaBoost(t)
