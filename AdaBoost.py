import math
import numpy as np
import random
from Rectangles import Rectangle
from collections import defaultdict


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

    # Initialize each point weight to be 1 / n:
    weights = np.empty(dataPointsNumber)
    weights.fill(1 / dataPointsNumber)

    # final results containers::
    hypothesis = defaultdict() # key: rectangle_t, value: alpha_t

    for i in range(numberOfModels):
        train, test = splitTestTrain(data)
        trainFeatures, trainLabels = splitToFeaturesLabels(train)
        # Using Rectangle to find a rectangle with minimum weighted error εt
        rectangle_t, error_t = Rectangle(trainFeatures, trainLabels, weights)
        # Rectangle alpha (the weight):
        alpha_t = float(0.5 * np.log((1.0 - error_t) / error_t))
        print("alpha_t: ", alpha_t)
        print("------")
        # Compute new weights for the points:
        for point in range(trainFeatures.shape[0]):
            # For an error on point:
            if rectangle_t.h(trainFeatures[point]) != trainLabels[point]:
                weights[point] = weights[point] * np.exp(-alpha_t)
            else: # Not an error on point:
                weights[point] = weights[point] * np.exp(alpha_t)
        # Normalize these weights:
        weights /= np.sum(weights)
    hypothesis[rectangle_t] = alpha_t



if __name__ == '__main__':
    r = 8
    for i in range(10):
        adaBoost(r)
