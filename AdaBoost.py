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


def adaBoost(trainFeatures, trainLabels, numberOfModels):

    dataPointsNumber = trainFeatures.shape[0]
    # Initialize each point weight to be 1 / n:
    weights = np.empty(dataPointsNumber)
    weights.fill(1 / dataPointsNumber)

    # final results containers::
    hypothesis = []
    alphas = []

    for i in range(numberOfModels):

        # Using Rectangle to find a rectangle with minimum weighted error εt
        rectangle_t, error_t = Rectangle(trainFeatures, trainLabels, weights)
        # Rectangle alpha (the weight):
        alpha_t = float(0.5 * np.log((1.0 - error_t) / error_t))
        #print("alpha_t: ", alpha_t)
        #print("------")
        # Compute new weights for the points:
        for point in range(trainFeatures.shape[0]):
            # For an error on point:
            if rectangle_t.h(trainFeatures[point]) != trainLabels[point]:
                weights[point] = weights[point] * np.exp(-alpha_t)
            else: # Not an error on point:
                weights[point] = weights[point] * np.exp(alpha_t)
        # Normalize these weights:
        weights /= np.sum(weights)
        hypothesis.append(rectangle_t)
        alphas.append(alpha_t)

    return hypothesis, alphas


if __name__ == '__main__':

    path = 'C:/Users/Roi Abramovitch/Downloads/לימודים מדעי המחשב/שנה ג/למידת מכונה/מטלות/מטלה 2/HC_Body_Temperature'
    data = read_data(path)
    dataPointsNumber = data.shape[0]

    R = 9
    iteration = 10
    for r in range(1, R):
        errorTest = 0.0
        errorTrain = 0.0
        for i in range(iteration):
            train, test = splitTestTrain(data)
            trainFeatures, trainLabels = splitToFeaturesLabels(train)
            testFeatures, testLabels = splitToFeaturesLabels(test)
            H, alphas = adaBoost(trainFeatures, trainLabels, r)

            for point in range(testFeatures.shape[0]):
                sign = 0
                for index_t in range(len(H)):
                    sign += alphas[index_t] * H[index_t].h(testFeatures[point])
                errorTest += int(sign * testLabels[point] < 0)
            for point in range(trainFeatures.shape[0]):
                sign = 0
                for index_t in range(len(H)):
                    sign += alphas[index_t] * H[index_t].h(trainFeatures[point])
                errorTrain += int(sign * trainLabels[point] < 0)
        print('r:', r, 'average error:',  (errorTest / iteration)/65)
        print('r:', r, 'average error:', (errorTrain / iteration) / 65)


