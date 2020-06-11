
import numpy as np
from Rectangles import Rectangle
from Circles import Circle
import matplotlib.patches as patches
import matplotlib.pyplot as plt

"""
Read data from Hope College data set Temperature.
Note: The first column is Body temperature in degrees Fahrenheit,
      the second is Gender (1 = male, 2 = female)
      and the third is Heart rate in beats per minute
"""


def read_data(path):
    file = np.loadtxt(path)
    file = np.where(file == 2, -1, file)
    x1 = np.array(file[:, 0])
    x2 = np.array(file[:, 2])
    y = np.array(file[:, 1])
    data = np.vstack((x1, x2, y)).T
    return data


"""
In the experiment we randomly divide the data-points for each iteration 
into 65 training points and 65 test points
"""


def splitTestTrain(data):
    np.random.shuffle(data)
    train = data[: 65, :]
    test = data[66:, :]
    return train, test


"""
Split into 2 columns of features (Body temperature, Heart rate)
and one vector of labels (gender)
"""


def splitToFeaturesLabels(data):
    X = np.array(data[:, 0:2])
    Y = np.array(data[:, -1])
    return X, Y


"""
Draw the points and the model
"""


def draw(modelType, trainFeatures, trainLabels, model):
    fig, ax = plt.subplots()
    fig = plt.gcf()
    ax = fig.gca()

    # Draw all the shape in model list:
    if modelType == 'Rectangle':  # Create a Rectangle patch
        for h in range(len(model)):
            width = model[h].bottomRight[0] - model[h].bottomLeft[0]
            height = model[h].upperLeft[1] - model[h].bottomLeft[1]
            rect = patches.Rectangle(model[h].bottomLeft, width, height, linewidth=3, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(rect)
    elif modelType == 'Circle':  # Create a Circle patch
        for h in range(len(model)):
            circle = patches.Circle(model[h].center, model[h].radius, linewidth=3, edgecolor='r', facecolor='none')
            # Add the patch to the Axes
            ax.add_patch(circle)

    # Draw points:
    for i in range(trainFeatures.shape[0]):
        if trainLabels[i] == 1:  # Positive points:
            plt.scatter(trainFeatures[i, 0], trainFeatures[i, 1], color="green", s=30)
        else:  # Negative points:
            plt.scatter(trainFeatures[i, 0], trainFeatures[i, 1], color="blue", s=30)
        ax.grid()

    # Add legend to colors:
    green_patch = patches.Patch(color='green', label='Positive points (male)')
    blue_patch = patches.Patch(color='blue', label='Negative points (female)')
    plt.legend(handles=[green_patch, blue_patch])
    plt.show()


"""
Returns the model minimizes the weighted error on the points,
          model type is circle or rectangle.
"""


def model(trainFeatures, trainLabels, weights, modelType):
    if modelType == 'Rectangle':
        return Rectangle(trainFeatures, trainLabels, weights)
    elif modelType == 'Circle':
        return Circle(trainFeatures, trainLabels, weights)


"""
AdaBoost main function:
1. Initialize each point weight to be 1/n:  D0(xi) = 1/n 
2. For round t=1,…,r
    a. Use Rectangle to find a rectangle with minimum weighted error εt
        and call this rectangle ht 
    b. Compute the weight    άt = (1/2) ln ((1- εt)/εt) 
    c. Compute new weights for the points: 
        i. For an error on point xi:  Dt(xi) = Dt-1(xi) exp(άt) 
        ii. Not an error on point xi:  Dt(xi) = Dt-1(xi) exp(-άt) 
    d. Normalize these weights:   Dt(xi) = Dt(xi) / ∑j Dt(xj) 
"""


def adaBoost(trainFeatures, trainLabels, numberOfModels, modelType):
    dataPointsNumber = trainFeatures.shape[0]
    # Initialize each point weight to be 1 / n:
    weights = np.empty(dataPointsNumber)
    weights.fill(1 / dataPointsNumber)

    # final results containers::
    hypothesis = []
    alphas = []

    for i in range(numberOfModels):
        # Using Rectangle to find a rectangle with minimum weighted error εt
        h_t, error_t = model(trainFeatures, trainLabels, weights, modelType)
        # Rectangle alpha (the weight):
        alpha_t = 0.5 * float(np.log((1.0 - error_t) / error_t))
        # Compute new weights for the points:
        for point in range(trainFeatures.shape[0]):
            # For an error on point:
            if h_t.h(trainFeatures[point]) != trainLabels[point]:
                weights[point] = weights[point] * np.exp(alpha_t)
            else:  # Not an error on point:
                weights[point] = weights[point] * np.exp(-alpha_t)
        # Normalize these weights:
        weights /= np.sum(weights)
        hypothesis.append(h_t)
        alphas.append(alpha_t)
    return hypothesis, alphas


"""
AdaBoost experiment:
We are running the adaBoost algorithm 100 times for each of r=1,…,8.
For each run, randomly dividing the points into 50% training points R and 50% test points T.
Then run AdaBoost on R, and after computing the final hypothesis, find its error T.
We will use the HC Temperature data set. It contains 130 data points. The label (1 and -1)
will be the gender, and the temperature and heartrate define the 2-dimensional point.
"""


def adaBoostExperiment(R, iteration, data, modelType):
    for r in range(1, R):
        errorTest = 0.0
        errorTrain = 0.0
        for i in range(iteration):
            train, test = splitTestTrain(data)
            trainFeatures, trainLabels = splitToFeaturesLabels(train)
            testFeatures, testLabels = splitToFeaturesLabels(test)
            H, alphas = adaBoost(trainFeatures, trainLabels, r, modelType)
            # Find test - error
            for point in range(testFeatures.shape[0]):
                sign = 0
                for index_t in range(len(H)):
                    sign += alphas[index_t] * H[index_t].h(testFeatures[point])
                errorTest += int(sign * testLabels[point] < 0)
            # Find train - error
            for point in range(trainFeatures.shape[0]):
                sign = 0
                for index_t in range(len(H)):
                    sign += alphas[index_t] * H[index_t].h(trainFeatures[point])
                errorTrain += int(sign * trainLabels[point] < 0)
            # draw
            # draw(modelType, trainFeatures, trainLabels, H)
        print('r:', r, 'test average error:', (errorTest / iteration) / 65, 'train  average error:',
              (errorTrain / iteration) / 65)


if __name__ == '__main__':
    # initiailize data set
    path = 'C:/Users/owner/Desktop/dataset.txt'
    data = read_data(path)

    # Run the experiment:
    # for 0-R number of models, find the best number for test accuracy
    R = 9
    iteration = 100
    print('-----------')
    print('Rectangles: ')
    adaBoostExperiment(R, iteration, data, 'Rectangle')
    print('-----------')
    print('Circles: ')
    adaBoostExperiment(R, iteration, data, 'Circle')
