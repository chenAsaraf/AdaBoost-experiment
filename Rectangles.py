import numpy as np
from itertools import combinations
import matplotlib.patches as patches
import matplotlib.pyplot as plt

"""
 Set of axis-parallel rectangles
 for which the inside is positive and the outside is negative.
 Rectangle can c by the 2 points on its diagonal.
 """
class RectangleModel:

    """
    Input:
           - Two points defining the rectangle diagonal
           - Data set:
                * X the points - nx2 matrix
                * y the labels of the points
                * weights of the points
     Note: This implementation based on assumption that
            the points whose real label is positive stored sequentially
            from the beginning of the data-list.
    """
    def __init__(self, upper, bottom, X, y, weights):
        self.data = X
        self.labels = y
        self.weights = weights
        self.__initPoints(upper, bottom)
        self.truePositive = positives(y)


    # initialize rectangle borders
    def __initPoints(self,upper,bottom):
        if upper[0] <= bottom[0]: # if x coordinate is smaller in upper point
            self.upperLeft = upper
            self.bottomRight = bottom
            self.upperRight = [bottom[0], upper[1]]
            self.bottomLeft = [upper[0], bottom[1]]
        else:
            self.upperRight = upper
            self.bottomLeft = bottom
            self.upperLeft = [bottom[0], upper[1]]
            self.bottomRight = [upper[0], bottom[1]]



    # Returns the lable that the rectangle gives to the point
    def h(self,point):
        # if y-coordinate is between the y's coordinates of the rectangle:
        #print(point)
        #print("self.upperLeft[0]", self.upperLeft[0], "self.bottomLeft[0]", self.bottomLeft[0])
        if point[1] < self.upperLeft[1] and point[1] > self.bottomRight[1]:
            # between x-coordinates:
            if point[0] > self.upperLeft[0] and point[0] < self.bottomRight[0]:
                return 1
        return -1


    # Error function:
    # returns the average sum of weights of wrongly placed points.
    def error(self):
        sum = 0
        for point in range(self.data.shape[0]):
            # if h(x) != y:
            if self.h(self.data[point]) != self.labels[point]:
                sum = sum + weights[point]
        numberOfSamples = self.data.shape[0]
        return sum/numberOfSamples


    # Draw the points and the rectangle
    def draw(self):
        fig, ax = plt.subplots()
        fig = plt.gcf()
        ax = fig.gca()
        # Positive points:
        for i in range(self.truePositive):
            plt.scatter(self.data[i, 0], self.data[i, 1], color="green", s=30)
        # Negative points:
        for i in range(self.data.shape[0] - self.truePositive):
            plt.scatter(self.data[self.truePositive + i, 0], self.data[self.truePositive + i, 1], color="blue", s=30)
        ax.grid()
        # Create a Rectangle patch
        width = self.bottomRight[0] - self.bottomLeft[0]
        height = self.upperLeft[1] - self.bottomLeft[1]
        rect = patches.Rectangle(self.bottomLeft, width, height, linewidth=3, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(rect)
        # Add legend to colors:
        green_patch = patches.Patch(color='green', label='Positive points (male)')
        blue_patch = patches.Patch(color='blue', label='Negative points (female)')
        plt.legend(handles=[green_patch, blue_patch])
        plt.show()


# Returns the number of points whose real label is positive
def positives(labels):
    quantity = 0
    for index in range(labels.shape[0]):
        if labels[index] == 1:
            quantity += 1
        else:
            break
    return quantity

"""
Rectangle function:
    given: * a set of labelled and weighted points,
           * and the hypothesis class of all axis-parallel rectangles for which
            the inside is positive and the outside is negative,
    finds the rectangle which minimizes the weighted error on the points.
Algo:
    1. Look for any possible rectangle:
            by any possible combination of two points labeled positive,
            which define the diagonal of the rectangle.
    2. For each rectangle compute the weighted error
    3. Choose the one that has the smallest error  
"""
def Rectangle(X , y , weights):
    positiveQuantity = positives(y)
    minError = float('inf')
    bestRectangle = []
    allPossible = combinations(X[:positiveQuantity], 2)
    for comb in list(allPossible):
        firstPoint = comb[0]
        secondPoint = comb[1]
        # Create rectangle
        if firstPoint[1] < secondPoint[1]:
            rectangle = RectangleModel(secondPoint, firstPoint, X, y, weights)
        else:
            rectangle = RectangleModel(firstPoint, secondPoint, X, y, weights)
        #rectangle.draw()
        # Check the error
        newError = rectangle.error()
        if minError > newError:
            minError = newError
            bestRectangle = rectangle
    bestRectangle.draw()
    return bestRectangle


# TODO: need to be in AdaBoost file 
def read_data(path):
    file = np.loadtxt(path)
    file = np.where(file == 2, -1, file)
    x1 = np.array(file[:, 0])
    x2 = np.array(file[:, 2])
    weights = np.empty(x1.shape[0])
    weights.fill(1/x1.shape[0])
    y = np.array(file[:, 1])
    X = np.vstack((x1, x2)).T
    return X , y, weights


# --------------------------------- #
# Demo:

path= 'C:/Users/owner/Desktop/dataset.txt'
features, labels, weights = read_data(path)
Rectangle(features, labels, weights )
