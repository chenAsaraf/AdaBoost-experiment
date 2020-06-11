import numpy as np
from itertools import combinations
import matplotlib.patches as patches
import matplotlib.pyplot as plt

"""
 Set of axis-parallel rectangles
 for which the inside is *positive* and the outside is negative.
 Rectangle can defined by the 2 points on its diagonal.
 """


class RectangleModel:
    """
    Input:
           - Two points defining the rectangle diagonal
           - Data set:
                * X the points - nx2 matrix
                * y the labels of the points
                * weights of the points
    """

    def _init_(self, firstPoint, secondPoint, X, y, w):
        self.data = X
        self.labels = y
        self.weights = w
        self.__initPoints(firstPoint, secondPoint)

    # initialize rectangle borders
    def __initPoints(self, firstPoint, secondPoint):
        up = 0
        down = 0
        left = 0
        right = 0
        if firstPoint[0] <= secondPoint[0]:  # if x coordinate is smaller in first point
            right = secondPoint[0]
            left = firstPoint[0]
        if firstPoint[0] > secondPoint[0]:
            right = firstPoint[0]
            left = secondPoint[0]
        if firstPoint[1] <= secondPoint[1]:  # if y coordinate is smaller in first point
            up = secondPoint[1]
            down = firstPoint[1]
        else:
            up = firstPoint[1]
            down = secondPoint[1]
        self.upperLeft = [left, up]
        self.upperRight = [right, up]
        self.bottomRight = [right, down]
        self.bottomLeft = [left, down]

    # Returns the lable that the rectangle gives to the point
    def h(self, point):
        # if y-coordinate is between the y's coordinates of the rectangle:
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
                sum = sum + self.weights[point]
        return sum

    # Draw the points and the rectangle
    def draw(self):
        fig, ax = plt.subplots()
        fig = plt.gcf()
        ax = fig.gca()
        for i in range(self.data.shape[0]):
            if self.labels[i] == 1:  # Positive points:
                plt.scatter(self.data[i, 0], self.data[i, 1], color="green", s=30)
            else:  # Negative points:
                plt.scatter(self.data[i, 0], self.data[i, 1], color="blue", s=30)
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


"""
 Set of axis-parallel rectangles
 for which the inside is *negative* and the outside is positive.
 Rectangle can defined by the 2 points on its diagonal.
 """


class NegativeRectangleModel(RectangleModel):
    def _init_(self, firstPoint, secondPoint, X, y, w):
        super(NegativeRectangleModel, self)._init_(firstPoint, secondPoint, X, y, w)

    # Returns the lable that the rectangle gives to the point
    def h(self, point):
        # if y-coordinate is between the y's coordinates of the rectangle:
        if point[1] < self.upperLeft[1] and point[1] > self.bottomRight[1]:
            # between x-coordinates:
            if point[0] > self.upperLeft[0] and point[0] < self.bottomRight[0]:
                return -1
        return 1


"""
Rectangle function:
    given: * a set of labelled and weighted points,
           * and the hypothesis class of all axis-parallel rectangles for which
            the inside is positive and the outside is negative, or the opposite,
    finds the rectangle which minimizes the weighted error on the points.
Algo:
    1. Look for any possible rectangle:
            by any possible combination of two points,
            which define the diagonal of the rectangle,
            whether within the rectangle is defined as positive or vice versa.
    2. For each rectangle compute the weighted error
    3. Choose the one that has the smallest error  
"""


def Rectangle(X, y, weights):
    minError = float('inf')
    bestRectangle = []
    allPossible = combinations(X, 2)
    for comb in list(allPossible):
        firstPoint = comb[0]
        secondPoint = comb[1]
        # Create rectangles: first inside positive then inside negative:
        negRectangle = NegativeRectangleModel(firstPoint, secondPoint, X, y, weights)
        posRectangle = RectangleModel(firstPoint, secondPoint, X, y, weights)
        # Check the error
        negError = negRectangle.error()
        posError = posRectangle.error()
        if minError > negError:
            minError = negError
            bestRectangle = negRectangle
        if minError > posError:
            minError = posError
            bestRectangle = posRectangle
    return bestRectangle, minError
