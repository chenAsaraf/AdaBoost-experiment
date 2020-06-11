import numpy as np
from itertools import combinations
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import math

"""
 Set of circles
 for which the inside is **positive** and the outside is negative.
 Circle can be defined by the 2 points- his center and his radius.
 """
class CircleModel:
    """
    Input:
           - Two points defining the circle
           - Data set:
                * X the points - nx2 matrix
                * y the labels of the points
                * weights of the points
    """
    def __init__(self, radius_point, center_point, X, y, w):
        self.center = center_point
        self.radius = self.distance(radius_point,center_point)
        self.data = X
        self.labels = y
        self.weights = w

    """
    Returns the label that the circle gives to the point.
    A point p is inside the circle if the distance between
    p and the center of the circle, is smaller then the radius.  
    """
    def h(self, point):
        dist = self.distance(point, self.center)
        if dist < self.radius:
            return 1
        return -1

    def distance(self,point1, point2):
        diffX = point1[0] - point2[0]
        diffY = point1[1] - point2[1]
        d = math.sqrt(pow(diffX,2) + pow(diffY,2))
        return d

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
            if self.labels[i] == 1: # Positive points:
                plt.scatter(self.data[i, 0], self.data[i, 1], color="green", s=30)
            else:# Negative points:
                plt.scatter(self.data[i, 0], self.data[i, 1], color="blue", s=30)
        ax.grid()
        # Create a Circle patch
        circle = patches.Circle(self.center, self.radius, linewidth=3, edgecolor='r', facecolor='none')
        # Add the patch to the Axes
        ax.add_patch(circle)
        # Add legend to colors:
        green_patch = patches.Patch(color='green', label='Positive points (male)')
        blue_patch = patches.Patch(color='blue', label='Negative points (female)')
        plt.legend(handles=[green_patch, blue_patch])
        plt.show()



class NegativeCircleModel(CircleModel):
    def __init__(self, radius_point, center_point, X, y, w):
        super(NegativeCircleModel, self).__init__(radius_point, center_point, X, y, w)

    """
    Returns the label that the circle gives to the point.
    A point p is inside the circle if the distance between
    p and the center of the circle, is smaller then the radius.  
    """
    def h(self, point):
        dist = self.distance(point, self.center)
        if dist < self.radius:
            return -1
        return 1


"""
Circle function:
    given: * a set of labelled and weighted points,
           * and the hypothesis class of all circles in 2D, for which
             the inside is positive and the outside is negative, or the opposite, 
    finds the circle which minimizes the weighted error on the points.
Algo:
    1. Look for any possible circle:
            by any possible combination of two points,
            which define the radius and the center point of the circle,
            whether within the circle is defined as positive or vice versa.            
    2. For each circle compute the weighted error
    3. Choose the one that has the smallest error  
"""
def Circle(X, y, weights):
    minError = float('inf')
    bestCircle = []
    allPossible = combinations(X, 2)
    for comb in list(allPossible):
        firstPoint = comb[0]
        secondPoint = comb[1]

        # Create circles: first inside positive then inside negative:
        posCircle_1 = CircleModel(firstPoint, secondPoint, X, y, weights)
        posCircle_2 = CircleModel(secondPoint, firstPoint, X, y, weights)
        negCircle_1 = NegativeCircleModel(firstPoint, secondPoint, X, y, weights)
        negCircle_2 = NegativeCircleModel(secondPoint, firstPoint, X, y, weights)
        circles = [posCircle_1, posCircle_2, negCircle_1, negCircle_2]

        # Check the error
        for i in range(len(circles)):
            if minError > circles[i].error():
                minError = circles[i].error()
                bestCircle = circles[i]
    return bestCircle, minError

