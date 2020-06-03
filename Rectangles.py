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
           - Data set
     Note: This implementation based on assumption that
           data set is nx4 matrix, where
            -  n is the number of samples
            -  first column is y coordinate
            -  second column is x coordinate
            -  third column is true label of the point
            -  fourth column is point's weight
            -  the points whose real label is positive stored sequentially
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
        if point[1] < self.upperLeft[1] and point[1] > self.bottomLeft[1]:
            # between x-coordinates:
            if point[0] < self.upperLeft[0] and point[0] > self.bottomLeft[0]:
                return 1
        return -1

    """
    Error:
    Sum of weights of wrongly placed points.
    """
    def error(self):
        sum = 0
        for point in range(self.data.shape[0]):
            if self.h(self.data[point]) != self.labels[point]:
                sum = sum + weights[point]
        return sum


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
        # rectangle.draw()
        # Check the error
        newError = rectangle.error()
        if minError > newError:
            minError = newError
            bestRectangle = rectangle
    bestRectangle.draw()
    return bestRectangle


# TODO: need to be in AdaBoost file 
def read_data(file):
    file = np.loadtxt('C:/Users/owner/Desktop/dataset.txt')
    file = np.where(file == 2, -1, file)
    x1 = np.array(file[:, 0])
    x2 = np.array(file[:, 2])
    weights = np.empty(x1.shape[0])
    weights.fill(1/x1.shape[0])
    y = np.array(file[:, 1])
    X = np.vstack((x1, x2)).T
    print(X.shape)
    return X , y, weights


# --------------------------------- #
# Demo:

# Demo Data:
#order of columns:x  y label weight
data = np.array([[0,  0,  1, 1],
                 [0,  1,  1, 1],
                 [0.2,0.5,1, 1],
                 [1,  0, -1, 1],
                 [1,  1, -1, 1],
                 [-1, 0, -1, 1]])



features, labels, weights = read_data(data)
Rectangle(features, labels, weights )
#print(np.arange(0,3))