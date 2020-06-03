import math

import numpy as np
import Rectangles
def adaBoost():
    # Initialization of utility variables
    path = 'C:/Users/Roi Abramovitch/Downloads/HC_Body_Temperature'
    features, labels, weights = Rectangles.read_data(path)

    # For r = 1 to r
    for m in range(2):
        for i in range(100):
            rec_ht = Rectangles.rectangle(features, labels, weights)
            print(rec_ht[1])
            new_weights = 0.5 * np.log((1.0 - rec_ht[1])/rec_ht[1])
            print(new_weights)
            for point in range(features.shape[0]):
                if update_weights(rec_ht[0],features[point]) != labels[point]:
                    weights[point] = weights[point] * np.exp(-new_weights)
                else:
                    weights[point] = weights[point] * np.exp(new_weights)
            weights /= np.sum(weights)


def update_weights(rectangle, features):
    # if y-coordinate is between the y's coordinates of the rectangle:
    # print(point)
    # print("self.upperLeft[0]", self.upperLeft[0], "self.bottomLeft[0]", self.bottomLeft[0])
    if features[1] < rectangle.upperLeft[1] and features[1] > rectangle.bottomRight[1]:
        # between x-coordinates:
        if features[0] > rectangle.upperLeft[0] and features[0] < rectangle.bottomRight[0]:
            return 1
    return -1

if __name__ == '__main__':
    adaBoost()