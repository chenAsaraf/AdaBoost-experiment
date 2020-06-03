import numpy as np
from itertools import combinations

file = np.loadtxt('C:/Users/Roi Abramovitch/Downloads/HC_Body_Temperature')

file = np.where(file == 2, -1, file)

x1 = np.array(file[:, 0])
x2 = np.array(file[:, 2])
y = np.array(file[:, 1])
X = np.vstack((x1, x2)).T
com = combinations(X,2)
for i in list(com):
    print(i[0][1])
