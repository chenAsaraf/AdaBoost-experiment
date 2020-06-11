# AdaBoost Algorithm


### About the project:
In this project, we implemented AdaBoost algorithm and experimented to find the ideal number of models.
The dataset is the Hope College Temperature data set.
The data described by body temperature in degrees Fahrenheit, the gender (1 = male, 2 = female) and the heart rate in beats per minute.

We changed the gender label  into points when men are +1 and women are -1.
The hypothesis models in the experiment are rectangles and circles. 

We are running the adaBoost algorithm 100 times for each of r=1,…,8.
For each run, randomly dividing the points into 50% training points R and 50% test points T.
Then run AdaBoost on R, and after computing the final hypothesis, find its error T.
Dataset contains 130 data points. The label (1 and -1) will be the gender, and the temperature and heartrate define the 2-dimensional point.

## Overfitting:


### Exemples:

![pic](https://user-images.githubusercontent.com/44756354/84368978-e74e9e00-abde-11ea-98e3-021cfc5f1280.png)

###
Chen Asaraf

Roi Abramovitch
