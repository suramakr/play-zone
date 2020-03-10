from sklearn.model_selection import train_test_split
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# for animation
from matplotlib import animation, rc
from IPython.display import HTML

# animation function. This is called sequentially


def animate(i):
    # list of range of wt_matrix 1,2,3....columns
    x = list(range(wt_matrix.shape[1]))
    # ith row all possible values
    y = wt_matrix[i, :]
    line.set_data(x, y)
    return (line,)


class Perceptron:

    def __init__(self):
        self.b = None  # b will be a scalar
        self.w = None  # w will be an array

    # return 1 if >= b, else 0
    # given x compute y
    def model(self, x):
        return 1 if np.dot(self.w, x) >= self.b else 0

    # Train and Test vectors, predict across them
    # Take a vector of X and return a vector of Y
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)  # convert list to array

    # learning algorithm to find value of b
    # we have to give the training data
    def fit(self, X, Y, epochs=1, lr=1):
        # we can't have a loop for b like we did for MP Neuron
        # since we have real values of b and infinite values for w
        wt_matrix = []

        # 0 gives u rows, 1 columns
        self.w = np.ones(X.shape[1])
        self.b = 0

        accuracy = {}  # a dictionary
        max_accuracy = 0

        # a list to be updated in end of every epoch
        # wt_matrix = []

        # we also now introduce a concept called learning rate via lr

        # how we look at the data to start this counter of incrementation
        # impacts accuracy of the algm. Look at the first element of X
        # Iterate till we achieve convergence, Parameter space w 30, b 1=31 dimensions
        # Epoch we have is just 1.
        for i in range(epochs):
            for x, y in zip(X, Y):
                y_pred = self.model(x)  # ground truth in y
                if y == 1 and y_pred == 0:
                    # if there is difference
                    self.w = self.w + lr * x
                    self.b = self.b + lr * 1
                elif y == 0 and y_pred == 1:
                    self.w = self.w - lr * x
                    self.b = self.b - lr * 1
                    # after each epoch
            wt_matrix.append(self.w)

            accuracy[i] = accuracy_score(self.predict(X), Y)
            if (accuracy[i] > max_accuracy):
                max_accuracy = accuracy[i]
                # we keep over-writing when accuracy increases
                chkptw = self.w
                chkptb = self.b

        # later lets reset with the higest checkpoint value we found
        # maximizing training accuracy, since we don't have access to test data
        self.w = chkptw
        self.b = chkptb

        # np.array(d.values()).astype(float)
        # since it is dict get values
        # Convert the dict_values object to list first so the array is created from the content of the list.
        # NumPy cannot build an array of the contained items from dict_values, it instead creates an array
        # of type object and puts the dict_values object inside:
        #   plt.plot(accuracy.values())  this doesn't work
        print('max accuracy is ')
        print(max_accuracy)
        plt.ylim([0, 1])
        plt.plot(np.array(list(accuracy.values())).astype(float))
        # plt.show()

        # instead of list return nparray
        # each row is one set of weights
        return np.array(wt_matrix)


# Load data
breast_cancer = sklearn.datasets.load_breast_cancer()
X = breast_cancer.data
Y = breast_cancer.target

data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)
# add a column to a pandas data frame
data['class'] = breast_cancer.target

# Initialize X, Y
X = data.drop('class', axis=1)
Y = data['class']

# Get data ready
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1)
X_train = X_train.values  # converting to numpy Arrays
X_test = X_test.values

# Use of class, train model
perceptron = Perceptron()
# increase the epochs increases the accuracy, and oscillation happens
# 20 is a HYPER parameter, and it impacts the accuracy
# and there is still oscilaation even after 20, for e.g. with 50
# you will see slightly better numbers.
# perceptron.fit(X_train, Y_train, 100)
# make very small changes using learning rate of .0001
# Hyper parameter Tuning of Epoch, Checkpointing
# few iteration for aninmation
wt_matrix = perceptron.fit(X_train, Y_train, 100)
# later we can use a grid system to investigate the epoch vs checkpointing to get
# the best value

# plt.plot(perceptron.w)
# plt.show()

# ANIMATE
# First set up the figure, the axis, and the plot element we want to animate
fig, ax = plt.subplots()

# initialize for animation
# no need for plot handles
# number of columns
ax.set_xlim((0, wt_matrix.shape[1]))
print('Shape is ')
print(wt_matrix.shape[1])

# plot the weight matrix and look at the yaxis labels to get limit values
ax.set_ylim((-10000, 25000))
line, = ax.plot([], [], lw=2)


# call the animator. blit=True means only re-draw the parts that have changed.
# As the number of epochs goes up, certain features start going up in weight and certain negatively contribute. Over time when the epocs number is very high, and learning rate is lowered also, we start seeing problems of OVERFITTING!!!

anim = animation.FuncAnimation(
    fig, animate, frames=100, interval=20, blit=True)
plt.show()

# Training accuracy
Y_train_pred = perceptron.predict(X_train)
accuracy_train = accuracy_score(Y_train_pred, Y_train)
print("Accuracy of training:")
print(accuracy_train)

# Test the model and find accuracy
Y_test_pred = perceptron.predict(X_test)
accuracy_test = accuracy_score(Y_test_pred, Y_test)
print("Accuracy of test:")
print(accuracy_test)
