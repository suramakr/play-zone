# Finding the right balance in learning rate is important
# Large learning rates - drastic change in decision boundary
# Small learning rates - within epochs can’t find optimal solution

from sklearn.model_selection import train_test_split
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import accuracy_score

my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["red", "yellow", "green"])


class SigmoidNeuron:
    def __init__(self):
        self.w = None
        self.b = None

    def perceptron(self, x):
        return np.dot(x, self.w.T) + self.b

    # gives graded output

    def sigmoid(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    # function that computes derivative of loss fn w.r.t b

    def grad_b(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred)

    # function that computes derivative of loss fn w.r.t w
    # gradient descent learning algorithm
    # y_pred is between -1 to 1, so we need to normalize x, now it is between -10 to 10,  gradients to
    # w are large, so learning rate is small.
    # large learning rates can introduce drastic changes, and too small in the epochs we have
    # might not produce the change we need
    def grad_w(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    def fit(self, X, Y, epochs=1, learning_rate=1, initialize=True):

        # initialize w, b
        if initialize:
            self.w = np.random.randn(1, X.shape[1])
            self.b = 0

        # iterate over the data
        # compute yhat
        # compute loss with w, b
        # if not happy with loss, adjust w, b
        # till satisfied

        for i in range(epochs):
            # compute gradient
            dw = 0
            db = 0
            # go thru inputs
            for x, y in zip(X, Y):
                # derivate of loss function w.r.t w
                dw += self.grad_w(x, y)
                # derivate of loss function w.r.t b
                db += self.grad_b(x, y)
            self.w -= learning_rate * dw
            self.b = learning_rate * db
        return self

# define some plotting functions


def plot_sn(X, Y, sn, ax):
    X1 = np.linspace(-10, 10, 100)
    X2 = np.linspace(-10, 10, 100)
    XX1, XX2 = np.meshgrid(X1, X2)
    YY = np.zeros(XX1.shape)
    # ordering matters X2, X1
    for i in range(X2.size):
        for j in range(X1.size):
            val = np.asarray([X1[j], X2[i]])
            YY[i, j] = sn.sigmoid(sn.perceptron(val))
    ax.contourf(XX1, XX2, YY, cmap=my_cmap, alpha=0.6)
    ax.scatter(X[:, 0], X[:, 1], c=Y)
    ax.plot()
    plt.show()


# linearly seperable data
X = np.asarray([[2.5, 2.5], [4, -1],  [1, -4], [-3, 1.25], [-2, -4], [1, 5]])

# Ground truth data
Y = [1, 1, 1, 0, 0, 0]

# instantiate a sigmoid neuron
sn = SigmoidNeuron()
sn.fit(X, Y, 1, 0.05, True)
N = 15
plt.figure(figsize=(10, N * 5))
plt.show()
for i in range(N):
    print(sn.w, sn.b)  # weight and bias
    # have n linear plots
    ax = plt.subplot(N, 1, i + 1)
    plot_sn(X, Y, sn, ax)
    sn.fit(X, Y, 1, 0.25, False)
# plt.show()
