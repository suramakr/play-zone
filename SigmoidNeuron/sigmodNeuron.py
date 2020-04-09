from sklearn.model_selection import train_test_split
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


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
        return (y_pred - y) * y_pred * (1 - y_pred) * x

    # function that computes derivative of loss fn w.r.t w
    # gradient descent learning algorithm
    def grad_w(self, x, y):
        y_pred = self.sigmoid(self.perceptron(x))
        return (y_pred - y) * y_pred * (1 - y_pred)

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


# linearly seperable data
X = np.asarray([[2.5, 2.5], [4, -1],  [1, -4], [3, 1.25], [2, 4], [1, 5]])
# Ground truth data
Y = [1, 1, 1, 0, 0, 0]


# instantiate a sigmoid neuron
sn = SigmoidNeuron()
sn.fit(X, Y, 1, 0.25, True)
for i in range(10):
    print(sn.w, sn.b)  # weight and bias
    sn.fit(X, Y, 1, 0.25, False)
