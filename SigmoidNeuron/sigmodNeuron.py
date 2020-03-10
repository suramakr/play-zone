from sklearn.model_selection import train_test_split
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


X = [0.5, 2.5]
Y = [0.2, 0.9]


def sigmoid_f(w, b, x):
    # sigmoid with parameters w,b
    return 1.0 / (1.0 + np.exp(-(w*x + b)))


# goes over the data points in all of X, Y
# compute the true value of fn and calculate error loss

def error(w, b):
    error = 0.0
    for x, y in zip(X, Y):
        fx = sigmoid_f(w, b, x)
        err += 0.5 * (fx - y) ** 2
    return err


# function that computes derivative of loss fn w.r.t b
def grad_b(w, b, x, y):
    fx = sigmoid_f(w, b, x)
    return (fx - y) * fx * (1 - fx)

# function that computes derivative of loss fn w.r.t w


def grad_w(w, b, x, y):
    fx = sigmoid_f(w, b, x)
    return (fx - y) * fx * (1 - fx) * x


# main function
def do_gradient_descent():
    # initialize w,b , learning rate as 1.0
    w, b, eta = -2, -2, 1.0
    max_epochs = 1000  # 1000 passes over my data

    # compute derivatives and update weights in each pass
    for i in range(max_epochs):
        dw, db = 0, 0
        for x, y in zip(X, Y):
            dw += grad_w(w, b, x, y)
            db += grad_b(w, b, x, y)
        plt.plot(dw, db)
    w = w - eta * dw
    b = b - eta * db
    plt.show()


plt.plot(X, Y)
plt.show()
do_gradient_descent()
