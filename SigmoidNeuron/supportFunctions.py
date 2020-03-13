import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d


def sigmoid(x, w, b):
    return 1/(1 + np.exp(-(w*x + b)))


sigmoid(1, 0.5, 0)

# For many values of X, Y can we generate a graph
# rate of fall of the sigmoid function
w = 0.5
# position of fall w.r.t 0
b = -6
# 100 equally spaced numbers between -10 and 10
X = np.linspace(-10, 10, 100)
Y = sigmoid(X, w, b)
print(type(X))
print(type(Y))

# show a sigmoid function
# plt.plot(X, Y)
# plt.show()

# 2d Graph


def sigmoid_2d(x1, x2, w1, w2, b):
    return 1/(1 + np.exp(-(w1*x1 + w2*x2 + b)))


print(sigmoid_2d(1, 0, 0.5, 0, 0))


# 3d graph
# https://jakevdp.github.io/PythonDataScienceHandbook/04.12-three-dimensional-plotting.html
X1 = np.linspace(-10, 10, 100)
X2 = np.linspace(-10, 10, 100)

XX1, XX2 = np.meshgrid(X1, X2)
print(X1.shape, X2.shape, XX1.shape, XX2.shape)

w1 = 0.5
w2 = 0.5
b = 0
Y = sigmoid_2d(XX1, XX2, w1, w2, b)

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(XX1, XX2, Y, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')
# plt.show()

# create a slice of the graph and watch the curve surface as a sigmoid
ax.view_init(30, 270)
fig
plt.show()
