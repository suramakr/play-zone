from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors
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

w1 = 2
w2 = 0.5
b = 0
Y = sigmoid_2d(XX1, XX2, w1, w2, b)


my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["red", "yellow", "green"])
# Contour plot example
# alpha introduces transparency and grid lines
# faster transition is shown by a thinner transition
plt.contourf(XX1, XX2, Y, cmap=my_cmap, alpha=0.6)
# plt.show()

fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(XX1, XX2, Y, cmap='viridis')
ax.set_xlabel('X1')
ax.set_ylabel('X2')
ax.set_zlabel('Y')


# create a slice of the graph and watch the curve surface as a sigmoid
# ax.view_init(30, 270)

# see dominating aspect of X2
ax.view_init(30, 180)
# plt.show()


# how to computer loss for a given dataset
w_unknown = 0.5
b_unknown = 0.25
X = np.random.random(25) * 20 - 10
Y = sigmoid(X, w_unknown, b_unknown)


def calculate_loss(X, Y, w_est, b_est):
    loss = 0
    for x, y in zip(X, Y):
        loss += (y - sigmoid(x, w_est, b_est)) ** 2
    return loss


# plt.plot(X, Y, '*')
# plt.show()

W = np.linspace(0, 2, 100)
B = np.linspace(-1, 1, 100)
WW, BB = np.meshgrid(W, B)
Loss = np.zeros(WW.shape)
print(WW.shape)
for i in range(WW.shape[0]):
    for j in range(WW.shape[1]):
        Loss[i, j] = calculate_loss(X, Y, WW[i, j], BB[i, j])


# z axis is loss
fig = plt.figure()
ax = plt.axes(projection='3d')
ax.plot_surface(WW, BB, Loss, cmap='viridis')
ax.set_xlabel('w')
ax.set_ylabel('b')
ax.set_zlabel('Loss')
ax.view_init(30, 180)  # see from bias perspective
# plt.show()


# we want to find min w, b
# seeing the graph, we observe -w is bad, so refine the space of W
# change W = np.linspace(-1, 1, 100) to
# W = np.linspace(0, 2, 100)
ij = np.argmin(Loss)
i = int(np.floor(ij/Loss.shape[1]))
j = int(ij-i * Loss.shape[1])

print(i, j)
print(WW[i, j], BB[i, j])


# test standardisation
# plt.close()
plt.clf()
R = np.random.random([100, 1])
print(R.shape)
plt.plot(range(100), R)
plt.show()
print(np.mean(R))
print(np.std(R))

scaler = StandardScaler(copy=True, with_mean=True, with_std=True)
print("StandardScaler")
scaler.fit(R)
print(scaler.mean_)

# Use the scaler to transform any data, having learnt
print("transformation")
RT = scaler.transform(R)
print(np.mean(RT))  # mean is almost 0
print(np.std(RT))  # std is 1
plt.plot(RT)
plt.show()
