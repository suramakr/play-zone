import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def sigmoid(x, w, b):
    return 1/(1 + np.exp(-(w*x + b)))


sigmoid(1, 0.5, 0)

# For many values of X, Y can we generate a graph
w = 0.5
b = -6
# 100 equally spaced numbers between -10 and 10
X = np.linspace(-10, 10, 100)
Y = sigmoid(X, w, b)
print(type(X))
print(type(Y))

# show a sigmoid function
plt.plot(X, Y)
plt.show()
