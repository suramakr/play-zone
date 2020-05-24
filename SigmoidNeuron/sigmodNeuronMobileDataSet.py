from sklearn.model_selection import train_test_split
import sklearn.datasets
import numpy as np
# data structures for manipulating tables and time series
import pandas as pd
import matplotlib.colors
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split

my_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "", ["red", "yellow", "green"])


class SigmoidNeuron:
    def __init__(self):
        self.w = None
        self.b = None

    # compute output
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

    # iterating epocs, updating w, b and applying gradients, with learning rate param,
    # initialize if you do not want to train from beginning set initialize=False
    def fit(self, X, Y, epochs=1, learning_rate=1, initialize=True, display_loss=False):

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

    # for a vector X input, product Y_pred
    def predict(self, X):
        Y_pred = []
        for x in X:
            y_pred = self.sigmoid(self.perceptron(x))
            Y_pred.append(y_pred)
        return np.array(Y_pred)


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
    # plt.show()


# Test Sigmoid Function before actual use with a test dataset
# linearly seperable data
X = np.asarray([[2.5, 2.5], [4, -1],  [1, -4], [-3, 1.25], [-2, -4], [1, 5]])

# Ground truth data
Y = [1, 1, 1, 0, 0, 0]

# instantiate a sigmoid neuron
sn = SigmoidNeuron()
sn.fit(X, Y, 1, 0.05, True)
N = 15
plt.figure(figsize=(10, N * 5))
# plt.show()
for i in range(N):
    print(sn.w, sn.b)  # weight and bias
    # have n linear plots
    ax = plt.subplot(N, 1, i + 1)
    plot_sn(X, Y, sn, ax)
    sn.fit(X, Y, 1, 0.25, False)
# plt.show()

# STEP 1: DATA PREPARATION

# load cleaned data set 341 rows i.e phones and 88 features
path = '/Users/sramakrishnan/work/python/play-zone/SigmoidNeuron'
data = pd.read_csv(path + '/mobile_cleaned.csv')

# first understand shape, rows and features
print(data.shape)

# drop column
X = data.drop('Rating', axis=1)
print(X.shape)

# convert to values for easier processing, Y.head() to inspect
# convert DataFrame which has columns to a numpyArray
Y = data['Rating'].values
print(Y.shape)

# real value Y is used for training
# compute accuracy on the binarized value of Y
threshold = 4
data['Class'] = (data['Rating'] >= threshold).astype(np.int)

# check if the data is skewed
print(data['Class'].value_counts(normalize=True))
# you will note 1 has 238, and 0 has only 103, many more 1s than 0s
# use normalize to see in percentage

# How to solve this? For binary classification, baseline becomes the class that is larger
# We ideally want 0.5, 0.5 spread in "Y" for linear classification
# Increase threadhold = 4.1]
threshold = 4.2
data.drop('Class', axis=1)
data['Class'] = (data['Rating'] >= threshold).astype(np.int)
print(data['Class'].value_counts(normalize=True))

Y_binarised = data['Class'].values

# STEP 2: Standardization
#
# stratify ensure similar ratio of split of Y_binarised in both
# test and train dataset
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, random_state=0, stratify=Y_binarised)

# standardize column input features X after split
scaler = StandardScaler()
X_scaled_train = scaler.fit_transform(X_train)
X_scaled_test = scaler.transform(X_test)

# standardize Y that matches sigmoid output
minmax_scaler = MinMaxScaler()
Y_scaled_train = minmax_scaler.fit_transform(Y_train.reshape(-1, 1))

# for accuracy computation we have to do only transform
Y_scaled_test = minmax_scaler.transform(Y_test.reshape(-1, 1))

scaled_threshold = list(minmax_scaler.transform(
    np.array([threshold]).reshape(-1, 1)))[0][0]

Y_binarised_train = (Y_scaled_train > scaled_threshold).astype("int").ravel()
Y_binarised_test = (Y_scaled_test > scaled_threshold).astype("int").ravel()


# Step 3: Train on Real Data

# instantiate a sigmoid neuron
sn = SigmoidNeuron()
sn.fit(X_scaled_train, Y_scaled_train, epochs=10000, learning_rate=0.2)

# how does it perform
Y_pred_train = sn.predict(X_scaled_train)
Y_pred_test = sn.predict(X_scaled_test)
Y_pred_binarized_train = (
    Y_pred_train > scaled_threshold).astype("int").ravel()
Y_pred_binarized_test = (Y_pred_test > scaled_threshold).astype("int").ravel()

accuracy_train = accuracy_score(Y_pred_binarized_train, Y_binarised_train)
accuracy_test = accuracy_score(Y_pred_binarized_test, Y_binarised_test)

print(accuracy_train, accuracy_test)

# How to set the values of epoch and learning rate
# Let's plot the loss functions to get insight
