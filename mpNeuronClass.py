from sklearn.model_selection import train_test_split
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score


class MPNeuron:

    def __init__(self):
        self.b = None

    # return 1 if >= b, else 0
    def model(self, x):
        return (sum(x) >= self.b)

    # Train and Test vectors, predict across them
    def predict(self, X):
        Y = []
        for x in X:
            result = self.model(x)
            Y.append(result)
        return np.array(Y)

    # learning algorithm to find value of b
    # we have to give the training data
    def fit(self, X, Y):
        # let's have a dictionary of values
        accuracy = {}
        # of colunns with shape 1
        for b in range(X.shape[1] + 1):
            self.b = b
            Y_pred = self.predict(X)
            accuracy[b] = accuracy_score(Y_pred, Y)

        # find largest value
        best_b = max(accuracy, key=accuracy.get)
        self.b = best_b

        print('Optimal value of b', best_b)
        print('Highest accuracy is', accuracy[best_b])


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

# binarise it for MP Neuron
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1, 0])
X_binarised_train = X_binarised_train.values
X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1, 0])
X_binarised_test = X_binarised_test.values

# Use of class, train model
mp_neuron = MPNeuron()
mp_neuron.fit(X_binarised_train, Y_train)

# Test the model and find accuracy
Y_test_pred = mp_neuron.predict(X_binarised_test)
accuracy_test = accuracy_score(Y_test_pred, Y_test)

print(accuracy_test)
