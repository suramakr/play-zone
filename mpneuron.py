from sklearn.model_selection import train_test_split
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# real world data
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html
breast_cancer = sklearn.datasets.load_breast_cancer()

# read x as features, and y as class label
# binary classification of tissue - malignant or benign


#### LOAD DATASET ########

# Series of data items, each row contains sequence of features for
# each sample
X = breast_cancer.data
# Y contains 0 and 1
Y = breast_cancer.target

print(X, Y)

# (569, 30) (569,)
# X has 569 samples, 30 columns or features, and
# Y has a corresponding scalar value for the 569 samples
print(X.shape, Y.shape)

data = pd.DataFrame(breast_cancer.data, columns=breast_cancer.feature_names)

# add a column to a pandas data frame
data['class'] = breast_cancer.target

# look at first 5 rows, 31 columns (30 feature + 1 class)
print(data.head())

# get some statistics to get better meaning
# mean, std etc, class
print(data.describe())

# class has mean of 0.62
# mean 0.62 (more 1s then 0s)
# 1    357
# 0    212
# Name: class, dtype: int64
# ['malignant' 'benign']
print(data['class'].value_counts())
print(breast_cancer.target_names)

# For each feature we can see aggregated mean
#        mean radius  ...  worst fractal dimension
# class               ...
# 0        17.462830  ...                 0.091530
# 1        12.146524  ...                 0.079442
plt


# SPLITTING AND TRAINING DATASET
# Generalize, and ensure we don't overfit
# for columns use axis =1
# we are doing this so that we can have the data as DataFrames
X = data.drop('class', axis=1)
Y = data['class']

# class 'pandas.core.frame.DataFrame'
print(type(X))

# X will get split into X_train, X_test, Y_train, and Y_test
X_train, X_test, Y_train, Y_test = train_test_split(X, Y)

# 80% data for train
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)

# For deep learning with larger train data, we can lower test data like 5%
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1)
# 80% data for train
print(X.shape, X_train.shape, X_test.shape)
print(Y.shape, Y_train.shape, Y_test.shape)

# Do we have a similar ration of benign and malign?
print(Y.mean(), Y_train.mean(), Y_test.mean())

# if you see output, 0.6274165202108963 0.6328125 0.5789473684210527
# the y_test has only 57% malign cases
# let's stratify the data to fix this

# This stratify parameter makes a split so that the proportion of values
# in the sample produced will be the same as the proportion of values provided to parameter
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y)
print(Y.mean(), Y_train.mean(), Y_test.mean())
print(X.mean(), X_train.mean(), X_test.mean())

# everytime you run the train_test_split you get
# different types of split. which is not ACCEPTABLE!!!
# Make the split deterministic
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.1, stratify=Y, random_state=1)


# Handling inputs
# If input is in real values, but let's say you want to use
# MP Neuron as a AI method to categorize, we need to convert real
# to binary form
# BINARISATION OF INPUT
plt.plot(X_train, '*')
# Each feature is show in a different color
# x axis is the 500+ different items, and veritical
# axis is the values it takes on.
plt.show()

# now plot the transpose of this to see if it looks better
plt.plot(X_train.T, '*')

# at times the labels might not show well
plt.xticks(rotation='vertical')
plt.show()

# If you see the graph, you will find two features
# that have peaks. Let's choose "mean area"
# That way we can now start binarize the data at say 1000
# > 1000, 1, using mean
X_binarised_3_train = X_train['mean area'].map(lambda x: 0 if x < 1000 else 1)
plt.plot(X_binarised_3_train, "+")
plt.show()

# instead of doing this column wise. We can do it across all
# features. Use the pd.cut call, and into 2 bins, with labels 0,1
# X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[0, 1])
# if you see above we had labels 0,1 instead can we try 1,0
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1, 0])
plt.plot(X_binarised_train, "*")
plt.xticks(rotation='vertical')
plt.show()

# Will the transpose look better?
plt.plot(X_binarised_train.T, "*")
plt.xticks(rotation='vertical')
plt.show()

# Use pd.cut to do this auto for ALL FEATURES
# do it for test also
#X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[0, 1])
X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[1, 0])
# convert them back to numpy arrays
X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values

# they will be numpy Arrays and we can now apply the model on them
type(X_binarised_test)


# INFERENCE AND SEARCH
# MP NEURON, use "b" value

b = 3
i = 100  # row instead of 100
if (np.sum(X_binarised_train[i, :]) >= b):
    print("MP Neuron Inference is malignant")
else:
    print("MP Neuron Inference is benign")
# ans: MP Neuron Inference is benign

print(Y.shape, Y_train.shape, Y_test.shape)
if (Y_train[i] == 1):
    print('Ground truth is malignant')
else:
    print('Ground truth is benign')

# Goal is to find a value b where the accuracy is maximized
# for .e.g for row 208 we will have a problem

b = 3
Y_pred_train = []  # predicted
accurate_rows = 0

# use zip to iterate over two iterators together
for x, y in zip(X_binarised_train, Y_train):
    y_pred = (np.sum(x) >= b)
    Y_pred_train.append(y_pred)
    accurate_rows += (y == y_pred)

print('CALCULATING ACCURACY NOW')
print(accurate_rows, accurate_rows/X_binarised_train.shape[0])
# 77 0.150390625
# this is low, 62% of the malignant baseline we need to get


for b in range(X_binarised_train.shape[1] + 1):
    Y_pred_train = []  # predicted
    accurate_rows = 0
    # use zip to iterate over two iterators together
    for x, y in zip(X_binarised_train, Y_train):
        y_pred = (np.sum(x) >= b)
        Y_pred_train.append(y_pred)
        accurate_rows += (y == y_pred)

    print("b accurate rows accuracy:")
    print(b, accurate_rows, accurate_rows/X_binarised_train.shape[0])

# Why is the model only able to do as good as b=3
# if you see the means of 0/1 u will see the 0 value beningn
# cases have larger feature set. So having a b to seperate
# the features based on sizes is not working.
# X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[1, 0])
# after changing the labels we see 84% accuracy
# b = 28 is greater than baseline of 0.62 and higher accuracy
# style of binarization impacts


# INFERENCE
# on train data set we got 80+% accuracy with b=28
# can we get that in test data
b = 28
Y_pred_test = []  # predicted

# use zip to iterate over two iterators together
for x in X_binarised_test:
    y_pred = (np.sum(x) >= b)
    Y_pred_test.append(y_pred)

accuracy = accuracy_score(Y_pred_test, Y_test)
print('value of b and accuracy')
print(b, accuracy)
# on test data the accuracy is .79
