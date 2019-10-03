from sklearn.model_selection import train_test_split
import sklearn.datasets
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# real world data
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
print(data.groupby('class').mean())


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
X_binarised_train = X_train.apply(pd.cut, bins=2, labels=[0, 1])
plt.plot(X_binarised_train, "*")
plt.xticks(rotation='vertical')
plt.show()

# Will the transpose look better?
plt.plot(X_binarised_train.T, "*")
plt.xticks(rotation='vertical')
plt.show()

# do it for test also
X_binarised_test = X_test.apply(pd.cut, bins=2, labels=[0, 1])

# convert them back to numpy arrays
X_binarised_test = X_binarised_test.values
X_binarised_train = X_binarised_train.values

# they will be numpy Arrays and we can now apply the model on them
type(X_binarised_test)
