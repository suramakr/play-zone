import sklearn.datasets
import numpy as np
import pandas as pd

# real world data
breast_cancer = sklearn.datasets.load_breast_cancer()

# read x as features, and y as class label
# binary classification of tissue - malignant or benign


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
