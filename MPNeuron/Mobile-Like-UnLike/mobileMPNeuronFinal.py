# Task: To predict whether the user likes the mobile phone or not. <br>
# Assumption: If the average rating of mobile >= threshold, then the user likes it, otherwise not.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder,MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error, log_loss
import operator
import json
from IPython import display
import os
import warnings

np.random.seed(0)
warnings.filterwarnings("ignore")
THRESHOLD = 4


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


def data_clean(data):

    # Let's first remove all missing value features
    columns_to_remove = ['Also Known As', 'Applications', 'Audio Features', 'Bezel-less display'
                         'Browser', 'Build Material', 'Co-Processor', 'Browser'
                         'Display Colour', 'Mobile High-Definition Link(MHL)',
                         'Music', 'Email', 'Fingerprint Sensor Position',
                         'Games', 'HDMI', 'Heart Rate Monitor', 'IRIS Scanner',
                         'Optical Image Stabilisation', 'Other Facilities',
                         'Phone Book', 'Physical Aperture', 'Quick Charging',
                         'Ring Tone', 'Ruggedness', 'SAR Value', 'SIM 3', 'SMS',
                         'Screen Protection', 'Screen to Body Ratio (claimed by the brand)',
                         'Sensor', 'Software Based Aperture', 'Special Features',
                         'Standby time', 'Stylus', 'TalkTime', 'USB Type-C',
                         'Video Player', 'Video Recording Features', 'Waterproof',
                         'Wireless Charging', 'USB OTG Support', 'Video Recording', 'Java']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    # Features having very low variance
    columns_to_remove = ['Architecture', 'Audio Jack', 'GPS',
                         'Loudspeaker', 'Network', 'Network Support', 'VoLTE']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    # Multivalued:
    columns_to_remove = ['Architecture', 'Launch Date', 'Audio Jack', 'GPS',
                         'Loudspeaker', 'Network', 'Network Support', 'VoLTE', 'Custom UI']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    # Not much important
    columns_to_remove = ['Bluetooth', 'Settings', 'Wi-Fi', 'Wi-Fi Features']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    return data


def for_integer(test):
    try:
        test = test.strip()
        return int(test.split(' ')[0])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass


def for_string(test):
    try:
        test = test.strip()
        return (test.split(' ')[0])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass


def for_float(test):
    try:
        test = test.strip()
        return float(test.split(' ')[0])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass


def find_freq(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[2][0] == '(':
            return float(test[2][1:])
        return float(test[2])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass


def for_Internal_Memory(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[1] == 'GB':
            return int(test[0])
        if test[1] == 'MB':
            #             print("here")
            return (int(test[0]) * 0.001)
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass


def find_freq(test):
    try:
        test = test.strip()
        test = test.split(' ')
        if test[2][0] == '(':
            return float(test[2][1:])
        return float(test[2])
    except IOError:
        pass
    except ValueError:
        pass
    except:
        pass


def data_clean_2(x):
    data = x.copy()
    data['Capacity'] = data['Capacity'].apply(for_integer)
    data['Height'] = data['Height'].apply(for_float)
    data['Height'] = data['Height'].fillna(data['Height'].mean())

    data['Internal Memory'] = data['Internal Memory'].apply(
        for_Internal_Memory)

    data['Pixel Density'] = data['Pixel Density'].apply(for_integer)

    data['Internal Memory'] = data['Internal Memory'].fillna(
        data['Internal Memory'].median())
    data['Internal Memory'] = data['Internal Memory'].astype(int)

    data['RAM'] = data['RAM'].apply(for_integer)
    data['RAM'] = data['RAM'].fillna(data['RAM'].median())
    data['RAM'] = data['RAM'].astype(int)

    data['Resolution'] = data['Resolution'].apply(for_integer)
    data['Resolution'] = data['Resolution'].fillna(data['Resolution'].median())
    data['Resolution'] = data['Resolution'].astype(int)

    data['Screen Size'] = data['Screen Size'].apply(for_float)

    data['Thickness'] = data['Thickness'].apply(for_float)
    data['Thickness'] = data['Thickness'].fillna(data['Thickness'].mean())
    data['Thickness'] = data['Thickness'].round(2)

    data['Type'] = data['Type'].fillna('Li-Polymer')

    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].apply(
        for_float)
    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].fillna(
        data['Screen to Body Ratio (calculated)'].mean())
    data['Screen to Body Ratio (calculated)'] = data['Screen to Body Ratio (calculated)'].round(
        2)

    data['Width'] = data['Width'].apply(for_float)
    data['Width'] = data['Width'].fillna(data['Width'].mean())
    data['Width'] = data['Width'].round(2)

    data['Flash'][data['Flash'].isna() == True] = "Other"

    data['User Replaceable'][data['User Replaceable'].isna() == True] = "Other"

    data['Num_cores'] = data['Processor'].apply(for_string)
    data['Num_cores'][data['Num_cores'].isna() == True] = "Other"

    data['Processor_frequency'] = data['Processor'].apply(find_freq)
    # because there is one entry with 208MHz values, to convert it to GHz
    data['Processor_frequency'][data['Processor_frequency'] > 200] = 0.208
    data['Processor_frequency'] = data['Processor_frequency'].fillna(
        data['Processor_frequency'].mean())
    data['Processor_frequency'] = data['Processor_frequency'].round(2)

    data['Camera Features'][data['Camera Features'].isna() == True] = "Other"

    # simplifyig Operating System to os_name for simplicity
    data['os_name'] = data['Operating System'].apply(for_string)
    data['os_name'][data['os_name'].isna() == True] = "Other"

    data['Sim1'] = data['SIM 1'].apply(for_string)

    data['SIM Size'][data['SIM Size'].isna() == True] = "Other"

    data['Image Resolution'][data['Image Resolution'].isna() == True] = "Other"

    data['Fingerprint Sensor'][data['Fingerprint Sensor'].isna() ==
                               True] = "Other"

    data['Expandable Memory'][data['Expandable Memory'].isna() == True] = "No"

    data['Weight'] = data['Weight'].apply(for_integer)
    data['Weight'] = data['Weight'].fillna(data['Weight'].mean())
    data['Weight'] = data['Weight'].astype(int)

    data['SIM 2'] = data['SIM 2'].apply(for_string)
    data['SIM 2'][data['SIM 2'].isna() == True] = "Other"

    return data


def data_clean_3(x):

    data = x.copy()

    columns_to_remove = ['User Available Storage', 'SIM Size', 'Chipset', 'Processor', 'Autofocus', 'Aspect Ratio', 'Touch Screen',
                         'Bezel-less display', 'Operating System', 'SIM 1', 'USB Connectivity', 'Other Sensors', 'Graphics', 'FM Radio',
                         'NFC', 'Shooting Modes', 'Browser', 'Display Colour']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    columns_to_remove = ['Screen Resolution', 'User Replaceable', 'Camera Features',
                         'Thickness', 'Display Type']

    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    columns_to_remove = ['Fingerprint Sensor', 'Flash', 'Rating Count', 'Review Count', 'Image Resolution', 'Type', 'Expandable Memory',
                         'Colours', 'Width', 'Model']
    columns_to_retain = list(set(data.columns)-set(columns_to_remove))
    data = data[columns_to_retain]

    return data


# MAIN CODE
np.random.seed(0)
warnings.filterwarnings("ignore")
THRESHOLD = 4

path = '/Users/sramakrishnan/work/python/play-zone/Mobile-Like-UnLike'


# read data from file
train = pd.read_csv(path + '/train.csv')
test = pd.read_csv(path + '/test.csv')

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])

# Removing unwanted features
train = data_clean(train)
test = data_clean(test)

# removing all those data points in which more than 15 features are missing
train = train[(train.isnull().sum(axis=1) <= 15)]
# IMPORTANT: don't remove data from test set!!!!


# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])

# Filling Missing values

train = data_clean_2(train)
test = data_clean_2(test)

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


# Remove Not very important features
train = data_clean_3(train)
test = data_clean_3(test)

# check the number of features and data points in train
print("Number of data points in train: %d" % train.shape[0])
print("Number of features in train: %d" % train.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test.shape[0])
print("Number of features in test: %d" % test.shape[1])


# one hot encoding
train_ids = train['PhoneId']
test_ids = test['PhoneId']

cols = list(test.columns)
cols.remove('PhoneId')
cols.insert(0, 'PhoneId')

combined = pd.concat([train.drop('Rating', axis=1)[cols], test[cols]])
print(combined.shape)
print(combined.columns)

combined = pd.get_dummies(combined)
print(combined.shape)
print(combined.columns)

train_new = combined[combined['PhoneId'].isin(train_ids)]
test_new = combined[combined['PhoneId'].isin(test_ids)]

train_new = train_new.merge(train[['PhoneId', 'Rating']], on='PhoneId')

# check the number of features and data points in train
print("Number of data points in train: %d" % train_new.shape[0])
print("Number of features in train: %d" % train_new.shape[1])

# check the number of features and data points in test
print("Number of data points in test: %d" % test_new.shape[0])
print("Number of features in test: %d" % test_new.shape[1])

print(train_new.head())
print(test_new.head())

X_train = train_new.drop(['PhoneId'], axis=1)  # DataFrame
X_train['Rating'] = X_train['Rating'].apply(lambda x: 1 if x >= THRESHOLD else 0)
#X_train_thin.groupby(['Rating']).describe()
X_train.groupby(['Rating']).count()

## Insert code here to play with data
## refer to mpneurondatatestcode.py

import seaborn as sns
sns.set()

X_train_thin = X_train[['Capacity', 'Weight','RAM','Resolution','Internal Memory',
                        'Pixel Density', 'Screen Size', 'Rating']]
sns.pairplot(X_train_thin, diag_kind='hist', hue='Rating')

X_train_thin = X_train[['RAM', 'Rating']]
sns.pairplot(X_train_thin, diag_kind='hist', hue='Rating')

plt.figure(figsize=(4,2))
sns.scatterplot(x='RAM',y='Rating',data=X_train)


# Initialize X, Y Dataframes, for columns use axis=1
# no need to binarize PhoneId, ir-relevant
X_train = train_new.drop(['Rating', 'PhoneId'], axis=1)  # DataFrame

Y_train = pd.DataFrame({'Rating': train_new['Rating']})
# 0.7008797653958945 is the accuracy with this threshold technique
Y_binary_train = Y_train['Rating'].map(lambda x: 1 if x >= THRESHOLD else 0)
Y_binary_train = Y_binary_train.values

# Using this binarization changed the optimal value of b 8
# Highest accuracy is 0.8328445747800587 Accuracy obtained during learning 0.8328445747800587
#Y_binary_train = Y_train.apply(pd.cut, bins=2, labels=[0, 1]) # use binarization

# To test now
X_test = test_new.drop(['PhoneId'], axis=1)

X_binary_train = np.array([
        X_train['RAM'].map(lambda x: 1 if x < 5 else 0),
        X_train['Capacity'].map(lambda x: 1 if x>=2000 else 0), #/check
        X_train['Weight'].map(lambda x: 1 if x > 150 else 0), 
        X_train['Screen to Body Ratio (calculated)'].map(lambda x: 1 if x>=56 else 0),
        X_train['Resolution'].map(lambda x: 1 if x > 9 else 0),
        X_train['Height'].map(lambda x: 1 if x>=0 else 0), #no difference to like or disklike
        X_train['Pixel Density'].map(lambda x: 1 if x >= 250 else 0),
        X_train['Processor_frequency'].map(lambda x: 1 if x > 1.75 else 0),
        X_train['Internal Memory'].map(lambda x: 1 if x > 16 else 0),
        X_train['Screen Size'].map(lambda x: 1 if x > 4.8 else 0),
        X_train['SIM Slot(s)_Dual SIM, GSM+CDMA'],
        X_train['SIM Slot(s)_Dual SIM, GSM+GSM'].map(lambda x: 1 if x>=1 else 0),
        X_train['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'].map(lambda x: 1 if x>=1 else 0),
        X_train['SIM Slot(s)_Single SIM, GSM'],
        X_train['Sim1_2G'],
        X_train['Sim1_3G'],
        X_train['Sim1_4G'].map(lambda x: 1 if x>=1 else 0),
        X_train['os_name_Android'].map(lambda x: 1 if x>=1 else 0),
        X_train['os_name_Blackberry'],
        X_train['os_name_KAI'],
        X_train['os_name_Nokia'],
        X_train['os_name_Other'].map(lambda x: 1 if x>=1 else 0),
        X_train['os_name_Tizen'],
        X_train['os_name_iOS'].map(lambda x: 1 if x>=1 else 0),
        X_train['Brand_10.or'], 
        X_train['Brand_Apple'].map(lambda x: 1 if x>=1 else 0),
        X_train['Brand_Asus'],
        X_train['Brand_Billion'],
        X_train['Brand_Blackberry'],
        X_train['Brand_Comio'],
        X_train['Brand_Coolpad'], 
        X_train['Brand_Do'], 
        X_train['Brand_Gionee'],
        X_train['Brand_Google'].map(lambda x: 1 if x>=1 else 0), 
        X_train['Brand_HTC'].map(lambda x: 0 if x>=1 else 1),
        X_train['Brand_Honor'],
        X_train['Brand_Huawei'],
        X_train['Brand_InFocus'], 
        X_train['Brand_Infinix'],
        X_train['Brand_Intex'],
        X_train['Brand_Itel'],
        X_train['Brand_Jivi'], 
        X_train['Brand_Karbonn'],
        X_train['Brand_LG'].map(lambda x: 0 if x>=1 else 1),
        X_train['Brand_Lava'], 
        X_train['Brand_LeEco'], 
        X_train['Brand_Lenovo'], 
        X_train['Brand_Lephone'], 
        X_train['Brand_Lyf'],
        X_train['Brand_Meizu'],
        X_train['Brand_Micromax'], 
        X_train['Brand_Mobiistar'], 
        X_train['Brand_Moto'],
        X_train['Brand_Motorola'],
        X_train['Brand_Nokia'], 
        X_train['Brand_Nubia'].map(lambda x: 0 if x>=1 else 1),
        X_train['Brand_OPPO'],
        X_train['Brand_OnePlus'],
        X_train['Brand_Oppo'],
        X_train['Brand_Panasonic'].map(lambda x: 0 if x>=1 else 1), 
        X_train['Brand_Razer'].map(lambda x: 0 if x>=1 else 1),
        X_train['Brand_Realme'],
        X_train['Brand_Reliance'].map(lambda x: 0 if x>=1 else 1), 
        X_train['Brand_Samsung'].map(lambda x: 1 if x>=1 else 0),
        X_train['Brand_Sony'],
        X_train['Brand_Spice'], 
        X_train['Brand_Tecno'],
        X_train['Brand_Ulefone'],
        X_train['Brand_VOTO'].map(lambda x: 0 if x>=1 else 1), 
        X_train['Brand_Vivo'],
        X_train['Brand_Xiaomi'],
        X_train['Brand_Xiaomi Poco'],
        X_train['Brand_Yu'].map(lambda x: 0 if x>=1 else 1),         
        X_train['Brand_iVooMi'].map(lambda x: 0 if x>=1 else 1), 
        X_train['SIM Slot(s)_Dual SIM, GSM+CDMA'],
        X_train['SIM Slot(s)_Dual SIM, GSM+GSM'].map(lambda x: 0 if x>=1 else 1),
        X_train['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'].map(lambda x: 1 if x>=1 else 0),
        X_train['SIM Slot(s)_Single SIM, GSM'],
        X_train['Num_cores_312'].map(lambda x: 0 if x>=1 else 1),
        X_train['Num_cores_Deca'].map(lambda x: 0 if x>=1 else 1),
        X_train['Num_cores_Dual'].map(lambda x: 0 if x>=1 else 1),
        X_train['Num_cores_Hexa'].map(lambda x: 0 if x>=1 else 1),
        X_train['Num_cores_Octa'].map(lambda x: 1 if x>=1 else 0),
        X_train['Num_cores_Other'].map(lambda x: 0 if x>=1 else 1),
        X_train['Num_cores_Quad'].map(lambda x: 0 if x>=1 else 1),
        X_train['Num_cores_Tru-Octa'].map(lambda x: 0 if x>=1 else 1)
        ])

# Find the shapes of everything to ensure they look ok
X_binary_train = X_binary_train.T
print(X_binary_train.shape)
print(Y_binary_train.shape)

print(X_binary_train[321])


# Use of class, train model
mp_neuron = MPNeuron()
mp_neuron.fit(X_binary_train, Y_binary_train)

# Test the model with your split data first and find accuracy
Y_test_pred = mp_neuron.predict(X_binary_train)
accuracy_test = accuracy_score(Y_test_pred, Y_binary_train)
print("Accuracy obtained during learning")
print(accuracy_test)

X_binary_test = np.array([
        X_test['RAM'].map(lambda x: 1 if x < 5 else 0),
        X_test['Capacity'].map(lambda x: 1 if x>=2000 else 0), #/check
        X_test['Weight'].map(lambda x: 1 if x > 150 else 0), 
        X_test['Screen to Body Ratio (calculated)'].map(lambda x: 1 if x>=56 else 0), #check
        X_test['Resolution'].map(lambda x: 1 if x > 9 else 0),
        X_test['Height'].map(lambda x: 1 if x>=0 else 0), #no difference to like or disklike
        X_test['Pixel Density'].map(lambda x: 1 if x >= 250 else 0),
        X_test['Processor_frequency'].map(lambda x: 1 if x > 1.75 else 0),
        X_test['Internal Memory'].map(lambda x: 1 if x > 16 else 0),
        X_test['Screen Size'].map(lambda x: 1 if x > 4.8 else 0),
        X_test['SIM Slot(s)_Dual SIM, GSM+CDMA'],
        X_test['SIM Slot(s)_Dual SIM, GSM+GSM'].map(lambda x: 1 if x>=1 else 0),
        X_test['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'].map(lambda x: 1 if x>=1 else 0),
        X_test['SIM Slot(s)_Single SIM, GSM'],
        X_test['Sim1_2G'],
        X_test['Sim1_3G'],
        X_test['Sim1_4G'].map(lambda x: 1 if x>=1 else 0),
        X_test['os_name_Android'].map(lambda x: 1 if x>=1 else 0),
        X_test['os_name_Blackberry'],
        X_test['os_name_KAI'],
        X_test['os_name_Nokia'],
        X_test['os_name_Other'].map(lambda x: 1 if x>=1 else 0),
        X_test['os_name_Tizen'],
        X_test['os_name_iOS'].map(lambda x: 1 if x>=1 else 0),
        X_test['Brand_10.or'], 
        X_test['Brand_Apple'].map(lambda x: 1 if x>=1 else 0),
        X_test['Brand_Asus'],
        X_test['Brand_Billion'],
        X_test['Brand_Blackberry'],
        X_test['Brand_Comio'],
        X_test['Brand_Coolpad'], 
        X_test['Brand_Do'], 
        X_test['Brand_Gionee'],
        X_test['Brand_Google'].map(lambda x: 1 if x>=1 else 0), 
        X_test['Brand_HTC'].map(lambda x: 0 if x>=1 else 1),
        X_test['Brand_Honor'],
        X_test['Brand_Huawei'],
        X_test['Brand_InFocus'], 
        X_test['Brand_Infinix'],
        X_test['Brand_Intex'],
        X_test['Brand_Itel'],
        X_test['Brand_Jivi'], 
        X_test['Brand_Karbonn'],
        X_test['Brand_LG'].map(lambda x: 0 if x>=1 else 1),
        X_test['Brand_Lava'], 
        X_test['Brand_LeEco'], 
        X_test['Brand_Lenovo'], 
        X_test['Brand_Lephone'], 
        X_test['Brand_Lyf'],
        X_test['Brand_Meizu'],
        X_test['Brand_Micromax'], 
        X_test['Brand_Mobiistar'], 
        X_test['Brand_Moto'],
        X_test['Brand_Motorola'],
        X_test['Brand_Nokia'], 
        X_test['Brand_Nubia'].map(lambda x: 0 if x>=1 else 1),
        X_test['Brand_OPPO'],
        X_test['Brand_OnePlus'],
        X_test['Brand_Oppo'],
        X_test['Brand_Panasonic'].map(lambda x: 0 if x>=1 else 1), 
        X_test['Brand_Razer'].map(lambda x: 0 if x>=1 else 1),
        X_test['Brand_Realme'],
        X_test['Brand_Reliance'].map(lambda x: 0 if x>=1 else 1), 
        X_test['Brand_Samsung'].map(lambda x: 1 if x>=1 else 0),
        X_test['Brand_Sony'],
        X_test['Brand_Spice'], 
        X_test['Brand_Tecno'],
        X_test['Brand_Ulefone'],
        X_test['Brand_VOTO'].map(lambda x: 0 if x>=1 else 1), 
        X_test['Brand_Vivo'],
        X_test['Brand_Xiaomi'],
        X_test['Brand_Xiaomi Poco'],
        X_test['Brand_Yu'].map(lambda x: 0 if x>=1 else 1),         
        X_test['Brand_iVooMi'].map(lambda x: 0 if x>=1 else 1), 
        X_test['SIM Slot(s)_Dual SIM, GSM+CDMA'],
        X_test['SIM Slot(s)_Dual SIM, GSM+GSM'].map(lambda x: 0 if x>=1 else 1),
        X_test['SIM Slot(s)_Dual SIM, GSM+GSM, Dual VoLTE'].map(lambda x: 1 if x>=1 else 0),
        X_test['SIM Slot(s)_Single SIM, GSM'],
        X_test['Num_cores_312'].map(lambda x: 0 if x>=1 else 1),
        X_test['Num_cores_Deca'].map(lambda x: 0 if x>=1 else 1),
        X_test['Num_cores_Dual'].map(lambda x: 0 if x>=1 else 1),
        X_test['Num_cores_Hexa'].map(lambda x: 0 if x>=1 else 1),
        X_test['Num_cores_Octa'].map(lambda x: 1 if x>=1 else 0),
        X_test['Num_cores_Other'].map(lambda x: 0 if x>=1 else 1),
        X_test['Num_cores_Quad'].map(lambda x: 0 if x>=1 else 1),
        X_test['Num_cores_Tru-Octa'].map(lambda x: 0 if x>=1 else 1)
        ])
                          
                          
X_binary_test = X_binary_test.T
# Find the shapes of everything to ensure they look ok
print(X_binary_test.shape)


# Now try with test_new data
#X_binary_test = X_test.apply(pd.cut, bins=2, labels=[0, 1])
#X_binary_test = X_binary_test.values

test_y_pred = mp_neuron.predict(X_binary_test)

print(type(test_y_pred))

test_new['Class'] = test_y_pred
test_new['Class'] = test_new['Class'].map(lambda x: 1 if x == True else 0)
test_new['Class']
print(test_y_pred)

submission = pd.DataFrame({'PhoneId':test_new['PhoneId'], 'Class':test_new['Class']})
submission = submission[['PhoneId', 'Class']]
with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
    print(submission)