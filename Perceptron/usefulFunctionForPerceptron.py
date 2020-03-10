# use of masks to remove outliers
from scipy import stats


def remove_outliers(df, q=0.05):
    upper = df.quantile(1-q)
    lower = df.quantile(q)
    mask = (df < upper) & (df > lower)
    return mask


t = pd.DataFrame({'train': [1, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9],
                  'y': [1, 0, 0, 1, 1, 0, 0, 1, 1, 1, 0]})

mask = remove_outliers(t['train'], 0.1)

print(t[mask])

# min max scaling code for X_train['RAM]
rom sklearn import preprocessing

# test with one column
# Get column names first
names = X_train.columns
names = ['RAM']

fig, ((ax1, ax2)) = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

for name in names:
    x_np = np.asarray(X_train[name])
    y_pos = [0 for i in range(len(X_train[name]))]
    print(x_np)
    # Min-Max scaling
    np_minmax = (x_np - x_np.min()) / (x_np.max() - x_np.min())
    print(np_minmax)
    ax1.scatter(np_minmax, y_pos, color='b')
    ax1.set_title('Python NumPy Min-Max scaling', color='b')

    ax2.scatter(x_np, y_pos, color='b')
    ax2.set_title('Python NumPy original data', color='r')

    plt.tight_layout()


# Lets remove the impact of outliers from our data
z = np.abs(stats.zscore(X_train))
threshold_z = 3
print(np.where(z > threshold_z or z < -threshold_z))

#  measure of the dispersion similar to standard deviation or variance
Q1 = X_train.quantile(0.25)
Q3 = X_train.quantile(0.75)
IQR = Q3 - Q1


def f(x): return 1 if x == True else 0


q = IQR.map(f)

# it’s time to get hold on outliers.
# The data point where we have False that means these values are
# valid whereas True indicates presence of an outlier.
#print(X_train < (Q1 - 1.5 * q)) |(X_train > (Q3 + 1.5 * q))

boston_df_o = X_train[(z < threshold_z).all(axis=1)]
print(X_train.shape)
print(boston_df_o.shape)
