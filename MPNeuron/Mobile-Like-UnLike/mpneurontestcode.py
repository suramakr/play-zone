from scipy import stats
X_train = train_new.drop(['PhoneId'], axis=1)  # DataFrame
X_train['Rating'] = X_train['Rating'].apply(
    lambda x: 1 if x >= THRESHOLD else 0)
# X_train_thin.groupby(['Rating']).describe()
X_train.groupby(['Rating']).count()


X_train_thin = X_train[['Capacity', 'Rating']]
sns.pairplot(X_train_thin, diag_kind='hist', hue='Rating')

print(X_train[X_train['Rating'] == 1]['Capacity'].describe())
print(X_train[X_train['Rating'] == 0]['Capacity'].describe())

print(np.median(X_train[X_train['Rating'] == 1]['Capacity']))
print(np.median(X_train[X_train['Rating'] == 0]['Capacity']))


X_train_thin = X_train[['Internal Memory', 'Rating']]
sns.pairplot(X_train_thin, diag_kind='hist', hue='Rating')

print(X_train[X_train['Rating'] == 1]['Internal Memory'].describe())
print(X_train[X_train['Rating'] == 0]['Internal Memory'].describe())

print(np.median(X_train[X_train['Rating'] == 1]['Internal Memory']))
print(np.median(X_train[X_train['Rating'] == 0]['Internal Memory']))


X_train_thin = X_train[['RAM', 'Rating']]
sns.pairplot(X_train_thin, diag_kind='hist', hue='Rating')

print(X_train[X_train['Rating'] == 1]['RAM'].describe())
print(X_train[X_train['Rating'] == 0]['RAM'].describe())

print(np.median(X_train[X_train['Rating'] == 1]['RAM']))
print(np.median(X_train[X_train['Rating'] == 0]['RAM']))


print(X_train[X_train['Rating'] == 1]['RAM'].describe())
print(X_train[X_train['Rating'] == 0]['RAM'].describe())


X_train_thin = X_train[['Weight', 'Rating']]
sns.pairplot(X_train_thin, diag_kind='hist', hue='Rating')

print(X_train[X_train['Rating'] == 1]['Weight'].describe())
print(X_train[X_train['Rating'] == 0]['Weight'].describe())

print(np.median(X_train[X_train['Rating'] == 1]['Weight']))
print(np.median(X_train[X_train['Rating'] == 0]['Weight']))
# mode value
mode = stats.mode(X_train[X_train['Rating'] == 0]['Weight'])
print(mode)
mode = stats.mode(X_train[X_train['Rating'] == 1]['Weight'])
print(mode)


X_train_thin = X_train[['Screen to Body Ratio (calculated)', 'Rating']]
sns.pairplot(X_train_thin, diag_kind='hist', hue='Rating')

print(X_train[X_train['Rating'] == 1]
      ['Screen to Body Ratio (calculated)'].describe())
print(X_train[X_train['Rating'] == 0]
      ['Screen to Body Ratio (calculated)'].describe())

print(np.median(X_train[X_train['Rating'] == 1]
                ['Screen to Body Ratio (calculated)']))
print(np.median(X_train[X_train['Rating'] == 0]
                ['Screen to Body Ratio (calculated)']))
# mode value
mode = stats.mode(X_train[X_train['Rating'] == 0]
                  ['Screen to Body Ratio (calculated)'])
print(mode)
mode = stats.mode(X_train[X_train['Rating'] == 1]
                  ['Screen to Body Ratio (calculated)'])
print(mode)


X_train.to_csv("X_train.csv", index=False)


X_train_thin = X_train[['Resolution', 'Rating']]
sns.pairplot(X_train_thin, diag_kind='hist', hue='Rating')

print(X_train[X_train['Rating'] == 1]['Resolution'].describe())
print(X_train[X_train['Rating'] == 0]['Resolution'].describe())

print(np.median(X_train[X_train['Rating'] == 1]['Resolution']))
print(np.median(X_train[X_train['Rating'] == 0]['Resolution']))
# mode value
mode = stats.mode(X_train[X_train['Rating'] == 0]['Resolution'])
print(mode)
mode = stats.mode(X_train[X_train['Rating'] == 1]['Resolution'])
print(mode)

plt.figure(figsize=(4, 2))
sns.scatterplot(x='Resolution', y='Rating', data=X_train)

plt.plot(X_train.T, '*')
plt.xticks(rotation='vertical')
plt.show()
