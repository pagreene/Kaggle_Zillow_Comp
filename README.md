# Zillow Competition Tools

These are my tools for analyzing the data for the Zillow challenge.

## Cleaning

The cleaning object may be used to clean the data. You must first prime the cleaner with the training data, to prevent data leakage.

Example:

```
from sklearn.model_selection import train_test_split
from analysis import loadData, Cleaner

df = loadData()
X = df.drop('logerror', axis=1)
y = df['logerror']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

c = Cleaner()

X_train, y_train = c.cleanData(X_train, y_train)
X_test, y_test = c.cleanData(X_test, y_test)
```
