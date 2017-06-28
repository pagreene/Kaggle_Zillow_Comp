# Zillow Competition Tools

These are my tools for analyzing the data for the Zillow challenge.

## Cleaning

The cleaning object may be used to clean the data. You must first prime the cleaner with the training data, to prevent data leakage.

Example:

```
from analysis import loadData, splitAndClean

df = loadData()
X_train, X_test, y_train, y_test, cleaner = splitAndClean(df)
```
