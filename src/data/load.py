from pandas import read_parquet
from sklearn.model_selection import KFold

import seeder

df = read_parquet("data/c2db.parquet")

# Features and target views
x = df.iloc[:, 1:]
y = df.iloc[:, 0]

# Default k-fold
k = 5
kf = KFold(n_splits=k)


def k_fold(x, y):
  for train_indices, test_indices in kf.split(x):
    x_train = x.iloc[train_indices]
    y_train = y.iloc[train_indices]
    x_test = x.iloc[test_indices]

    # Yield training data, testing input and the indices to put predictions
    yield x_train, y_train, x_test, test_indices
