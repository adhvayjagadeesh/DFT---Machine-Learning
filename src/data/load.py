from os import environ

from numpy.random import seed
from pandas import read_parquet
from sklearn.model_selection import KFold

if "RANDOM" not in environ:
  # Fixed random seed for reproducibility, NOT A HYPERPARAM
  seed(67)  # SIX SEVEN

# Read C2DB
df = read_parquet(
  "data/c2db.parquet",
  engine="pyarrow",
)

# Target and features
y = df.pop("HSE06 band gap (eV)")
x = df

# Default k-fold
k = 4
kf = KFold(n_splits=k)


def k_fold(x, y):
  for train_indices, test_indices in kf.split(x):
    x_train = x.iloc[train_indices]
    y_train = y.iloc[train_indices]
    x_test = x.iloc[test_indices]

    # Yield training data, testing input and the indices to put predictions
    yield x_train, y_train, x_test, test_indices
