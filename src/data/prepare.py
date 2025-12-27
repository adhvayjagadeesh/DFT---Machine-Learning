from os import environ

from numpy.random import seed
from pandas import read_csv
from sklearn.model_selection import KFold

if "RANDOM" not in environ:
  # Fixed random seed for reproducibility, NOT A HYPERPARAM
  seed(67)  # SIX SEVEN

# Load c2db
df = read_csv("data/c2db.csv")

# Drop calculated band gap, expensive DFT columns, and useless formula column
df.drop(
  columns=[
    "Formula",
    "Direct band gap (PBE) (eV)",
    "Direct band gap (PBE) (eV).1",
    "Band gap (PBE) (eV)",
    "Band gap (G₀W₀) (eV)",
    "Direct band gap (G₀W₀) (eV)",
    "Direct band gap (HSE06) (eV)",
    "Direct band gap (HSE06) (eV).1",
    "CBM wrt. vacuum (PBE) (eV)",
    "VBM wrt. vacuum (PBE) (eV)",
  ],
  inplace=True,
)

# Drop columns with >90% missing data + identifiers
drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=drop_cols, inplace=True, errors="ignore")

# Fill missing numerical values with the mean
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and target
y = df.pop("Band gap (HSE06) (eV)")
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
