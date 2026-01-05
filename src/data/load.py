from os import environ

from numpy.random import seed
from pandas import MultiIndex, read_csv
from sklearn.model_selection import KFold

if "RANDOM" not in environ:
  # Fixed random seed for reproducibility, NOT A HYPERPARAM
  seed(67)  # SIX SEVEN

# Read C2DB while ignoring columns
df = read_csv(
  "data/c2db.csv",
  usecols=lambda col: col
  not in (
    # Non-HSE06 band gap
    "Direct band gap (PBE) (eV)",
    "Direct band gap (PBE) (eV).1",
    "Band gap (PBE) (eV)",
    "Band gap (G₀W₀) (eV)",
    "Direct band gap (G₀W₀) (eV)",
    "Direct band gap (HSE06) (eV)",
    "Direct band gap (HSE06) (eV).1",
    # Expensive DFT columns
    "CBM wrt. vacuum (PBE) (eV)",
    "VBM wrt. vacuum (PBE) (eV)",
    # Useless formula cuz it basically an index
    "Formula",
  ),
  engine="c",
)

# Target and features
y = df.pop("Band gap (HSE06) (eV)")
x = df

# Temporary indexing for features as a level
x.columns = MultiIndex.from_arrays(
  [x.columns, [str(i) for i in range(1, len(x.columns) + 1)]]
)

# Unimportant features
x.drop(
  columns=[
    "Magnetic anisotropy energy, xz (meV/unit cell)",
    "Magnetic anisotropy energy, yz (meV/unit cell)",
  ],
  level=0,  # Drop by level 1 (name)
  inplace=True,
)

# Collect remaining feature indices
feat_indices = x.columns.get_level_values(1)

# Drop the indexing level
x.columns = x.columns.droplevel(1)

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
