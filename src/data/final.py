from os import environ

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder

if "RANDOM" not in environ:
  # Fixed random seed for reproducibility, NOT A HYPERPARAM
  np.random.seed(67)  # SIX SEVEN

# Load c2db
df = pd.read_csv("data/Final_rect_materials_filled_in_correctly.csv")

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

# Encode categorical columns, should only be the formula column for now
cat_cols = df.select_dtypes(include="object").columns
label_encoders = {}
for col in cat_cols:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col].astype(str))
  label_encoders[col] = le

# Features and target
y = df["Band gap (HSE06) (eV)"]
df.drop(columns=["Band gap (HSE06) (eV)"], inplace=True)
x = df

# Default k-fold object
kf = KFold(n_splits=4, shuffle=True)


def k_fold(x=x, y=y):
  for train_indices, test_indices in kf.split(x):
    x_train = x.iloc[train_indices]
    y_train = y.iloc[train_indices]
    x_test = x.iloc[test_indices]

    # Yield training data, testing input and the indices to put predictions
    yield x_train, y_train, x_test, test_indices
