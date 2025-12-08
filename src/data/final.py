import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import RobustScaler as DefaultScaler

# Fixed random seed for reproducibility, in practice use None, NOT A HYPERPARAM
rng_seed = 67  # SIX SEVEN
np.random.seed(rng_seed)

# Load c2db
df = pd.read_csv("data/Final_rect_materials_filled_in_correctly.csv")

# Drop calculated band gap + DFT columns
df.drop(
  columns=[
    "Direct band gap (PBE) [eV]",
    "Direct band gap (PBE) [eV].1",
    "Band gap (PBE) [eV]",
    "Band gap (G₀W₀) [eV]",
    "Direct band gap (G₀W₀) [eV]",
    "Direct band gap (HSE06) [eV]",
    "Direct band gap (HSE06) [eV].1",
    "CBM wrt. vacuum (PBE) [eV]",
    "VBM wrt. vacuum (PBE) [eV]",
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

# Feature count
feat_cnt = df.shape[1]

# Features and target
y = df["Band gap (HSE06) [eV]"]
df.drop(columns=["Band gap (HSE06) [eV]"], inplace=True)
x = df


# Split data
def split(x=x, y=y, second_size=0.2):
  x1, x2, y1, y2 = tts(x, y, test_size=second_size)
  return x1, y1, x2, y2


# Default k for k-fold
k_ = 4


def k_fold(x=x, y=y, k=k_, scale=True):
  kf = KFold(n_splits=k, shuffle=True)
  for train_indices, test_indices in kf.split(x):
    xtrain = x.iloc[train_indices]
    ytrain = y.iloc[train_indices]
    xtest = x.iloc[test_indices]
    ytest = y.iloc[test_indices]

    if scale:
      scaler = DefaultScaler()
      xtrain = scaler.fit_transform(xtrain)
      xtest = scaler.transform(xtest)

    yield xtrain, ytrain, xtest, ytest
