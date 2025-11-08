import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split as tts, KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

# Fixed random seed for reproducibility, in practice use None, NOT A HYPERPARAM
rng_seed = 67 # SIX SEVEN
np.random.seed(rng_seed)

# Load c2db
df = pd.read_csv("data/Final_rect_materials_filled_in_correctly.csv")

# Drop column
df = df.drop(columns=[
  'Direct band gap (PBE) [eV]',
  'Direct band gap (PBE) [eV].1',
  'Band gap (PBE) [eV]',
  'Band gap (G₀W₀) [eV]',
  'Direct band gap (G₀W₀) [eV]',
  'Direct band gap (HSE06) [eV]',
  'Direct band gap (HSE06) [eV].1',
  'CBM wrt. vacuum (PBE) [eV]',
  'VBM wrt. vacuum (PBE) [eV]',
])

# Drop columns with >90% missing data + identifiers
drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Fill missing numerical values with the mean
df.fillna(df.mean(numeric_only = True), inplace = True)

# Encode categorical columns, should only be the formula column for now
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col].astype(str))
  label_encoders[col] = le

# Feature count
feat_cnt = df.shape[1]

# Features and target
x_ = df.drop(columns=["Band gap (HSE06) [eV]"])
y_ = df["Band gap (HSE06) [eV]"]

def split(x = x_, y = y_, second_size = 0.15):
  x_train, x_test, y_train, y_test = tts(x, y, test_size = second_size)
  return x_train, y_train, x_test, y_test

# Default k for k-fold
k_ = 4
def k_fold(x = x_, y = y_, k = k_, scale = True):
  kf = KFold(n_splits = k, shuffle = True)
  for train_indices, test_indices in kf.split(x):
    x_train = x.iloc[train_indices]
    y_train = y.iloc[train_indices]
    x_test = x.iloc[test_indices]
    y_test = y.iloc[test_indices]

    if scale:
      scaler = StandardScaler()
      x_train = scaler.fit_transform(x_train)
      x_test = scaler.transform(x_test)

    yield x_train, y_train, x_test, y_test