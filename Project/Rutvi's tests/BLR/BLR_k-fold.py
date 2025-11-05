import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt

# Load and prepare the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")

drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=[
  'Direct band gap (PBE) [eV]',
  'Direct band gap (PBE) [eV].1',
  'Band gap (PBE) [eV]',
  'Band gap (G₀W₀) [eV]',
  'Direct band gap (G₀W₀) [eV]',
  'Direct band gap (HSE06) [eV]',
  'Direct band gap (HSE06) [eV].1',
  'CBM wrt. vacuum (PBE) [eV]',
  'VBM wrt. vacuum (PBE) [eV]'
], inplace=True)
df.drop(columns=drop_cols, inplace=True, errors='ignore')

cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
  df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df.drop(columns=["Band gap (HSE06) [eV]"])
y = df["Band gap (HSE06) [eV]"]
X = X.fillna(X.mean())
X_scaled = StandardScaler().fit_transform(X)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)
maes, rmses, r2s = [], [], []

for train_index, test_index in kf.split(X_scaled):
  X_train, X_test = X_scaled[train_index], X_scaled[test_index]
  y_train, y_test = y.iloc[train_index], y.iloc[test_index]

  model = BayesianRidge()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)

  maes.append(mean_absolute_error(y_test, y_pred))
  rmses.append(sqrt(mean_squared_error(y_test, y_pred)))
  r2s.append(r2_score(y_test, y_pred))

print("Bayesian Ridge - 5-Fold Cross-Validation:")
print(f"MAE  : {np.mean(maes):.4f} ± {np.std(maes):.4f}")
print(f"RMSE : {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
print(f"R²   : {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
