import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.utils import resample
from math import sqrt

# Load and prepare the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")

# Drop unnecessary and high-missing columns
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

# Encode categorical features
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

# Define features and target
X = df.drop(columns=["Band gap (HSE06) [eV]"])
y = df["Band gap (HSE06) [eV]"]

# Impute and scale
X = X.fillna(X.mean())
X_scaled = StandardScaler().fit_transform(X)

# K-Fold + Bootstrapping
kf = KFold(n_splits=5, shuffle=True, random_state=42)
maes, rmses, r2s = [], [], []

np.random.seed(42)
for fold, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for i in range(100):
        X_resampled, y_resampled = resample(X_train, y_train, replace=True, random_state=i)
        model = BayesianRidge()
        model.fit(X_resampled, y_resampled)
        y_pred = model.predict(X_test)

        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(sqrt(mean_squared_error(y_test, y_pred)))
        r2s.append(r2_score(y_test, y_pred))

print("Bayesian Ridge - 5-Fold CV + 100 Bootstraps per Fold:")
print(f"MAE  : {np.mean(maes):.4f} ± {np.std(maes):.4f}")
print(f"RMSE : {np.mean(rmses):.4f} ± {np.std(rmses):.4f}")
print(f"R²   : {np.mean(r2s):.4f} ± {np.std(r2s):.4f}")
