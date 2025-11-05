import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVR
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=[
  'Direct band gap (PBE) [eV]', 'Direct band gap (PBE) [eV].1',
  'Band gap (PBE) [eV]', 'Band gap (G₀W₀) [eV]',
  'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]',
  'Direct band gap (HSE06) [eV].1', 'CBM wrt. vacuum (PBE) [eV]',
  'VBM wrt. vacuum (PBE) [eV]'
])

target = 'Band gap (HSE06) [eV]'
X = df.drop(columns=[target])
y = df[target]

numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()

numeric_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='mean')),
  ('scaler', StandardScaler())
])

categorical_transformer = Pipeline(steps=[
  ('imputer', SimpleImputer(strategy='most_frequent')),
  ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(transformers=[
  ('num', numeric_transformer, numeric_cols),
  ('cat', categorical_transformer, categorical_cols)
])

kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to collect scores across all folds and bootstraps
all_r2 = []
all_mae = []
all_rmse = []

n_bootstraps = 100
np.random.seed(42)

fold_num = 1
for train_index, test_index in kf.split(X):
  print(f"\nStarting fold {fold_num}/5")
  X_train_full, X_test = X.iloc[train_index], X.iloc[test_index]
  y_train_full, y_test = y.iloc[train_index], y.iloc[test_index]

  # Store bootstrap metrics per fold
  fold_r2 = []
  fold_mae = []
  fold_rmse = []

  for b in range(n_bootstraps):
    # Bootstrap sample from training fold
    X_train, y_train = resample(X_train_full, y_train_full, replace=True, random_state=fold_num*1000 + b)

    # Define and fit pipeline
    model_pipeline = Pipeline(steps=[
      ('preprocessor', preprocessor),
      ('regressor', SVR(kernel='rbf'))
    ])

    model_pipeline.fit(X_train, y_train)

    # Predict on fold test set (not bootstrapped, original fold test set)
    y_pred = model_pipeline.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    fold_r2.append(r2)
    fold_mae.append(mae)
    fold_rmse.append(rmse)

    if (b + 1) % 20 == 0:
      print(f"  Bootstrap {b + 1}/{n_bootstraps} done")

  # Aggregate fold bootstrap results
  print(f"Fold {fold_num} results:")
  print(f"  Mean R²  : {np.mean(fold_r2):.4f} ± {np.std(fold_r2):.4f}")
  print(f"  Mean MAE : {np.mean(fold_mae):.4f} ± {np.std(fold_mae):.4f}")
  print(f"  Mean RMSE: {np.mean(fold_rmse):.4f} ± {np.std(fold_rmse):.4f}")

  all_r2.extend(fold_r2)
  all_mae.extend(fold_mae)
  all_rmse.extend(fold_rmse)

  fold_num += 1

print("\nOverall nested CV + bootstrap results:")
print(f"Mean R²  : {np.mean(all_r2):.4f} ± {np.std(all_r2):.4f}")
print(f"Mean MAE : {np.mean(all_mae):.4f} ± {np.std(all_mae):.4f}")
print(f"Mean RMSE: {np.mean(all_rmse):.4f} ± {np.std(all_rmse):.4f}")
