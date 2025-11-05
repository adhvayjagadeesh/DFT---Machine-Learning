import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")

# Drop irrelevant columns
df = df.drop(columns=[
  'Direct band gap (PBE) [eV]',
  'Direct band gap (PBE) [eV].1',
  'Band gap (PBE) [eV]',
  'Band gap (G₀W₀) [eV]',
  'Direct band gap (G₀W₀) [eV]',
  'Direct band gap (HSE06) [eV]',
  'Direct band gap (HSE06) [eV].1',
  'CBM wrt. vacuum (PBE) [eV]',
  'VBM wrt. vacuum (PBE) [eV]'
])

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col].astype(str))

# Define features and target
target = 'Band gap (HSE06) [eV]'
X = df.drop(columns=[target]).reset_index(drop=True)
y = df[target].reset_index(drop=True)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_bootstraps = 100

# Containers to hold metrics and predictions across all folds and bootstraps
all_y_test = []
all_y_pred = []
fold_metrics = []

np.random.seed(42)

for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X)):
  print(f"\nFold {fold_idx+1}")
  X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
  y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

  r2_scores = []
  mae_scores = []
  rmse_scores = []

  fold_y_test = []
  fold_y_pred = []

  n_train_samples = X_train.shape[0]

  for b in range(n_bootstraps):
    # Bootstrap sample from training data
    bootstrap_idx = np.random.choice(n_train_samples, n_train_samples, replace=True)
    X_boot = X_train.iloc[bootstrap_idx]
    y_boot = y_train.iloc[bootstrap_idx]

    # Train Random Forest on bootstrap sample
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_boot, y_boot)

    # Predict on test fold
    y_pred = rf.predict(X_test)

    # Metrics
    r2_scores.append(r2_score(y_test, y_pred))
    mae_scores.append(mean_absolute_error(y_test, y_pred))
    rmse_scores.append(sqrt(mean_squared_error(y_test, y_pred)))

    fold_y_test.extend(y_test)
    fold_y_pred.extend(y_pred)

  # Print mean and std metrics for fold
  print(f"  Mean R²  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
  print(f"  Mean MAE : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
  print(f"  Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

  # Collect fold predictions
  all_y_test.extend(fold_y_test)
  all_y_pred.extend(fold_y_pred)
  fold_metrics.append({
    'r2_mean': np.mean(r2_scores),
    'r2_std': np.std(r2_scores),
    'mae_mean': np.mean(mae_scores),
    'mae_std': np.std(mae_scores),
    'rmse_mean': np.mean(rmse_scores),
    'rmse_std': np.std(rmse_scores),
  })

# Convert all predictions to arrays for plotting
all_y_test = np.array(all_y_test)
all_y_pred = np.array(all_y_pred)
errors = all_y_test - all_y_pred

# Overall aggregated metrics
print("\nOverall metrics across all folds and bootstraps:")
print(f"R²:  {r2_score(all_y_test, all_y_pred):.4f}")
print(f"MAE: {mean_absolute_error(all_y_test, all_y_pred):.4f}")
print(f"RMSE:{sqrt(mean_squared_error(all_y_test, all_y_pred)):.4f}")

# Regression plot: Actual vs Predicted (all folds + bootstraps)
plt.figure(figsize=(8, 6))
sns.regplot(x=all_y_test, y=all_y_pred, line_kws={"color": "red"}, scatter_kws={"alpha":0.3, "edgecolor":"k"})
plt.xlabel('Actual Band Gap (HSE06) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('5-Fold CV with 100 Bootstraps: Predicted vs Actual')
plt.grid(True)
plt.tight_layout()
plt.show()

# Error distribution histogram
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=40, edgecolor='k', alpha=0.7)
plt.title('Error Distribution (Actual - Predicted)')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
