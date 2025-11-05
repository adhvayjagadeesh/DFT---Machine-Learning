import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
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

# Set target
target_col = 'Band gap (HSE06) [eV]'
X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encode
X = pd.get_dummies(X)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_bootstraps = 100
r2_scores, mae_scores, rmse_scores = [], [], []
all_y_true, all_y_pred = [], []

# Cross-validation with bootstraps
for fold_idx, (train_idx, test_idx) in enumerate(kf.split(X_scaled)):
  X_train_fold, X_test_fold = X_scaled[train_idx], X_scaled[test_idx]
  y_train_fold, y_test_fold = y.iloc[train_idx], y.iloc[test_idx]
  
  for b in range(n_bootstraps):
    X_resampled, y_resampled = resample(X_train_fold, y_train_fold, random_state=fold_idx * 100 + b)
    
    model = LassoCV(cv=5, random_state=42)
    model.fit(X_resampled, y_resampled)
    
    y_pred = model.predict(X_test_fold)
    
    r2_scores.append(r2_score(y_test_fold, y_pred))
    mae_scores.append(mean_absolute_error(y_test_fold, y_pred))
    rmse_scores.append(np.sqrt(mean_squared_error(y_test_fold, y_pred)))
    
    all_y_true.extend(y_test_fold)
    all_y_pred.extend(y_pred)

# Metrics
print("\nLASSO with 5-Fold CV + 100 Bootstraps/Fold")
print(f"Average R²  : {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Average MAE : {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(all_y_true, all_y_pred, alpha=0.3, edgecolors='k', s=20)
plt.plot([min(all_y_true), max(all_y_true)], [min(all_y_true), max(all_y_true)], 'r--', lw=2)
plt.xlabel('Actual Bandgap [eV]')
plt.ylabel('Predicted Bandgap [eV]')
plt.title('LASSO: 5-Fold CV + 100 Bootstraps/ Fold')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Residual Error Histogram
errors = np.array(all_y_true) - np.array(all_y_pred)
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
plt.title('Residual Error Distribution (5-Fold CV with Bootstraps)')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
