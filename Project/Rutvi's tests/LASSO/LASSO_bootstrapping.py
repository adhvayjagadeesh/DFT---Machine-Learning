import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
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

# One-hot encode categoricals
X = pd.get_dummies(X)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
feature_names = imputer.get_feature_names_out(X.columns)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Set up storage
n_bootstraps = 100
r2_list, mae_list, rmse_list = [], [], []
y_test_all = []
y_pred_all = []

for i in range(n_bootstraps):
    X_resampled, y_resampled = resample(X_scaled, y, random_state=42 + i)
    split_idx = int(0.8 * len(X_resampled))
    X_train, X_test = X_resampled[:split_idx], X_resampled[split_idx:]
    y_train, y_test = y_resampled[:split_idx], y_resampled[split_idx:]
    
    model = LassoCV(cv=5, random_state=42)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    r2_list.append(r2_score(y_test, y_pred))
    mae_list.append(mean_absolute_error(y_test, y_pred))
    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    y_test_all.extend(y_test)
    y_pred_all.extend(y_pred)

# Convert predictions for final plots
y_test_all = np.array(y_test_all)
y_pred_all = np.array(y_pred_all)

# Final evaluation
print("\nLASSO with 100 Bootstraps")
print(f"Average R²  : {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
print(f"Average MAE : {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Average RMSE: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test_all, y_pred_all, alpha=0.4, edgecolors='k', s=20)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Bandgap [eV]')
plt.ylabel('Predicted Bandgap [eV]')
plt.title('LASSO (100 Bootstraps): Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Residual error distribution
errors = y_test_all - y_pred_all
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=40, edgecolor='black', alpha=0.7)
plt.title('Prediction Error Distribution (LASSO 100 Bootstraps)')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
