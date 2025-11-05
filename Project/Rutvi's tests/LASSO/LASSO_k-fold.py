import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# LassoCV with 5-fold CV
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train, y_train)

# Predictions
y_pred = lasso.predict(X_test)

# Metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"\nLASSO with 5-Fold CV")
print(f"Optimal alpha: {lasso.alpha_:.6f}")
print(f"R² Score: {r2:.4f}")
print(f"MAE   : {mae:.4f}")
print(f"RMSE  : {rmse:.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Bandgap [eV]')
plt.ylabel('Predicted Bandgap [eV]')
plt.title('LASSO (5-Fold CV): Actual vs Predicted')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot: Error distribution
errors = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Prediction Error Distribution (LASSO 5-Fold CV)')
plt.xlabel('Error (Actual - Predicted)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
