import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv")
df = df.drop(columns=[
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (PBE) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]'
])

# Set the target column
target_col = 'Band gap (PBE) [eV]'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encode categorical variables
X = pd.get_dummies(X)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)
feature_names = imputer.get_feature_names_out(X.columns)
X = pd.DataFrame(X_imputed, columns=feature_names)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Fit LASSO with cross-validation to select alpha
lasso_cv = LassoCV(cv=5, random_state=42)
lasso_cv.fit(X_train_scaled, y_train)
optimal_alpha = lasso_cv.alpha_

print(f"\nOptimal alpha selected by LASSO CV: {optimal_alpha:.6f}")

# ----------------------------
# Bootstrapping
# ----------------------------
n_iterations = 100
rng = np.random.default_rng(42)

mae_bootstrap = []
r2_bootstrap = []
bootstrap_errors = []

for i in range(n_iterations):
    # Sample with replacement
    indices = rng.integers(0, len(X_train_scaled), len(X_train_scaled))
    X_boot = X_train_scaled[indices]
    y_boot = y_train.iloc[indices]

    # Fit LASSO with selected alpha
    model = Lasso(alpha=optimal_alpha)
    model.fit(X_boot, y_boot)

    # Predict on test set
    y_pred_boot = model.predict(X_test_scaled)
    mae_bootstrap.append(mean_absolute_error(y_test, y_pred_boot))
    r2_bootstrap.append(r2_score(y_test, y_pred_boot))

    # Collect errors for bootstrapping distribution
    bootstrap_errors.append(y_test - y_pred_boot)

# ----------------------------
# Final model on full train set
# ----------------------------
final_model = Lasso(alpha=optimal_alpha)
final_model.fit(X_train_scaled, y_train)
y_pred = final_model.predict(X_test_scaled)

mae_final = mean_absolute_error(y_test, y_pred)
r2_final = r2_score(y_test, y_pred)

print(f"\nFinal Model Performance on Test Set:")
print(f"MAE: {mae_final:.4f}")
print(f"R² : {r2_final:.4f}")

print(f"\nBootstrapping Summary ({n_iterations} iterations):")
print(f"MAE: Mean = {np.mean(mae_bootstrap):.4f}, Std = {np.std(mae_bootstrap):.4f}")
print(f"R² : Mean = {np.mean(r2_bootstrap):.4f}, Std = {np.std(r2_bootstrap):.4f}")

# ----------------------------
# Plots
# ----------------------------

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Bandgap [eV]')
plt.ylabel('Predicted Bandgap [eV]')
plt.title('LASSO: Actual vs Predicted Bandgap (Test Set)')
plt.grid(True)
plt.tight_layout()
plt.show()

# Histograms of bootstrap errors
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(mae_bootstrap, bins=20, color='skyblue', edgecolor='black')
plt.axvline(mae_final, color='red', linestyle='--', label=f'Final MAE = {mae_final:.3f}')
plt.title('Bootstrap MAE Distribution')
plt.xlabel('MAE')
plt.ylabel('Frequency')
plt.legend()

plt.subplot(1, 2, 2)
plt.hist(r2_bootstrap, bins=20, color='lightgreen', edgecolor='black')
plt.axvline(r2_final, color='red', linestyle='--', label=f'Final R² = {r2_final:.3f}')
plt.title('Bootstrap R² Distribution')
plt.xlabel('R²')
plt.ylabel('Frequency')
plt.legend()

plt.tight_layout()
plt.show()

# ----------------------------
# Error Distribution of Final Model (Test Set)
# ----------------------------
test_errors = y_test - y_pred
plt.figure(figsize=(8, 5))
plt.hist(test_errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Prediction Error Distribution (Test Set)')
plt.xlabel('Error (Actual - Predicted Bandgap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# ----------------------------
# Feature Importance
# ----------------------------
coef_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': final_model.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 most influential features:")
print(coef_df.head(10))

# ----------------------------
# Error Distribution for Bootstrapped Iterations
# ----------------------------
all_bootstrap_errors = np.concatenate(bootstrap_errors)

plt.figure(figsize=(8, 5))
plt.hist(all_bootstrap_errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Bootstrap Prediction Error Distribution')
plt.xlabel('Error (Actual - Predicted Bandgap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
