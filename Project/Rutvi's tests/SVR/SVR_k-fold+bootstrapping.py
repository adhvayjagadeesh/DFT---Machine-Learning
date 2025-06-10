import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]'])

# Separate features and target
X = df.drop(columns=['Band gap (HSE06) [eV]'])  
y = df['Band gap (HSE06) [eV]']

# Keep only numeric features
X = X.select_dtypes(include=[float, int])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# K-Fold + Bootstrapping
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_bootstrap = 50
mae_scores, rmse_scores, r2_scores = [], [], []

fold_idx = 1
for train_idx, test_idx in kf.split(X_scaled):
    print(f"\nFold {fold_idx}...")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    for i in range(n_bootstrap):
        # Bootstrap sampling
        bootstrap_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot, y_boot = X_train[bootstrap_idx], y_train.iloc[bootstrap_idx]

        # Train SVR
        model = SVR(kernel='rbf')
        model.fit(X_boot, y_boot)

        # Predict
        y_pred = model.predict(X_test)

        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        rmse_scores.append(rmse)
        r2_scores.append(r2)

        # Save final fold + final bootstrap predictions
        if fold_idx == 5 and i == n_bootstrap - 1:
            final_y_test = y_test
            final_y_pred = y_pred
            final_mae = mae
            final_rmse = rmse
            final_r2 = r2

    fold_idx += 1

# Summary of metrics across all bootstrapped folds
print("\nBootstrapped K-Fold Results (All Folds)")
print(f"Average MAE:  {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Average RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Average R²:   {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# Final fold's last bootstrap metrics
print("\nFinal Fold - Last Bootstrap Metrics")
print(f"MAE:  {final_mae:.4f}")
print(f"RMSE: {final_rmse:.4f}")
print(f"R²:   {final_r2:.4f}")

# --- Regression Plot (Final Fold - Last Bootstrap) ---
plt.figure(figsize=(10, 6))
plt.scatter(final_y_test, final_y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR: Actual vs Predicted (Final Fold - Last Bootstrap)')
plt.legend()
plt.show()

# --- Error Distribution Plot ---
errors = final_y_test - final_y_pred

plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, edgecolor='black', alpha=0.7)
plt.title("Prediction Error Distribution (Final Fold - Last Bootstrap)")
plt.xlabel("Error (Actual - Predicted Band Gap)")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
