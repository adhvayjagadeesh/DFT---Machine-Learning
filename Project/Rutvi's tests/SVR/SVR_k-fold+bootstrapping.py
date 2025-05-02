import pandas as pd
from sklearn.model_selection import KFold, train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score
import numpy as np
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv")
df = df.drop(columns=['Band gap (G₀W₀) [eV]', 'Direct band gap (PBE) [eV]', 'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]'])

# Separate features and target
X = df.drop(columns=['Band gap (PBE) [eV]'])  # Replace with your actual target column name
y = df['Band gap (PBE) [eV]']

# Ensure features are numeric (if any non-numeric, convert or drop them)
X = X.select_dtypes(include=[float, int])  # Select only numeric columns

# Step 1: Impute missing values (mean imputation)
imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or other strategies
X_imputed = imputer.fit_transform(X)

# Step 2: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 3: K-Fold cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
n_bootstrap = 50  # Number of bootstrap iterations
mae_scores = []
r2_scores = []

# Step 4: K-Fold + Bootstrapping
fold_idx = 1
for train_idx, test_idx in kf.split(X_scaled):
    print(f"\nFold {fold_idx}...")
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
    
    for i in range(n_bootstrap):
        # Bootstrapping: Sample with replacement
        bootstrap_idx = np.random.choice(len(X_train), size=len(X_train), replace=True)
        X_boot = X_train[bootstrap_idx]
        y_boot = y_train.iloc[bootstrap_idx]

        # Train SVR model
        model = SVR(kernel='rbf')  # You can try different kernels like 'linear' or 'poly'
        model.fit(X_boot, y_boot)

        # Predict and evaluate
        y_pred = model.predict(X_test)

        # Calculate metrics for this bootstrap
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        mae_scores.append(mae)
        r2_scores.append(r2)

        # Save predictions from last bootstrap for plotting
        if fold_idx == 5 and i == n_bootstrap - 1:
            final_y_test = y_test
            final_y_pred = y_pred

    fold_idx += 1

# Results summary
print(f"\nBootstrapped K-Fold MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Bootstrapped K-Fold R²: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

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
