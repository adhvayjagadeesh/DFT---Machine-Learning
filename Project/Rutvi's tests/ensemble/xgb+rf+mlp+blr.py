# type: ignore

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor
from scipy.optimize import minimize
from scipy.stats import spearmanr

# Load dataset
df = pd.read_csv("../../c2db_data/Final_rect_materials_filled_in_correctly.csv")

# Drop columns with >90% missing data + identifiers
drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=[
    'Direct band gap (PBE) [eV]', 'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]', 'Band gap (G₀W₀) [eV]',
    'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1', 'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]'
], inplace=True)
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Fill missing numerical values
df.fillna(df.mean(numeric_only=True), inplace=True)

# Features and target
X = df.drop(columns=["Band gap (HSE06) [eV]"])
y = df["Band gap (HSE06) [eV]"]

# Train-test split
X_train, X_holdout, y_train_full, y_test_final = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_holdout_scaled = scaler.transform(X_holdout)

# K-Fold setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for predictions
all_true = []
xgb_all = []
rf_all = []
mlp_all = []
blr_all = []

# Cross-validation loop
for train_idx, val_idx in kf.split(X_train_scaled):
    X_train, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_train, y_val = y_train_full.iloc[train_idx], y_train_full.iloc[val_idx]

    # Define models
    xgb = XGBRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
    blr = BayesianRidge()

    # Train models
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)
    blr.fit(X_train, y_train)

    # Predict
    xgb_pred = xgb.predict(X_val)
    rf_pred = rf.predict(X_val)
    mlp_pred = mlp.predict(X_val)
    blr_pred = blr.predict(X_val)

    # Store results
    all_true.extend(y_val)
    xgb_all.extend(xgb_pred)
    rf_all.extend(rf_pred)
    mlp_all.extend(mlp_pred)
    blr_all.extend(blr_pred)

# Convert predictions to numpy arrays
y_true = np.array(all_true)
pred_matrix = np.vstack([xgb_all, rf_all, mlp_all, blr_all]).T

# Define loss and optimize weights
def loss_fn(weights):
    blended = np.dot(pred_matrix, weights)
    return mean_absolute_error(y_true, blended)

init_weights = [1/4] * 4
bounds = [(0, 1)] * 4
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

result = minimize(loss_fn, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Final training on full data
xgb_final = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42)
rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
mlp_final = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)
blr_final = BayesianRidge()

xgb_final.fit(X_train_scaled, y_train_full)
rf_final.fit(X_train_scaled, y_train_full)
mlp_final.fit(X_train_scaled, y_train_full)
blr_final.fit(X_train_scaled, y_train_full)

# Final predictions on holdout set
xgb_pred_final = xgb_final.predict(X_holdout_scaled)
rf_pred_final = rf_final.predict(X_holdout_scaled)
mlp_pred_final = mlp_final.predict(X_holdout_scaled)
blr_pred_final = blr_final.predict(X_holdout_scaled)

# Hybrid prediction
pred_matrix_final = np.vstack([xgb_pred_final, rf_pred_final, mlp_pred_final, blr_pred_final]).T

hybrid_pred_final = np.dot(pred_matrix_final, optimal_weights)

# Evaluation
mae = mean_absolute_error(y_test_final, hybrid_pred_final)
r2 = r2_score(y_test_final, hybrid_pred_final)
rmse = np.sqrt(mean_squared_error(y_test_final, hybrid_pred_final))
spearman, _ = spearmanr(y_holdout, hybrid_pred)
n = len(y_test_final)
k = X.shape[1]
adjusted_r2 = 1 - (1 - r2) * (n - 1) / (n - k - 1)

print("Optimal Weights (XGB, RF, MLP, BLR):", optimal_weights)
print(f"Hybrid Model MAE (Test Set): {mae:.4f}")
print(f"Hybrid Model R² (Test Set): {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
print(f"Spearman correlation: {spearman:.4f}")
print(f"Hybrid Model Adjusted R² (Test Set): {adjusted_r2:.4f}")

# Plot Actual vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_test_final, hybrid_pred_final, color='purple', alpha=0.7, label="Hybrid Prediction")
plt.plot([min(y_test_final), max(y_test_final)], [min(y_test_final), max(y_test_final)], 'r--')
plt.xlabel("Actual Band gap (HSE06) [eV]")
plt.ylabel("Predicted Band gap (HSE06) [eV]")
plt.title("Hybrid Model: Actual vs Predicted Band gap")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot error distribution
errors = hybrid_pred_final - y_test_final
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=50, color='teal', alpha=0.7, edgecolor='black')
plt.title("Hybrid Prediction Error Distribution")
plt.xlabel("Prediction Error (eV)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
