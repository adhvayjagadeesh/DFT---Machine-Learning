import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, r2_score
from scipy.optimize import minimize
import xgboost as xgb

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")

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

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train_val, X_holdout, y_train_val, y_holdout = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Initialize K-Fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Storage for true and predicted values
all_true = []
svr_all = []
rf_all = []
xgb_all = []
blr_all = []

# Cross-validation loop
for train_idx, val_idx in kf.split(X_train_val):
    X_train, X_val = X_train_val[train_idx], X_train_val[val_idx]
    y_train, y_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    # Define models
    svr = SVR(kernel='rbf', C=10, epsilon=0.1)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb_model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
    blr = BayesianRidge()

    # Train models
    svr.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    xgb_model.fit(X_train, y_train)
    blr.fit(X_train, y_train)

    # Predict
    svr_pred = svr.predict(X_val)
    rf_pred = rf.predict(X_val)
    xgb_pred = xgb_model.predict(X_val)
    blr_pred = blr.predict(X_val)

    # Store predictions
    all_true.extend(y_val)
    svr_all.extend(svr_pred)
    rf_all.extend(rf_pred)
    xgb_all.extend(xgb_pred)
    blr_all.extend(blr_pred)

# Convert to arrays
y_true = np.array(all_true)
pred_matrix = np.vstack([svr_all, rf_all, xgb_all, blr_all]).T

# Optimize weights for hybrid model
def loss_fn(weights):
    blended = np.dot(pred_matrix, weights)
    return mean_absolute_error(y_true, blended)

init_weights = [1/4, 1/4, 1/4, 1/4]
bounds = [(0, 1)] * 4
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

result = minimize(loss_fn, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x

# Retrain models on full training data and evaluate on holdout
svr_final = SVR(kernel='rbf', C=10, epsilon=0.1)
rf_final = RandomForestRegressor(n_estimators=100, random_state=42)
xgb_final = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=100, random_state=42)
blr_final = BayesianRidge()

svr_final.fit(X_train_val, y_train_val)
rf_final.fit(X_train_val, y_train_val)
xgb_final.fit(X_train_val, y_train_val)
blr_final.fit(X_train_val, y_train_val)

# Final predictions on holdout set
svr_holdout = svr_final.predict(X_holdout)
rf_holdout = rf_final.predict(X_holdout)
xgb_holdout = xgb_final.predict(X_holdout)
blr_holdout = blr_final.predict(X_holdout)

pred_holdout_matrix = np.vstack([svr_holdout, rf_holdout, xgb_holdout, blr_holdout]).T
hybrid_pred = np.dot(pred_holdout_matrix, optimal_weights)

# Evaluate
mae = mean_absolute_error(y_holdout, hybrid_pred)
r2 = r2_score(y_holdout, hybrid_pred)
n = len(y_holdout)
p = X.shape[1]
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

print("Optimal Weights (SVR, RF, XGB, BLR):", optimal_weights)
print(f"Hybrid Model MAE (Test Set): {mae:.4f}")
print(f"Hybrid Model R² (Test Set): {r2:.4f}")
print(f"Hybrid Model Adjusted R² (Test Set): {adj_r2:.4f}")

# Plot: Actual vs Predicted
plt.figure(figsize=(6, 5))
plt.scatter(y_holdout, hybrid_pred, color='purple', alpha=0.7, label="Hybrid Prediction")
plt.plot([min(y_holdout), max(y_holdout)], [min(y_holdout), max(y_holdout)], 'r--')
plt.xlabel("Actual Band gap (HSE06) [eV]")
plt.ylabel("Predicted Band gap (HSE06) [eV]")
plt.title("Hybrid Model (Holdout Set): Actual vs Predicted Band gap")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# Plot error distribution
errors = hybrid_pred - y_holdout
plt.figure(figsize=(6, 4))
plt.hist(errors, bins=50, color='teal', alpha=0.7, edgecolor='black')
plt.title("Hybrid Prediction Error Distribution")
plt.xlabel("Prediction Error (eV)")
plt.ylabel("Frequency")
plt.grid(True)
plt.tight_layout()
plt.show()
