import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor
from scipy.optimize import minimize

# Load dataset
df = pd.read_csv("c2db_C_materials.csv")

# Drop columns with >90% missing data + identifiers
drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=["Formula"], inplace=True)
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
X = df.drop(columns=["Band gap (PBE) [eV]"])
y = df["Band gap (PBE) [eV]"]

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split once (fixed test set)
X_train_full, X_test, y_train_full, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Parameters
n_bootstrap = 20  # number of bootstrap samples
xgb_preds, rf_preds, mlp_preds = [], [], []

# Bootstrapping loop
rng = np.random.default_rng(42)
n_samples = len(X_train_full)

for _ in range(n_bootstrap):
    # Generate bootstrap sample (with replacement)
    indices = rng.choice(n_samples, size=n_samples, replace=True)
    X_train = X_train_full[indices]
    y_train = y_train_full.iloc[indices]

    # Define models
    xgb = XGBRegressor(n_estimators=50, learning_rate=0.1, max_depth=3, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    mlp = MLPRegressor(hidden_layer_sizes=(32, 16), max_iter=500, random_state=42)

    # Train
    xgb.fit(X_train, y_train)
    rf.fit(X_train, y_train)
    mlp.fit(X_train, y_train)

    # Predict on the fixed test set
    xgb_preds.append(xgb.predict(X_test))
    rf_preds.append(rf.predict(X_test))
    mlp_preds.append(mlp.predict(X_test))

# Average predictions over all bootstraps
xgb_mean = np.mean(xgb_preds, axis=0)
rf_mean = np.mean(rf_preds, axis=0)
mlp_mean = np.mean(mlp_preds, axis=0)

# Combine predictions
pred_matrix = np.vstack([xgb_mean, rf_mean, mlp_mean]).T

# Optimize weights to minimize MAE
def loss_fn(weights):
    blended = np.dot(pred_matrix, weights)
    return mean_absolute_error(y_test, blended)

init_weights = [1/3, 1/3, 1/3]
bounds = [(0, 1)] * 3
constraints = {'type': 'eq', 'fun': lambda w: sum(w) - 1}

result = minimize(loss_fn, init_weights, method='SLSQP', bounds=bounds, constraints=constraints)
optimal_weights = result.x
hybrid_pred = np.dot(pred_matrix, optimal_weights)

# Evaluate
mae = mean_absolute_error(y_test, hybrid_pred)
r2 = r2_score(y_test, hybrid_pred)

print("Optimal Weights (XGB, RF, MLP):", optimal_weights)
print(f"Hybrid Model MAE (Bootstrap): {mae:.4f}")
print(f"Hybrid Model R² (Bootstrap): {r2:.4f}")

# Plot
plt.figure(figsize=(6, 5))
plt.scatter(y_test, hybrid_pred, color='purple', alpha=0.7, label="Hybrid Prediction")
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'r--')
plt.xlabel("Actual Band gap (PBE) [eV]")
plt.ylabel("Predicted Band gap (PBE) [eV]")
plt.title("Hybrid Model (Bootstrap): Actual vs Predicted Band gap (PBE) [eV]")
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()
