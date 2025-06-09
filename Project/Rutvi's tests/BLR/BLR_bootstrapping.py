import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")

# Drop columns with >90% missing data + identifiers
drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df.drop(columns=[
    'Formula',
    'Band gap (HSE06) [eV]',
    'Direct band gap (PBE) [eV]', 
    'Direct band gap (HSE06) [eV]'
], inplace=True)
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Define features and target
X = df.drop(columns=["Band gap (PBE) [eV]"])
y = df["Band gap (PBE) [eV]"]

# Fill missing values in X if any
X = X.fillna(X.mean())

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Number of bootstrap samples
n_bootstrap = 50

# Store predictions for each bootstrap model
all_bootstrap_preds = np.zeros((len(y), n_bootstrap))

for b in range(n_bootstrap):
    # Bootstrap sample indices (with replacement)
    bootstrap_indices = np.random.choice(len(X_scaled), size=len(X_scaled), replace=True)
    X_bootstrap = X_scaled[bootstrap_indices]
    y_bootstrap = y.iloc[bootstrap_indices]

    # Train BLR on bootstrap sample
    blr = BayesianRidge()
    blr.fit(X_bootstrap, y_bootstrap)

    # Predict on entire dataset
    all_bootstrap_preds[:, b] = blr.predict(X_scaled)

# Average prediction across bootstraps
y_pred_mean = all_bootstrap_preds.mean(axis=1)
# Standard deviation as uncertainty measure
y_pred_std = all_bootstrap_preds.std(axis=1)

# Calculate metrics
mae = mean_absolute_error(y, y_pred_mean)
rmse = sqrt(mean_squared_error(y, y_pred_mean))
r2 = r2_score(y, y_pred_mean)

print(f"Bootstrapped BLR Results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

# Plot Actual vs Predicted with error bars (± std dev)
plt.figure(figsize=(7, 6))
plt.errorbar(
    y, y_pred_mean, yerr=y_pred_std, fmt='o',
    ecolor='orange', alpha=0.6, label='Predictions ± Std Dev'
)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Band gap (PBE) [eV]')
plt.ylabel('Predicted Band gap (PBE) [eV]')
plt.title('Bootstrapped Bayesian Linear Regression\nPredictions with Uncertainty')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
