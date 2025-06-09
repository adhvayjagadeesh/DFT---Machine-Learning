import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
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

# Setup K-Fold cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Lists to store metrics and predictions for plotting
mae_scores = []
rmse_scores = []
r2_scores = []
fold_preds = []
fold_trues = []

for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    # Train BLR model
    blr = BayesianRidge()
    blr.fit(X_train, y_train)

    # Predict
    y_pred = blr.predict(X_val)

    # Calculate metrics
    mae = mean_absolute_error(y_val, y_pred)
    rmse = sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    mae_scores.append(mae)
    rmse_scores.append(rmse)
    r2_scores.append(r2)
    fold_preds.extend(y_pred)
    fold_trues.extend(y_val)

    print(f"Fold {fold} - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

# Overall metrics averaged
print("\n=== Cross-Validated Metrics ===")
print(f"Mean MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Mean RMSE: {np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")
print(f"Mean R^2: {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")

# Convert to numpy arrays for plotting
fold_trues = np.array(fold_trues)
fold_preds = np.array(fold_preds)
residuals = fold_trues - fold_preds

# Plot: Actual vs Predicted with error bars
plt.figure(figsize=(7, 6))
plt.errorbar(
    fold_trues, fold_preds, yerr=np.abs(residuals),
    fmt='o', ecolor='lightcoral', alpha=0.6, label='Predictions with Error Bars'
)
plt.plot([fold_trues.min(), fold_trues.max()], [fold_trues.min(), fold_trues.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Band gap (PBE) [eV]')
plt.ylabel('Predicted Band gap (PBE) [eV]')
plt.title('Bayesian Linear Regression: Actual vs Predicted with Error Bars (K-Fold CV)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot residual error histogram
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=40, color='teal', alpha=0.7, edgecolor='black')
plt.title('Residual Error Distribution (K-Fold CV)')
plt.xlabel('Prediction Error (eV)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
