import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import BayesianRidge
from sklearn.model_selection import train_test_split
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

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42
)

# Train Bayesian Linear Regression model
blr = BayesianRidge()
blr.fit(X_train, y_train)

# Predict on test set
y_pred = blr.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"Bayesian Linear Regression Results:")
print(f"MAE: {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2: {r2:.4f}")

# Regression plot with error bars
residuals = y_test - y_pred

plt.figure(figsize=(7, 6))
plt.errorbar(
    y_test, y_pred, yerr=np.abs(residuals),
    fmt='o', ecolor='lightcoral', alpha=0.6, label='Predictions with Error Bars'
)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2, label='Ideal Fit')
plt.xlabel('Actual Band gap (PBE) [eV]')
plt.ylabel('Predicted Band gap (PBE) [eV]')
plt.title('Bayesian Linear Regression: Actual vs Predicted with Error Bars')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Residual error histogram
plt.figure(figsize=(6, 4))
plt.hist(residuals, bins=40, color='teal', alpha=0.7, edgecolor='black')
plt.title('Residual Error Distribution')
plt.xlabel('Prediction Error (eV)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()
