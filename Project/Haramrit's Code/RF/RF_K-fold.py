import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt

# Load dataset
df = pd.read_csv("Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
# Drop irrelevant or redundant columns
df = df.drop(columns=[
  'Direct band gap (PBE) [eV]',
  'Direct band gap (PBE) [eV].1',
  'Band gap (PBE) [eV]',
  'Band gap (G₀W₀) [eV]',
  'Direct band gap (G₀W₀) [eV]',
  'Direct band gap (HSE06) [eV]',
  'Direct band gap (HSE06) [eV].1',
  'CBM wrt. vacuum (PBE) [eV]',
  'VBM wrt. vacuum (PBE) [eV]'
])

# Encode categorical columns BEFORE defining X
cat_cols = df.select_dtypes(include='object').columns
label_encoders = {}
for col in cat_cols:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col].astype(str))
  label_encoders[col] = le

# Define features and target
target = 'Band gap (HSE06) [eV]'
X = df.drop(columns=[target])
y = df[target]

# Split data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(
  X, y, test_size=0.2, random_state=42
)

# Train the Random Forest model
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)

# Predict and evaluate
y_pred = rf.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
rmse = sqrt(mean_squared_error(y_test, y_pred))
errors = y_test - y_pred

print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

# Plot error distribution
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Prediction Error Distribution (Random Forest)')
plt.xlabel('Error (Actual - Predicted Band Gap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot regression: Predicted vs Actual
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"}, scatter_kws={"alpha":0.7, "edgecolor":"k"})
plt.xlabel('Actual Band Gap (HSE06) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('Random Forest: Predicted vs Actual Band Gap')
plt.grid(True)
plt.tight_layout()
plt.show()

from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

T = 1.0
y_test_bin = (y_test > T).astype(int)  # true binary labels for test set
y_scores_from_reg = y_pred         # continuous regressor predictions from RFRegressor

fpr, tpr, _ = roc_curve(y_test_bin, y_scores_from_reg)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(7, 6))
plt.plot(fpr, tpr, label=f'ROC (regressor as score) AUC={roc_auc:.3f}')
plt.plot([0, 1], [0, 1], '--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC — regressor predictions as score')
plt.legend(loc='lower right')
plt.grid(True)
plt.show()