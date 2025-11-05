import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt

# Load dataset
df = pd.read_csv("Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
# Drop irrelevant columns
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

# Encode categorical columns
cat_cols = df.select_dtypes(include='object').columns
for col in cat_cols:
  le = LabelEncoder()
  df[col] = le.fit_transform(df[col].astype(str))

# Define features and target
target = 'Band gap (HSE06) [eV]'
X = df.drop(columns=[target])
y = df[target].values

# Bootstrap parameters
n_bootstraps = 100
n_samples = X.shape[0]

# Containers for metrics and predictions
r2_scores = []
mae_scores = []
rmse_scores = []
all_y_test = []
all_y_pred = []

np.random.seed(42)

for i in range(n_bootstraps):
  # Bootstrap sampling with replacement
  indices = np.random.choice(n_samples, n_samples, replace=True)
  X_boot = X.iloc[indices]
  y_boot = y[indices]
  
  # Out-Of-Bag (OOB) samples (not in bootstrap sample)
  oob_mask = ~np.isin(range(n_samples), indices)
  X_oob = X.iloc[oob_mask]
  y_oob = y[oob_mask]
  
  if len(y_oob) == 0:
    # In rare cases, all samples might be selected; skip this bootstrap
    continue
  
  # Train model on bootstrap sample
  rf = RandomForestRegressor(n_estimators=100, random_state=42)
  rf.fit(X_boot, y_boot)
  
  # Predict on OOB samples
  y_pred_oob = rf.predict(X_oob)
  
  # Store metrics
  r2_scores.append(r2_score(y_oob, y_pred_oob))
  mae_scores.append(mean_absolute_error(y_oob, y_pred_oob))
  rmse_scores.append(sqrt(mean_squared_error(y_oob, y_pred_oob)))
  
  # Collect predictions for plotting
  all_y_test.extend(y_oob)
  all_y_pred.extend(y_pred_oob)

# Aggregate metrics
print(f"Bootstrap results ({n_bootstraps} iterations):")
print(f"Mean R²:  {np.mean(r2_scores):.4f} ± {np.std(r2_scores):.4f}")
print(f"Mean MAE: {np.mean(mae_scores):.4f} ± {np.std(mae_scores):.4f}")
print(f"Mean RMSE:{np.mean(rmse_scores):.4f} ± {np.std(rmse_scores):.4f}")

# Convert lists to arrays for plotting
all_y_test = np.array(all_y_test)
all_y_pred = np.array(all_y_pred)
errors = all_y_test - all_y_pred

# Regression plot (all OOB predictions)
plt.figure(figsize=(8, 6))
sns.regplot(x=all_y_test, y=all_y_pred, line_kws={"color": "red"}, scatter_kws={"alpha":0.5, "edgecolor":"k"})
plt.xlabel('Actual Band Gap (HSE06) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('Random Forest Bootstrap: Predicted vs Actual Band Gap')
plt.grid(True)
plt.tight_layout()
plt.show()

# Error histogram
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Bootstrap Prediction Error Distribution')
plt.xlabel('Error (Actual - Predicted Band Gap)')
plt.ylabel('Frequency')
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