import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc, RocCurveDisplay
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
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

# Separate features and target
X = df.drop(columns=['Band gap (HSE06) [eV]']) 
y = df['Band gap (HSE06) [eV]']

# Convert categorical variables to dummies
X = pd.get_dummies(X, drop_first=True)

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Initialize model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

n_samples = X_imputed.shape[0]
n_bootstrap = 100
bootstrap_preds = np.zeros((n_bootstrap, n_samples))

np.random.seed(42)
for i in range(n_bootstrap):
  bootstrap_idx = np.random.choice(n_samples, n_samples, replace=True)
  X_bootstrap = X_imputed[bootstrap_idx]
  y_bootstrap = y.iloc[bootstrap_idx]

  model.fit(X_bootstrap, y_bootstrap)
  preds = model.predict(X_imputed)
  bootstrap_preds[i] = preds

# Aggregate predictions by mean
y_pred_bootstrap = bootstrap_preds.mean(axis=0)

mae = mean_absolute_error(y, y_pred_bootstrap)
rmse = np.sqrt(mean_squared_error(y, y_pred_bootstrap))
r2 = r2_score(y, y_pred_bootstrap)

print(f"Bootstrap Mean MAE: {mae:.4f}")
print(f"Bootstrap Mean RMSE: {rmse:.4f}")
print(f"Bootstrap Mean R²: {r2:.4f}")

# Plot error distribution
error = y - y_pred_bootstrap
plt.figure(figsize=(8,6))
sns.histplot(error, kde=True, bins=30, color='lightcoral', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted) - Bootstrap')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Regression plot
plt.figure(figsize=(8,6))
sns.regplot(x=y, y=y_pred_bootstrap, scatter_kws={'s':50, 'alpha':0.6}, line_kws={'color':'blue'})
plt.title('Regression Plot: True vs Predicted Bandgap - Bootstrap')
plt.xlabel('True Bandgap')
plt.ylabel('Predicted Bandgap')
plt.grid(True)
plt.show()

T = 1.0  # threshold in eV to define positive class (change as needed)
y_bin = (y > T).astype(int)

if y_bin.nunique() < 2:
  print(f"ROC skipped: need both classes present for threshold T={T}. Found classes: {y_bin.unique()}")
else:
  y_scores = y_pred_bootstrap
  fpr, tpr, _ = roc_curve(y_bin, y_scores)
  roc_auc = auc(fpr, tpr)

  disp = RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc)
  fig, ax = plt.subplots(figsize=(7, 6))
  disp.plot(ax=ax)
  ax.plot([0, 1], [0, 1], '--', color='gray')
  ax.set_title(f'ROC Curve (regressor scores, T={T} eV) AUC={roc_auc:.3f}')
  fig.tight_layout()
  fig.savefig('gbdt_bootstrap_roc.png', dpi=200)
  plt.show()
