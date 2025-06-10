import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
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
X = X.select_dtypes(include=[float, int])  # Keep only numeric columns

# Impute missing values with mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=3,
    random_state=42
)

# Bootstrapping parameters
n_bootstrap = 100
n_samples = X_imputed.shape[0]
bootstrap_preds = np.zeros((n_bootstrap, n_samples))

# Bootstrapping loop
np.random.seed(42)
for i in range(n_bootstrap):
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_bootstrap = X_imputed[bootstrap_indices]
    y_bootstrap = y.iloc[bootstrap_indices]

    model.fit(X_bootstrap, y_bootstrap)
    preds = model.predict(X_imputed)
    bootstrap_preds[i] = preds

# Aggregate predictions
y_pred_bootstrap = bootstrap_preds.mean(axis=0)

# Calculate evaluation metrics
mae_bootstrap = mean_absolute_error(y, y_pred_bootstrap)
rmse_bootstrap = np.sqrt(mean_squared_error(y, y_pred_bootstrap))
r2_bootstrap = r2_score(y, y_pred_bootstrap)

# Print metrics
print("Model Performance - Bootstrapping on Full Dataset:")
print(f"  R²   : {r2_bootstrap:.4f}")
print(f"  MAE  : {mae_bootstrap:.4f}")
print(f"  RMSE : {rmse_bootstrap:.4f}")

# Plot error distribution
error_bootstrap = y - y_pred_bootstrap
plt.figure(figsize=(8, 6))
sns.histplot(error_bootstrap, kde=True, bins=30, color='lightcoral', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted) - Bootstrap')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot regression (true vs predicted)
plt.figure(figsize=(8, 6))
sns.regplot(x=y, y=y_pred_bootstrap, scatter_kws={'s': 50, 'alpha': 0.6}, line_kws={'color': 'blue'})
plt.title('Regression Plot: True vs Predicted Bandgap - Bootstrap')
plt.xlabel('True Bandgap (eV)')
plt.ylabel('Predicted Bandgap (eV)')
plt.grid(True)
plt.show()
