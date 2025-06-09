import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]','Band gap (HSE06) [eV]', 'Direct band gap (HSE06) [eV]'])

# Separate features and target
X = df.drop(columns=['Band gap (PBE) [eV]']) 
y = df['Band gap (PBE) [eV]']
X = X.select_dtypes(include=[float, int])  # Select only numeric columns

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Parameters for bootstrapping
n_bootstrap = 100
n_samples = X_imputed.shape[0]

# Storage for predictions from each bootstrap model
bootstrap_preds = np.zeros((n_bootstrap, n_samples))

# Perform bootstrapping
np.random.seed(42)
for i in range(n_bootstrap):
    # Sample indices with replacement for bootstrap sample
    bootstrap_indices = np.random.choice(n_samples, size=n_samples, replace=True)
    X_bootstrap = X_imputed[bootstrap_indices]
    y_bootstrap = y.iloc[bootstrap_indices]
    
    # Fit model on bootstrap sample
    model.fit(X_bootstrap, y_bootstrap)
    
    # Predict on the full dataset (out-of-bag prediction)
    preds = model.predict(X_imputed)
    bootstrap_preds[i] = preds

# Aggregate predictions by averaging over bootstrap models
y_pred_bootstrap = bootstrap_preds.mean(axis=0)

# Evaluate metrics
mae_bootstrap = mean_absolute_error(y, y_pred_bootstrap)
r2_bootstrap = r2_score(y, y_pred_bootstrap)

print(f'Mean Absolute Error (MAE) - Bootstrap: {mae_bootstrap:.4f}')
print(f'R² - Bootstrap: {r2_bootstrap:.4f}')

# Plot error distribution (actual vs predicted errors for bootstrap)
error_bootstrap = y - y_pred_bootstrap
plt.figure(figsize=(8, 6))
sns.histplot(error_bootstrap, kde=True, bins=30, color='lightcoral', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted) - Bootstrap')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

# Plot regression plot (true vs predicted values for bootstrap)
plt.figure(figsize=(8, 6))
sns.regplot(x=y, y=y_pred_bootstrap, scatter_kws={'s': 50}, line_kws={'color': 'blue'})
plt.title('Regression Plot: True vs Predicted Bandgap - Bootstrap')
plt.xlabel('True Bandgap')
plt.ylabel('Predicted Bandgap')
plt.show()
