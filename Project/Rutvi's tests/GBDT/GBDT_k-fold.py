# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, KFold, cross_val_predict
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv")
df = df.drop(columns=['Band gap (G₀W₀) [eV]', 'Direct band gap (PBE) [eV]', 'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]'])

# Separate features and target
X = df.drop(columns=['Band gap (PBE) [eV]']) 
y = df['Band gap (PBE) [eV]']
X = X.select_dtypes(include=[float, int])  # Select only numeric columns

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Define KFold Cross-validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Use cross-validation to get predictions
y_pred_cv = cross_val_predict(model, X_imputed, y, cv=kf)

# Calculate MAE and R2 using cross-validation
mae_cv = mean_absolute_error(y, y_pred_cv)
r2_cv = r2_score(y, y_pred_cv)

print(f'Mean Absolute Error (MAE) - K-Fold CV: {mae_cv}')
print(f'R² - K-Fold CV: {r2_cv}')

# Plot error distribution (actual vs predicted errors for cross-validation)
error_cv = y - y_pred_cv
plt.figure(figsize=(8, 6))
sns.histplot(error_cv, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted) - K-Fold CV')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

# Plot regression plot (true vs predicted values for cross-validation)
plt.figure(figsize=(8, 6))
sns.regplot(x=y, y=y_pred_cv, scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Regression Plot: True vs Predicted Bandgap - K-Fold CV')
plt.xlabel('True Bandgap')
plt.ylabel('Predicted Bandgap')
plt.show()
