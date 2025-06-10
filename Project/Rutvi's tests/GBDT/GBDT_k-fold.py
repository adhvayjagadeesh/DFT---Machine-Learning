# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_predict, KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

# Load the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
drop_cols = df.columns[df.isnull().mean() > 0.9].tolist()
df = df.drop(columns=['Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]'])
df.drop(columns=drop_cols, inplace=True, errors='ignore')

# Separate features and target
X = df.drop(columns=['Band gap (HSE06) [eV]']) 
y = df['Band gap (HSE06) [eV]']
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

# Calculate evaluation metrics
mae_cv = mean_absolute_error(y, y_pred_cv)
rmse_cv = np.sqrt(mean_squared_error(y, y_pred_cv))
r2_cv = r2_score(y, y_pred_cv)

# Print results
print(f'Mean Absolute Error (MAE) - K-Fold CV: {mae_cv:.4f}')
print(f'Root Mean Squared Error (RMSE) - K-Fold CV: {rmse_cv:.4f}')
print(f'R² - K-Fold CV: {r2_cv:.4f}')

# Plot error distribution
error_cv = y - y_pred_cv
plt.figure(figsize=(8, 6))
sns.histplot(error_cv, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted) - K-Fold CV')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot regression plot (true vs predicted values)
plt.figure(figsize=(8, 6))
sns.regplot(x=y, y=y_pred_cv, scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Regression Plot: True vs Predicted Bandgap - K-Fold CV')
plt.xlabel('True Bandgap')
plt.ylabel('Predicted Bandgap')
plt.grid(True)
plt.show()
