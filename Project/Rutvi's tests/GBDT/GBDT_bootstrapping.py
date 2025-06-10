# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

# Load the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]'])

# Separate features and target
X = df.drop(columns=['Band gap (HSE06) [eV]']) 
y = df['Band gap (HSE06) [eV]']
X = X.select_dtypes(include=[float, int])  # Select only numeric columns

# Handle missing values by imputing with the mean
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Re-split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Bootstrapping
n_iterations = 100
mae_list = []
rmse_list = []
r2_list = []
errors = []
last_y_pred = None  # Save the last prediction for plotting

for i in range(n_iterations):
    X_train_resampled, y_train_resampled = resample(X_train, y_train, random_state=i)
    
    model.fit(X_train_resampled, y_train_resampled)
    y_pred = model.predict(X_test)

    mae_list.append(mean_absolute_error(y_test, y_pred))
    rmse_list.append(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2_list.append(r2_score(y_test, y_pred))
    errors.append(y_test - y_pred)

    if i == n_iterations - 1:
        last_y_pred = y_pred  # Save last prediction for regression plot

# Mean and standard deviation of metrics
print(f'Mean Absolute Error (MAE) - Bootstrapping: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}')
print(f'Root Mean Squared Error (RMSE) - Bootstrapping: {np.mean(rmse_list):.4f} ± {np.std(rmse_list):.4f}')
print(f'R² - Bootstrapping: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}')

# Flatten the list of errors for plotting
errors_flat = np.concatenate(errors)

# Plot error distribution
plt.figure(figsize=(8, 6))
sns.histplot(errors_flat, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted) - Bootstrapping')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# Plot regression: last iteration's prediction
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=last_y_pred, scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Regression Plot: True vs Predicted Bandgap - Last Bootstrap Iteration')
plt.xlabel('True Bandgap')
plt.ylabel('Predicted Bandgap')
plt.grid(True)
plt.show()
