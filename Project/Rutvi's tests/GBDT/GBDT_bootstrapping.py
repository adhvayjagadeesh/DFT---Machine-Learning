# Import necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

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

# Re-split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Initialize the Gradient Boosting model
model = GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Bootstrapping: Perform bootstrapping with replacement to train and evaluate the model multiple times
n_iterations = 100  # Number of bootstrap iterations
mae_list = []
r2_list = []
errors = []  # To store the errors for each iteration

for i in range(n_iterations):
    # Bootstrap sampling
    X_train_resampled, y_train_resampled = resample(X_train, y_train, random_state=i)
    
    # Train the model on the bootstrapped sample
    model.fit(X_train_resampled, y_train_resampled)
    
    # Predict on the test set
    y_pred = model.predict(X_test)
    
    # Calculate the MAE and R² for each iteration
    mae_list.append(mean_absolute_error(y_test, y_pred))
    r2_list.append(r2_score(y_test, y_pred))
    
    # Store the errors (actual - predicted) for each iteration
    errors.append(y_test - y_pred)

# Calculate the average MAE and R² from the bootstrap iterations
mae_bootstrap = np.mean(mae_list)
r2_bootstrap = np.mean(r2_list)

print(f'Mean Absolute Error (MAE) - Bootstrapping: {mae_bootstrap}')
print(f'R² - Bootstrapping: {r2_bootstrap}')

# Flatten the list of errors for plotting
errors_flat = np.concatenate(errors)

# Plot error distribution (actual vs predicted errors)
plt.figure(figsize=(8, 6))
sns.histplot(errors_flat, kde=True, bins=30, color='skyblue', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted) - Bootstrapping')
plt.xlabel('Error')
plt.ylabel('Frequency')
plt.show()

# Plot regression plot (true vs predicted values for bootstrapping)
plt.figure(figsize=(8, 6))
sns.regplot(x=y_test, y=np.mean(errors_flat.reshape(n_iterations, -1), axis=0) + y_test, scatter_kws={'s': 50}, line_kws={'color': 'red'})
plt.title('Regression Plot: True vs Predicted Bandgap - Bootstrapping')
plt.xlabel('True Bandgap')
plt.ylabel('Predicted Bandgap')
plt.show()
