import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from math import sqrt

# Load and clean the dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]'])

# Define features and target
target = 'Band gap (HSE06) [eV]'
X = df.drop(columns=[target])
y = df[target]

# Train-test split (fixed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize metrics
n_iterations = 100
mae_list = []
r2_list = []
all_y_true = []
all_y_pred = []

# Bootstrapping loop
for i in range(n_iterations):
    # Sample with replacement from training set
    indices = np.random.choice(len(X_train), size=len(X_train), replace=True)
    X_sample = X_train.iloc[indices]
    y_sample = y_train.iloc[indices]

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=i)
    rf.fit(X_sample, y_sample)

    # Predict on fixed test set
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Store metrics
    mae_list.append(mae)
    r2_list.append(r2)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# Convert predictions to numpy arrays for plotting
all_y_true = np.array(all_y_true)
all_y_pred = np.array(all_y_pred)
errors = all_y_true - all_y_pred

# Print metrics
print(f"\nBootstrapped MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Bootstrapped R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")
rmse = sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")


# --- Plot 1: Error distribution ---
plt.figure(figsize=(8, 5))
plt.hist(errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Prediction Error Distribution (Bootstrapped)')
plt.xlabel('Error (Actual - Predicted Band Gap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# --- Plot 2: Regression (Predicted vs Actual) ---
plt.figure(figsize=(8, 6))
sns.regplot(x=all_y_true, y=all_y_pred, line_kws={"color": "red"}, scatter_kws={"alpha": 0.3})
plt.xlabel('Actual Band Gap (HSE06) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('Regression Plot: Predicted vs Actual (Bootstrapped)')
plt.grid(True)
plt.show()
