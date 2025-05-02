import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error, r2_score

# Load and clean the dataset
df = pd.read_csv('/workspaces/DFT---Machine-Learning/Project/c2db_data/Final - rectangular_materials_sortedby_bandgap_HSE06.csv')
df = df.drop(columns=['Formula', 'Layer group', 'Stoichiometry', 'Band gap (G₀W₀) [eV]', 
                      'Direct band gap (PBE) [eV]', 'Direct band gap (G₀W₀) [eV]', 
                      'Direct band gap (HSE06) [eV]'])

# Define features and target
target = 'Band gap (PBE) [eV]'
X = df.drop(columns=[target])
y = df[target]

# Ensure all data is numeric
X = X.select_dtypes(include=[np.number])

# Initialize k-fold
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# Store metrics and predictions
mae_list = []
r2_list = []
all_errors = []
all_y_true = []
all_y_pred = []

# Loop through folds
for train_index, test_index in kf.split(X):
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    # Train model
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = rf.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    errors = y_test - y_pred

    # Store results
    mae_list.append(mae)
    r2_list.append(r2)
    all_errors.extend(errors)
    all_y_true.extend(y_test)
    all_y_pred.extend(y_pred)

# Report average performance
print(f"\nAverage MAE: {np.mean(mae_list):.4f} ± {np.std(mae_list):.4f}")
print(f"Average R²: {np.mean(r2_list):.4f} ± {np.std(r2_list):.4f}")

# --- Plot 1: Error distribution ---
plt.figure(figsize=(8, 5))
plt.hist(all_errors, bins=30, edgecolor='k', alpha=0.7)
plt.title('Prediction Error Distribution (5-Fold CV)')
plt.xlabel('Error (Actual - Predicted Band Gap)')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()

# --- Plot 2: Linear regression (y_true vs. y_pred) ---
plt.figure(figsize=(8, 6))
sns.regplot(x=all_y_true, y=all_y_pred, line_kws={'color': 'red'})
plt.xlabel('Actual Band Gap (PBE) [eV]')
plt.ylabel('Predicted Band Gap [eV]')
plt.title('Predicted vs Actual Band Gap (5-Fold CV)')
plt.grid(True)
plt.show()
