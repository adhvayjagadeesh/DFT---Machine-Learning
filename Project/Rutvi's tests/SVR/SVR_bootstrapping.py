import pandas as pd
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv")
df = df.drop(columns=['Direct band gap (PBE) [eV]','Band gap (HSE06) [eV]', 'Direct band gap (HSE06) [eV]'])

# Separate features and target
X = df.drop(columns=['Band gap (PBE) [eV]'])
y = df['Band gap (PBE) [eV]']

# Keep only numeric features
X = X.select_dtypes(include=[float, int])

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Bootstrapping
n_iterations = 100
mae_scores = []
r2_scores = []
errors = []

np.random.seed(42)

for i in range(n_iterations):
    # Resample with replacement
    X_resampled, y_resampled = resample(X_scaled, y, replace=True, random_state=42 + i)

    # Split into train/test sets
    split_index = int(0.8 * len(X_resampled))
    X_train, X_test = X_resampled[:split_index], X_resampled[split_index:]
    y_train, y_test = y_resampled[:split_index], y_resampled[split_index:]

    # Train SVR model
    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    error = y_test - y_pred

    mae_scores.append(mae)
    r2_scores.append(r2)
    errors.extend(error)

# Summary statistics
print("Bootstrapped Evaluation over", n_iterations, "iterations")
print("Average MAE:", np.mean(mae_scores))
print("Average R²:", np.mean(r2_scores))

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='orange', edgecolor='black')
plt.title("Bootstrapped Error Distribution (Actual - Predicted)")
plt.xlabel("Prediction Error")
plt.ylabel("Frequency")
plt.grid(True)
plt.show()
