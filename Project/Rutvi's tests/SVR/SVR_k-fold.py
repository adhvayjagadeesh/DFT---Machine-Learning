import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv")
df = df.drop(columns=['Band gap (G₀W₀) [eV]', 'Direct band gap (PBE) [eV]', 'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]'])

# Separate features and target
X = df.drop(columns=['Band gap (PBE) [eV]'])
y = df['Band gap (PBE) [eV]']
X = X.select_dtypes(include=[float, int])  # Keep only numeric features

# Impute missing values
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X)

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# K-Fold Cross Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)

mae_scores = []
r2_scores = []
errors = []

for train_index, test_index in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_index], X_scaled[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    model = SVR(kernel='rbf')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    error = y_test - y_pred  # residuals

    mae_scores.append(mae)
    r2_scores.append(r2)
    errors.extend(error)

# Summary of results
print("Average MAE:", np.mean(mae_scores))
print("Average R² Score:", np.mean(r2_scores))

# Plot error distribution
plt.figure(figsize=(10, 6))
plt.hist(errors, bins=30, color='skyblue', edgecolor='black')
plt.title('Error Distribution (Actual - Predicted)')
plt.xlabel('Prediction Error')
plt.ylabel('Frequency')
plt.grid(True)
plt.show()
