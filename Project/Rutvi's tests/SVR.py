import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv")
df = df.drop(columns=['Band gap (G₀W₀) [eV]', 'Direct band gap (PBE) [eV]', 'Direct band gap (G₀W₀) [eV]', 'Direct band gap (HSE06) [eV]'])


# Separate features and target
X = df.drop(columns=['Band gap (PBE) [eV]'])  # Replace with your actual target column name
y = df['Band gap (PBE) [eV]']

# Ensure features are numeric (if any non-numeric, convert or drop them)
X = X.select_dtypes(include=[float, int])  # Select only numeric columns

# Step 1: Impute missing values (mean imputation)
imputer = SimpleImputer(strategy='mean')  # You can also use 'median' or other strategies
X_imputed = imputer.fit_transform(X)

# Step 2: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Step 3: Split data
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Step 4: Train SVR model
model = SVR(kernel='rbf')  # You can try different kernels like 'linear' or 'poly'
model.fit(X_train, y_train)

# Step 5: Predict and evaluate
y_pred = model.predict(X_test)

# Calculate Mean Squared Error and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print evaluation metrics
print("Mean Squared Error:", mse)
print("R² Score:", r2)

# Step 6: Plot Actual vs Predicted Values
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs Actual')
plt.plot([y.min(), y.max()], [y.min(), y.max()], color='red', linestyle='--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('SVR: Actual vs Predicted')
plt.legend()
plt.show()
