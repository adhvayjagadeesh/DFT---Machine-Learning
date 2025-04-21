import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, r2_score

# Load your dataset
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Materials from c2db - rectangular_materials_sortedby_bandgap_HSE06.csv")

# Set the target column name (ensure this matches exactly)
target_col = 'Band gap (PBE) [eV]'
if target_col not in df.columns:
    raise ValueError(f"Target column '{target_col}' not found in dataset.")

# Separate features and target
X = df.drop(columns=[target_col])
y = df[target_col]

# One-hot encode categorical variables
X_encoded = pd.get_dummies(X)

# Impute missing values using column means
imputer = SimpleImputer(strategy='mean')
X_imputed = imputer.fit_transform(X_encoded)

# Now split dataset into training and testing sets (80-20)
X_train, X_test, y_train, y_test = train_test_split(X_imputed, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Apply Polynomial Transformation (degree 2)
poly = PolynomialFeatures(degree=2)

# Transform the features into polynomial features
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# Print the shapes of the data before and after transformation
print(f"Transformed X_train shape: {X_train_poly.shape}")
print(f"Transformed X_test shape: {X_test_poly.shape}")

# Train Linear Regression model on polynomial features
poly_regressor = LinearRegression()
poly_regressor.fit(X_train_poly, y_train)

# Predict and evaluate the model
y_pred = poly_regressor.predict(X_test_poly)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Output results
print(f"Mean Absolute Error (MAE): {mae:.4f}")
print(f"R² Score: {r2:.4f}")

# Plot actual vs predicted
plt.figure(figsize=(6, 6))
plt.scatter(y_test, y_pred, alpha=0.7, edgecolors='k')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2)
plt.xlabel('Actual Bandgap [eV]')
plt.ylabel('Predicted Bandgap [eV]')
plt.title('Polynomial Regression: Actual vs Predicted Bandgap')
plt.grid(True)
plt.tight_layout()
plt.show()

# Show feature importance based on coefficients
coef_df = pd.DataFrame({
    'Feature': poly.get_feature_names_out(X_encoded.columns),
    'Coefficient': poly_regressor.coef_
}).sort_values(by='Coefficient', key=abs, ascending=False)

print("\nTop 10 most influential features:")
print(coef_df.head(10))
