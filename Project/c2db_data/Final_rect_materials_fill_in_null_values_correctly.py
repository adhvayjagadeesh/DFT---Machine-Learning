import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# Load the CSV
df = pd.read_csv("Project/c2db_data/rectangular_materials_sortedby_bandgap_HSE06.csv")
df.columns = df.columns.str.strip()

# Work only on numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols].copy()

# Drop columns with more than 50% null values
null_ratio = df_numeric.isnull().mean()
df_numeric = df_numeric.drop(columns=null_ratio[null_ratio > 0.5].index)

# Normalize numeric data (zero mean, unit variance)
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df_numeric), columns=df_numeric.columns, index=df.index)

# Prepare output dataframe
df_filled = df_scaled.copy()
imputation_report = []

# Impute missing values
for col in df_scaled.columns:
    missing = df_scaled[col].isna().sum()
    if missing == 0:
        imputation_report.append((col, "None", "No missing values"))
        continue

    std_dev = df_scaled[col].std()
    skew = df_scaled[col].skew()
    method = ""

    if std_dev < 3:
        # KNN Imputation
        imputer = KNNImputer(n_neighbors=3)
        df_filled[col] = imputer.fit_transform(df_scaled)[..., df_scaled.columns.get_loc(col)]
        method = "KNN (std < 3)"
    else:
        # Regression Imputation
        not_null = df_scaled[df_scaled[col].notna()]
        null_rows = df_scaled[df_scaled[col].isna()]

        X_train = not_null.drop(columns=[col])
        y_train = not_null[col]

        X_pred = null_rows.drop(columns=[col])

        try:
            reg = LinearRegression()
            reg.fit(X_train, y_train)
            predictions = reg.predict(X_pred)
            df_filled.loc[df_scaled[col].isna(), col] = predictions
            method = f"Linear Regression (std ≥ 3, skew = {skew:.2f})"
        except Exception as e:
            # Fallback if regression fails
            df_filled[col] = df_scaled[col].fillna(df_scaled[col].mean())
            method = f"Fallback to Mean (std ≥ 3, regression failed: {e})"

    imputation_report.append((col, method, f"Missing: {missing}"))

# Inverse transform to return data to original scale
df_filled_original_scale = pd.DataFrame(scaler.inverse_transform(df_filled), columns=df_filled.columns, index=df.index)

# Save the cleaned dataset
df_filled_original_scale.to_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_filled_conditional_v2.csv", index=False)

# Print report
print("\n=== Imputation Report ===")
for col, method, details in imputation_report:
    print(f"{col}: {method} ({details})")
