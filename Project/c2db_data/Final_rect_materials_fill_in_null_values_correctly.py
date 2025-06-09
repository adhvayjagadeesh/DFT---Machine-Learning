import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Load the CSV
df = pd.read_csv("Project/c2db_data/rectangular_materials_sortedby_bandgap_HSE06.csv")
df.columns = df.columns.str.strip()

# Step 1: Preserve metadata columns (if they exist)
possible_meta_cols = [
    'Formula', 
    'Band gap (G₀W₀) [eV]', 
    'Direct band gap (G₀W₀) [eV]',
    'Band gap (HSE06) [eV] ▲',
    'Direct band gap (PBE) [eV]', 
    'Direct band gap (HSE06) [eV]'
]
meta_cols = [col for col in possible_meta_cols if col in df.columns]
df_meta = df[meta_cols].copy()

# Step 2: Select numeric features
numeric_cols = df.select_dtypes(include=[np.number]).columns
df_numeric = df[numeric_cols].copy()

# Step 3: Drop columns with more than 50% null values
null_ratio = df_numeric.isnull().mean()
df_numeric = df_numeric.drop(columns=null_ratio[null_ratio > 0.5].index)

# Step 4: Normalize the numeric data
scaler = StandardScaler()
df_scaled = pd.DataFrame(
    scaler.fit_transform(df_numeric), 
    columns=df_numeric.columns, 
    index=df_numeric.index
)

# Prepare for imputation
df_filled = df_scaled.copy()
imputation_report = []

# Step 5: Impute column by column with skew check
for col in df_scaled.columns:
    missing = df_scaled[col].isna().sum()
    if missing == 0:
        imputation_report.append((col, "None", "No missing values"))
        continue

    std_dev = df_scaled[col].std()
    skew = df_scaled[col].skew()
    method = ""

    if abs(skew) > 1:
        # Highly skewed → use median
        median_value = df_scaled[col].median()
        df_filled[col] = df_scaled[col].fillna(median_value)
        method = f"Median (high skew = {skew:.2f})"
    elif std_dev < 3:
        # Low variance → KNN
        imputer = KNNImputer(n_neighbors=3)
        df_filled[col] = imputer.fit_transform(df_scaled)[..., df_scaled.columns.get_loc(col)]
        method = f"KNN (std < 3, skew = {skew:.2f})"
    else:
        # High variance, low skew → Regression
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
            df_filled[col] = df_scaled[col].fillna(df_scaled[col].mean())
            method = f"Fallback to Mean (regression failed: {e})"

    imputation_report.append((col, method, f"Missing: {missing}"))

# Step 6: Reverse normalization
df_filled_original_scale = pd.DataFrame(
    scaler.inverse_transform(df_filled), 
    columns=df_filled.columns, 
    index=df_filled.index
)

# Step 7: Recombine metadata + imputed numeric data
df_final = pd.concat([df_meta, df_filled_original_scale], axis=1)

# Step 8: Save final cleaned dataset
df_final.to_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_rect_materials_filled_in_correctly.csv", index=False)

# Step 9: Print imputation summary
print("\n=== Imputation Report ===")
for col, method, details in imputation_report:
    print(f"{col}: {method} ({details})")
