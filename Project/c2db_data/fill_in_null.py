import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# Load the CSV
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final - rectangular_materials_sortedby_bandgap_HSE06.csv")

# Strip column names of extra whitespace
df.columns = df.columns.str.strip()

# Create a copy for imputation
df_filled = df.copy()

# Select only numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns

# Prepare an imputation report
imputation_report = []

# Loop through each numeric column
for col in numeric_cols:
  null_ratio = df[col].isna().mean()
  std_dev = df[col].std()

  if null_ratio > 0.5:
    # Skip if more than 50% of values are missing
    imputation_report.append((col, "Skipped", f"Null ratio = {null_ratio:.2%}"))
    continue

  if std_dev < 3:
    # Apply KNN imputation
    print(f"Applying KNN to: {col} (std={std_dev:.3f})")
    numeric_df = df[numeric_cols].dropna(axis=1, how='all')
    imputer = KNNImputer(n_neighbors=3)
    imputed_data = imputer.fit_transform(numeric_df)
    imputed_df = pd.DataFrame(imputed_data, columns=numeric_df.columns, index=df.index)
    df_filled[col] = df[col].combine_first(imputed_df[col])
    imputation_report.append((col, "KNN", f"std = {std_dev:.3f}, nulls = {df[col].isna().sum()}"))
  else:
    # Fill with mean
    mean_value = df[col].mean()
    df_filled[col] = df[col].fillna(mean_value)
    imputation_report.append((col, "Mean", f"std = {std_dev:.3f}, nulls = {df[col].isna().sum()}"))

# Optional: Save the cleaned DataFrame
df_filled.to_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final_filled_conditional.csv", index=False)

# Print a summary
for col, method, notes in imputation_report:
  print(f"{col}: {method} ({notes})")
