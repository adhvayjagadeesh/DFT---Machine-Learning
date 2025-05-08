import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

# Load the CSV
df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final - rectangular_materials_sortedby_bandgap_HSE06.csv")

# Plot 'CBM'
cbm_col = 'CBM wrt. vacuum (PBE) [eV]'
plt.figure(figsize=(10, 5))
plt.plot(df[cbm_col], marker='o')
plt.title('CBM Column Plot')
plt.xlabel('Index')
plt.ylabel(cbm_col)
plt.grid(True)
plt.show()

# Calculate standard deviation
cbm_std = df[cbm_col].std()
print(f"Standard Deviation of '{cbm_col}': {cbm_std}")

# Initialize filled DataFrame
df_filled = df.copy()

# If std < 3, use KNN imputation on all numeric columns
if cbm_std < 3:
    print("Standard deviation < 3. Performing KNN imputation...")

    # Extract numeric columns and drop all-NaN columns
    numeric_df = df.select_dtypes(include=[np.number])
    numeric_df_cleaned = numeric_df.dropna(axis=1, how='all')

    # Apply KNN imputation
    knn_imputer = KNNImputer(n_neighbors=3)
    imputed_array = knn_imputer.fit_transform(numeric_df_cleaned)

    # Create DataFrame from imputed result
    imputed_df = pd.DataFrame(imputed_array, columns=numeric_df_cleaned.columns, index=df.index)

    # Replace nulls in the CBM column only
    df_filled[cbm_col] = df[cbm_col].combine_first(imputed_df[cbm_col])
    print("Filled null CBM values with KNN-imputed values.")

else:
    print("Standard deviation >= 3. Using mean imputation instead.")

    # Fill CBM using mean
    mean_value = df[cbm_col].mean()
    df_filled[cbm_col] = df[cbm_col].fillna(mean_value)
    print("Filled null CBM values with mean value.")

# Log how many were imputed
null_count = df[cbm_col].isna().sum()
print(f"Total null values originally in CBM column: {null_count}")
remaining_nulls = df_filled[cbm_col].isna().sum()
print(f"Remaining null values after imputation: {remaining_nulls}")


