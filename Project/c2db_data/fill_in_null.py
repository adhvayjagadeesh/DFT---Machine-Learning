import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer

df = pd.read_csv("/workspaces/DFT---Machine-Learning/Project/c2db_data/Final - rectangular_materials_sortedby_bandgap_HSE06.csv")


plt.figure(figsize=(10, 5))
plt.plot(df['CBM wrt. vacuum (PBE) [eV]'], marker='o')
plt.title('CBM Column Plot')
plt.xlabel('Index')
plt.ylabel('CBM wrt. vacuum (PBE) [eV]')
plt.grid(True)
plt.show()

# Check standard deviation
cbm_std = df['CBM wrt. vacuum (PBE) [eV]'].std()
print(f"Standard Deviation of 'CBM wrt. vacuum (PBE) [eV]': {cbm_std}")

# If std < 3, apply KNN using all numeric columns
if cbm_std < 3:
    print("Standard deviation < 3. Performing KNN imputation...")

    df_knn_filled = df.copy()

    # Extract numeric columns and drop columns that are all NaN
    numeric_df = df_knn_filled.select_dtypes(include=[np.number])
    numeric_df_cleaned = numeric_df.dropna(axis=1, how='all')

    # Apply KNN imputation
    knn_imputer = KNNImputer(n_neighbors=3)
    imputed_array = knn_imputer.fit_transform(numeric_df_cleaned)

    # Convert imputed array back to DataFrame
    imputed_df = pd.DataFrame(imputed_array, columns=numeric_df_cleaned.columns, index=df_knn_filled.index)

    # Replace only the imputed columns
    df_knn_filled[imputed_df.columns] = imputed_df

    print("KNN-imputed 'CBM wrt. vacuum (PBE) [eV]' column:")
    print(df_knn_filled['CBM wrt. vacuum (PBE) [eV]'])
else:
    print("Standard deviation >= 3. Skipping KNN imputation.")

# Mean imputation as fallback
df_mean_filled = df.copy()
df_mean_filled['CBM wrt. vacuum (PBE) [eV]'] = df_mean_filled['CBM wrt. vacuum (PBE) [eV]'].fillna(
    df_mean_filled['CBM wrt. vacuum (PBE) [eV]'].mean()
)

print("Mean-imputed 'CBM wrt. vacuum (PBE) [eV]' column:")
print(df_mean_filled['CBM wrt. vacuum (PBE) [eV]'])

# Fill original NaNs in CBM column with KNN-imputed values
df_filled = df.copy()
original_cbm_col = 'CBM wrt. vacuum (PBE) [eV]'

# Replace only the null values with KNN-imputed values
df_filled[original_cbm_col] = df[original_cbm_col].combine_first(df_knn_filled[original_cbm_col])

print("Filled null CBM values with KNN-imputed values and saved to 'Final_filled_knn.csv'.")
