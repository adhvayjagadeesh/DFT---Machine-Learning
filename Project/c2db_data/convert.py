import pandas as pd

# Load the CSV file
df = pd.read_csv('/workspaces/DFT---Machine-Learning/Project/c2db_data/Final - rectangular_materials_sortedby_bandgap_HSE06.csv')

# Convert "yes"/"no" in the desired column, e.g., 'your_column_name'
df['Magnetic'] = df['Magnetic'].map({'Yes': 1, 'No': 0})

# Save the modified CSV if needed
df.to_csv('/workspaces/DFT---Machine-Learning/Project/c2db_data/Final - rectangular_materials_sortedby_bandgap_HSE06.csv', index=False)
