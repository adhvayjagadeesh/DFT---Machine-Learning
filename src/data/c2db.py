import pandas as pd

# Load dataset
df = pd.read_csv("data/Final_rect_materials_filled_in_correctly.csv")

# Drop column
df = df.drop(columns=[
    'Direct band gap (PBE) [eV]',
    'Direct band gap (PBE) [eV].1',
    'Band gap (PBE) [eV]',
    'Band gap (G₀W₀) [eV]',
    'Direct band gap (G₀W₀) [eV]',
    'Direct band gap (HSE06) [eV]',
    'Direct band gap (HSE06) [eV].1',
    'CBM wrt. vacuum (PBE) [eV]',
    'VBM wrt. vacuum (PBE) [eV]',
])
