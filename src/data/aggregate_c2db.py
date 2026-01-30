from argparse import ArgumentParser

import pandas as pd
from ase.db import connect
from numpy import empty
from pymatgen.core import Composition

parser = ArgumentParser(
  "python -m data.aggregate_c2db",
  description="Aggregate and preprocess raw C2DB data",
)

parser.add_argument("db", help="C2DB ASE database file")
args = parser.parse_args()

con = connect(args.db)

n = con.count("gap_hse,bravais_search!=Hexagonal,bravais_search!=Oblique")
arr = empty((n, 2))

for i, row in enumerate(
  con.select(
    "gap_hse,bravais_search!=Hexagonal,bravais_search!=Oblique",
    include_data=False,
  )
):
  comp = Composition(row.formula)
  arr[i] = (row.gap_hse, comp.average_electroneg)

df = pd.DataFrame(
  arr, columns=("HSE06 band gap (eV)", "Average electronegativity")
)

df.to_parquet("data/c2db.parquet")
