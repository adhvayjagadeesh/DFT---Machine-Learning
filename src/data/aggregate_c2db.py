from argparse import ArgumentParser

from ase.db import connect
from numpy import empty
from numpy.random import shuffle
from pandas import DataFrame
from pymatgen.core import Composition

import seeder

parser = ArgumentParser(
  "python -m data.aggregate_c2db",
  description="Aggregate raw C2DB data",
)

parser.add_argument("db", help="C2DB ASE database file")
args = parser.parse_args()

con = connect(args.db)

query = "gap_hse,bravais_search!=Hexagonal,bravais_search!=Oblique,bravais_search!=Square"
n = con.count(query)

cols = (
  "HSE06 Band gap (eV)",
  "Mean electronegativity",
  "Atomic mass (amu)",
  "Atom count",
)
arr = empty((n, len(cols)))

for i, row in enumerate(con.select(query, include_data=False)):
  comp = Composition(row.formula)
  arr[i] = (
    row.gap_hse,
    comp.average_electroneg,
    comp.weight,
    comp.num_atoms,
  )

shuffle(arr)
DataFrame(
  arr,
  columns=cols,
).to_parquet("data/c2db.parquet", compression="zstd")
