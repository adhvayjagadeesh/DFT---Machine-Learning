from argparse import ArgumentParser

from ase.db import connect
from numpy import empty
from numpy.random import shuffle
from pandas import DataFrame
from pymatgen.core import Composition, Element

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
  "Mean ionization energy (kJ/mol)",
  "Mean atomic radius (Å)",
  "Mean electron affinity (eV)",
)
arr = empty((n, len(cols)))

for i, row in enumerate(con.select(query, include_data=False)):
  comp = Composition(row.formula)
  n_atoms = comp.num_atoms
  ionization_energy = 0
  atomic_radius = 0
  electron_affinity = 0

  for el, n in comp.get_el_amt_dict().items():
    el = Element(el)

    # All elements should have this, idk y it's even | None in the first place
    assert el.ionization_energy is not None
    ionization_energy += el.ionization_energy * n

    # Noble gases doesn't have this, but there's no noble gases in C2DB
    assert el.atomic_radius is not None
    atomic_radius += el.atomic_radius * n

    electron_affinity += el.electron_affinity * n

  arr[i] = (
    row.gap_hse,
    comp.average_electroneg,
    comp.weight,
    n_atoms,
    ionization_energy / n_atoms,
    atomic_radius / n_atoms,
    electron_affinity / n_atoms,
  )

# Remove any ordering C2DB might have
shuffle(arr)

DataFrame(
  arr,
  columns=cols,
).to_parquet("data/c2db.parquet", compression="zstd")
