from __future__ import annotations

import argparse
import logging
import tarfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ase import Atoms
from ase.io import read


#aggregation utilities
def compute_formula(atoms: Atoms) -> str:
    """
    compute reduced chemical formula from an ASE Atoms object
    """
    symbols = atoms.get_chemical_symbols()
    counts: Dict[str, int] = {}
    for s in symbols:
        counts[s] = counts.get(s, 0) + 1

    #reduce formula by greatest common divisor
    values = list(counts.values())
    gcd = values[0]
    for v in values[1:]:
        gcd = np.gcd(gcd, v)

    parts = []
    for el in sorted(counts.keys()):
        n = counts[el] // gcd
        parts.append(f"{el}{n if n > 1 else ''}")
    return "".join(parts)


def is_2d_structure(atoms: Atoms, tol: float = 5.0) -> bool:
    """
    detect 2D materials by vacuum spacing along one lattice direction
    """
    cell = atoms.get_cell().lengths()
    #2D materials usually have one large vacuum dimension
    large = sum(l > tol for l in cell)
    return large == 1


def is_rectangular_lattice(atoms: Atoms, angle_tol: float = 1.0) -> bool:
    """
    rectangular / orthorhombic lattice:
    a ⟂ b, angles ~90°, not hexagonal
    """
    angles = atoms.get_cell().angles()
    for a in angles:
        if abs(a - 90.0) > angle_tol:
            return False
    return True


#file helpers
def find_c2db_path(data_root: Path) -> Path:
    """
    find C2DB folder or archive inside data/
    """
    for p in data_root.iterdir():
        if p.is_dir() and "c2db" in p.name.lower():
            return p
        if p.is_file() and p.name.lower() in ("c2db.tar.gz", "c2db.tgz", "c2db.zip"):
            return p
    raise FileNotFoundError("C2DB folder or archive not found inside data/")


def extract_if_archive(path: Path) -> Path:
    """
    extract C2DB archive if needed
    """
    if path.is_dir():
        return path

    out = path.parent / "C2DB_extracted"
    out.mkdir(exist_ok=True)

    if path.suffixes[-2:] == [".tar", ".gz"] or path.suffix == ".tgz":
        with tarfile.open(path, "r:*") as t:
            t.extractall(out)
    elif path.suffix == ".zip":
        with zipfile.ZipFile(path) as z:
            z.extractall(out)
    else:
        raise ValueError(f"Unsupported archive format: {path}")

    return out


#core logic
def collect_structures(c2db_root: Path) -> List[Path]:
    """
    collect structure files recursively
    """
    patterns = ("*.traj", "*.xyz", "*.cif", "*.vasp", "*.POSCAR")
    files: List[Path] = []
    for pat in patterns:
        files.extend(c2db_root.rglob(pat))
    return files


def aggregate_c2db(
    data_dir: str,
    output_csv: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    locate C2DB, filter 2D rectangular materials, compute formula
    """
    data_root = Path(data_dir)
    c2db_path = find_c2db_path(data_root)
    c2db_root = extract_if_archive(c2db_path)

    if verbose:
        logging.info("Using C2DB path: %s", c2db_root)

    structure_files = collect_structures(c2db_root)
    if not structure_files:
        raise RuntimeError("No structure files found in C2DB")

    records = []

    for f in structure_files:
        try:
            atoms = read(f)
        except Exception:
            continue

        #filter: ONLY 2D rectangular materials
        if not is_2d_structure(atoms):
            continue
        if not is_rectangular_lattice(atoms):
            continue

        formula = compute_formula(atoms)
        material_id = f.stem

        records.append(
            {
                "material_id": material_id,
                "formula": formula,
            }
        )

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)

    if verbose:
        logging.info(
            "Saved %d 2D rectangular materials to %s",
            len(df),
            output_csv,
        )

    return df


#CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract 2D rectangular materials from C2DB and compute formulas."
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Project data directory containing C2DB",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/c2db_rectangular_2d.csv",
        help="Output CSV file",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    level = logging.INFO if not args.quiet else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")

    aggregate_c2db(
        data_dir=args.data_dir,
        output_csv=args.output,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
