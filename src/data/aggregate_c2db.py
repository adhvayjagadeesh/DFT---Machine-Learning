from __future__ import annotations

import argparse
import logging
import os
import re
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


#aggregation utilities
def mode_or_first(series: pd.Series):
    try:
        m = series.mode(dropna=True)
        if len(m) > 0:
            return m.iloc[0]
    except Exception:
        pass
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            parts = [str(c) for c in col if c is not None and str(c) != ""]
            new_cols.append("_".join(parts))
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


def resolve_duplicated_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = list(df.columns)
    base_map: Dict[str, List[str]] = {}
    suffix_patterns = ("_x", "_y", "_left", "_right")
    for c in cols:
        for suf in suffix_patterns:
            if c.endswith(suf):
                base = c[: -len(suf)]
                base_map.setdefault(base, []).append(c)
                break

    for base, variants in base_map.items():
        preferred = [v for v in variants]
        if base in df.columns:
            preferred = [base] + preferred
        combined = df[preferred[0]].copy()
        for other in preferred[1:]:
            combined = combined.combine_first(df[other])
        df[base] = combined
        for v in variants:
            if v in df.columns:
                df.drop(columns=[v], inplace=True)
    return df


#file/folder helpers
def find_c2db_source(root: Path) -> Optional[Path]:
    for dirpath, dirnames, filenames in os.walk(root):
        for d in dirnames:
            if "c2db" == d.lower() or "c2db" in d.lower():
                return Path(dirpath) / d
    for dirpath, dirnames, filenames in os.walk(root):
        for f in filenames:
            if f.lower() in ("c2db.tar.gz", "c2db.tgz", "c2db.zip"):
                return Path(dirpath) / f
    return None


def extract_archive_if_needed(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    if path.is_dir():
        return path
    tmp = Path(tempfile.mkdtemp(prefix="c2db_"))
    lower = path.name.lower()
    if lower.endswith((".tar.gz", ".tgz")):
        with tarfile.open(path, "r:*") as t:
            t.extractall(tmp)
    elif lower.endswith(".zip"):
        with zipfile.ZipFile(path, "r") as z:
            z.extractall(tmp)
    else:
        raise ValueError(f"Unsupported archive type: {path}")
    return tmp


#core logic - loading/merging
def load_c2db_folder(folder: Path, base_filename: Optional[str] = None) -> Tuple[str, pd.DataFrame, Dict[str, pd.DataFrame]]:
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"C2DB folder not found: {folder}")

    csv_files = sorted(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    dfs: Dict[str, pd.DataFrame] = {}
    for csv in csv_files:
        try:
            dfs[csv.stem] = pd.read_csv(csv)
            logging.debug("Loaded %s rows from %s", len(dfs[csv.stem]), csv)
        except Exception as e:
            logging.warning("Failed to read %s: %s", csv, e)

    if base_filename:
        base_stem = Path(base_filename).stem
        if base_stem not in dfs:
            raise FileNotFoundError(f"Requested base file '{base_filename}' not found in folder")
        base_df = dfs.pop(base_stem)
        return base_stem, base_df, dfs

    if "materials" in dfs:
        base_df = dfs.pop("materials")
        return "materials", base_df, dfs

    best = max(dfs.items(), key=lambda kv: len(kv[1]))
    base_stem, base_df = best[0], best[1]
    dfs.pop(base_stem)
    return base_stem, base_df, dfs


def merge_other_tables(base_df: pd.DataFrame, other_tables: Dict[str, pd.DataFrame], material_id_candidates: List[str]) -> pd.DataFrame:
    merged = base_df.copy()
    for name, df in other_tables.items():
        join_cols = [c for c in material_id_candidates if c in merged.columns and c in df.columns]
        if join_cols:
            on = join_cols
        else:
            common = list(set(merged.columns) & set(df.columns))
            if not common:
                logging.info("Skipping merge with %s: no shared columns", name)
                continue
            on = common
        merged = merged.merge(df, on=on, how="left", suffixes=("_left", "_right"))
    merged = resolve_duplicated_suffix_columns(merged)
    return merged


#detection/filtering for 2D rectangular bravais
def find_bravais_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(tok in lc for tok in ("brav", "latt", "lattice", "crystal", "system", "spacegroup", "bravais")):
            candidates.append(c)
    return candidates


def find_dimensionality_columns(df: pd.DataFrame) -> List[str]:
    candidates = []
    for c in df.columns:
        lc = c.lower()
        if any(tok in lc for tok in ("dim", "dimension", "ndim", "dimensional", "thickness", "layers", "is_2d", "monolayer")):
            candidates.append(c)
    return candidates


def is_2d_value(val: object) -> bool:
    if pd.isna(val):
        return False
    s = str(val).strip().lower()
    if s in ("2", "2.0", "2d", "2-d", "2 d", "two", "monolayer", "mono", "single-layer", "singlelayer", "layer"):
        return True
    if re.search(r"\b2d\b", s):
        return True
    # check numeric
    try:
        if float(s) == 2.0:
            return True
    except Exception:
        pass
    return False


def filter_rectangular_2d(df: pd.DataFrame, bravais_cols: List[str], dim_cols: List[str]) -> pd.DataFrame:
    if not bravais_cols:
        logging.warning("No bravais/lattice-like columns found; cannot filter to rectangular materials reliably.")
        return df.iloc[0:0]  # empty
    if not dim_cols:
        logging.warning("No dimensionality-like columns found; cannot filter to 2D materials reliably.")
        return df.iloc[0:0]  # empty

    # build masks
    # bravais mask: any bravais column contains rectangle/orthorhomb keywords
    br_keywords = ("rect", "rectang", "orthorhomb", "rectangle")
    br_mask = pd.Series(False, index=df.index)
    for c in bravais_cols:
        vals = df[c].astype(str).fillna("").str.lower()
        for kw in br_keywords:
            br_mask = br_mask | vals.str.contains(kw)

    # dimensionality mask: any dim column indicates 2D
    dim_mask = pd.Series(False, index=df.index)
    for c in dim_cols:
        vals = df[c]
        # test each value with is_2d_value
        dim_mask = dim_mask | vals.apply(is_2d_value)

    combined = br_mask & dim_mask
    filtered = df[combined]
    logging.info("Filtered 2D-rectangular: from %d -> %d rows (bravais_cols=%s, dim_cols=%s)", len(df), len(filtered), bravais_cols, dim_cols)
    return filtered


#formula extraction helpers
_FORMULA_TOKEN_RE = re.compile(r"([A-Z][a-z]?)(\d*\.?\d*)")


def parse_formula(formula: str) -> Dict[str, float]:
    if not isinstance(formula, str) or formula.strip() == "":
        return {}
    tokens = _FORMULA_TOKEN_RE.findall(formula)
    counts: Dict[str, float] = {}
    for el, cnt in tokens:
        if cnt == "":
            num = 1.0
        else:
            try:
                num = float(cnt)
            except Exception:
                num = 1.0
        counts[el] = counts.get(el, 0.0) + num
    return counts


def formula_from_element_columns(row: pd.Series, df_columns: List[str]) -> Optional[str]:
    elem_cols = [c for c in df_columns if re.match(r"^[A-Z][a-z]?$", c) and pd.api.types.is_numeric_dtype(row.get(c, np.nan))]
    if not elem_cols:
        for c in df_columns:
            m = re.match(r"^([A-Z][a-z]?)[\_\-].*$", c)
            if m and pd.api.types.is_numeric_dtype(row.get(c, np.nan)):
                elem_cols.append(c)
    counts = {}
    for c in elem_cols:
        m = re.match(r"^([A-Z][a-z]?)", c)
        if m:
            el = m.group(1)
        else:
            continue
        try:
            val = float(row.get(c, 0.0))
        except Exception:
            val = 0.0
        if val and not np.isnan(val) and val > 0:
            counts[el] = counts.get(el, 0.0) + val
    if not counts:
        return None
    parts = []
    for el in sorted(counts.keys()):
        cnt = counts[el]
        if float(cnt).is_integer():
            cnt_s = str(int(cnt))
        else:
            cnt_s = str(cnt)
        parts.append(f"{el}{cnt_s if cnt != 1 else ''}")
    return "".join(parts)


def compute_formula_for_row(row: pd.Series, df_columns: List[str], formula_columns: List[str]) -> Optional[str]:
    for fc in formula_columns:
        val = row.get(fc, None)
        if isinstance(val, str) and val.strip():
            return val.strip()
    f = formula_from_element_columns(row, df_columns)
    if f:
        return f
    candidates = [c for c in df_columns if any(tok in c.lower() for tok in ("formula", "composition", "stoich", "unit_cell"))]
    for c in candidates:
        v = row.get(c, None)
        if isinstance(v, str) and v.strip():
            parsed = parse_formula(v)
            if parsed:
                parts = []
                for el in sorted(parsed.keys()):
                    cnt = parsed[el]
                    if float(cnt).is_integer():
                        cnt_s = str(int(cnt))
                    else:
                        cnt_s = str(cnt)
                    parts.append(f"{el}{cnt_s if cnt != 1 else ''}")
                return "".join(parts)
    return None


#aggregation helpers (fast)
def aggregate_numeric_groupby(df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame(index=df[groupby_col].unique())
    agg_map = {col: ["mean", "std", "min", "max"] for col in numeric_cols}
    grouped = df.groupby(groupby_col, dropna=False)[numeric_cols].agg(agg_map)
    grouped = flatten_multiindex_columns(grouped)
    grouped.index.name = groupby_col
    return grouped


def aggregate_categorical_groupby(df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != groupby_col]
    if not categorical_cols:
        return pd.DataFrame(index=df[groupby_col].unique())
    agg_funcs = {col: (lambda x, col=col: mode_or_first(x)) for col in categorical_cols}
    grouped_cat = df.groupby(groupby_col, dropna=False)[categorical_cols].agg(agg_funcs)
    grouped_cat.index.name = groupby_col
    return grouped_cat


def aggregate_c2db_folder(
    c2db_folder: Optional[str],
    output_csv: str,
    material_id: Optional[str] = None,
    base_filename: Optional[str] = None,
    only_rectangular: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    folder_path: Path
    if c2db_folder:
        folder_path = Path(c2db_folder)
        if not folder_path.exists():
            found = find_c2db_source(Path("."))
            if found:
                folder_path = found
            else:
                raise FileNotFoundError(f"Provided c2db-folder not found: {c2db_folder}")
    else:
        found = find_c2db_source(Path("."))
        if not found:
            raise FileNotFoundError("Could not locate C2DB folder or archive in project tree; pass --c2db-folder explicitly")
        folder_path = found

    if not folder_path.is_dir():
        folder_path = extract_archive_if_needed(folder_path)

    base_name, base_df, other_tables = load_c2db_folder(folder_path, base_filename=base_filename)

    merged = merge_other_tables(base_df, other_tables, material_id_candidates=["material_id", "c2db_id", "id", "mp_id"])

    mid = detect_material_id(merged, provided=material_id)

    if only_rectangular:
        bravais_cols = find_bravais_columns(merged)
        dim_cols = find_dimensionality_columns(merged)
        merged = filter_rectangular_2d(merged, bravais_cols, dim_cols)
        if merged.empty:
            logging.warning("No 2D rectangular materials found after filtering; returning empty aggregated dataset.")

    if verbose:
        logging.info("Using material id column: %s", mid)
        logging.info("Total rows after merge/filter: %d", len(merged))

    df_cols = list(merged.columns)
    formula_cols = [c for c in df_cols if any(tok in c.lower() for tok in ("formula", "composition", "stoich", "unit_cell"))]
    computed = []
    for idx, row in merged.iterrows():
        cf = compute_formula_for_row(row, df_cols, formula_cols)
        computed.append(cf if cf is not None else "")
    merged["computed_formula"] = computed

    numeric_agg = aggregate_numeric_groupby(merged, groupby_col=mid)
    categorical_agg = aggregate_categorical_groupby(merged, groupby_col=mid)

    result = pd.concat([numeric_agg, categorical_agg], axis=1)
    result = result.reset_index()

    if "computed_formula" not in result.columns:
        formula_map = merged.groupby(mid)["computed_formula"].first()
        result = result.merge(formula_map.reset_index(), on=mid, how="left")

    result.to_csv(output_csv, index=False)
    if verbose:
        logging.info("Saved aggregated CSV to %s (shape=%s)", output_csv, result.shape)
    return result


#CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate a C2DB folder into an ML-ready CSV (optionally 2D-rectangular-only).")
    parser.add_argument("--c2db-folder", "-f", default=None, help="Path to C2DB folder or archive (auto-detected if omitted)")
    parser.add_argument("--output", "-o", default="data/c2db_aggregated.csv", help="Output aggregated CSV")
    parser.add_argument("--material-id", "-m", default=None, help="Material identifier column (auto-detected if omitted)")
    parser.add_argument("--base", "-b", default=None, help="Base CSV filename to use (stem or filename). Prefer 'materials.csv' if not provided.")
    parser.add_argument("--only-rectangular", action="store_true", help="Filter to 2D rectangular (rect/orthorhombic) Bravais-type materials only")
    parser.add_argument("--quiet", action="store_true", help="Suppress logging")
    return parser.parse_args()


def main():
    args = parse_args()
    level = logging.INFO if not args.quiet else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    aggregate_c2db_folder(
        c2db_folder=args.c2db_folder,
        output_csv=args.output,
        material_id=args.material_id,
        base_filename=args.base,
        only_rectangular=args.only_rectangular,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
