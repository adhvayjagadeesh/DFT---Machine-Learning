from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


#aggregation utilities
def mode_or_first(series: pd.Series):
    """Return the mode if available, otherwise the first non-null value."""
    try:
        m = series.mode(dropna=True)
        if len(m) > 0:
            return m.iloc[0]
    except Exception:
        pass
    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) else np.nan


def flatten_multiindex_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Flatten MultiIndex columns produced by groupby.agg into single-level names."""
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    new_cols = []
    for col in df.columns:
        if isinstance(col, tuple):
            #join non-empty parts with underscore
            parts = [str(c) for c in col if c is not None and str(c) != ""]
            new_cols.append("_".join(parts))
        else:
            new_cols.append(str(col))
    df.columns = new_cols
    return df


def resolve_duplicated_suffix_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    resolve columns with duplicated suffixes like _x/_y or _left/_right
    for each base name, prefer non-null values from the left variant, then right
    if multiple variants exist, combine them in left-to-right order using combine_first
    """
    cols = list(df.columns)
    #map base -> list of variants
    base_map: Dict[str, List[str]] = {}
    suffix_patterns = ("_x", "_y", "_left", "_right")
    for c in cols:
        for suf in suffix_patterns:
            if c.endswith(suf):
                base = c[: -len(suf)]
                base_map.setdefault(base, []).append(c)
                break

    for base, variants in base_map.items():
        #also include an exact base column if present
        preferred = [v for v in variants]  #e.g., base_x, base_y
        #if base itself exists, put it first
        if base in df.columns:
            preferred = [base] + preferred
        #combine them: start with first, combine_first successive
        combined = df[preferred[0]].copy()
        for other in preferred[1:]:
            combined = combined.combine_first(df[other])
        #assign combined to base (overwrite or create)
        df[base] = combined
        #drop the variant columns
        for v in variants:
            if v in df.columns:
                df.drop(columns=[v], inplace=True)
    return df


#core logic
def load_c2db_folder(folder: Path, base_filename: Optional[str] = None) -> Tuple[str, pd.DataFrame]:
    """
    load CSV files from folder and choose a base table
    if base_filename provided, prefer that file stem (without .csv)
    otherwise prefer 'materials.csv' (stem 'materials'), else pick the largest CSV by rows
    returns (base_name, base_df) and keeps other tables in memory only to merge
    """
    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"C2DB folder not found: {folder}")

    csv_files = list(folder.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    #read all csv files metadata (rows) to choose deterministic base if not provided
    dfs: Dict[str, pd.DataFrame] = {}
    for csv in csv_files:
        try:
            df = pd.read_csv(csv)
            dfs[csv.stem] = df
            logging.debug("Loaded %s rows from %s", len(df), csv)
        except Exception as e:
            logging.warning("Failed to read %s: %s", csv, e)

    if base_filename:
        base_stem = Path(base_filename).stem
        if base_stem not in dfs:
            raise FileNotFoundError(f"Requested base file '{base_filename}' not found in folder")
        base_df = dfs.pop(base_stem)
        return base_stem, base_df

    #prefer 'materials' if present
    if "materials" in dfs:
        base_df = dfs.pop("materials")
        return "materials", base_df

    #otherwise pick the CSV with the most rows
    best = max(dfs.items(), key=lambda kv: len(kv[1]))
    base_stem, base_df = best[0], best[1]
    dfs.pop(base_stem)
    return base_stem, base_df


def merge_other_tables(base_df: pd.DataFrame, other_tables: Dict[str, pd.DataFrame], material_id_candidates: List[str]) -> pd.DataFrame:
    """
    merge other_tables into base_df. Prefer merging on detected material id if present in both
    otherwise merge on any intersection of column names. Use suffixes to avoid losing data
    """
    merged = base_df.copy()
    for name, df in other_tables.items():
        #find best join column: prefer any material id candidate present in both
        join_cols = [c for c in material_id_candidates if c in merged.columns and c in df.columns]
        if join_cols:
            on = join_cols
        else:
            common = list(set(merged.columns) & set(df.columns))
            if not common:
                logging.info("Skipping merge with %s: no shared columns", name)
                continue
            on = common
        #perform merge with deterministic suffixes
        merged = merged.merge(df, on=on, how="left", suffixes=("_left", "_right"))
    #resolve duplicate suffixes deterministically
    merged = resolve_duplicated_suffix_columns(merged)
    return merged


def detect_material_id(df: pd.DataFrame, provided: Optional[str] = None) -> str:
    """
    auto-detect material identifier column. If provided and exists, return it
    otherwise check common candidate names (case-insensitive).
    raise KeyError if none found.
    """
    if provided:
        if provided in df.columns:
            logging.info("Using provided material id column: %s", provided)
            return provided
        #allow case-insensitive match
        lower_map = {c.lower(): c for c in df.columns}
        if provided.lower() in lower_map:
            resolved = lower_map[provided.lower()]
            logging.info("Using provided material id (case-insensitive) resolved to: %s", resolved)
            return resolved
        raise KeyError(f"Provided material id '{provided}' not found in columns")

    candidates = ["material_id", "c2db_id", "id", "mp_id", "materialid", "mat_id"]
    lower_cols = {c.lower(): c for c in df.columns}
    for cand in candidates:
        if cand in lower_cols:
            logging.info("Auto-detected material id column: %s", lower_cols[cand])
            return lower_cols[cand]
    #if nothing found, try to pick a unique-ish column: one with many unique values approximately equal to rows
    for c in df.columns:
        nunique = df[c].nunique(dropna=False)
        if nunique > max(1, len(df) * 0.5):
            logging.info("Fallback detected material id candidate (high uniqueness): %s", c)
            return c
    raise KeyError("Could not detect a material identifier column automatically; please pass --material-id")


def aggregate_numeric_groupby(df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
    """
    fast numeric aggregation using groupby.agg for numeric columns
    produces columns like: 'band_gap_mean', 'band_gap_std', etc
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        return pd.DataFrame(index=df[groupby_col].unique())

    agg_map = {col: ["mean", "std", "min", "max"] for col in numeric_cols}
    grouped = df.groupby(groupby_col, dropna=False)[numeric_cols].agg(agg_map)
    grouped = flatten_multiindex_columns(grouped)
    #ensure index name is groupby_col for merging with categorical results
    grouped.index.name = groupby_col
    return grouped


def aggregate_categorical_groupby(df: pd.DataFrame, groupby_col: str) -> pd.DataFrame:
    """
    aggregate categorical/non-numeric columns deterministically using mode_or_first
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in df.columns if c not in numeric_cols and c != groupby_col]
    if not categorical_cols:
        # return empty DataFrame with correct index
        return pd.DataFrame(index=df[groupby_col].unique())

    #groupby.agg with a dict of functions
    agg_funcs = {col: (lambda x: mode_or_first(x)) for col in categorical_cols}
    grouped_cat = df.groupby(groupby_col, dropna=False)[categorical_cols].agg(agg_funcs)
    #grouped_cat will have same columns as categorical_cols
    grouped_cat.index.name = groupby_col
    return grouped_cat


def aggregate_c2db_folder(
    c2db_folder: str,
    output_csv: str,
    material_id: Optional[str] = None,
    base_filename: Optional[str] = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    main orchestration: load folder, merge tables, detect material_id, aggregate numerics & categoricals, save CSV
    returns aggregated DataFrame
    """
    folder = Path(c2db_folder)
    base_name, base_df = load_c2db_folder(folder, base_filename=base_filename)

    #load other tables into dict for merging
    all_csvs = {p.stem: p for p in folder.glob("*.csv")}
    other_tables = {}
    for stem, path in all_csvs.items():
        if stem == base_name:
            continue
        try:
            other_tables[stem] = pd.read_csv(path)
        except Exception as e:
            logging.warning("Failed to read %s: %s", path, e)

    merged = merge_other_tables(base_df, other_tables, material_id_candidates=["material_id", "c2db_id", "id", "mp_id"])

    #detect material id
    mid = detect_material_id(merged, provided=material_id)

    if verbose:
        logging.info("Using material id column: %s", mid)
        logging.info("Total rows after merge: %d", len(merged))

    #numeric aggregation (fast)
    numeric_agg = aggregate_numeric_groupby(merged, groupby_col=mid)

    #categorical aggregation
    categorical_agg = aggregate_categorical_groupby(merged, groupby_col=mid)

    #combine numeric and categorical results into a single DataFrame
    result = pd.concat([numeric_agg, categorical_agg], axis=1)
    #ensure material id is a column
    result = result.reset_index()

    #save
    result.to_csv(output_csv, index=False)
    if verbose:
        logging.info("Aggregated dataset saved to %s (shape: %s)", output_csv, result.shape)
    return result


#CLI
def parse_args():
    parser = argparse.ArgumentParser(description="Aggregate a C2DB folder into an ML-ready CSV.")
    parser.add_argument("--c2db-folder", "-f", default="data/C2DB", help="Path to C2DB folder containing CSV files")
    parser.add_argument("--output", "-o", default="data/c2db_aggregated.csv", help="Output aggregated CSV")
    parser.add_argument("--material-id", "-m", default=None, help="Material identifier column (auto-detected if omitted)")
    parser.add_argument("--base", "-b", default=None, help="Base CSV filename to use (stem or filename). Prefer 'materials.csv' if not provided.")
    parser.add_argument("--quiet", action="store_true", help="Suppress logging output")
    return parser.parse_args()


def main():
    args = parse_args()
    #configure logging
    level = logging.INFO if not args.quiet else logging.WARNING
    logging.basicConfig(level=level, format="%(levelname)s: %(message)s")
    aggregate_c2db_folder(
        c2db_folder=args.c2db_folder,
        output_csv=args.output,
        material_id=args.material_id,
        base_filename=args.base,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
