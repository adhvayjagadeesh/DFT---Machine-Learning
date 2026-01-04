from __future__ import annotations

import argparse
from typing import List

import numpy as np
import pandas as pd


#aggregation helpers
def aggregate_numeric(df: pd.DataFrame) -> pd.DataFrame:
    "aggregate numeric columns using statistical summaries
    
    return df.agg(["mean", "std", "min", "max"])


def aggregate_categorical(series: pd.Series):
    "Deterministic aggregation for categorical columns.
    
    try:
        mode = series.mode(dropna=True)
        if len(mode) > 0:
            return mode.iloc[0]
    except Exception:
        pass

    non_null = series.dropna()
    return non_null.iloc[0] if len(non_null) > 0 else np.nan


def aggregate_group(group: pd.DataFrame) -> pd.Series:
    "aggregate one material group into a single row
    out = {}

    numeric_cols = group.select_dtypes(include=[np.number]).columns
    categorical_cols = [c for c in group.columns if c not in numeric_cols]

    #numeric aggregation
    for col in numeric_cols:
        stats = aggregate_numeric(group[col])
        for stat_name, value in stats.items():
            out[f"{col}_{stat_name}"] = value

    #categorical aggregation
    for col in categorical_cols:
        out[col] = aggregate_categorical(group[col])

    return pd.Series(out)



#main aggregation logic
def aggregate_c2db(
    input_csv: str,
    output_csv: str,
    material_id_col: str,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Aggregate C2DB dataset into one row per material.
    """
    if verbose:
        print(f"Loading C2DB data from: {input_csv}")

    df = pd.read_csv(input_csv)

    if material_id_col not in df.columns:
        raise KeyError(
            f"Material ID column '{material_id_col}' not found.\n"
            f"Available columns: {list(df.columns)}"
        )

    if verbose:
        n_materials = df[material_id_col].nunique()
        print(f"Found {len(df)} rows corresponding to {n_materials} materials")

    # Group and aggregate
    aggregated = (
        df.groupby(material_id_col, dropna=False)
        .apply(aggregate_group)
        .reset_index()
    )

    if verbose:
        print(f"Aggregated dataset shape: {aggregated.shape}")
        print(f"Saving aggregated dataset to: {output_csv}")

    aggregated.to_csv(output_csv, index=False)
    return aggregated


#CLI
def parse_args():
    parser = argparse.ArgumentParser(
        description="Aggregate raw C2DB dataset into ML-ready CSV."
    )
    parser.add_argument(
        "--input",
        "-i",
        default="data/c2db.csv",
        help="Path to raw C2DB CSV (default: data/c2db.csv)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="data/c2db_aggregated.csv",
        help="Path to write aggregated CSV (default: data/c2db_aggregated.csv)",
    )
    parser.add_argument(
        "--material-id",
        "-m",
        default="c2db_id",
        help="Column that uniquely identifies a material (default: c2db_id)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress logging output",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    aggregate_c2db(
        input_csv=args.input,
        output_csv=args.output,
        material_id_col=args.material_id,
        verbose=not args.quiet,
    )


if __name__ == "__main__":
    main()
