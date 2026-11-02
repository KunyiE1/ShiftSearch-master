"""Apply the paper's tie-aware PSM FDR procedure to top-1 search results."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from ShiftSearch.evaluation.fdr import add_tie_aware_q_values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path, help="Top-1 results.best.tsv, including target AND decoy rows.")
    parser.add_argument("--output", required=True, type=Path, help="Passing target PSMs.")
    parser.add_argument("--all-output", type=Path, help="Optional full top-1 table with recalculated threshold FDR and q-values.")
    parser.add_argument("--fdr", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def parse_decoy_flags(series):
    """Parse TSV booleans without treating the string 'False' as True."""
    true = {"true", "1", "yes", "y", "t"}
    false = {"false", "0", "no", "n", "f"}
    text = series.astype(str).str.strip().str.lower()
    invalid = ~text.isin(true | false)
    if invalid.any():
        raise ValueError(f"Invalid is_decoy values: {sorted(set(text[invalid]))[:5]}")
    return text.isin(true)


def main(argv=None):
    args = parse_args(argv)
    if not 0 <= args.fdr <= 1:
        raise ValueError("--fdr must be between 0 and 1")
    paths = [args.output] + ([args.all_output] if args.all_output else [])
    resolved = [path.resolve() for path in paths]
    if len(resolved) != len(set(resolved)) or args.input.resolve() in resolved:
        raise ValueError("Input, output and all-output must be different files")
    for path in paths:
        if path.exists() and not args.overwrite:
            raise FileExistsError(f"Output already exists: {path}; use --overwrite deliberately")
    table = pd.read_csv(args.input, sep="\t", low_memory=False)
    missing = {"score", "is_decoy"} - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if "rank" in table and not pd.to_numeric(table["rank"], errors="coerce").eq(1).all():
        raise ValueError("Provide results.best.tsv, not the top-k result table")
    table["is_decoy"] = parse_decoy_flags(table["is_decoy"])
    table["score"] = pd.to_numeric(table["score"], errors="coerce")
    scored = add_tie_aware_q_values(table)
    passing = scored.loc[(~scored["is_decoy"]) & scored["q_value"].le(args.fdr)]
    for path in paths:
        path.parent.mkdir(parents=True, exist_ok=True)
    passing.to_csv(args.output, sep="\t", index=False)
    if args.all_output:
        scored.to_csv(args.all_output, sep="\t", index=False)
    print(f"[paper-fdr] top-1 PSMs={len(scored)}; passing targets={len(passing)}; q<={args.fdr:g}")
    print(f"[paper-fdr] saved {args.output}")


if __name__ == "__main__":
    main()
