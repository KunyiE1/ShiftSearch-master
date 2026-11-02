"""Paper target-decoy FDR calculation, reused from the original analysis code.

The original search module remains unchanged for compatibility. Paper analyses
instead group tied scores at one threshold, use D/T without a +1 correction,
clip threshold FDR to 1, and take reverse cumulative minima for q-values.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_tie_aware_q_values(
    df: pd.DataFrame,
    metric_column: str = "score",
    higher_is_better: bool = True,
    threshold_fdr_column: str = "threshold_fdr",
    q_value_column: str = "q_value",
) -> pd.DataFrame:
    """Assign target-decoy q-values while treating metric ties as one threshold.

    ``df`` must contain one winning PSM per spectrum and a Boolean ``is_decoy``
    column. This function is the unchanged calculation used by the paper's
    original ``analyze_search_results.add_tie_aware_q_values`` implementation.
    """
    if df.empty:
        result = df.copy()
        result[threshold_fdr_column] = np.nan
        result[q_value_column] = np.nan
        return result

    metric = pd.to_numeric(df[metric_column], errors="coerce")
    if metric.isna().any() or not np.isfinite(metric.to_numpy(float)).all():
        raise ValueError(f"Non-finite values in ranking metric {metric_column!r}")

    score_table = (
        df.assign(_ranking_metric=metric)
        .groupby("_ranking_metric", sort=False)["is_decoy"]
        .agg(total="size", decoys="sum")
        .reset_index()
        .sort_values("_ranking_metric", ascending=not higher_is_better)
        .reset_index(drop=True)
    )
    score_table["targets"] = score_table["total"] - score_table["decoys"]
    score_table["cum_decoys"] = score_table["decoys"].cumsum()
    score_table["cum_targets"] = score_table["targets"].cumsum()
    score_table[threshold_fdr_column] = (
        score_table["cum_decoys"] / score_table["cum_targets"].clip(lower=1)
    ).clip(upper=1.0)
    score_table[q_value_column] = np.minimum.accumulate(
        score_table[threshold_fdr_column].to_numpy(dtype=float)[::-1]
    )[::-1]
    fdr_map = score_table.set_index("_ranking_metric")[threshold_fdr_column]
    q_map = score_table.set_index("_ranking_metric")[q_value_column]

    result = df.copy()
    result[threshold_fdr_column] = metric.map(fdr_map)
    result[q_value_column] = metric.map(q_map)
    return result.sort_values(
        [metric_column, "is_decoy"],
        ascending=[not higher_is_better, True],
    ).reset_index(drop=True)
