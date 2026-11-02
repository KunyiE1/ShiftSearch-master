from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def compute_psm_q_values(df: pd.DataFrame, score_col: str = "score", is_decoy_col: str = "is_decoy") -> pd.DataFrame:
    if df.empty:
        out = df.copy()
        out["q_value"] = []
        return out

    ordered = df.sort_values(score_col, ascending=False).reset_index(drop=True).copy()
    targets = 0
    decoys = 0
    fdrs = []
    for _, row in ordered.iterrows():
        if bool(row[is_decoy_col]):
            decoys += 1
        else:
            targets += 1
        fdrs.append(decoys / max(targets, 1))
    ordered["q_value"] = np.minimum.accumulate(np.asarray(fdrs, dtype=np.float64)[::-1])[::-1]
    return ordered
