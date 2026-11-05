"""Uncertainty fingerprints: the cell of the (U_SC, U_SE) plane an answer falls in.

Each axis is cut at its median, so building a fingerprint needs no correctness labels.
"""
from __future__ import annotations

from typing import Optional, Tuple

import pandas as pd

REGIONS = ["Confident", "Gap-like", "Shallow", "Surface-varying"]


def fit_medians(df: pd.DataFrame) -> Tuple[float, float]:
    """Median thresholds (tau_SC, tau_SE). Uses no labels."""
    return float(df["U_sc"].median()), float(df["U_se"].median())


def assign_region(sc: float, se: float, tau_sc: float, tau_se: float) -> str:
    """Region of one answer. Ties go to the 'low' side (a value equal to the median is low)."""
    sc_high, se_high = sc > tau_sc, se > tau_se
    if not sc_high and not se_high:
        return "Confident"
    if sc_high and se_high:
        return "Gap-like"
    if not sc_high and se_high:
        return "Shallow"
    return "Surface-varying"


def add_regions(df: pd.DataFrame, tau_sc: Optional[float] = None, tau_se: Optional[float] = None) -> pd.DataFrame:
    """Add a ``region`` column. Fits the medians on ``df`` unless thresholds are given."""
    if tau_sc is None or tau_se is None:
        tau_sc, tau_se = fit_medians(df)
    df = df.copy()
    df["region"] = [assign_region(a, b, tau_sc, tau_se) for a, b in zip(df["U_sc"], df["U_se"])]
    return df


def region_summary(df: pd.DataFrame, label_col: str = "label_containment") -> pd.DataFrame:
    """Number of questions and error rate in each region (needs labels)."""
    d = df.assign(wrong=1 - df[label_col])
    out = d.groupby("region").agg(n=("wrong", "size"), error_rate=("wrong", "mean"))
    out = out.reindex(REGIONS)
    out["n"] = out["n"].fillna(0).astype(int)
    return out


def unanimous_errors(df: pd.DataFrame, label_col: str = "label_containment") -> pd.DataFrame:
    """Confident fabrications, absolute definition: all samples identical (U_SC = 0) and still wrong."""
    return df[(df["U_sc"] == 0) & (df[label_col] == 0)]
