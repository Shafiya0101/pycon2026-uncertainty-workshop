"""The four uncertainty signals. Convention: higher value = more uncertain.

The formulas are identical to the workshop's ``uncertainty_lib`` and to the thesis
pipeline, and are checked against the paper in ``tests/test_reproduce_paper.py``.
"""
from __future__ import annotations

import re
import string
from collections import Counter
from typing import Dict, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SIGNALS = ["U_lp", "U_sc", "U_se", "U_nli"]
SCORES = SIGNALS + ["Comb3", "Comb4"]
Scaler = Dict[str, Tuple[float, float]]


def normalize(text: str) -> str:
    """Lower-case, drop the articles a/an/the, delete punctuation, collapse whitespace.

    Identical to ``normalize_answer()`` in the thesis pipeline.
    """
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def self_consistency_uncertainty(samples: Sequence[str]) -> float:
    """U_SC = 1 - (size of the largest group of identical normalized answers) / N."""
    if len(samples) == 0:
        raise ValueError("need at least one sample")
    counts = Counter(normalize(s) for s in samples)
    return 1.0 - max(counts.values()) / len(samples)


def greedy_cluster(bidir: Sequence[Sequence[int]]) -> list:
    """Greedy meaning clusters. ``bidir[i][j] == 1`` if answers i and j entail each other.

    Each answer joins the first cluster whose representative (first member) it
    mutually entails; otherwise it starts a new cluster. Returns one id per answer.
    """
    ids, reps = [-1] * len(bidir), []
    for i in range(len(bidir)):
        for c, r in enumerate(reps):
            if bidir[i][r] == 1 and bidir[r][i] == 1:
                ids[i] = c
                break
        else:
            reps.append(i)
            ids[i] = len(reps) - 1
    return ids


def semantic_entropy(cluster_ids: Sequence[int]) -> float:
    """U_SE = -sum_k p_k log p_k, with p_k = share of samples in meaning cluster k."""
    counts = np.array(list(Counter(cluster_ids).values()), dtype=float)
    p = counts / len(cluster_ids)
    return max(0.0, float(-np.sum(p * np.log(p + 1e-12))))  # clip the -1e-12 artefact


def logprob_uncertainty(token_logprobs: Sequence[float]) -> float:
    """U_LP = minus the mean log-probability of the greedy answer's tokens."""
    return float(-np.mean(token_logprobs))


def nli_uncertainty(contra_probs: Sequence[float]) -> float:
    """U_NLI = mean P(contradiction) between the greedy answer and each sample (SelfCheckGPT-NLI)."""
    return float(np.mean(contra_probs))


def minmax(x, lo: Optional[float] = None, hi: Optional[float] = None) -> np.ndarray:
    """Min-max scale to [0, 1]. Pass ``lo``/``hi`` to reuse bounds fitted elsewhere
    (values outside them are clipped)."""
    x = np.asarray(x, dtype=float)
    lo = x.min() if lo is None else lo
    hi = x.max() if hi is None else hi
    rng = hi - lo
    return np.clip((x - lo) / rng, 0.0, 1.0) if rng > 0 else np.zeros_like(x)


def fit_scaler(df: pd.DataFrame) -> Scaler:
    """Min/max of each raw signal, used to build Comb3/Comb4 on new data."""
    return {c: (float(df[c].min()), float(df[c].max())) for c in SIGNALS}


def add_signals(df: pd.DataFrame, scaler: Optional[Mapping[str, Tuple[float, float]]] = None) -> pd.DataFrame:
    """Add ``cluster_ids``, U_lp, U_sc, U_se, U_nli, Comb3 (LP+SC+SE) and Comb4 (+NLI).

    Needs columns ``samples``, ``token_logprobs``, ``nli_contra`` and ``bidir_entail``.
    If ``scaler`` is None the combined scores are min-max scaled on ``df`` itself
    (the workshop / paper behaviour). For live data, pass the scaler fitted on a
    calibration set (see :class:`uncertainty_fingerprint.Detector`).
    """
    df = df.copy()
    df["cluster_ids"] = df["bidir_entail"].apply(greedy_cluster)
    df["U_lp"] = df["token_logprobs"].apply(logprob_uncertainty)
    df["U_sc"] = df["samples"].apply(self_consistency_uncertainty)
    df["U_se"] = df["cluster_ids"].apply(semantic_entropy)
    df["U_nli"] = df["nli_contra"].apply(nli_uncertainty)
    sc = fit_scaler(df) if scaler is None else scaler
    n = {c: minmax(df[c], *sc[c]) for c in SIGNALS}
    df["Comb3"] = (n["U_lp"] + n["U_sc"] + n["U_se"]) / 3
    df["Comb4"] = (n["U_lp"] + n["U_sc"] + n["U_se"] + n["U_nli"]) / 4
    return df


def auroc_table(df: pd.DataFrame, label_col: str = "label_containment", scores: Sequence[str] = SCORES) -> Dict[str, float]:
    """AUROC of each score for ranking WRONG answers above correct ones (1 = correct in ``label_col``)."""
    wrong = 1 - df[label_col].to_numpy()
    return {c: float(roc_auc_score(wrong, df[c])) for c in scores}
