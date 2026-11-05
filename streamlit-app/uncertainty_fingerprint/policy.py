"""Answer / verify / abstain policies and the matched-coverage comparison."""
from __future__ import annotations

from typing import Dict, Sequence

import numpy as np
import pandas as pd

from .fingerprint import assign_region, fit_medians
from .signals import SCORES


def risk_coverage_curve(score, correct):
    """Answer the lowest-uncertainty questions first. Returns (coverage, accuracy) arrays."""
    order = np.argsort(np.asarray(score), kind="stable")
    c = np.asarray(correct)[order]
    k = np.arange(1, len(c) + 1)
    return k / len(c), np.cumsum(c) / k


def accuracy_at_coverage(score, correct, coverage: float, rng=None) -> float:
    """Accuracy when keeping the ``coverage`` fraction with the lowest score (random tie-break)."""
    rng = rng or np.random.default_rng(0)
    score, correct = np.asarray(score, float), np.asarray(correct)
    order = np.lexsort((rng.random(len(score)), score))
    k = max(1, int(round(coverage * len(score))))
    return float(correct[order[:k]].mean())


def three_action(score, t_answer: float, t_abstain: float) -> np.ndarray:
    """score < t_answer -> 'answer'; t_answer <= score < t_abstain -> 'verify'; else 'abstain'."""
    if t_answer > t_abstain:
        raise ValueError("t_answer must be <= t_abstain")
    score = np.asarray(score, float)
    return np.where(score < t_answer, "answer", np.where(score < t_abstain, "verify", "abstain"))


def evaluate_actions(actions, correct, verify_cost: float = 2.0, verify_accuracy: float = 0.9) -> Dict[str, float]:
    """Expected accuracy, coverage and cost of a three-action policy.

    ASSUMPTIONS (toy cost model, replace with your own numbers): sampling costs 1.0 per
    question; 'verify' sends the question to an extra check (e.g. retrieval or a person)
    that costs ``verify_cost`` more and yields a correct answer with probability
    ``verify_accuracy``; 'abstain' gives no answer.
    """
    actions, correct = np.asarray(actions), np.asarray(correct)
    n = len(actions)
    a, v = actions == "answer", actions == "verify"
    n_served = a.sum() + v.sum()
    acc = (correct[a].sum() + verify_accuracy * v.sum()) / n_served if n_served else float("nan")
    return {"coverage": float(n_served / n), "accuracy": float(acc),
            "cost_per_question": float(1.0 + verify_cost * v.sum() / n)}


def _policy_mask(sc, se, tau_sc, tau_se, policy):
    regs = np.array([assign_region(a, b, tau_sc, tau_se) for a, b in zip(sc, se)])
    if policy == "A":
        return regs != "Gap-like"
    if policy == "B":
        return regs == "Confident"
    raise ValueError("policy must be 'A' (abstain on Gap-like) or 'B' (answer only Confident)")


def matched_coverage_comparison(df: pd.DataFrame, policy: str = "A", label_col: str = "label_containment",
                                n_rep: int = 200, seed: int = 0,
                                baselines: Sequence[str] = ("U_sc", "U_se", "U_lp", "U_nli", "Comb3", "Comb4")) -> pd.DataFrame:
    """Does a fingerprint policy beat a plain threshold that answers the SAME fraction?

    Each repetition: resample questions, split 50/50, fit medians on the calibration half,
    apply the policy to the evaluation half, then compare with each single-score threshold
    that keeps the same fraction of answers. Returns the mean accuracy difference and a 95%
    percentile interval per baseline. (Combined scores are normalized on the full data;
    this uses no labels.) The paper uses ``n_rep=500``.
    """
    rng = np.random.default_rng(seed)
    diffs = {b: [] for b in baselines}
    n = len(df)
    for _ in range(n_rep):
        idx = rng.integers(0, n, n)
        perm = rng.permutation(n)
        cal, ev = df.iloc[idx[perm[: n // 2]]], df.iloc[idx[perm[n // 2:]]]
        tau_sc, tau_se = fit_medians(cal)
        mask = _policy_mask(ev["U_sc"].to_numpy(), ev["U_se"].to_numpy(), tau_sc, tau_se, policy)
        if mask.sum() == 0:
            continue
        y = ev[label_col].to_numpy()
        pol_acc, cov = y[mask].mean(), mask.mean()
        for b in baselines:
            diffs[b].append(pol_acc - accuracy_at_coverage(ev[b].to_numpy(), y, cov, rng))
    rows = []
    for b in baselines:
        x = np.array(diffs[b])
        lo, hi = np.percentile(x, [2.5, 97.5])
        rows.append({"baseline": b, "mean_diff": x.mean(), "ci_low": lo, "ci_high": hi,
                     "significant": bool(lo > 0 or hi < 0)})
    return pd.DataFrame(rows).set_index("baseline")
