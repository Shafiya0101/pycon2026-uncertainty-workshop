"""Input format, validation and the bundled study data.

One row per question. Columns used by the library:

==================  ==========================  ===============================================
column              type                        meaning
==================  ==========================  ===============================================
question            str                         question text (optional, for reading)
greedy              str                         the answer the user sees (T = 0)
samples             list[str] (N)               N sampled answers (paper: N = 10, T = 0.7, top-p 0.95)
token_logprobs      list[float]                 log-probability of each token of ``greedy``
nli_contra          list[float] (N)             P(contradiction) between ``greedy`` and each sample
bidir_entail        N x N list of 0/1           1 if samples i and j entail each other
label_containment   0/1 (optional)              1 = correct; any label column works
==================  ==========================  ===============================================

``nli_contra`` and ``bidir_entail`` can be computed with :mod:`uncertainty_fingerprint.nli`.
"""
from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Iterable, List, Union

import pandas as pd

SETTINGS = ["mistral_triviaqa", "qwen2_7b_triviaqa", "qwen2_1_5b_triviaqa", "mistral_nq", "mistral_hotpotqa"]
REQUIRED = ["greedy", "samples", "token_logprobs", "nli_contra", "bidir_entail"]


def read_jsonl(path: Union[str, Path]) -> pd.DataFrame:
    """Read a JSON Lines file (one question per line) into a DataFrame."""
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f if line.strip()])


def write_jsonl(df: pd.DataFrame, path: Union[str, Path]) -> None:
    """Write a DataFrame to JSON Lines (lists and numpy values are converted)."""
    with open(path, "w", encoding="utf-8") as f:
        for rec in df.to_dict(orient="records"):
            f.write(json.dumps(rec, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o)) + "\n")


def validate(df: pd.DataFrame, required: Iterable[str] = REQUIRED, label_cols: Iterable[str] = ()) -> List[str]:
    """Return a list of problems (empty list = OK). Checks columns, list lengths and labels."""
    problems = [f"missing column '{c}'" for c in list(required) + list(label_cols) if c not in df.columns]
    if problems:
        return problems
    for i, r in enumerate(df.itertuples(index=False)):
        n = len(r.samples)
        where = f"row {i}"
        if n < 2:
            problems.append(f"{where}: need at least 2 samples")
        if len(r.nli_contra) != n:
            problems.append(f"{where}: nli_contra has {len(r.nli_contra)} values, expected {n}")
        if len(r.bidir_entail) != n or any(len(x) != n for x in r.bidir_entail):
            problems.append(f"{where}: bidir_entail must be {n}x{n}")
        if len(r.token_logprobs) == 0:
            problems.append(f"{where}: token_logprobs is empty")
        for c in label_cols:
            if getattr(r, c) not in (0, 1):
                problems.append(f"{where}: {c} must be 0 or 1")
        if len(problems) > 20:
            problems.append("... (stopping after 20 problems)")
            break
    return problems


def load_setting(name: str) -> pd.DataFrame:
    """Load one of the five study settings (200 questions each) bundled with the package.

    These are the real precomputed results from the paper, as used in the PyCon ES 2026 workshop.
    """
    if name not in SETTINGS:
        raise ValueError(f"unknown setting '{name}'. Choose from {SETTINGS}")
    path = resources.files("uncertainty_fingerprint") / "data" / f"{name}.jsonl"
    with resources.as_file(path) as p:
        df = read_jsonl(p)
    df.insert(0, "setting", name)
    return df


def load_all() -> pd.DataFrame:
    """All five settings in one DataFrame (1,000 rows)."""
    return pd.concat([load_setting(s) for s in SETTINGS], ignore_index=True)
