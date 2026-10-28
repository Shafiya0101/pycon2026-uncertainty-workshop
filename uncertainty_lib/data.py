"""Load the precomputed workshop data (JSON Lines, one question per line).

Real results go in data/precomputed/. If a file is missing we fall back to the
SYNTHETIC placeholder in data/synthetic/ and flag it in df.attrs['synthetic'].
"""
import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SETTINGS = ["mistral_triviaqa", "qwen2_7b_triviaqa", "qwen2_1_5b_triviaqa",
            "mistral_nq", "mistral_hotpotqa"]
REQUIRED = ["qid", "question", "gold", "greedy", "samples", "token_logprobs",
            "nli_contra", "bidir_entail", "label_containment", "label_judge"]


def _read(path):
    with open(path, encoding="utf-8") as f:
        return pd.DataFrame([json.loads(line) for line in f])


def load_setting(name, force_synthetic=False):
    real, fake = ROOT / "data/precomputed" / f"{name}.jsonl", ROOT / "data/synthetic" / f"{name}.jsonl"
    use_real = real.exists() and not force_synthetic
    df = _read(real if use_real else fake)
    df.insert(0, "setting", name)
    df.attrs["synthetic"] = not use_real
    return df


def load_all(force_synthetic=False):
    dfs = [load_setting(s, force_synthetic) for s in SETTINGS]
    out = pd.concat(dfs, ignore_index=True)
    out.attrs["synthetic"] = any(d.attrs["synthetic"] for d in dfs)
    return out
