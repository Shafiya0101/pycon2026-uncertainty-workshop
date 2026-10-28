"""TEMPLATE: convert the thesis-pipeline outputs into data/precomputed/<setting>.jsonl.

I (the assistant) could not see how run_pipeline.py / run_multimodel.py store their raw per-question
results, so the loading part below is a TODO. Fill in `load_thesis_rows(setting)` so that it yields
one dict per question with the fields in data/README.md, then run:

    python scripts/export_precomputed_TEMPLATE.py
    python scripts/validate_data.py
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
THESIS_REPO = Path("../metacognitive-uncertainty-layer")      # TODO: path to your thesis repo clone
SETTINGS = ["mistral_triviaqa", "qwen2_7b_triviaqa", "qwen2_1_5b_triviaqa", "mistral_nq", "mistral_hotpotqa"]


def load_thesis_rows(setting):
    """TODO: read your saved per-question results (pickle / csv / json) and yield dicts with keys:
    qid, question, gold, greedy, samples, token_logprobs, nli_contra, bidir_entail,
    label_containment, label_judge.
    If you did not save the 10x10 bidirectional-entailment matrix, recompute it once on your GPU."""
    raise NotImplementedError("Write load_thesis_rows for your result format")


if __name__ == "__main__":
    out = ROOT / "data" / "precomputed"
    out.mkdir(parents=True, exist_ok=True)
    for s in SETTINGS:
        with open(out / f"{s}.jsonl", "w", encoding="utf-8") as f:
            for row in load_thesis_rows(s):
                f.write(json.dumps(row) + "\n")
        print("wrote", s)
