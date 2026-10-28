"""Check that every file in data/precomputed/ matches the schema in data/README.md."""
import json, sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uncertainty_lib.data import REQUIRED, SETTINGS

bad = 0
for name in SETTINGS:
    p = ROOT / "data/precomputed" / f"{name}.jsonl"
    if not p.exists():
        print(f"- {name}: missing (synthetic placeholder will be used)"); continue
    for i, line in enumerate(p.open(encoding="utf-8")):
        r = json.loads(line)
        miss = [k for k in REQUIRED if k not in r]
        probs = []
        if miss: probs.append(f"missing {miss}")
        else:
            if len(r["samples"]) != 10: probs.append("need 10 samples")
            if len(r["nli_contra"]) != 10: probs.append("need 10 nli_contra values")
            if len(r["bidir_entail"]) != 10 or any(len(x) != 10 for x in r["bidir_entail"]): probs.append("bidir_entail must be 10x10")
            if r["label_containment"] not in (0, 1) or r["label_judge"] not in (0, 1): probs.append("labels must be 0/1")
        if probs:
            bad += 1; print(f"✗ {name} line {i}: {'; '.join(probs)}"); break
    else:
        print(f"✓ {name}: OK ({i+1} rows)")
sys.exit(1 if bad else 0)
