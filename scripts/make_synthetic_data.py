"""Generate SYNTHETIC placeholder data (NOT real model output) so every notebook runs
before the real precomputed results are exported. Questions are obviously fake on purpose.

    python scripts/make_synthetic_data.py
"""
import json
from pathlib import Path
import numpy as np

OUT = Path(__file__).resolve().parents[1] / "data" / "synthetic"
# name: (target accuracy, P(unanimous wrong | wrong), P(verbose paraphrases))
CONFIG = {
    "mistral_triviaqa": (0.765, 0.08, 0.25), "qwen2_7b_triviaqa": (0.61, 0.10, 0.25),
    "qwen2_1_5b_triviaqa": (0.425, 0.0, 0.25), "mistral_nq": (0.33, 0.04, 0.25),
    "mistral_hotpotqa": (0.355, 0.05, 0.25),
}
N_Q, N_S = 200, 10


def make_question(rng, i, acc, fab, verbose_p):
    c = rng.beta(3 * acc + 0.4, 3 * (1 - acc) + 0.4)          # latent confidence
    correct = rng.random() < c
    unanimous_wrong = (not correct) and rng.random() < fab
    s = 1.0 if unanimous_wrong else float(np.clip(c + rng.normal(0, 0.1), 0.05, 1.0))
    verbose = rng.random() < verbose_p
    names = [f"Gold-{i}"] + [f"Wrong-{i}{k}" for k in "abc"]       # meaning 0 is the gold answer
    g = 0 if correct else int(rng.integers(1, 4))

    def surface(m):
        n = names[m]
        forms = [n, n.lower(), f"The answer to the question is: {n}"]
        if verbose:
            forms += [f"{n}, I believe", f"It is {n}.", f"{n} (most likely)"]
        return forms[int(rng.integers(len(forms)))]

    meanings = [g if rng.random() < s else int(rng.choice([m for m in range(4) if m != g]))
                for _ in range(N_S)]
    samples = [surface(m) for m in meanings]
    bidir = [[int(meanings[a] == meanings[b] and (a == b or rng.random() < 0.97))
              for b in range(N_S)] for a in range(N_S)]
    for a in range(N_S):
        for b in range(a):
            bidir[b][a] = bidir[a][b]
    contra = [round(float(rng.beta(1, 20) if m == g else rng.beta(8, 2)), 3) for m in meanings]
    mean_lp = -(0.08 + 1.3 * (1 - s) ** 1.2)
    lps = [round(-abs(float(rng.normal(mean_lp, 0.15))) - 0.001, 3) for _ in range(5)]
    judge = int(correct) if rng.random() > 0.04 else 1 - int(correct)
    return {"qid": f"syn-{i}", "question": f"[SYNTHETIC] Question number {i}?",
            "gold": [names[0], names[0].lower()], "greedy": surface(g), "samples": samples,
            "token_logprobs": lps, "nli_contra": contra, "bidir_entail": bidir,
            "label_containment": int(correct), "label_judge": judge}


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    for k, (name, (acc, fab, vp)) in enumerate(CONFIG.items()):
        rng = np.random.default_rng(42 + k)
        with open(OUT / f"{name}.jsonl", "w", encoding="utf-8") as f:
            for i in range(N_Q):
                f.write(json.dumps(make_question(rng, i, acc, fab, vp)) + "\n")
        print("wrote", name)
