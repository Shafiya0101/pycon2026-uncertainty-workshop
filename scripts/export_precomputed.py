"""Export the thesis results to the workshop format (data/precomputed/<setting>.jsonl).

Run it with the THESIS virtual environment (needs torch + transformers + numpy).
Use --limit 5 for a quick test first, then run again without --limit (about 20-40 min on a GPU).
It recomputes the NLI results that the thesis pickles did not keep, using the code from run_pipeline.py,
and CHECKS them against the values saved in the pickles and against Table IV of the paper.
"""
import argparse, json, os, pickle, re, string, sys, time
from collections import Counter
import numpy as np

# setting: (folder in thesis dir, prefix in judge_labels.json, containment acc, judge acc [paper Table IV])
SETTINGS = {
    "mistral_triviaqa":    ("results",           "Mistral-7B | TriviaQA",  0.765, 0.730),
    "qwen2_7b_triviaqa":   ("results_llama3",    "Qwen2-7B | TriviaQA",    0.610, 0.590),
    "qwen2_1_5b_triviaqa": ("results_phi3",      "Qwen2-1.5B | TriviaQA",  0.425, 0.385),
    "mistral_nq":          ("results_naturalqa", "Mistral-7B | NaturalQA", 0.330, 0.360),
    "mistral_hotpotqa":    ("results_hotpotqa",  "Mistral-7B | HotpotQA",  0.355, 0.375),
}
N = 10
TEMPLATE = "The answer to the question is: "
NLI_NAME = "microsoft/deberta-large-mnli"


def normalize_answer(text):
    text = text.lower().strip()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


class NLI:
    """DeBERTa-large-MNLI as loaded in run_pipeline.py. Labels: 0 contradiction, 1 neutral, 2 entailment."""
    def __init__(self):
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.torch = torch
        self.tok = AutoTokenizer.from_pretrained(NLI_NAME)
        self.model = AutoModelForSequenceClassification.from_pretrained(NLI_NAME).eval()
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.dev)
        print(f"NLI model on {self.dev}", flush=True)

    def logits(self, a, b):
        inp = self.tok(a, b, return_tensors="pt", truncation=True, max_length=256).to(self.dev)
        with self.torch.no_grad():
            return [float(x) for x in self.model(**inp).logits[0].tolist()]


def entail_matrix(resps, nli, cache):
    """B[i][j] = 1 if responses i and j entail each other (both directions). Blank responses -> 0."""
    n = len(resps)
    def ent(a, b):
        if (a, b) not in cache:
            cache[(a, b)] = int(np.argmax(nli.logits(a, b)) == 2)
        return cache[(a, b)]
    B = [[0] * n for _ in range(n)]
    for i in range(n):
        B[i][i] = 1
        for j in range(i + 1, n):
            if resps[i].strip() and resps[j].strip():
                B[i][j] = B[j][i] = int(ent(resps[i], resps[j]) and ent(resps[j], resps[i]))
    return B


def contradiction_scores(greedy, resps, nli, cache):
    """Per-sample P(contradiction), as in step 5 of run_pipeline.py (0.5 for blank text)."""
    out = []
    for sa in resps:
        if not sa.strip() or not greedy.strip():
            out.append(0.5); continue
        key = ("SC", sa, greedy)
        if key not in cache:
            l = nli.logits(TEMPLATE + sa, TEMPLATE + greedy)
            ze, zc = l[2], l[0]
            cache[key] = float(np.exp(zc) / (np.exp(ze) + np.exp(zc)))
        out.append(cache[key])
    return out


def thesis_clusters(B, resps):
    """do_cluster() from run_pipeline.py: compare with the FIRST member of each cluster; blanks skipped."""
    cls = []
    for i, r in enumerate(resps):
        if not r.strip():
            continue
        for c in cls:
            if B[c[0]][i]:
                c.append(i); break
        else:
            cls.append([i])
    return cls


def sem_entropy(cls, n):
    if not cls:
        return 0.0
    ps = np.array([len(c) / n for c in cls])
    return float(-np.sum(ps * np.log(ps + 1e-12)))


def export_setting(name, records, judge, nli, limit=None):
    """Returns (rows, report). `records` = final_results.pkl['results']."""
    cache, rows = {}, []
    bad = Counter()
    recs = records[:limit] if limit else records
    t0 = time.time()
    for k, rec in enumerate(recs):
        resps = [s["response"] for s in rec["samples"]]
        assert len(resps) == N, f"{name} q{rec['idx']}: {len(resps)} samples"
        B = entail_matrix(resps, nli, cache)
        contra = contradiction_scores(rec["greedy_response"], resps, nli, cache)
        cls = thesis_clusters(B, resps)
        lp = [float(x) for x in rec["log_probs"]]
        norm = [normalize_answer(r) for r in resps]
        # --- compare with what the thesis pipeline saved ---
        if len(cls) != rec["n_semantic_clusters"]:
            bad["n_clusters"] += 1
        if abs(sem_entropy(cls, N) - rec["uncertainty_se"]) > 1e-6:
            bad["semantic_entropy"] += 1
        if abs(float(np.mean(contra)) - rec["uncertainty_selfcheck"]) > 1e-4:
            bad["selfcheck_mean"] += 1
        if abs((1 - Counter(norm).most_common(1)[0][1] / N) - rec["uncertainty_sc"]) > 1e-9:
            bad["self_consistency"] += 1
        if abs(-float(np.mean(lp)) - rec["avg_neg_log_prob"]) > 1e-6:
            bad["avg_neg_log_prob"] += 1
        if any(not r.strip() for r in resps):
            bad["has_blank_sample(info)"] += 1
        verdict = judge.get(f"{judge['__prefix__']}#{rec['idx']}")
        if verdict is None:
            bad["judge_label_missing"] += 1
        rows.append({
            "qid": f"{name}-{rec['idx']}", "question": rec["question"],
            "gold": [str(g) for g in rec["ground_truths"]], "greedy": rec["greedy_response"],
            "samples": resps, "token_logprobs": lp, "nli_contra": [round(c, 6) for c in contra],
            "bidir_entail": B, "label_containment": int(bool(rec["is_correct"])),
            "label_judge": int(verdict == "CORRECT"),
        })
        if (k + 1) % 10 == 0 or k + 1 == len(recs):
            el = time.time() - t0
            print(f"  {name}: {k + 1}/{len(recs)}  ({el / (k + 1):.1f}s/question, ~{el / (k + 1) * (len(recs) - k - 1) / 60:.1f} min left)", flush=True)
    return rows, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--thesis-dir", required=True)
    ap.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "precomputed"))
    ap.add_argument("--settings", nargs="*", default=list(SETTINGS))
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions (test run, writes to <out>/_test)")
    ap.add_argument("--force", action="store_true", help="overwrite existing files")
    a = ap.parse_args()

    out = os.path.join(a.out, "_test") if a.limit else a.out
    os.makedirs(out, exist_ok=True)
    jl = json.load(open(os.path.join(a.thesis_dir, "analysis_out", "judge_labels.json"), encoding="utf-8"))
    nli = NLI()
    summary = []
    for name in a.settings:
        folder, prefix, acc_c, acc_j = SETTINGS[name]
        dst = os.path.join(out, f"{name}.jsonl")
        if os.path.exists(dst) and not a.force:
            print(f"- {name}: {dst} exists, skipping (use --force)"); continue
        d = pickle.load(open(os.path.join(a.thesis_dir, folder, "final_results.pkl"), "rb"))
        print(f"\n== {name}  <-  {folder}\\final_results.pkl   model={d.get('model')}", flush=True)
        j = dict(jl); j["__prefix__"] = prefix
        rows, bad = export_setting(name, d["results"], j, nli, a.limit)
        tmp = dst + ".part"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        os.replace(tmp, dst)
        n = len(rows)
        ac, aj = np.mean([r["label_containment"] for r in rows]), np.mean([r["label_judge"] for r in rows])
        problems = {k: v for k, v in bad.items() if not k.endswith("(info)")}
        ok = not problems and (a.limit or (abs(ac - acc_c) < 0.0051 and abs(aj - acc_j) < 0.0051))
        print(f"   wrote {n} rows -> {dst}")
        print(f"   accuracy containment {ac:.3f} (paper {acc_c:.3f}) | judge {aj:.3f} (paper {acc_j:.3f})")
        print(f"   differences from saved thesis values: {dict(bad) or 'none'}")
        summary.append((name, n, "OK" if ok else "CHECK"))
    print("\n" + "=" * 60)
    for s in summary:
        print(f"{s[0]:22s} {s[1]:4d} rows  {s[2]}")
    print("Next: python scripts/validate_data.py  then  python scripts/verify_against_paper.py")
    if a.limit:
        print(f"(test run: files are in {out}; delete that folder before the real export)")


if __name__ == "__main__":
    main()
