"""Build student notebooks (notebooks/) and full-solution notebooks (solutions/) from one source.
No nbformat needed:  python tools/build_notebooks.py"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class NB:
    def __init__(self, name):
        self.name, self.s, self.sol = name, [], []

    def md(self, t):
        c = {"cell_type": "markdown", "metadata": {}, "source": t.strip("\n").splitlines(True)}
        self.s.append(c); self.sol.append(c)

    def code(self, t):
        c = {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": t.strip("\n").splitlines(True)}
        self.s.append(c); self.sol.append(c)

    def todo(self, student, solution):
        mk = lambda t: {"cell_type": "code", "metadata": {}, "execution_count": None,
                        "outputs": [], "source": t.strip("\n").splitlines(True)}
        self.s.append(mk(student)); self.sol.append(mk(solution))

    def save(self):
        for folder, cells in (("notebooks", self.s), ("solutions", self.sol)):
            nb = {"cells": cells, "nbformat": 4, "nbformat_minor": 4,
                  "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python",
                                              "name": "python3"},
                               "language_info": {"name": "python"}}}
            (ROOT / folder / f"{self.name}.ipynb").write_text(json.dumps(nb, indent=1), encoding="utf-8")


BOOT = r'''
import sys, pathlib, warnings
from collections import Counter
REPO = "pycones2026-uncertainty-workshop"   # <- must match your GitHub repo name (used on Colab only)
if "google.colab" in sys.modules:
    ROOT = pathlib.Path("/content") / REPO
    if not ROOT.exists():
        import subprocess
        subprocess.run(["git", "clone", f"https://github.com/Shafiya0101/{REPO}.git", str(ROOT)], check=True)
else:
    ROOT = pathlib.Path.cwd()
    if ROOT.name in ("notebooks", "solutions"):
        ROOT = ROOT.parent
sys.path.insert(0, str(ROOT))
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib.pyplot as plt
from uncertainty_lib import *
from uncertainty_lib import checks
from uncertainty_lib.plotting import set_style, plot_region_grid, plot_risk_coverage
set_style()
pd.set_option("display.max_colwidth", 70)
'''

# ---------------------------------------------------------------- 00
n = NB("00_setup_check")
n.md("""# 00 · Setup check (2 minutes)
**Goal:** confirm your environment works. No downloads, no internet, no API key needed.

Run the two cells below (Shift+Enter). You should see ✅ at the end.""")
n.code(BOOT)
n.code(r'''
import platform, sklearn, matplotlib
print("Python     :", platform.python_version(), "(need 3.10 or newer)")
print("numpy      :", np.__version__)
print("pandas     :", pd.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib :", matplotlib.__version__)
assert sys.version_info >= (3, 10), "Please use Python 3.10+"

df = load_setting("mistral_triviaqa")
checks.banner(df)
print(f"Loaded {len(df)} questions. Each has 1 greedy answer + {len(df.samples[0])} sampled answers.")
print("✅ Setup OK. You are ready for notebook 01.")
df[["question", "greedy"]].head(3)
''')
n.md("""**Something failed?** Raise your hand. Fallback: open the repo on Colab (see README) or use the USB copy.""")
n.save()

# ---------------------------------------------------------------- 01
n = NB("01_uncertainty_signals")
n.md("""# 01 · The four uncertainty signals (≈ 12 min)
**Challenge 2: build a self-consistency detector.**

For each question we stored the model's *greedy answer* (the one a user would see) and **10 sampled answers**.
Four **uncertainty signals** (higher = more uncertain) are computed from that evidence:

| Signal | Abbreviation | Measures |
|---|---|---|
| Log-probability | LP | the model's own token confidence in the greedy answer |
| Self-consistency | SC | do the 10 samples give the same *string*? |
| Semantic entropy | SE | do the 10 samples give the same *meaning*? |
| NLI contradiction | NLI | do the samples contradict the greedy answer? (NLI = natural language inference) |

The heavy model calls were done in advance, so everything here runs on your laptop.""")
n.code(BOOT)
n.code(r'''
df = load_setting("mistral_triviaqa")
checks.banner(df)
df["cluster_ids"] = df["bidir_entail"].apply(greedy_cluster)   # meaning clusters, built from stored NLI results
row = df.iloc[0]
print("Question :", row.question)
print("Gold     :", row.gold)
print("Greedy   :", row.greedy)
print("Samples  :")
for s in row.samples:
    print("   -", s)
''')
n.md("""## Step 1 · Self-consistency (SC)
`U_SC = 1 − (size of the largest group of identical answers) / N`.
10/10 identical → 0.0. Five different answers, two each → 0.8.

**Predict first:** two answers like `Gold-1` and `The answer to the question is: Gold-1`: same group or different? (The helper `normalize()` decides; try it.)""")
n.code(r'''
print(normalize("The answer to the question is: Gold-1"), "|", normalize("Gold-1, I believe"))
''')
n.md("### 🛠 TODO 1: implement `my_self_consistency`")
n.todo(r'''
def my_self_consistency(samples):
    """Return 1 - (size of the largest group of identical normalized answers) / N."""
    # TODO 1a: normalize every sample with normalize()
    # TODO 1b: count how often each normalized answer appears (Counter)
    # TODO 1c: return 1 - biggest_count / len(samples)
    raise NotImplementedError("Fill in my_self_consistency")
''', r'''
def my_self_consistency(samples):
    groups = Counter(normalize(s) for s in samples)
    return 1 - max(groups.values()) / len(samples)
''')
n.code(r'''
checks.check_sc(my_self_consistency, self_consistency_uncertainty, df)
''')
n.md("""## Step 2 · Semantic entropy (SE)
SC compares *strings*, so a paraphrase looks like disagreement. SE first groups samples by **meaning**
(`cluster_ids`: samples with the same id mean the same thing), then measures how spread the samples are over those groups:

`U_SE = − Σ p_k log p_k`, with `p_k` = share of samples in cluster k.""")
n.md("### 🛠 TODO 2: implement `my_semantic_entropy`")
n.todo(r'''
def my_semantic_entropy(cluster_ids):
    """Return -sum(p_k * log(p_k)) where p_k = cluster size / N."""
    # TODO 2a: count the size of each cluster (Counter)
    # TODO 2b: turn counts into probabilities p
    # TODO 2c: return the entropy (natural log, np.log)
    raise NotImplementedError("Fill in my_semantic_entropy")
''', r'''
def my_semantic_entropy(cluster_ids):
    counts = np.array(list(Counter(cluster_ids).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())
''')
n.code(r'''
checks.check_se(my_semantic_entropy, semantic_entropy, df)
''')
n.md("""## Step 3 · All four signals
`add_signals` computes LP, SC, SE and NLI for every question, plus two averages: **Comb3** (LP+SC+SE) and **Comb4** (+NLI).""")
n.code(r'''
df = add_signals(df)
df[["greedy", "label_containment", "U_lp", "U_sc", "U_se", "U_nli"]].round(2).head(6)
''')
n.md("""### Where SC and SE disagree
A question where the *strings* differ (SC high) but the *meaning* is stable (SE low): a paraphrase, not real doubt.""")
n.code(r'''
cand = df[(df.U_sc >= 0.5) & (df.U_se <= 0.35)]
if len(cand):
    r = cand.iloc[0]
    print(f"U_SC = {r.U_sc:.2f}   U_SE = {r.U_se:.2f}")
    for s in r.samples:
        print("   -", s)
else:
    print("No such question in this setting. Try another one.")
''')
n.md("""## Step 4 · Which signal detects wrong answers best?
**Predict before you run:** which of the six scores has the highest AUROC?
(AUROC = chance that a random *wrong* answer gets a higher uncertainty than a random *correct* one. 0.5 = coin flip, 1.0 = perfect.)""")
n.code(r'''
auc = auroc_table(df)
for k, v in auc.items():
    print(f"{k:7s} AUROC = {v:.3f}")
fig, ax = plt.subplots()
ax.barh(list(auc), list(auc.values()), color="0.55")
for i, v in enumerate(auc.values()):
    ax.text(v + 0.005, i, f"{v:.2f}", va="center")
ax.axvline(0.5, ls="--", color="black"); ax.text(0.5, -0.9, "chance", ha="center")
ax.set_xlim(0.4, 1); ax.set_xlabel("AUROC (higher = better)"); ax.set_title("Detecting wrong answers")
plt.show()
''')
n.md("""**Discuss with your neighbour (1 min):** are the differences between signals large? With 200 questions, a gap of 0.02 is mostly noise. Keep that in mind for notebook 03.""")
n.save()

# ---------------------------------------------------------------- 02
n = NB("02_fingerprinting")
n.md("""# 02 · Uncertainty fingerprints (≈ 12 min)
**Challenge 3: build a fingerprint and find the confident fabrication the signals miss.**

A **fingerprint** is the cell of the (SC, SE) plane an answer falls in. We cut each axis at its **median**:

| Region | SC | SE | Reading |
|---|---|---|---|
| Confident | low | low | samples agree |
| Gap-like | high | high | samples disagree |
| Shallow | low | high | strings agree, meanings differ |
| Surface-varying | high | low | wording differs, meaning agrees |

No labels are used to build it, so it could run in a real system.""")
n.code(BOOT)
n.code(r'''
df = add_signals(load_setting("mistral_triviaqa"))
checks.banner(df)
from scipy.stats import spearmanr
rho = spearmanr(df.U_sc, df.U_se).statistic
print(f"Rank correlation SC vs SE: {rho:.2f}   (close to 1 means they carry almost the same information)")
tau_sc, tau_se = fit_medians(df)
print(f"Median thresholds: tau_SC = {tau_sc:.2f}, tau_SE = {tau_se:.2f}")
''')
n.md("### 🛠 TODO 3: write the region rule")
n.todo(r'''
def my_region(sc, se, tau_sc, tau_se):
    """Return 'Confident', 'Gap-like', 'Shallow' or 'Surface-varying'."""
    # TODO 3: compare each signal with its threshold. A value EQUAL to the median counts as LOW.
    raise NotImplementedError("Fill in my_region")
''', r'''
def my_region(sc, se, tau_sc, tau_se):
    sc_high, se_high = sc > tau_sc, se > tau_se
    if not sc_high and not se_high: return "Confident"
    if sc_high and se_high:         return "Gap-like"
    if not sc_high and se_high:     return "Shallow"
    return "Surface-varying"
''')
n.code(r'''
checks.check_region(my_region, tau_sc, tau_se)
''')
n.code(r'''
df["region"] = [my_region(a, b, tau_sc, tau_se) for a, b in zip(df.U_sc, df.U_se)]
summary = region_summary(df)
print(summary.round(3))
plot_region_grid(summary)
plt.show()
''')
n.md("""**Observe:** how many regions have enough questions to say anything? Usually only *Confident* and *Gap-like* are well populated.
The error rate in *Gap-like* is the highest. That is the information that actually matters.""")
n.md("""## Challenge 3 · The confident fabrication
"Confident" by the median is only *relative*. Use an **absolute** definition: all 10 samples identical (`U_SC = 0`) **and** the answer is wrong.""")
n.code(r'''
bad = unanimous_errors(df)
print(f"{len(bad)} unanimous-but-wrong answers out of {len(df)} questions")
pct = lambda c: df[c].rank(pct=True)          # 0 = most certain, 1 = most uncertain
view = bad.assign(LP_pct=pct("U_lp")[bad.index], NLI_pct=pct("U_nli")[bad.index],
                  Comb4_pct=pct("Comb4")[bad.index])
view[["question", "greedy", "gold", "region", "LP_pct", "NLI_pct", "Comb4_pct"]].round(2)
''')
n.md("""**Questions to answer in pairs (3 min)**
1. Which region do these answers fall in? Could a policy that abstains on *Gap-like* ever catch them?
2. Do LP or NLI flag them? (a percentile near 1.0 = flagged, near 0 = invisible)
3. What kind of evidence *outside the model* could catch them?

**Bonus (fast finishers):** change `"mistral_triviaqa"` to `"qwen2_1_5b_triviaqa"`. What happens to `tau_SC`, and to who counts as "Confident"?""")
n.save()

# ---------------------------------------------------------------- 03
n = NB("03_adaptive_policy")
n.md("""# 03 · Adaptive policy (≈ 12 min)
**Challenge 4: design an adaptive policy** that decides to **answer**, **verify** (run an extra, costlier check) or **abstain**.

Key ideas
* **Coverage** = fraction of questions the system answers. **Accuracy** is measured on answered questions only.
* Abstaining more always raises accuracy. So a fair test compares policies **at the same coverage**.""")
n.code(BOOT)
n.code(r'''
df = add_regions(add_signals(load_setting("mistral_triviaqa")))
checks.banner(df)
y = df.label_containment.to_numpy()
print(f"Answer everything: accuracy = {y.mean():.3f}, coverage = 1.00")
''')
n.md("""## Step 1 · Accuracy–coverage curves
Answer the lowest-uncertainty questions first. The two black markers are the fingerprint policies:
**Policy A** abstains on *Gap-like*; **Policy B** answers only *Confident*.""")
n.code(r'''
maskA, maskB = (df.region != "Gap-like").to_numpy(), (df.region == "Confident").to_numpy()
pts = [(maskA.mean(), y[maskA].mean(), "Policy A", "*"), (maskB.mean(), y[maskB].mean(), "Policy B", "D")]
for c, a, name, _ in pts:
    print(f"{name}: coverage {c:.2f}, accuracy {a:.3f}")
plot_risk_coverage(df, "label_containment", ["U_sc", "U_se", "U_lp", "U_nli", "Comb3"], extra_points=pts)
plt.show()
''')
n.md("""**Look closely:** do the markers sit above the curves of the simple scores, or on them?""")
n.md("### 🛠 TODO 4: write the three-action rule")
n.todo(r'''
def my_actions(score, t_answer, t_abstain):
    """score < t_answer -> 'answer';  t_answer <= score < t_abstain -> 'verify';  otherwise 'abstain'."""
    # TODO 4: return a NumPy array of strings, one per question (hint: np.where, nested)
    raise NotImplementedError("Fill in my_actions")
''', r'''
def my_actions(score, t_answer, t_abstain):
    score = np.asarray(score, float)
    return np.where(score < t_answer, "answer", np.where(score < t_abstain, "verify", "abstain"))
''')
n.code(r'''
checks.check_actions(my_actions)
''')
n.md("""## Challenge 4 · Your policy
Pick two thresholds on the **Comb4** score (quantiles of the score: 0.5 means the median question).

*Toy cost model (an assumption, see `evaluate_actions`):* sampling costs 1.0 per question; *verify* adds 2.0 and
fixes the answer 90% of the time; *abstain* gives no answer.

**Leaderboard rule:** highest accuracy with coverage ≥ 0.60. Ties are broken by lower cost.""")
n.code(r'''
q_answer, q_abstain = 0.45, 0.80          # <-- CHANGE THESE and re-run
t1, t2 = df.Comb4.quantile(q_answer), df.Comb4.quantile(q_abstain)
actions = my_actions(df.Comb4, t1, t2)
res = evaluate_actions(actions, y)
print(pd.Series(actions).value_counts().to_dict())
print(f"accuracy {res['accuracy']:.3f} | coverage {res['coverage']:.2f} | cost {res['cost_per_question']:.2f}")
print("Eligible for the leaderboard" if res["coverage"] >= 0.60 else "Coverage below 0.60: not eligible")
''')
n.md("""### Sweep: the trade-off at a glance
Each row = a policy that answers up to `q_answer`, verifies up to `q_abstain`, abstains above.""")
n.code(r'''
rows = []
for qa in (0.3, 0.5, 0.7):
    for qb in (0.8, 0.9, 1.0):
        r = evaluate_actions(my_actions(df.Comb4, df.Comb4.quantile(qa), df.Comb4.quantile(qb) + (1e-9 if qb == 1.0 else 0)), y)
        rows.append({"q_answer": qa, "q_abstain": qb, **{k: round(v, 3) for k, v in r.items()}})
pd.DataFrame(rows)
''')
n.md("""## Step 2 · Does the *pattern* help, or just abstaining?
Compare Policy A with a **plain threshold on one score that answers the same fraction** (500 repetitions in the paper; 200 here).
If the 95% interval of the difference contains 0, the pattern adds nothing detectable.""")
n.code(r'''
cmp = matched_coverage_comparison(df, policy="A", n_rep=200)
print(cmp.round(3))
print("\nAny interval entirely above 0?", bool((cmp.ci_low > 0).any()))
''')
n.md("""**Discuss (2 min):** what does it mean if a plain threshold matches the fingerprint? When would you still prefer the fingerprint? (Think: explaining *why* the system abstained.)

**Bonus:** repeat with `label_col="label_judge"`, or with `policy="B"`. Does the conclusion change?""")
n.save()

# ---------------------------------------------------------------- 04
n = NB("04_break_the_detector")
n.md("""# 04 · Break the detector (≈ 15 min, small groups)
**Challenge 5.** Find answers where the detector is wrong or overconfident, and say *why*.

Everything here runs offline on the stored results across all five settings.""")
n.code(BOOT)
n.code(r'''
frames = [add_signals(load_setting(s)) for s in SETTINGS]        # normalize per setting
allq = pd.concat(frames, ignore_index=True)
allq = pd.concat([add_regions(f) for f in frames], ignore_index=True)
checks.banner(allq)
print(len(allq), "answers in", len(SETTINGS), "settings")
''')
n.md("""## Step 1 · How often is *unanimous* still wrong?
Absolute definition: all 10 samples identical. If agreement guaranteed correctness, the last column would be 0%.""")
n.code(r'''
g = allq.groupby("setting")
tab = pd.DataFrame({
    "unanimous": g.apply(lambda d: int((d.U_sc == 0).sum())),
    "unanimous_wrong": g.apply(lambda d: int(((d.U_sc == 0) & (d.label_containment == 0)).sum())),
})
tab["share_wrong"] = (tab.unanimous_wrong / tab.unanimous.replace(0, np.nan)).round(3)
tab
''')
n.md("""## Step 2 · Catalogue the failures
Pick **up to 10** confident-but-wrong cases. For each, decide the cause:

* `fabrication`: the model really is wrong
* `strict_gold`: the answer is right but the gold string is too strict (e.g. "doctor" vs "physician")
* `label_noise`: the label is wrong (e.g. wrong answer credited because it mentions the gold string)
* `ambiguous`: reasonable people disagree""")
n.code(r'''
cases = allq[(allq.U_sc == 0) & (allq.label_containment == 0)].head(10)
cases[["setting", "qid", "question", "greedy", "gold", "label_judge"]]
''')
n.code(r'''
# Edit this dictionary with your group's verdicts, then run the next cell
my_verdicts = {
    # "qid": "fabrication",
}
print(pd.Series(my_verdicts).value_counts() if my_verdicts else "Add some verdicts above.")
''')
n.md("""## Step 3 · Where do the two labelings disagree?
`label_containment` is a string rule; `label_judge` comes from an LLM judge. Disagreements show how label noise moves results.""")
n.code(r'''
dis = allq[allq.label_containment != allq.label_judge]
print(f"{len(dis)} of {len(allq)} answers ({len(dis)/len(allq):.1%}) get different labels")
dis[["setting", "question", "greedy", "gold", "label_containment", "label_judge"]].head(8)
''')
n.md("""## Step 4 · The detector's most confident mistakes
Wrong answers with the **lowest** Comb4 uncertainty (the system would answer these without hesitation).""")
n.code(r'''
worst = allq[allq.label_containment == 0].nsmallest(10, "Comb4")
worst[["setting", "question", "greedy", "gold", "U_sc", "U_se", "U_lp", "U_nli", "Comb4"]].round(2)
''')
n.md("""## Discussion (for the last 8 minutes)
1. Which failures could a **better detector** fix, and which need **outside evidence** (retrieval, a database, a human)?
2. How much of what looks like "model error" is actually **label error**?
3. Your product answers medical or legal questions. Which **coverage** would you accept, and who handles the abstentions?
4. Would you add a fifth sampling signal, or something of a different kind?""")
n.save()
print("built", sorted(p.name for p in (ROOT / "notebooks").glob("*.ipynb")))
