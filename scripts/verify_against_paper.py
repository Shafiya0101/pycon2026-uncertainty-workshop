"""After exporting: do the workshop files reproduce the paper? Compares Table V (AUROC), Table VI
(region counts) and the unanimous-error counts of Section V-D. Run:  python scripts/verify_against_paper.py"""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from uncertainty_lib import *

# Table V: LP, SC, SE, NLI, Comb3, Comb4
AUROC = {"mistral_triviaqa": [.769, .768, .773, .792, .795, .813], "qwen2_7b_triviaqa": [.802, .800, .802, .808, .820, .822],
         "qwen2_1_5b_triviaqa": [.647, .700, .737, .802, .731, .780], "mistral_nq": [.694, .629, .656, .666, .666, .675],
         "mistral_hotpotqa": [.698, .680, .707, .688, .703, .711]}
# Table VI: tau_SC, n Confident, n Gap-like, n Shallow, n Surface-varying
REGION = {"mistral_triviaqa": (.20, 92, 87, 10, 11), "qwen2_7b_triviaqa": (.15, 100, 91, 0, 9),
          "qwen2_1_5b_triviaqa": (.70, 91, 79, 10, 20), "mistral_nq": (.80, 93, 64, 33, 10), "mistral_hotpotqa": (.65, 94, 93, 6, 7)}
# Section V-D (containment labels): (unanimous answers, of which wrong)
UNAN = {"mistral_triviaqa": (74, 5), "mistral_nq": (14, 4), "mistral_hotpotqa": (23, 5), "qwen2_7b_triviaqa": (86, 12), "qwen2_1_5b_triviaqa": (8, 0)}
COLS = ["U_lp", "U_sc", "U_se", "U_nli", "Comb3", "Comb4"]

bad = 0
for s in SETTINGS:
    df = load_setting(s)
    if df.attrs["synthetic"]:
        print(f"{s}: SYNTHETIC data in use, nothing to verify"); bad += 1; continue
    df = add_regions(add_signals(df))
    auc = auroc_table(df)
    print(f"\n== {s}")
    for c, ref in zip(COLS, AUROC[s]):
        ok = abs(auc[c] - ref) <= 0.0015
        bad += not ok
        print(f"   AUROC {c:6s} {auc[c]:.3f}  paper {ref:.3f}  {'OK' if ok else 'CHECK'}")
    tau, *_ = REGION[s]
    counts = tuple(int(df.region.eq(r).sum()) for r in REGIONS)
    ok = abs(fit_medians(df)[0] - tau) < 0.005 and counts == REGION[s][1:]
    bad += not ok
    print(f"   regions {counts}  tau_SC {fit_medians(df)[0]:.2f}  paper {REGION[s][1:]} / {tau}  {'OK' if ok else 'CHECK'}")
    un = (int((df.U_sc == 0).sum()), len(unanimous_errors(df)))
    ok = un == UNAN[s]
    bad += not ok
    print(f"   unanimous / wrong {un}  paper {UNAN[s]}  {'OK' if ok else 'CHECK'}")
print("\nALL MATCH THE PAPER" if not bad else f"\n{bad} item(s) to check (see CHECK above)")
