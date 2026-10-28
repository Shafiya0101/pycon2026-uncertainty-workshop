"""The four uncertainty signals (higher value = more uncertain)."""
import re
from collections import Counter
import numpy as np
from sklearn.metrics import roc_auc_score

_ARTICLES = re.compile(r"\b(a|an|the)\b")


def normalize(text):
    """Lower-case, drop the answer template, punctuation and articles."""
    t = text.lower().strip()
    t = re.sub(r"^the answer to the question is:?\s*", "", t)
    t = re.sub(r"[^\w\s]", " ", t)
    t = _ARTICLES.sub(" ", t)
    return " ".join(t.split())


def self_consistency_uncertainty(samples):
    """U_SC = 1 - (size of the largest group of identical normalized answers) / N."""
    counts = Counter(normalize(s) for s in samples)
    return 1.0 - max(counts.values()) / len(samples)


def greedy_cluster(bidir):
    """Greedy clustering. bidir[i][j] = 1 if answers i and j entail each other.
    Returns one cluster id per answer."""
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


def semantic_entropy(cluster_ids):
    """U_SE = -sum_k p_k log p_k, with p_k = share of samples in meaning cluster k."""
    counts = np.array(list(Counter(cluster_ids).values()), dtype=float)
    p = counts / counts.sum()
    return float(-(p * np.log(p)).sum())


def logprob_uncertainty(token_logprobs):
    """U_LP = minus the mean log-probability of the greedy answer's tokens."""
    return float(-np.mean(token_logprobs))


def nli_uncertainty(contra_probs):
    """U_NLI = mean contradiction probability between the greedy answer and each sample."""
    return float(np.mean(contra_probs))


def minmax(x):
    x = np.asarray(x, dtype=float)
    rng = x.max() - x.min()
    return (x - x.min()) / rng if rng > 0 else np.zeros_like(x)


def add_signals(df):
    """Add U_lp, U_sc, U_se, U_nli, Comb3 (LP+SC+SE) and Comb4 (+NLI) columns."""
    df = df.copy()
    df["cluster_ids"] = df["bidir_entail"].apply(greedy_cluster)
    df["U_lp"] = df["token_logprobs"].apply(logprob_uncertainty)
    df["U_sc"] = df["samples"].apply(self_consistency_uncertainty)
    df["U_se"] = df["cluster_ids"].apply(semantic_entropy)
    df["U_nli"] = df["nli_contra"].apply(nli_uncertainty)
    n = {c: minmax(df[c]) for c in ["U_lp", "U_sc", "U_se", "U_nli"]}
    df["Comb3"] = (n["U_lp"] + n["U_sc"] + n["U_se"]) / 3
    df["Comb4"] = (n["U_lp"] + n["U_sc"] + n["U_se"] + n["U_nli"]) / 4
    return df


def auroc_table(df, label_col="label_containment"):
    """AUROC of each score for ranking WRONG answers above correct ones."""
    wrong = 1 - df[label_col].to_numpy()
    cols = ["U_lp", "U_sc", "U_se", "U_nli", "Comb3", "Comb4"]
    return {c: float(roc_auc_score(wrong, df[c])) for c in cols}
