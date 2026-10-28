"""Friendly self-checks for the TODO cells."""
import numpy as np
from .fingerprint import assign_region
from .policy import three_action


def _report(ok, name, hint):
    print(f"✅ {name}: correct!" if ok else f"❌ {name}: not yet. {hint}")
    return ok


def check_sc(fn, ref, df, n=50):
    try:
        ok = all(np.isclose(fn(s), ref(s)) for s in df["samples"].head(n))
    except Exception as e:  # noqa: BLE001
        return _report(False, "self-consistency", f"Your function raised: {e!r}")
    return _report(ok, "self-consistency", "Hint: group NORMALIZED answers, take the biggest group.")


def check_se(fn, ref, df, n=50):
    try:
        ok = all(np.isclose(fn(c), ref(c)) for c in df["cluster_ids"].head(n))
    except Exception as e:  # noqa: BLE001
        return _report(False, "semantic entropy", f"Your function raised: {e!r}")
    return _report(ok, "semantic entropy", "Hint: p_k = cluster size / N, then -sum p log p.")


def check_region(fn, tau_sc, tau_se):
    try:
        pts = [(0, 0), (1, 1), (0, 1), (1, 0), (tau_sc, tau_se)]
        ok = all(fn(a, b, tau_sc, tau_se) == assign_region(a, b, tau_sc, tau_se) for a, b in pts)
    except Exception as e:  # noqa: BLE001
        return _report(False, "region function", f"Your function raised: {e!r}")
    return _report(ok, "region function", "Hint: a value EQUAL to the median counts as low.")


def check_actions(fn):
    try:
        s = np.array([0.1, 0.4, 0.8])
        ok = list(fn(s, 0.3, 0.6)) == list(three_action(s, 0.3, 0.6))
    except Exception as e:  # noqa: BLE001
        return _report(False, "three-action policy", f"Your function raised: {e!r}")
    return _report(ok, "three-action policy", "Hint: answer < t_answer <= verify < t_abstain <= abstain.")


def banner(df):
    """Warn loudly when the notebook is running on synthetic placeholder data."""
    if df.attrs.get("synthetic", False):
        print("⚠️  SYNTHETIC PLACEHOLDER DATA (data/synthetic). Numbers are NOT real results.")
    else:
        print("✅ Real precomputed results loaded (data/precomputed).")
