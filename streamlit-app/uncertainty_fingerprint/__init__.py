"""uncertainty-fingerprint: detect LLM hallucinations with four uncertainty signals,
uncertainty fingerprints and adaptive answer / verify / abstain policies.

Core (numpy, pandas, scikit-learn only)::

    import uncertainty_fingerprint as ufp
    df = ufp.add_regions(ufp.add_signals(ufp.load_setting("mistral_triviaqa")))
    ufp.auroc_table(df)

Optional extras: ``[nli]`` (NLIScorer), ``[generate]`` (HFGenerator), ``[plot]``.
"""
from .signals import (SIGNALS, SCORES, normalize, self_consistency_uncertainty, greedy_cluster,
                      semantic_entropy, logprob_uncertainty, nli_uncertainty, minmax, fit_scaler,
                      add_signals, auroc_table)
from .fingerprint import REGIONS, fit_medians, assign_region, add_regions, region_summary, unanimous_errors
from .policy import (risk_coverage_curve, accuracy_at_coverage, three_action, evaluate_actions,
                     matched_coverage_comparison)
from .data import SETTINGS, REQUIRED, read_jsonl, write_jsonl, validate, load_setting, load_all
from .detector import Detector

__version__ = "0.1.0"
__all__ = [n for n in dir() if not n.startswith("_")]
