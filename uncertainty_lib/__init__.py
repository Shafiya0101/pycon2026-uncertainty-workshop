"""Small helper library for the PyCon ES 2026 uncertainty-fingerprinting workshop."""
from .signals import (normalize, self_consistency_uncertainty, greedy_cluster,
                      semantic_entropy, logprob_uncertainty, nli_uncertainty,
                      minmax, add_signals, auroc_table)
from .fingerprint import (REGIONS, fit_medians, assign_region, add_regions,
                          region_summary, unanimous_errors)
from .policy import (risk_coverage_curve, accuracy_at_coverage, three_action,
                     evaluate_actions, matched_coverage_comparison)
from .data import SETTINGS, load_setting, load_all
