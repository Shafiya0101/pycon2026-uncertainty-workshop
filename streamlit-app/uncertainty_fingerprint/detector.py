"""A fitted detector for new, unlabeled answers.

The workshop functions fit medians and min-max scaling on the whole dataset at once.
In production you score answers one at a time, so the thresholds must be *learned on a
calibration set and then frozen*. :class:`Detector` does exactly that, using no labels.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .fingerprint import assign_region, fit_medians
from .policy import three_action
from .signals import add_signals, fit_scaler


@dataclass
class Detector:
    """Fit on unlabeled calibration answers, then score new answers.

    Example::

        det = Detector(score="Comb4").fit(calibration_df, q_answer=0.45, q_abstain=0.80)
        out = det.predict(new_df)      # adds signals, region and action columns
        det.save("detector.json")
    """
    score: str = "Comb4"
    scaler: Dict[str, Tuple[float, float]] = field(default_factory=dict)
    tau_sc: Optional[float] = None
    tau_se: Optional[float] = None
    t_answer: Optional[float] = None
    t_abstain: Optional[float] = None

    def fit(self, df: pd.DataFrame, q_answer: float = 0.5, q_abstain: float = 0.8) -> "Detector":
        """Learn scaling, fingerprint medians and the two policy cut-offs (as quantiles of
        ``score`` on the calibration data). Uses no correctness labels."""
        if not 0 <= q_answer <= q_abstain <= 1:
            raise ValueError("need 0 <= q_answer <= q_abstain <= 1")
        scored = add_signals(df)          # scales on df itself...
        self.scaler = fit_scaler(scored)  # ...and those bounds are frozen for later
        self.tau_sc, self.tau_se = fit_medians(scored)
        s = scored[self.score].to_numpy()
        self.t_answer = float(np.quantile(s, q_answer))
        # q_abstain = 1 means "never abstain"
        self.t_abstain = float(np.inf) if q_abstain >= 1 else float(np.quantile(s, q_abstain))
        return self

    def _check(self):
        if self.tau_sc is None:
            raise RuntimeError("call fit() or load() first")

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add the four signals, Comb3/Comb4, ``region`` and ``action`` columns."""
        self._check()
        out = add_signals(df, scaler=self.scaler)
        out["region"] = [assign_region(a, b, self.tau_sc, self.tau_se) for a, b in zip(out["U_sc"], out["U_se"])]
        out["action"] = three_action(out[self.score], self.t_answer, self.t_abstain)
        return out

    def predict_one(self, record: dict) -> dict:
        """Score a single answer given as a dict with the library's input fields."""
        row = self.predict(pd.DataFrame([record])).iloc[0]
        keys = ["U_lp", "U_sc", "U_se", "U_nli", "Comb3", "Comb4", "region", "action"]
        return {k: (row[k].item() if hasattr(row[k], "item") else row[k]) for k in keys}

    def save(self, path: Union[str, Path]) -> None:
        d = asdict(self)
        d["t_abstain"] = None if d["t_abstain"] == float("inf") else d["t_abstain"]
        Path(path).write_text(json.dumps(d, indent=2))

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Detector":
        d = json.loads(Path(path).read_text())
        d["scaler"] = {k: tuple(v) for k, v in d["scaler"].items()}
        if d["t_abstain"] is None:
            d["t_abstain"] = float("inf")
        return cls(**d)
