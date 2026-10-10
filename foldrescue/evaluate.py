"""Evaluation: the metrics, chosen thresholds, and multi-seed error bars.

WHY THESE METRICS
=================

**PR-AUC (average precision)** rather than ROC-AUC as the headline. With ~30%
positives and a decision that amounts to "which pairs go in the assay queue",
what matters is precision among the top-ranked pairs. ROC-AUC averages over
all thresholds including useless ones and is famously optimistic under class
imbalance (Saito & Rehmsmeier 2015). ROC-AUC is still reported, because it is
what most readers have calibrated intuitions for -- but PR-AUC is always
printed next to the base rate, because "PR-AUC 0.45" is excellent at 10%
prevalence and poor at 40%.

**MCC** (Matthews correlation coefficient) for the thresholded decision. MCC
is the one confusion-matrix summary that cannot be gamed by predicting the
majority class: it is the Pearson correlation between predicted and true
binary labels, and it is near zero for any constant predictor. Accuracy and F1
both flatter a degenerate model.

THE THRESHOLD PROBLEM, AND WHY 0.5 IS WRONG HERE
------------------------------------------------
A probability threshold of 0.5 is a convention from balanced binary problems.
At 31% prevalence a well-calibrated model will rarely exceed 0.5 for anything,
so thresholding there can yield *zero* positive predictions and an MCC of
exactly 0 -- which looks like a broken model and is in fact a broken metric
choice. (This repository hit precisely that during development.)

The threshold is therefore **chosen on the calibration split**, never on test,
by maximising MCC over a grid. Choosing it on test and reporting the resulting
MCC is fitting the test set with one parameter, and the resulting number is not
a held-out measurement.

ERROR BARS
----------
Every reported number is a mean +/- standard deviation over seeds. A single
split of ~250 test rows has a standard error on PR-AUC of several points: two
methods differing by 0.02 on one seed are indistinguishable. We vary the model
seed *and* the split seed, because split variance dominates at this n and
reporting only model-seed variance understates uncertainty badly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .calibration import brier_score, expected_calibration_error
from .utils import get_logger

__all__ = [
    "PairMetrics",
    "evaluate_predictions",
    "choose_threshold",
    "aggregate_over_seeds",
]

_log = get_logger(__name__)


def choose_threshold(
    y_calib: np.ndarray, p_calib: np.ndarray, *, objective: str = "mcc", n_grid: int = 199
) -> float:
    """Pick a decision threshold on held-out calibration data.

    Grid search over interior quantiles of the predicted scores, rather than a
    uniform grid on [0, 1]: when a model's scores all live in [0.1, 0.45], a
    uniform grid wastes most of its candidates and may miss the optimum
    entirely.
    """
    from sklearn.metrics import f1_score, matthews_corrcoef  # noqa: PLC0415

    y = np.asarray(y_calib).ravel()
    p = np.asarray(p_calib, dtype=float).ravel()
    if len(y) == 0 or len(np.unique(y)) < 2:
        _log.warning("cannot choose a threshold without both classes; defaulting to 0.5")
        return 0.5

    score_fn = matthews_corrcoef if objective == "mcc" else f1_score
    candidates = np.unique(np.quantile(p, np.linspace(0.01, 0.99, n_grid)))
    best_t, best_s = 0.5, -np.inf
    for t in candidates:
        s = float(score_fn(y, (p > t).astype(int)))
        if s > best_s:
            best_t, best_s = float(t), s
    _log.debug("threshold chosen on calib: %.4f (%s=%.4f)", best_t, objective, best_s)
    return best_t


@dataclass
class PairMetrics:
    """One evaluation of one model on one split."""

    split: str
    tier: str
    n_test: int
    prevalence: float
    pr_auc: float
    roc_auc: float
    mcc: float
    brier: float
    ece: float
    threshold: float
    n_predicted_positive: int
    oracle_pr_auc: float | None = None
    oracle_roc_auc: float | None = None
    extra: dict[str, float] = field(default_factory=dict)

    @property
    def pr_auc_lift_over_base(self) -> float:
        """PR-AUC above the no-skill baseline, which equals the prevalence."""
        return self.pr_auc - self.prevalence

    @property
    def fraction_of_ceiling(self) -> float | None:
        """PR-AUC as a fraction of what the Bayes-optimal predictor achieves.

        Only available on synthetic cohorts. It is the honest way to say "the
        model captured most of the available signal" versus "the model is bad",
        which raw PR-AUC cannot distinguish.
        """
        if self.oracle_pr_auc is None:
            return None
        denom = self.oracle_pr_auc - self.prevalence
        if denom <= 0:
            return None
        return float(self.pr_auc_lift_over_base / denom)

    def as_row(self) -> dict[str, object]:
        row: dict[str, object] = {
            "split": self.split,
            "tier": self.tier,
            "n_test": self.n_test,
            "prevalence": self.prevalence,
            "pr_auc": self.pr_auc,
            "roc_auc": self.roc_auc,
            "mcc": self.mcc,
            "brier": self.brier,
            "ece": self.ece,
            "threshold": self.threshold,
            "n_predicted_positive": self.n_predicted_positive,
        }
        if self.oracle_pr_auc is not None:
            row["oracle_pr_auc"] = self.oracle_pr_auc
            row["fraction_of_ceiling"] = self.fraction_of_ceiling
        row.update(self.extra)
        return row


def evaluate_predictions(
    y_test: np.ndarray,
    p_test: np.ndarray,
    *,
    split: str,
    tier: str,
    y_calib: np.ndarray | None = None,
    p_calib: np.ndarray | None = None,
    threshold: float | None = None,
    oracle_p_test: np.ndarray | None = None,
    n_bins: int = 10,
) -> PairMetrics:
    """Compute the metric set, with the threshold taken from calibration data.

    Exactly one of ``threshold`` or (``y_calib``, ``p_calib``) should be
    supplied. If neither is, the threshold falls back to 0.5 and a warning is
    emitted, because a silently-0.5 MCC on an imbalanced problem is a trap.
    """
    from sklearn.metrics import (  # noqa: PLC0415
        average_precision_score,
        matthews_corrcoef,
        roc_auc_score,
    )

    y = np.asarray(y_test).ravel()
    p = np.asarray(p_test, dtype=float).ravel()
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")

    if threshold is None:
        if y_calib is not None and p_calib is not None:
            threshold = choose_threshold(y_calib, p_calib)
        else:
            _log.warning(
                "no calibration split supplied for threshold selection; using 0.5. "
                "At low prevalence this can produce zero positive predictions and a "
                "meaningless MCC of 0."
            )
            threshold = 0.5

    pred = (p > threshold).astype(int)
    single_class = len(np.unique(y)) < 2
    return PairMetrics(
        split=split,
        tier=tier,
        n_test=int(len(y)),
        prevalence=float(y.mean()) if len(y) else float("nan"),
        pr_auc=float("nan") if single_class else float(average_precision_score(y, p)),
        roc_auc=float("nan") if single_class else float(roc_auc_score(y, p)),
        mcc=float(matthews_corrcoef(y, pred)) if not single_class else float("nan"),
        brier=brier_score(y, p),
        ece=expected_calibration_error(y, p, n_bins=n_bins),
        threshold=float(threshold),
        n_predicted_positive=int(pred.sum()),
        oracle_pr_auc=(
            None
            if oracle_p_test is None or single_class
            else float(average_precision_score(y, np.asarray(oracle_p_test).ravel()))
        ),
        oracle_roc_auc=(
            None
            if oracle_p_test is None or single_class
            else float(roc_auc_score(y, np.asarray(oracle_p_test).ravel()))
        ),
    )


def aggregate_over_seeds(rows: list[dict[str, object]], *, by: tuple[str, ...] = ("split", "tier")) -> pd.DataFrame:
    """Mean, SD and n over seeds, for the error bars every table needs.

    Reports the standard *deviation* across seeds rather than the standard
    error of the mean. SD answers "how much would this number move if I reran
    it?", which is the question a reader comparing two tiers actually has; SEM
    shrinks with more seeds and makes two methods look more separable than one
    rerun would show.
    """
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    numeric = [c for c in frame.columns if c not in by and pd.api.types.is_numeric_dtype(frame[c])]
    grouped = frame.groupby(list(by), observed=True)[numeric]
    out = grouped.agg(["mean", "std", "count"])
    out.columns = [f"{metric}_{stat}" for metric, stat in out.columns]
    return out.reset_index()
