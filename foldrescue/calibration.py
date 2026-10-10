"""Probability calibration and conformal prediction.

WHY CALIBRATION IS THE POINT AND NOT A FOOTNOTE
===============================================

A ranking model answers "which pair is most promising?". A *calibrated* model
answers "how likely is this pair to work?" -- and only the second supports the
decision this package exists to inform, which is how to spend a limited assay
budget.

The distinction is not academic. A model with excellent PR-AUC can be grossly
overconfident: gradient-boosted trees trained with logloss on imbalanced data
routinely push scores toward 0 and 1, and a reported "0.95" that is right 70%
of the time will burn a third of a screening budget on pairs that were never
going to work. Discrimination (ranking) and calibration (probability accuracy)
are mathematically independent -- you can have either without the other.

Two standard repairs, both fitted on a held-out calibration split:

**Temperature scaling** (Guo, Pleiss, Sun & Weinberger 2017, *On Calibration
of Modern Neural Networks*) divides the logit by a single learned scalar T.
Because it is monotone with one parameter, it cannot change the ranking at all
-- AUC is provably unchanged -- and cannot overfit a small calibration split.
T > 1 means the model was overconfident. It is the right default.

**Isotonic regression** (Zadrozny & Elkan 2002) fits a free monotone step
function. Strictly more expressive, so it can repair non-monotone
miscalibration that temperature cannot, at the cost of being able to overfit a
few-hundred-point calibration split and of producing piecewise-constant
outputs with ties. Use it when you have calibration data to spare.

MEASURING IT
------------
*Brier score*: mean squared error of the probability, ``mean((p - y)^2)``. A
proper scoring rule -- it is minimised only by the true probability -- so it
captures discrimination and calibration together.

*Expected calibration error (ECE)*: bin predictions by confidence, then
average ``|mean(p) - mean(y)|`` over bins, weighted by bin population. ECE is
interpretable ("off by 8 percentage points on average") but has a real flaw:
it depends on the binning, and with equal-width bins most mass can land in one
bin and hide the error. We therefore report equal-count (quantile) bins by
default and expose the scheme, rather than quietly picking the flattering one.

CONFORMAL PREDICTION
--------------------
An alternative uncertainty route with a genuinely different guarantee. Split
(inductive) conformal prediction (Vovk, Papadopoulos; Lei et al.) takes
nonconformity scores on a calibration set and returns intervals with
*finite-sample marginal coverage* at level 1-alpha, under exchangeability
alone -- no assumption that the model is correct or that probabilities are
calibrated. The catch is in the word *marginal*: coverage holds on average
over the population, not conditionally for your particular query. A conformal
interval is not a promise about one pair, and this module's docstrings say so
because the distinction is routinely lost in practice.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from .utils import get_logger

__all__ = [
    "VennAbersPredictor",
    "CalibrationReport",
    "TemperatureScaler",
    "IsotonicCalibrator",
    "brier_score",
    "expected_calibration_error",
    "reliability_curve",
    "fit_calibrator",
    "ConformalPredictor",
]

_log = get_logger(__name__)

_EPS = 1e-12

BinScheme = Literal["quantile", "uniform"]


def _as_prob(p: np.ndarray) -> np.ndarray:
    """Clip to the open interval so logits stay finite."""
    return np.clip(np.asarray(p, dtype=float).ravel(), _EPS, 1.0 - _EPS)


def _logit(p: np.ndarray) -> np.ndarray:
    p = _as_prob(p)
    return np.log(p / (1.0 - p))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    # Branch to avoid overflow in exp for large |z|.
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    """Mean squared error of probabilistic predictions. Lower is better."""
    y = np.asarray(y_true, dtype=float).ravel()
    p = _as_prob(p)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    scheme: BinScheme = "quantile",
) -> float:
    """Population-weighted mean gap between confidence and accuracy.

    ``scheme="quantile"`` uses equal-count bins, which keeps every bin
    populated and stops a single crowded bin from masking error. ``"uniform"``
    reproduces the equal-width convention from the original papers; we report
    it when comparing to published numbers and otherwise prefer quantile.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = _as_prob(p)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs p {p.shape}")
    if len(y) == 0:
        return float("nan")

    if scheme == "quantile":
        # Unique quantile edges; duplicates collapse when p is highly tied
        # (isotonic output does exactly this), and that is fine.
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    elif scheme == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    else:  # pragma: no cover
        raise ValueError(f"unknown bin scheme {scheme!r}")
    if len(edges) < 2:
        return float(abs(p.mean() - y.mean()))

    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        total += (m.sum() / len(y)) * abs(p[m].mean() - y[m].mean())
    return float(total)


def reliability_curve(
    y_true: np.ndarray,
    p: np.ndarray,
    *,
    n_bins: int = 10,
    scheme: BinScheme = "quantile",
) -> dict[str, np.ndarray]:
    """Points for a reliability diagram: mean predicted vs observed per bin.

    Returns bin counts too, because a reliability diagram without them is
    misleading -- a wildly off point carrying four samples looks identical to
    one carrying four hundred.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    p = _as_prob(p)
    if scheme == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    if len(edges) < 2:
        return {
            "mean_predicted": np.array([p.mean()]),
            "observed_frequency": np.array([y.mean()]),
            "count": np.array([len(y)]),
        }
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, len(edges) - 2)
    pred, obs, cnt = [], [], []
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        pred.append(p[m].mean())
        obs.append(y[m].mean())
        cnt.append(int(m.sum()))
    return {
        "mean_predicted": np.asarray(pred),
        "observed_frequency": np.asarray(obs),
        "count": np.asarray(cnt),
    }


@dataclass
class CalibrationReport:
    """Before/after metrics for one calibrator on one split.

    ``improved`` is the headline the cookbook reports: calibration that does
    not reduce ECE on held-out data has not helped, and saying so is the whole
    discipline here.
    """

    method: str
    n_calib: int
    n_eval: int
    brier_before: float
    brier_after: float
    ece_before: float
    ece_after: float
    ece_uniform_before: float
    ece_uniform_after: float
    params: dict[str, float]

    @property
    def ece_reduction(self) -> float:
        return self.ece_before - self.ece_after

    @property
    def improved(self) -> bool:
        return self.ece_after < self.ece_before

    def summary(self) -> str:
        arrow = "improved" if self.improved else "DID NOT IMPROVE"
        return (
            f"{self.method}: ECE {self.ece_before:.4f} -> {self.ece_after:.4f} "
            f"({arrow}), Brier {self.brier_before:.4f} -> {self.brier_after:.4f} "
            f"[n_calib={self.n_calib}, n_eval={self.n_eval}]"
        )


# ---------------------------------------------------------------------------
# Calibrators
# ---------------------------------------------------------------------------


class TemperatureScaler:
    """Single-parameter logit scaling: ``p' = sigmoid(logit(p) / T)``.

    Fitted by minimising negative log-likelihood over T. Because the map is
    monotone, **ranking and therefore ROC-AUC / PR-AUC are unchanged** -- which
    is why this is safe to apply by default: it can only alter the probability
    values, never the shortlist order.

    T is found by golden-section search over ``log T``. One dimension, smooth,
    convex in practice -- a bespoke optimiser is not worth a scipy dependency
    here, and search is more robust than a hand-rolled Newton step.
    """

    def __init__(self, t_bounds: tuple[float, float] = (0.05, 20.0)) -> None:
        self.t_bounds = t_bounds
        self.temperature_: float = 1.0
        self.fitted_: bool = False

    def _nll(self, log_t: float, z: np.ndarray, y: np.ndarray) -> float:
        t = float(np.exp(log_t))
        p = _as_prob(_sigmoid(z / t))
        return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))

    def fit(self, p: np.ndarray, y: np.ndarray) -> TemperatureScaler:
        z = _logit(p)
        y = np.asarray(y, dtype=float).ravel()
        if len(np.unique(y)) < 2:
            _log.warning(
                "temperature scaling needs both classes in the calibration split; "
                "got only %s. Leaving T=1 (identity).",
                np.unique(y),
            )
            self.temperature_, self.fitted_ = 1.0, True
            return self

        lo, hi = np.log(self.t_bounds[0]), np.log(self.t_bounds[1])
        invphi, invphi2 = (np.sqrt(5) - 1) / 2, (3 - np.sqrt(5)) / 2
        a, b = lo, hi
        h = b - a
        c, d = a + invphi2 * h, a + invphi * h
        fc, fd = self._nll(c, z, y), self._nll(d, z, y)
        for _ in range(80):  # ~1e-8 relative precision on log T
            if fc < fd:
                b, d, fd = d, c, fc
                h *= invphi
                c = a + invphi2 * h
                fc = self._nll(c, z, y)
            else:
                a, c, fc = c, d, fd
                h *= invphi
                d = a + invphi * h
                fd = self._nll(d, z, y)
            if h < 1e-9:
                break
        self.temperature_ = float(np.exp((a + b) / 2))
        self.fitted_ = True
        _log.info(
            "temperature scaling: T=%.4f (%s)",
            self.temperature_,
            "was overconfident" if self.temperature_ > 1 else "was underconfident",
        )
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("TemperatureScaler.fit must be called before transform")
        return _sigmoid(_logit(p) / self.temperature_)

    @property
    def params(self) -> dict[str, float]:
        return {"temperature": self.temperature_}


class IsotonicCalibrator:
    """Free monotone mapping via isotonic regression (PAVA).

    More expressive than temperature scaling, and correspondingly easier to
    overfit: with a few hundred calibration points it can chase noise and
    produce long flat plateaus. It also *can* change the ranking, by mapping
    distinct scores onto identical calibrated values (ties), so the shortlist
    order may shift relative to the raw model.
    """

    def __init__(self) -> None:
        self.model_ = None
        self.fitted_ = False

    def fit(self, p: np.ndarray, y: np.ndarray) -> IsotonicCalibrator:
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

        y = np.asarray(y, dtype=float).ravel()
        if len(np.unique(y)) < 2:
            _log.warning("isotonic calibration needs both classes; leaving identity.")
            self.model_, self.fitted_ = None, True
            return self
        self.model_ = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
        self.model_.fit(_as_prob(p), y)
        self.fitted_ = True
        return self

    def transform(self, p: np.ndarray) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("IsotonicCalibrator.fit must be called before transform")
        if self.model_ is None:
            return _as_prob(p)
        return _as_prob(self.model_.predict(_as_prob(p)))

    @property
    def params(self) -> dict[str, float]:
        # NB: do not use `or []` on the thresholds -- it is a numpy array and
        # truth-testing it raises. Number of knots is a useful overfit smell:
        # approaching n_calib means the step function is chasing noise.
        thresholds = getattr(self.model_, "X_thresholds_", None)
        return {"n_thresholds": 0.0 if thresholds is None else float(len(thresholds))}


def fit_calibrator(
    method: str,
    p_calib: np.ndarray,
    y_calib: np.ndarray,
    *,
    p_eval: np.ndarray | None = None,
    y_eval: np.ndarray | None = None,
    n_bins: int = 10,
) -> tuple[TemperatureScaler | IsotonicCalibrator, CalibrationReport]:
    """Fit a calibrator and measure it on data it was not fitted on.

    When ``p_eval`` is omitted the report is computed on the calibration split
    itself and is an *optimistic* in-sample figure; the method name is suffixed
    with ``(in-sample)`` so a report can never be mistaken for a held-out one.
    """
    cal: TemperatureScaler | IsotonicCalibrator
    if method == "temperature":
        cal = TemperatureScaler()
    elif method == "isotonic":
        cal = IsotonicCalibrator()
    else:
        raise ValueError(f"unknown calibration method {method!r}; use 'temperature' or 'isotonic'")
    cal.fit(p_calib, y_calib)

    in_sample = p_eval is None or y_eval is None
    pe = np.asarray(p_calib if in_sample else p_eval, dtype=float)
    ye = np.asarray(y_calib if in_sample else y_eval, dtype=float)
    pa = cal.transform(pe)

    report = CalibrationReport(
        method=method + (" (in-sample)" if in_sample else ""),
        n_calib=len(np.asarray(p_calib).ravel()),
        n_eval=len(pe.ravel()),
        brier_before=brier_score(ye, pe),
        brier_after=brier_score(ye, pa),
        ece_before=expected_calibration_error(ye, pe, n_bins=n_bins),
        ece_after=expected_calibration_error(ye, pa, n_bins=n_bins),
        ece_uniform_before=expected_calibration_error(ye, pe, n_bins=n_bins, scheme="uniform"),
        ece_uniform_after=expected_calibration_error(ye, pa, n_bins=n_bins, scheme="uniform"),
        params=cal.params,
    )
    _log.info(report.summary())
    return cal, report


# ---------------------------------------------------------------------------
# Conformal prediction
# ---------------------------------------------------------------------------


class ConformalPredictor:
    """Split-conformal intervals around a predicted probability.

    Method: on a calibration split, score each point by nonconformity
    ``s_i = |y_i - p_i|`` (absolute residual on the probability scale). Take
    the ``ceil((n+1)(1-alpha))/n`` empirical quantile ``q`` of those scores.
    The interval ``[p - q, p + q]``, clipped to [0, 1], then has marginal
    coverage of at least ``1 - alpha`` for an exchangeable future point.

    WHAT THE GUARANTEE DOES AND DOES NOT SAY
    ----------------------------------------
    It says: across many queries drawn like the calibration data, at least
    ``1-alpha`` of true labels fall inside. It does *not* say the guarantee
    holds for a particular query, nor for a query drawn from somewhere else --
    exchangeability is exactly what an out-of-domain query violates. Hence the
    interval width is constant here: a single global ``q``, honestly flat,
    rather than a per-point width that would imply conditional information the
    method does not provide.

    This is why conformal intervals and the applicability domain in
    ``domain.py`` are complements, not substitutes. The interval quantifies
    noise under exchangeability; the domain flag tells you whether
    exchangeability is plausible at all.
    """

    def __init__(self, alpha: float = 0.1) -> None:
        if not 0.0 < alpha < 1.0:
            raise ValueError(f"alpha must be in (0, 1), got {alpha}")
        self.alpha = alpha
        self.q_: float = 1.0
        self.n_calib_: int = 0
        self.fitted_: bool = False

    def fit(self, p_calib: np.ndarray, y_calib: np.ndarray) -> ConformalPredictor:
        p = _as_prob(p_calib)
        y = np.asarray(y_calib, dtype=float).ravel()
        if p.shape != y.shape:
            raise ValueError(f"shape mismatch: p {p.shape} vs y {y.shape}")
        n = len(p)
        self.n_calib_ = n
        if n == 0:
            raise ValueError("conformal calibration requires at least one point")

        scores = np.abs(y - p)
        # The finite-sample correction (n+1 rather than n) is what makes the
        # coverage guarantee exact rather than asymptotic. Dropping it is the
        # most common implementation error in conformal code.
        level = np.ceil((n + 1) * (1.0 - self.alpha)) / n
        if level > 1.0:
            # Too few calibration points to certify this alpha at all.
            _log.warning(
                "conformal: n_calib=%d is too small for alpha=%.3f (needs >= %d); "
                "falling back to the maximum observed nonconformity, which is "
                "CONSERVATIVE but carries no exact guarantee.",
                n,
                self.alpha,
                int(np.ceil(1.0 / self.alpha) - 1),
            )
            self.q_ = float(scores.max())
        else:
            self.q_ = float(np.quantile(scores, level, method="higher"))
        self.fitted_ = True
        _log.info(
            "conformal: q=%.4f at alpha=%.3f from n_calib=%d (interval half-width)",
            self.q_,
            self.alpha,
            n,
        )
        return self

    def interval(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return ``(lower, upper)`` clipped to [0, 1]."""
        if not self.fitted_:
            raise RuntimeError("ConformalPredictor.fit must be called before interval")
        p = _as_prob(p)
        return np.clip(p - self.q_, 0.0, 1.0), np.clip(p + self.q_, 0.0, 1.0)

    def empirical_coverage(self, p: np.ndarray, y: np.ndarray) -> float:
        """Fraction of true labels inside the interval. Should be >= 1-alpha."""
        lo, hi = self.interval(p)
        y = np.asarray(y, dtype=float).ravel()
        return float(np.mean((y >= lo - _EPS) & (y <= hi + _EPS)))

    def label_set(self, p: np.ndarray) -> list[tuple[int, ...]]:
        """Conformal *label sets* -- the meaningful conformal output for binary y.

        For each query, the set of labels not rejected at level ``alpha``:
        ``(0,)`` or ``(1,)`` is a confident call, ``(0, 1)`` is an explicit
        abstention, and ``()`` means both labels were rejected (which happens
        when the model is confidently wrong about this region).

        This is reported alongside the Venn-Abers interval because it answers
        a different question: the label set says "which outcomes are
        plausible?", the interval says "what is the probability?".
        """
        if not self.fitted_:
            raise RuntimeError("ConformalPredictor.fit must be called before label_set")
        p = _as_prob(p)
        out = []
        for pi in p:
            labels = tuple(lab for lab in (0, 1) if abs(lab - pi) <= self.q_ + _EPS)
            out.append(labels)
        return out


class VennAbersPredictor:
    """Probability *intervals* for a binary outcome (Vovk & Petej 2014).

    WHY NOT PLAIN CONFORMAL REGRESSION HERE
    ---------------------------------------
    Split-conformal regression with nonconformity ``|y - p|`` is valid for
    binary ``y``, but useless: because ``y`` is 0 or 1, the calibration
    residuals are large, the quantile ``q`` lands near 0.9, and every interval
    collapses to the whole of [0, 1]. (This repository shipped exactly that for
    one iteration -- every row read ``ci_low=0.0, ci_high=1.0``, which is
    honest and conveys nothing.)

    Venn-Abers is the right instrument for a *probability* interval on a binary
    label. The inductive version (IVAP) works like this: to score a query with
    raw model output ``p*``, fit isotonic regression on the calibration pairs
    twice -- once with ``(p*, 0)`` appended and once with ``(p*, 1)`` appended
    -- and read off the fitted value at ``p*`` each time. Those two numbers,
    ``[p_lo, p_hi]``, bracket the probability.

    The guarantee is *validity*: one of the two endpoints is a perfectly
    calibrated probability, so the interval is guaranteed to contain a
    calibrated estimate. The interval width is therefore a direct readout of
    how much the calibration data constrains this region -- wide where
    calibration data is sparse, narrow where it is dense. That is exactly the
    behaviour a bench scientist wants from an error bar.

    Cost: a naive implementation refits isotonic twice per query. We instead
    precompute on a grid of raw scores and interpolate, which is O(grid) fits
    rather than O(n_query), and is exact to the grid resolution.
    """

    def __init__(self, n_grid: int = 256) -> None:
        self.n_grid = int(n_grid)
        self.grid_: np.ndarray = np.array([])
        self.lo_: np.ndarray = np.array([])
        self.hi_: np.ndarray = np.array([])
        self.fitted_ = False

    def fit(self, p_calib: np.ndarray, y_calib: np.ndarray) -> VennAbersPredictor:
        from sklearn.isotonic import IsotonicRegression  # noqa: PLC0415

        p = _as_prob(p_calib)
        y = np.asarray(y_calib, dtype=float).ravel()
        if p.shape != y.shape:
            raise ValueError(f"shape mismatch: p {p.shape} vs y {y.shape}")
        if len(np.unique(y)) < 2:
            _log.warning("Venn-Abers needs both classes; returning a degenerate [0,1].")
            self.grid_ = np.linspace(0.0, 1.0, 2)
            self.lo_, self.hi_ = np.zeros(2), np.ones(2)
            self.fitted_ = True
            return self

        # Grid over the observed score range: querying outside it is
        # extrapolation, and np.interp clamps to the endpoints, which is the
        # conservative direction.
        self.grid_ = np.unique(
            np.clip(np.linspace(p.min(), p.max(), self.n_grid), _EPS, 1 - _EPS)
        )
        lo, hi = [], []
        for g in self.grid_:
            for label, sink in ((0.0, lo), (1.0, hi)):
                iso = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
                iso.fit(np.append(p, g), np.append(y, label))
                sink.append(float(iso.predict([g])[0]))
        self.lo_ = np.asarray(lo)
        self.hi_ = np.asarray(hi)
        # p0 <= p1 holds by construction; enforce it against float noise.
        swap = self.lo_ > self.hi_
        self.lo_[swap], self.hi_[swap] = self.hi_[swap], self.lo_[swap]
        self.fitted_ = True
        _log.info(
            "Venn-Abers fitted on %d calibration points; median interval width %.4f",
            len(p),
            float(np.median(self.hi_ - self.lo_)),
        )
        return self

    def interval(self, p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.fitted_:
            raise RuntimeError("VennAbersPredictor.fit must be called before interval")
        p = _as_prob(p)
        return (
            np.interp(p, self.grid_, self.lo_),
            np.interp(p, self.grid_, self.hi_),
        )

    def point(self, p: np.ndarray) -> np.ndarray:
        """Midpoint of the interval -- the usual Venn-Abers point prediction."""
        lo, hi = self.interval(p)
        return (lo + hi) / 2.0

    def empirical_coverage(self, p: np.ndarray, y: np.ndarray) -> float:
        """Fraction of outcomes bracketed, for the notebook's sanity check."""
        lo, hi = self.interval(p)
        y = np.asarray(y, dtype=float).ravel()
        return float(np.mean((y >= lo - _EPS) & (y <= hi + _EPS)))
