"""Tests for calibration, including the required synthetic-miscalibration case.

The specification requires: "calibration improves ECE on a synthetic
miscalibrated case". That is ``test_temperature_scaling_improves_ece_on_*``
below. The construction is deliberate -- we *know* the true temperature,
because we introduced it -- so the test checks both that ECE falls and that the
recovered parameter is close to the planted one. A test that only checks "ECE
went down" would pass for a calibrator that squashed everything to the base
rate.
"""

from __future__ import annotations

import numpy as np
import pytest

from foldrescue.calibration import (
    ConformalPredictor,
    IsotonicCalibrator,
    TemperatureScaler,
    VennAbersPredictor,
    brier_score,
    expected_calibration_error,
    fit_calibrator,
    reliability_curve,
)

TRUE_TEMPERATURE = 3.0


def make_miscalibrated(n: int = 4000, temperature: float = TRUE_TEMPERATURE, seed: int = 0):
    """Labels from a well-specified model, scores overconfident by ``temperature``.

    ``z`` is the true logit; ``y`` is drawn from ``sigmoid(z)``, so ``sigmoid(z)``
    is perfectly calibrated by construction. The reported score is
    ``sigmoid(z * temperature)``, i.e. the same ranking pushed toward 0 and 1 --
    exactly the failure mode of a boosted tree trained to minimise logloss.
    """
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, size=n)
    y = (rng.random(n) < 1.0 / (1.0 + np.exp(-z))).astype(int)
    p_overconfident = 1.0 / (1.0 + np.exp(-z * temperature))
    return p_overconfident, y


def test_temperature_scaling_improves_ece_on_synthetic_miscalibrated_case():
    """REQUIRED BY SPEC: calibration reduces ECE on held-out data."""
    p, y = make_miscalibrated()
    half = len(y) // 2
    _, report = fit_calibrator(
        "temperature", p[:half], y[:half], p_eval=p[half:], y_eval=y[half:]
    )

    assert report.improved, report.summary()
    assert report.ece_after < report.ece_before
    # Not a marginal win: the planted miscalibration is severe.
    assert report.ece_before > 0.10
    assert report.ece_after < 0.05
    assert report.ece_reduction > 0.05
    # Brier is a proper scoring rule, so it must also improve.
    assert report.brier_after < report.brier_before
    # And the same must hold under the equal-width binning convention, so the
    # result is not an artefact of quantile bins.
    assert report.ece_uniform_after < report.ece_uniform_before


def test_temperature_scaling_recovers_the_planted_temperature():
    p, y = make_miscalibrated()
    scaler = TemperatureScaler().fit(p, y)
    assert scaler.temperature_ == pytest.approx(TRUE_TEMPERATURE, rel=0.15)
    assert scaler.temperature_ > 1.0, "overconfidence must yield T > 1"


def test_temperature_scaling_cannot_change_the_ranking():
    """A one-parameter monotone map provably preserves AUC.

    This is the property that makes temperature scaling safe to apply by
    default: it can only change the numbers, never the shortlist order.
    """
    from sklearn.metrics import roc_auc_score

    p, y = make_miscalibrated()
    scaler = TemperatureScaler().fit(p, y)
    before = roc_auc_score(y, p)
    after = roc_auc_score(y, scaler.transform(p))
    assert after == pytest.approx(before, abs=1e-12)


def test_isotonic_improves_ece_too():
    p, y = make_miscalibrated()
    half = len(y) // 2
    _, report = fit_calibrator("isotonic", p[:half], y[:half], p_eval=p[half:], y_eval=y[half:])
    assert report.improved, report.summary()
    assert report.ece_after < 0.06


def test_in_sample_reports_are_labelled_as_such():
    """An in-sample report must never be mistakable for a held-out one."""
    p, y = make_miscalibrated()
    _, report = fit_calibrator("temperature", p, y)
    assert "in-sample" in report.method


def test_calibrating_an_already_calibrated_model_does_little():
    """Guard against a calibrator that "improves" everything by flattening."""
    p, y = make_miscalibrated(temperature=1.0)  # already calibrated
    scaler = TemperatureScaler().fit(p, y)
    assert scaler.temperature_ == pytest.approx(1.0, abs=0.25)


def test_single_class_calibration_split_is_survivable():
    """Degenerate split: leave the identity rather than crash or invent a T."""
    p = np.linspace(0.1, 0.9, 50)
    y = np.ones(50, dtype=int)
    scaler = TemperatureScaler().fit(p, y)
    assert scaler.temperature_ == 1.0
    assert np.allclose(scaler.transform(p), p)

    iso = IsotonicCalibrator().fit(p, y)
    assert np.allclose(iso.transform(p), p)


def test_transform_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        TemperatureScaler().transform(np.array([0.5]))
    with pytest.raises(RuntimeError, match="fit"):
        IsotonicCalibrator().transform(np.array([0.5]))


def test_unknown_calibration_method_rejected():
    with pytest.raises(ValueError, match="unknown calibration method"):
        fit_calibrator("sigmoid_platt_but_misspelled", np.array([0.5, 0.6]), np.array([0, 1]))


# --- metrics ---------------------------------------------------------------


def test_perfect_predictions_score_zero():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.0, 0.0, 1.0, 1.0])
    assert brier_score(y, p) == pytest.approx(0.0, abs=1e-20)
    assert expected_calibration_error(y, p) == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_a_constant_overconfident_predictor():
    y = np.zeros(100, dtype=int)
    p = np.full(100, 0.9)
    assert expected_calibration_error(y, p) == pytest.approx(0.9, abs=1e-6)


def test_ece_shape_mismatch_raises():
    with pytest.raises(ValueError, match="shape mismatch"):
        expected_calibration_error(np.zeros(5), np.zeros(4))


def test_reliability_curve_returns_bin_counts():
    """A reliability diagram without counts misleads; the counts must be there."""
    p, y = make_miscalibrated(n=500)
    curve = reliability_curve(y, p, n_bins=5)
    assert set(curve) == {"mean_predicted", "observed_frequency", "count"}
    assert curve["count"].sum() == len(y)
    assert len(curve["mean_predicted"]) == len(curve["count"])


# --- conformal / Venn-Abers ------------------------------------------------


@pytest.mark.parametrize("alpha", [0.05, 0.1, 0.2])
def test_conformal_coverage_is_near_nominal(alpha):
    p, y = make_miscalibrated(n=6000)
    half = len(y) // 2
    cp = ConformalPredictor(alpha=alpha).fit(p[:half], y[:half])
    coverage = cp.empirical_coverage(p[half:], y[half:])
    # Marginal coverage holds in expectation over splits, so a single split can
    # land slightly under; allow Monte Carlo slack but no systematic shortfall.
    assert coverage >= (1 - alpha) - 0.03, f"coverage {coverage:.4f} for alpha={alpha}"


def test_conformal_label_sets_are_informative():
    """For binary labels the useful conformal output is the label set."""
    p, y = make_miscalibrated(n=3000)
    half = len(y) // 2
    cp = ConformalPredictor(alpha=0.1).fit(p[:half], y[:half])
    sets = cp.label_set(p[half:])
    assert len(sets) == len(y) - half
    assert all(isinstance(s, tuple) for s in sets)
    # Some confident calls and some abstentions -- not all one or the other.
    singletons = sum(1 for s in sets if len(s) == 1)
    assert 0 < singletons < len(sets)


def test_conformal_warns_but_survives_tiny_calibration_sets():
    cp = ConformalPredictor(alpha=0.01).fit(np.array([0.2, 0.8]), np.array([0, 1]))
    assert cp.fitted_
    lo, hi = cp.interval(np.array([0.5]))
    assert 0.0 <= lo[0] <= hi[0] <= 1.0


def test_venn_abers_intervals_are_not_degenerate():
    """The defect this class exists to fix: never return [0, 1] everywhere.

    Split-conformal regression on |y - p| yields q ~ 0.9 for binary labels and
    therefore the whole unit interval for every query -- valid and useless.
    """
    p, y = make_miscalibrated(n=3000)
    half = len(y) // 2
    va = VennAbersPredictor().fit(p[:half], y[:half])
    lo, hi = va.interval(p[half:])

    assert (lo <= hi + 1e-12).all(), "interval endpoints must be ordered"
    assert (lo >= 0).all() and (hi <= 1).all()
    whole_interval = (lo <= 1e-9) & (hi >= 1 - 1e-9)
    assert not whole_interval.any(), "every interval collapsed to [0, 1]"
    assert np.median(hi - lo) < 0.25, "intervals should be informative, not vacuous"


def test_venn_abers_point_prediction_lies_inside_its_interval():
    p, y = make_miscalibrated(n=2000)
    va = VennAbersPredictor().fit(p, y)
    lo, hi = va.interval(p)
    mid = va.point(p)
    assert ((mid >= lo - 1e-12) & (mid <= hi + 1e-12)).all()


def test_venn_abers_interval_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        VennAbersPredictor().interval(np.array([0.5]))


def test_conformal_rejects_invalid_alpha():
    for bad in (0.0, 1.0, -1.0):
        with pytest.raises(ValueError, match="alpha"):
            ConformalPredictor(alpha=bad)
