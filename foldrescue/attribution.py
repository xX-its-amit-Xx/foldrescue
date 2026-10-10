"""Feature attribution, and the promiscuity interrogation.

TWO DIFFERENT QUESTIONS
=======================

*Which features does the model use?* -- answered by permutation importance and
SHAP.

*Did the model learn structure-function signal, or did it learn which
compounds are promiscuous?* -- NOT answered by either, and it is the question
that decides whether this package is useful. A model can have beautiful SHAP
plots dominated by ligand features and still be doing nothing but recalling
per-compound base rates.

This module implements both, and keeps them clearly separate.

PERMUTATION IMPORTANCE
----------------------
Shuffle one feature column, measure how much the score drops (Breiman 2001).
Advantages: model-agnostic, measured on held-out data, directly interpretable
as "how much does this feature contribute to generalisation?". The well-known
failure mode is correlated features: shuffling one of two collinear columns
barely hurts, because the model reads the signal off the other, so both look
unimportant. We therefore also permute **whole blocks** jointly, which is the
right granularity for the ablation question and is immune to within-block
collinearity.

SHAP
----
Shapley values (Shapley 1953, imported into ML by Lundberg & Lee 2017) are the
unique additive attribution satisfying local accuracy, missingness and
consistency. ``TreeSHAP`` computes them exactly for tree ensembles in
polynomial time, which is why Tier 1/3 get real Shapley values while Tier 2
would need a sampling approximation.

Caveat stated plainly: SHAP explains *the model*, not nature. A large SHAP
value on a promiscuity-correlated feature tells you the model relies on it, not
that it is mechanistically real.

THE PROMISCUITY INTERROGATION
-----------------------------
Four nested comparisons, each stripping away a way to cheat:

1. **Compound base-rate baseline.** Predict each compound's training positive
   rate, ignoring the variant entirely. This is the "promiscuity-only" model.
   Any honest claim of pair-level modelling must beat it.

2. **Variant base-rate baseline.** Symmetrically, per-gene training rate
   ignoring the compound.

3. **Within-variant compound shuffle.** Permute compound identity among the
   pairs sharing a variant. Compound marginals are preserved exactly, so a
   promiscuity-only model is unaffected; genuine pair-specific signal is
   destroyed. The drop isolates pair specificity.

4. **Recovered pair-signal share**, compared on synthetic data against the
   generative ``pair_signal_share``. This is the step real data cannot offer,
   and it is why the synthetic cohort is worth having: it validates that the
   diagnostic itself works.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .features import FeatureMatrix
from .utils import check_resources, get_logger

__all__ = [
    "BlockImportance",
    "PromiscuityReport",
    "permutation_importance_blocks",
    "permutation_importance_features",
    "shap_values_for_tree_model",
    "top_shap_features",
    "interrogate_promiscuity",
]

_log = get_logger(__name__)


def _score(y: np.ndarray, p: np.ndarray, metric: str = "pr_auc") -> float:
    from sklearn.metrics import average_precision_score, roc_auc_score  # noqa: PLC0415

    y = np.asarray(y).ravel()
    if len(np.unique(y)) < 2:
        return float("nan")
    if metric == "pr_auc":
        return float(average_precision_score(y, p))
    if metric == "roc_auc":
        return float(roc_auc_score(y, p))
    raise ValueError(f"unknown metric {metric!r}")


# ---------------------------------------------------------------------------
# Permutation importance
# ---------------------------------------------------------------------------


@dataclass
class BlockImportance:
    """Held-out score drop when each feature block is permuted jointly."""

    metric: str
    baseline: float
    drops: dict[str, float]
    sds: dict[str, float] = field(default_factory=dict)
    n_repeats: int = 5

    def to_frame(self) -> pd.DataFrame:
        return (
            pd.DataFrame(
                {
                    "block": list(self.drops),
                    f"{self.metric}_drop": [self.drops[b] for b in self.drops],
                    f"{self.metric}_drop_sd": [self.sds.get(b, float("nan")) for b in self.drops],
                }
            )
            .sort_values(f"{self.metric}_drop", ascending=False)
            .reset_index(drop=True)
        )

    def summary(self) -> str:
        ordered = sorted(self.drops.items(), key=lambda kv: -kv[1])
        body = ", ".join(f"{b}: {d:+.4f}" for b, d in ordered)
        return f"baseline {self.metric}={self.baseline:.4f} | block drops: {body}"


def permutation_importance_blocks(
    model: object,
    fm: FeatureMatrix,
    *,
    rows: np.ndarray | None = None,
    metric: str = "pr_auc",
    n_repeats: int = 5,
    seed: int = 0,
) -> BlockImportance:
    """Permute each block jointly and measure the held-out score drop.

    Block-level rather than column-level because the blocks are internally
    correlated by construction (1024 fingerprint bits of 12 compounds are
    massively redundant), and column-level permutation would report that no
    single fingerprint bit matters -- true, and completely uninformative about
    whether chemistry matters.
    """
    X = fm.X if rows is None else fm.X[np.asarray(rows, dtype=int)]
    y = fm.y if rows is None else fm.y[np.asarray(rows, dtype=int)]
    if y is None:
        raise ValueError("permutation importance needs labels")

    check_resources(X.shape[0], X.shape[1], dtype=X.dtype, label="permutation importance")
    baseline = _score(y, model.predict_proba(X), metric)  # type: ignore[attr-defined]
    rng = np.random.default_rng(seed)
    blocks = sorted(set(fm.block_of.values()))

    drops, sds = {}, {}
    for block in blocks:
        cols = fm.block_columns(block)
        if not cols:
            continue
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            # One shared permutation across the block's columns preserves
            # within-block structure while destroying the block's alignment
            # with the label -- i.e. it asks "what if this block described a
            # different, randomly chosen pair?".
            perm = rng.permutation(len(Xp))
            Xp[:, cols] = Xp[perm][:, cols]
            scores.append(_score(y, model.predict_proba(Xp), metric))  # type: ignore[attr-defined]
        drops[block] = float(baseline - np.mean(scores))
        sds[block] = float(np.std(scores))

    imp = BlockImportance(
        metric=metric, baseline=baseline, drops=drops, sds=sds, n_repeats=n_repeats
    )
    _log.info("block permutation importance: %s", imp.summary())
    return imp


def permutation_importance_features(
    model: object,
    fm: FeatureMatrix,
    *,
    rows: np.ndarray | None = None,
    metric: str = "pr_auc",
    n_repeats: int = 3,
    top_k: int = 30,
    seed: int = 0,
    candidate_columns: list[int] | None = None,
) -> pd.DataFrame:
    """Column-level permutation importance for the most promising columns.

    Permuting all ~1068 columns x n_repeats is wasteful, so by default this
    screens the non-fingerprint columns (the interpretable ones) plus the
    fingerprint bits with nonzero variance. Pass ``candidate_columns`` to
    override.
    """
    X = fm.X if rows is None else fm.X[np.asarray(rows, dtype=int)]
    y = fm.y if rows is None else fm.y[np.asarray(rows, dtype=int)]
    if y is None:
        raise ValueError("permutation importance needs labels")

    if candidate_columns is None:
        interpretable = [
            i for i, n in enumerate(fm.feature_names) if not n.startswith("ecfp_")
        ]
        variable_bits = [
            i
            for i, n in enumerate(fm.feature_names)
            if n.startswith("ecfp_") and float(np.std(X[:, i])) > 0
        ]
        candidate_columns = interpretable + variable_bits
    baseline = _score(y, model.predict_proba(X), metric)  # type: ignore[attr-defined]
    rng = np.random.default_rng(seed)

    recs = []
    for col in candidate_columns:
        scores = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[:, col] = rng.permutation(Xp[:, col])
            scores.append(_score(y, model.predict_proba(Xp), metric))  # type: ignore[attr-defined]
        recs.append(
            {
                "feature": fm.feature_names[col],
                "block": fm.block_of[fm.feature_names[col]],
                f"{metric}_drop": float(baseline - np.mean(scores)),
                f"{metric}_drop_sd": float(np.std(scores)),
            }
        )
    return (
        pd.DataFrame(recs)
        .sort_values(f"{metric}_drop", ascending=False)
        .head(top_k)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------------------
# SHAP
# ---------------------------------------------------------------------------


def shap_values_for_tree_model(
    model: object, fm: FeatureMatrix, *, rows: np.ndarray | None = None, max_rows: int = 2000
) -> np.ndarray:
    """Exact TreeSHAP values for a LightGBM-backed tier.

    Returns ``(n_rows, n_features)``. Subsamples above ``max_rows`` because
    TreeSHAP is polynomial but not free, and attribution summaries are stable
    well before the full set.
    """
    import shap  # noqa: PLC0415

    booster = getattr(model, "model_", None)
    if booster is None:
        raise TypeError(
            "shap_values_for_tree_model expects a fitted tree tier (Tier1GBM / "
            "Tier3StructureAware). Tier 2 is a neural model: use a sampling "
            "explainer (shap.GradientExplainer) instead, and note that its values "
            "are approximate."
        )
    X = fm.X if rows is None else fm.X[np.asarray(rows, dtype=int)]
    if len(X) > max_rows:
        sub = np.random.default_rng(0).choice(len(X), size=max_rows, replace=False)
        X = X[sub]
    check_resources(X.shape[0], X.shape[1], dtype=np.float64, label="TreeSHAP values")

    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(X, check_additivity=False)
    # Across shap/lightgbm versions this comes back as a list per class, or a
    # 3-d array, or a plain 2-d array for binary. Normalise to 2-d positive
    # class rather than trusting any one shape.
    if isinstance(values, list):
        values = values[-1]
    values = np.asarray(values)
    if values.ndim == 3:
        values = values[..., -1]
    return values


def top_shap_features(
    shap_values: np.ndarray, feature_names: list[str], *, k: int = 3
) -> list[str]:
    """Per-row top-k features by absolute SHAP, as ``name:+0.123`` strings.

    This is what lands in the CLI's ``top_shap_features`` column. Signed, so a
    reader can tell what pushed the prediction up from what pushed it down --
    an unsigned list of "important" features is much less useful at the bench.
    """
    out = []
    for row in np.asarray(shap_values):
        order = np.argsort(-np.abs(row))[:k]
        out.append("; ".join(f"{feature_names[i]}:{row[i]:+.4f}" for i in order))
    return out


# ---------------------------------------------------------------------------
# The promiscuity interrogation
# ---------------------------------------------------------------------------


@dataclass
class PromiscuityReport:
    """Did the model learn pair-specific signal, or compound base rates?

    Attributes
    ----------
    model_score / compound_baseline_score / variant_baseline_score:
        Held-out scores for the model and the two marginal baselines.
    shuffled_score:
        Model score when compound identity is permuted *within* each variant.
    pair_specificity:
        ``(model - shuffled) / (model - prevalence)`` -- the share of the
        model's lift over chance that survives only if it is using the actual
        pairing. 1.0 means all of the lift is pair-specific; 0.0 means none is
        and the model is a promiscuity lookup table.
    beats_compound_baseline:
        The minimum bar for calling this a pair model at all.
    ground_truth_pair_share:
        Generative value on synthetic cohorts, for validating the diagnostic.
    """

    metric: str
    model_score: float
    compound_baseline_score: float
    variant_baseline_score: float
    shuffled_score: float
    prevalence: float
    shap_block_share: dict[str, float] = field(default_factory=dict)
    ground_truth_pair_share: float | None = None
    n_shuffles: int = 5
    shuffled_sd: float = float("nan")

    @property
    def lift_over_chance(self) -> float:
        return self.model_score - self.prevalence

    @property
    def pair_specificity(self) -> float:
        if self.lift_over_chance <= 0:
            return float("nan")
        return float((self.model_score - self.shuffled_score) / self.lift_over_chance)

    @property
    def beats_compound_baseline(self) -> bool:
        return self.model_score > self.compound_baseline_score

    def verdict(self) -> str:
        """A sentence a reader can act on, not a number to interpret."""
        if not self.beats_compound_baseline:
            return (
                "FAILS the promiscuity test: the model does not beat a baseline that "
                "ignores the variant entirely and predicts each compound's training "
                "base rate. On this evidence it is a promiscuity lookup table, and "
                "its pair-level predictions should not be used to prioritise assays."
            )
        ps = self.pair_specificity
        if np.isnan(ps):
            return "INCONCLUSIVE: the model has no lift over chance to decompose."
        if ps >= 0.5:
            return (
                f"PASSES: {ps:.0%} of the model's lift over chance is destroyed by "
                f"permuting compound identity within each variant, so most of what it "
                f"knows is genuinely pair-specific rather than per-compound."
            )
        if ps >= 0.2:
            return (
                f"MIXED: only {ps:.0%} of the lift is pair-specific; the majority comes "
                f"from per-compound effects. Useful for ranking compounds, weak for "
                f"ranking variants within a compound."
            )
        return (
            f"MOSTLY PROMISCUITY: just {ps:.0%} of the lift is pair-specific. The model "
            f"is close to a per-compound base-rate predictor dressed up as a pair model."
        )

    def to_frame(self) -> pd.DataFrame:
        row = {
            "metric": self.metric,
            "model": self.model_score,
            "compound_baseline": self.compound_baseline_score,
            "variant_baseline": self.variant_baseline_score,
            "within_variant_shuffled": self.shuffled_score,
            "within_variant_shuffled_sd": self.shuffled_sd,
            "prevalence": self.prevalence,
            "pair_specificity": self.pair_specificity,
            "beats_compound_baseline": self.beats_compound_baseline,
        }
        if self.ground_truth_pair_share is not None:
            row["ground_truth_pair_share"] = self.ground_truth_pair_share
        row.update({f"shap_share_{k}": v for k, v in self.shap_block_share.items()})
        return pd.DataFrame([row])


def interrogate_promiscuity(
    model: object,
    fm_test: FeatureMatrix,
    *,
    train_pairs: pd.DataFrame,
    test_pairs: pd.DataFrame,
    cohort: object | None = None,
    metric: str = "pr_auc",
    n_shuffles: int = 5,
    seed: int = 0,
    shap_block_share: dict[str, float] | None = None,
) -> PromiscuityReport:
    """Run the four-way interrogation described in the module docstring.

    ``fm_test`` must have been built from ``test_pairs`` in the same row order;
    ``train_pairs`` supplies the base rates for the marginal baselines, and
    must come from the training split only.
    """
    y = fm_test.y
    if y is None:
        raise ValueError("promiscuity interrogation needs test labels")
    model_score = _score(y, model.predict_proba(fm_test.X), metric)  # type: ignore[attr-defined]
    prevalence = float(np.mean(y))

    # 1 & 2: marginal baselines, fitted on train, applied to test.
    global_rate = float(train_pairs["label"].mean())
    cmp_rate = train_pairs.groupby("compound_id", observed=True)["label"].mean()
    p_cmp = test_pairs["compound_id"].map(cmp_rate).fillna(global_rate).to_numpy(float)
    compound_baseline = _score(y, p_cmp, metric)

    if cohort is not None and hasattr(cohort, "variants"):
        genes = cohort.variants["gene"]  # type: ignore[index]
        train_gene = train_pairs["variant_id"].map(genes)
        gene_rate = train_pairs.assign(_g=train_gene).groupby("_g", observed=True)["label"].mean()
        p_var = (
            test_pairs["variant_id"].map(genes).map(gene_rate).fillna(global_rate).to_numpy(float)
        )
    else:
        p_var = np.full(len(y), global_rate)
    variant_baseline = _score(y, p_var, metric)

    # 3: within-variant compound shuffle. Compound marginals are preserved
    # exactly; only the pairing is broken.
    rng = np.random.default_rng(seed)
    ligand_cols = fm_test.block_columns("ligand") + fm_test.block_columns("interaction")
    shuffled_scores = []
    if ligand_cols:
        groups = test_pairs.reset_index(drop=True).groupby("variant_id", observed=True).indices
        for _ in range(n_shuffles):
            Xs = fm_test.X.copy()
            for _vid, idx in groups.items():
                if len(idx) < 2:
                    continue
                perm = rng.permutation(idx)
                Xs[np.asarray(idx)[:, None], ligand_cols] = Xs[perm[:, None], ligand_cols]
            shuffled_scores.append(_score(y, model.predict_proba(Xs), metric))  # type: ignore[attr-defined]
    else:
        # No ligand representation at all: the model cannot be pair-specific,
        # and the shuffle is a no-op by definition.
        shuffled_scores = [model_score]

    gt = None
    if cohort is not None:
        weights = getattr(cohort, "ground_truth_weights", None)
        if weights:
            gt = float(weights.get("pair_signal_share", float("nan")))

    report = PromiscuityReport(
        metric=metric,
        model_score=model_score,
        compound_baseline_score=compound_baseline,
        variant_baseline_score=variant_baseline,
        shuffled_score=float(np.mean(shuffled_scores)),
        shuffled_sd=float(np.std(shuffled_scores)),
        prevalence=prevalence,
        shap_block_share=shap_block_share or {},
        ground_truth_pair_share=gt,
        n_shuffles=n_shuffles,
    )
    _log.info("promiscuity interrogation: %s", report.verdict())
    return report


def shap_block_shares(shap_values: np.ndarray, fm: FeatureMatrix) -> dict[str, float]:
    """Fraction of total absolute SHAP mass carried by each feature block.

    Mass share, not a count: 1024 fingerprint columns will always *outnumber*
    17 stability columns, so any count-based summary says only that the
    fingerprint is wide.
    """
    total = float(np.abs(shap_values).sum())
    if total <= 0:
        return {}
    out = {}
    for block in sorted(set(fm.block_of.values())):
        cols = fm.block_columns(block)
        if cols:
            out[block] = float(np.abs(shap_values[:, cols]).sum() / total)
    return out
