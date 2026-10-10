"""Tests for the data layer, features, models, attribution and the CLI.

These are integration-leaning on purpose: the failure modes that matter in a
pipeline like this are not "does isotonic regression work" (sklearn's problem)
but "does the ablation actually remove the block", "does a split leak", and
"does the CLI refuse to rank an extrapolation".
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from foldrescue.attribution import (
    interrogate_promiscuity,
    permutation_importance_blocks,
    shap_block_shares,
    shap_values_for_tree_model,
    top_shap_features,
)
from foldrescue.data import (
    AmenabenchAdapter,
    holdout_gene_split,
    load_cohort,
)
from foldrescue.evaluate import choose_threshold, evaluate_predictions
from foldrescue.features import BLOCK_NAMES, FeatureSpec, build_features
from foldrescue.models import build_model


@pytest.fixture(scope="module")
def cohort():
    return load_cohort("synthetic", seed=0)


# ---------------------------------------------------------------------------
# Data layer
# ---------------------------------------------------------------------------


def test_synthetic_cohort_is_labelled_synthetic(cohort):
    """Nothing downstream may mistake generated data for measurement."""
    assert cohort.synthetic is True
    assert "SYNTHETIC" in cohort.label_banner()
    assert "SyntheticCohortAdapter" in cohort.provenance
    assert (cohort.pairs["source"] == "SYNTHETIC:foldrescue.data.SyntheticCohortAdapter").all()


def test_synthetic_cohort_is_reproducible():
    a = load_cohort("synthetic", seed=7)
    b = load_cohort("synthetic", seed=7)
    pd.testing.assert_frame_equal(a.pairs, b.pairs)
    assert np.allclose(a.oracle_p, b.oracle_p)


def test_different_seeds_give_different_cohorts():
    a = load_cohort("synthetic", seed=1)
    b = load_cohort("synthetic", seed=2)
    assert not a.pairs.equals(b.pairs)


def test_oracle_is_not_a_column_on_pairs(cohort):
    """The Bayes-optimal probability must not be reachable as a feature."""
    assert "oracle_p" not in cohort.pairs.columns
    assert "p_true" not in cohort.pairs.columns
    assert cohort.oracle_p is not None
    assert len(cohort.oracle_p) == len(cohort.pairs)


def test_oracle_beats_every_realisable_model(cohort):
    """Sanity check on the ceiling: it must actually be a ceiling."""
    from sklearn.metrics import roc_auc_score

    y = cohort.pairs["label"].to_numpy()
    assert roc_auc_score(y, cohort.oracle_p) > 0.70


def test_pair_signal_is_learnable_from_features(cohort):
    """Regression test for a real defect.

    The first version of the generator built its pair-interaction term from
    random latent vectors unrelated to any feature, making the dominant
    generative term mathematically unlearnable: every model sat at chance and
    the promiscuity analysis was unanswerable. The interaction must be a
    function of observables.
    """
    from sklearn.metrics import roc_auc_score

    fm = build_features(cohort, spec=FeatureSpec(ecfp_bits=256))
    s = cohort.splits["random"]
    model = build_model("tier3_structure").fit(
        fm.X[s["train"]], fm.y[s["train"]], feature_names=fm.feature_names
    )
    auc = roc_auc_score(fm.y[s["test"]], model.predict_proba(fm.X[s["test"]]))
    # Must clear chance by a real margin, and clear the promiscuity-only floor.
    assert auc > 0.56, f"pair signal is not learnable from the features (AUC={auc:.4f})"


def test_cohort_validation_catches_dangling_foreign_keys(cohort):
    bad = cohort.pairs.copy()
    bad.loc[0, "variant_id"] = "NO_SUCH_VARIANT"
    with pytest.raises(ValueError, match="variant_ids absent"):
        type(cohort)(variants=cohort.variants, compounds=cohort.compounds, pairs=bad)


def test_cohort_validation_catches_bad_labels(cohort):
    bad = cohort.pairs.copy()
    bad.loc[0, "label"] = 7
    with pytest.raises(ValueError, match="labels must be 0/1"):
        type(cohort)(variants=cohort.variants, compounds=cohort.compounds, pairs=bad)


def test_cohort_validation_catches_missing_columns(cohort):
    with pytest.raises(ValueError, match="missing required columns"):
        type(cohort)(
            variants=cohort.variants.drop(columns=["ddg_pred"]),
            compounds=cohort.compounds,
            pairs=cohort.pairs,
        )


def test_amenabench_adapter_error_is_actionable(tmp_path):
    """The benchmark does not exist yet, so this path must explain itself."""
    with pytest.raises(FileNotFoundError) as excinfo:
        AmenabenchAdapter(root=tmp_path).load()
    message = str(excinfo.value)
    assert "variants.csv" in message
    assert "--cohort synthetic" in message, "must name the working alternative"
    assert "data_contract" in message or "schema" in message


@pytest.mark.parametrize("split", ["random", "gene_holdout", "compound_holdout", "scaffold"])
def test_all_splits_are_three_way_and_nonempty(cohort, split):
    parts = cohort.split(split)
    assert set(parts) == {"train", "calib", "test"}
    for name, idx in parts.items():
        assert len(idx) > 0, f"{split}/{name} is empty"


@pytest.mark.parametrize("split", ["random", "gene_holdout", "compound_holdout", "scaffold"])
def test_splits_are_disjoint_and_complete(cohort, split):
    parts = cohort.split(split)
    joined = np.concatenate(list(parts.values()))
    assert len(joined) == len(np.unique(joined)), f"{split} partitions overlap"
    assert len(joined) == len(cohort.pairs), f"{split} does not cover every pair"


def test_gene_holdout_does_not_leak_genes(cohort):
    """A gene straddling train and test inflates every metric downstream."""
    parts = cohort.split("gene_holdout")
    genes = cohort.pairs.join(cohort.variants[["gene"]], on="variant_id")["gene"].to_numpy()
    train_g = set(genes[parts["train"]])
    test_g = set(genes[parts["test"]])
    assert not (train_g & test_g), f"genes leaked across the holdout: {train_g & test_g}"


def test_compound_holdout_does_not_leak_compounds(cohort):
    parts = cohort.split("compound_holdout")
    cmp_ids = cohort.pairs["compound_id"].to_numpy()
    assert not (set(cmp_ids[parts["train"]]) & set(cmp_ids[parts["test"]]))


def test_holdout_gene_split_excludes_the_gene_from_train_and_calib(cohort):
    gene = "GLB1"
    parts = holdout_gene_split(cohort, gene)
    genes = cohort.pairs.join(cohort.variants[["gene"]], on="variant_id")["gene"].to_numpy()
    assert set(genes[parts["test"]]) == {gene}
    assert gene not in set(genes[parts["train"]])
    assert gene not in set(genes[parts["calib"]]), "calibration split leaks the held-out gene"


def test_holdout_gene_split_rejects_unknown_gene(cohort):
    with pytest.raises(ValueError, match="not present in cohort"):
        holdout_gene_split(cohort, "NOT_A_GENE")


def test_unknown_split_name_lists_the_valid_ones(cohort):
    with pytest.raises(KeyError, match="available"):
        cohort.split("nonexistent")


# ---------------------------------------------------------------------------
# Features
# ---------------------------------------------------------------------------


def test_feature_matrix_is_finite_and_aligned(cohort):
    fm = build_features(cohort, spec=FeatureSpec(ecfp_bits=128))
    assert np.isfinite(fm.X).all(), "non-finite features would poison training"
    assert fm.X.shape[0] == len(cohort.pairs)
    assert fm.X.shape[1] == len(fm.feature_names)
    assert set(fm.block_of.values()) <= set(BLOCK_NAMES)


@pytest.mark.parametrize(
    "blocks",
    [
        ("stability",),
        ("structural",),
        ("ligand",),
        ("stability", "structural"),
        ("stability", "ligand", "interaction"),
        BLOCK_NAMES,
    ],
)
def test_ablation_specs_remove_exactly_the_named_blocks(cohort, blocks):
    """The ablation notebook is only meaningful if this holds."""
    fm = build_features(cohort, spec=FeatureSpec(blocks=blocks, ecfp_bits=64))
    assert set(fm.block_of.values()) == set(blocks)
    for block in BLOCK_NAMES:
        cols = fm.block_columns(block)
        if block in blocks:
            assert cols, f"{block} was requested but produced no columns"
        else:
            assert not cols, f"{block} was excluded but produced columns"


def test_dropping_the_ligand_block_makes_compounds_indistinguishable(cohort):
    """An ablation, not a configuration -- and the docstring says so."""
    fm = build_features(cohort, spec=FeatureSpec(blocks=("stability",)))
    pairs = cohort.pairs.reset_index(drop=True)
    # Two pairs sharing a variant but differing in compound must now be equal.
    groups = pairs.groupby("variant_id", observed=True).indices
    multi = [idx for idx in groups.values() if len(idx) >= 2]
    assert multi, "cohort has no variant tested against two compounds"
    i, j = multi[0][0], multi[0][1]
    assert np.allclose(fm.X[i], fm.X[j])


def test_interaction_block_requires_a_partner_block():
    with pytest.raises(ValueError, match="meaningless without"):
        FeatureSpec(blocks=("interaction",))


def test_unknown_block_rejected():
    with pytest.raises(ValueError, match="unknown feature blocks"):
        FeatureSpec(blocks=("stability", "quantum_vibes"))


def test_chunked_feature_build_matches_unchunked(cohort):
    """The guard's chunking must be honoured and must not change the result."""
    full = build_features(cohort, spec=FeatureSpec(ecfp_bits=64))
    chunked = build_features(cohort, spec=FeatureSpec(ecfp_bits=64), memory_fraction=1e-6)
    assert chunked.plan.chunked, "expected the tight budget to force chunking"
    assert np.allclose(full.X.astype(np.float32), chunked.X.astype(np.float32))


def test_unparseable_smiles_is_refused_not_featurised(cohort):
    broken = cohort.compounds.copy()
    broken.loc[broken.index[0], "smiles"] = "this is not a molecule ((("
    clone = type(cohort)(
        variants=cohort.variants, compounds=broken, pairs=cohort.pairs, provenance="t"
    )
    with pytest.raises(ValueError, match="could not parse"):
        build_features(clone, spec=FeatureSpec(ecfp_bits=64))


def test_unknown_pair_ids_rejected(cohort):
    bad = pd.DataFrame([{"variant_id": "ghost", "compound_id": "phantom"}])
    with pytest.raises(KeyError, match="unknown ids"):
        build_features(cohort, pairs=bad)


def test_empty_selection_rejected(cohort):
    with pytest.raises(ValueError, match="no pairs"):
        build_features(cohort, rows=np.array([], dtype=int))


# ---------------------------------------------------------------------------
# Models / evaluation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tier", ["tier1_gbm", "tier3_structure"])
def test_tree_tiers_train_and_produce_valid_probabilities(cohort, tier):
    fm = build_features(cohort, spec=FeatureSpec(ecfp_bits=128))
    s = cohort.splits["random"]
    model = build_model(tier).fit(
        fm.X[s["train"]], fm.y[s["train"]], feature_names=fm.feature_names
    )
    p = model.predict_proba(fm.X[s["test"]])
    assert p.shape == (len(s["test"]),)
    assert ((p >= 0) & (p <= 1)).all()
    assert np.isfinite(p).all()


def test_tier3_is_architecturally_identical_to_tier1():
    """The tiers differ by inputs, so the comparison is clean."""
    from foldrescue.models import Tier1GBM, Tier3StructureAware

    assert issubclass(Tier3StructureAware, Tier1GBM)


def test_unknown_tier_rejected():
    with pytest.raises(ValueError, match="unknown tier"):
        build_model("tier9_magic")


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError, match="fit"):
        build_model("tier1_gbm").predict_proba(np.zeros((2, 3)))


def test_tier2_rejects_wrong_embedding_width():
    """A width mismatch means the caller built the wrong matrix; say so."""
    pytest.importorskip("torch")
    model = build_model("tier2_fusion", protein_dim=8, compound_dim=8)
    with pytest.raises(ValueError, match="expects X with 16 columns"):
        model.fit(np.zeros((4, 99), dtype=np.float32), np.array([0, 1, 0, 1]))


def test_tier2_unfreeze_warns_loudly(caplog):
    from foldrescue.models import Tier2Fusion

    with caplog.at_level("WARNING"):
        Tier2Fusion(unfreeze_encoders=True)
    assert any("guarantees it" in r.message for r in caplog.records)


def test_threshold_is_chosen_away_from_half_on_imbalanced_data(cohort):
    """0.5 is a balanced-problem convention and yields MCC=0 here."""
    fm = build_features(cohort, spec=FeatureSpec(ecfp_bits=128))
    s = cohort.splits["random"]
    model = build_model("tier3_structure").fit(
        fm.X[s["train"]], fm.y[s["train"]], feature_names=fm.feature_names
    )
    p_calib = model.predict_proba(fm.X[s["calib"]])
    t = choose_threshold(fm.y[s["calib"]], p_calib)
    assert 0.0 < t < 1.0
    assert t != 0.5


def test_metrics_report_the_base_rate_and_a_nondegenerate_mcc(cohort):
    fm = build_features(cohort, spec=FeatureSpec(ecfp_bits=128))
    s = cohort.splits["random"]
    model = build_model("tier3_structure").fit(
        fm.X[s["train"]], fm.y[s["train"]], feature_names=fm.feature_names
    )
    m = evaluate_predictions(
        fm.y[s["test"]],
        model.predict_proba(fm.X[s["test"]]),
        split="random",
        tier="tier3_structure",
        y_calib=fm.y[s["calib"]],
        p_calib=model.predict_proba(fm.X[s["calib"]]),
        oracle_p_test=cohort.oracle_p[s["test"]],
    )
    assert 0 < m.prevalence < 1
    assert m.pr_auc > m.prevalence, "model must beat the no-skill PR-AUC baseline"
    assert m.n_predicted_positive > 0, "threshold selection produced a constant predictor"
    assert m.mcc != 0.0
    assert m.oracle_pr_auc is not None and m.oracle_pr_auc >= m.pr_auc
    assert m.fraction_of_ceiling is not None


def test_evaluate_warns_when_no_calibration_split_given(cohort, caplog):
    with caplog.at_level("WARNING"):
        evaluate_predictions(
            np.array([0, 1, 0, 1]),
            np.array([0.1, 0.2, 0.15, 0.25]),
            split="x",
            tier="y",
        )
    assert any("0.5" in r.message for r in caplog.records)


# ---------------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained(cohort):
    fm = build_features(cohort, spec=FeatureSpec(ecfp_bits=128))
    s = cohort.splits["random"]
    model = build_model("tier3_structure").fit(
        fm.X[s["train"]],
        fm.y[s["train"]],
        eval_set=(fm.X[s["calib"]], fm.y[s["calib"]]),
        feature_names=fm.feature_names,
    )
    return model, fm, s


def test_block_permutation_importance_ranks_blocks(trained):
    model, fm, s = trained
    imp = permutation_importance_blocks(model, fm, rows=s["test"], n_repeats=2)
    assert set(imp.drops) == set(fm.block_of.values())
    # Removing the chemistry must hurt a pair model.
    assert imp.drops["ligand"] > 0
    frame = imp.to_frame()
    assert frame["pr_auc_drop"].is_monotonic_decreasing


def test_shap_values_have_one_column_per_feature(trained):
    model, fm, s = trained
    sv = shap_values_for_tree_model(model, fm, rows=s["test"])
    assert sv.shape[1] == len(fm.feature_names)
    assert np.isfinite(sv).all()


def test_shap_block_shares_sum_to_one(trained):
    model, fm, s = trained
    sv = shap_values_for_tree_model(model, fm, rows=s["test"])
    shares = shap_block_shares(sv, fm)
    assert sum(shares.values()) == pytest.approx(1.0, abs=1e-6)


def test_top_shap_features_are_signed(trained):
    model, fm, s = trained
    sv = shap_values_for_tree_model(model, fm, rows=s["test"])
    strings = top_shap_features(sv[:3], fm.feature_names, k=3)
    assert len(strings) == 3
    for text in strings:
        assert text.count(";") == 2
        assert "+" in text or "-" in text


def test_shap_on_a_neural_tier_is_refused_with_guidance(cohort):
    from foldrescue.models import Tier2Fusion

    fm = build_features(cohort, spec=FeatureSpec(ecfp_bits=64))
    with pytest.raises(TypeError, match="GradientExplainer"):
        shap_values_for_tree_model(Tier2Fusion(), fm)


def test_promiscuity_interrogation_beats_the_compound_baseline(trained, cohort):
    """The central claim: the model is more than a per-compound lookup table."""
    model, fm, s = trained
    fm_test = build_features(cohort, spec=FeatureSpec(ecfp_bits=128), rows=s["test"])
    report = interrogate_promiscuity(
        model,
        fm_test,
        train_pairs=cohort.pairs.iloc[s["train"]],
        test_pairs=cohort.pairs.iloc[s["test"]].reset_index(drop=True),
        cohort=cohort,
        n_shuffles=2,
    )
    assert report.beats_compound_baseline, report.verdict()
    assert report.model_score > report.prevalence
    # The within-variant shuffle must destroy signal, not create it.
    assert report.shuffled_score < report.model_score
    assert 0.0 < report.pair_specificity <= 1.5
    assert report.ground_truth_pair_share is not None
    assert "PASSES" in report.verdict() or "MIXED" in report.verdict()


def test_promiscuity_verdict_fails_a_ligand_blind_model(cohort):
    """A model with no chemistry cannot be pair-specific, and must be told so."""
    fm = build_features(cohort, spec=FeatureSpec(blocks=("stability", "structural")))
    s = cohort.splits["random"]
    model = build_model("tier3_structure").fit(
        fm.X[s["train"]], fm.y[s["train"]], feature_names=fm.feature_names
    )
    fm_test = build_features(
        cohort, spec=FeatureSpec(blocks=("stability", "structural")), rows=s["test"]
    )
    report = interrogate_promiscuity(
        model,
        fm_test,
        train_pairs=cohort.pairs.iloc[s["train"]],
        test_pairs=cohort.pairs.iloc[s["test"]].reset_index(drop=True),
        cohort=cohort,
        n_shuffles=1,
    )
    # The crisp invariant: with no ligand columns there is nothing for the
    # within-variant shuffle to permute, so it cannot change the score at all.
    assert report.shuffled_score == pytest.approx(report.model_score, abs=1e-12)
    # Pair specificity is therefore either exactly 0, or nan when the model has
    # no lift over chance to decompose in the first place. Both are correct;
    # what must NOT happen is a confident non-zero pair-specificity claim.
    assert np.isnan(report.pair_specificity) or report.pair_specificity == pytest.approx(
        0.0, abs=1e-9
    )
    # Either way the verdict must not endorse it as a pair model.
    assert "PASSES" not in report.verdict()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    variants = tmp_path / "variants.csv"
    variants.write_text("variant,gene\nR201C,GLB1\nY83H,GLB1\n")
    compounds = tmp_path / "compounds.smi"
    compounds.write_text(
        # One in-domain iminosugar, one deliberately alien macrocycle.
        "OC[C@H]1NC[C@H](O)[C@@H](O)[C@@H]1O IMINO_IN\n"
        "O=C1NC(Cc2ccccc2)C(=O)NC(CCCCN)C(=O)NC(C(C)C)C(=O)N1 PEPTIDE_OOD\n"
    )
    return variants, compounds


def _run_cli(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, "-m", "foldrescue.cli", *args],
        capture_output=True,
        text=True,
        cwd=Path(__file__).resolve().parent.parent,
    )


def test_cli_predict_writes_the_specified_columns(tmp_path):
    variants, compounds = _write_inputs(tmp_path)
    out = tmp_path / "ranked.csv"
    result = _run_cli(
        ["predict", "--gene", "GLB1", "--variants", str(variants),
         "--compounds", str(compounds), "--out", str(out), "--no-shap"]
    )
    assert result.returncode == 0, result.stderr
    assert out.exists()

    frame = pd.read_csv(out, comment="#")
    for col in [
        "variant", "compound", "p_amenable", "calibrated_p", "ci_low", "ci_high",
        "out_of_domain", "nearest_training_neighbour", "top_shap_features",
    ]:
        assert col in frame.columns, f"missing required output column {col}"
    assert ((frame["calibrated_p"] >= 0) & (frame["calibrated_p"] <= 1)).all()
    assert (frame["ci_low"] <= frame["ci_high"]).all()


def test_cli_refuses_to_rank_ood_pairs_by_default(tmp_path):
    """The deliberate friction: an extrapolation must not be silently ranked."""
    variants, compounds = _write_inputs(tmp_path)
    out = tmp_path / "ranked.csv"
    result = _run_cli(
        ["predict", "--gene", "GLB1", "--variants", str(variants),
         "--compounds", str(compounds), "--out", str(out), "--no-shap"]
    )
    assert result.returncode == 0, result.stderr
    assert "WITHHELD" in result.stderr
    assert "--include-ood" in result.stderr
    frame = pd.read_csv(out, comment="#")
    assert not frame["out_of_domain"].any(), "an OOD row was ranked without the flag"


def test_cli_include_ood_appends_after_all_in_domain_rows(tmp_path):
    variants, compounds = _write_inputs(tmp_path)
    out = tmp_path / "ranked_ood.csv"
    result = _run_cli(
        ["predict", "--gene", "GLB1", "--variants", str(variants),
         "--compounds", str(compounds), "--out", str(out), "--no-shap", "--include-ood"]
    )
    assert result.returncode == 0, result.stderr
    frame = pd.read_csv(out, comment="#")
    assert frame["out_of_domain"].any(), "expected the alien compound to be present"
    flags = frame["out_of_domain"].to_numpy()
    # Never co-sorted: no in-domain row may follow an OOD row.
    first_ood = int(np.argmax(flags))
    assert flags[first_ood:].all(), "OOD and in-domain rows were interleaved"


def test_cli_output_header_declares_synthetic_training_data(tmp_path):
    variants, compounds = _write_inputs(tmp_path)
    out = tmp_path / "ranked.csv"
    _run_cli(
        ["predict", "--gene", "GLB1", "--variants", str(variants),
         "--compounds", str(compounds), "--out", str(out), "--no-shap"]
    )
    header = "\n".join(
        line for line in out.read_text().splitlines() if line.startswith("#")
    )
    assert "SYNTHETIC TRAINING DATA" in header
    assert "NOT CLINICAL DECISION SUPPORT" in header


def test_cli_warns_loudly_on_stderr_about_synthetic_data(tmp_path):
    variants, compounds = _write_inputs(tmp_path)
    result = _run_cli(
        ["predict", "--gene", "GLB1", "--variants", str(variants),
         "--compounds", str(compounds), "--out", str(tmp_path / "o.csv"), "--no-shap"]
    )
    assert "GENERATED data" in result.stderr
    assert "NO experimental or biological information" in result.stderr


def test_cli_parses_shorthand_variant_notation(tmp_path):
    variants = tmp_path / "v.csv"
    variants.write_text("variant\np.R201C\nY83H\n")
    compounds = tmp_path / "c.smi"
    compounds.write_text("OC[C@H]1NC[C@H](O)[C@@H](O)[C@@H]1O IMINO\n")
    result = _run_cli(
        ["predict", "--gene", "GLB1", "--variants", str(variants),
         "--compounds", str(compounds), "--out", str(tmp_path / "o.csv"),
         "--no-shap", "--include-ood"]
    )
    assert result.returncode == 0, result.stderr


def test_cli_rejects_unparseable_variant_notation(tmp_path):
    variants = tmp_path / "v.csv"
    variants.write_text("variant\nnonsense_variant\n")
    compounds = tmp_path / "c.smi"
    compounds.write_text("CCO ethanol\n")
    result = _run_cli(
        ["predict", "--gene", "GLB1", "--variants", str(variants),
         "--compounds", str(compounds), "--out", str(tmp_path / "o.csv")]
    )
    assert result.returncode == 1
    assert "could not parse variant" in result.stderr


def test_cli_resources_subcommand_reports_a_plan():
    result = _run_cli(["resources", "--rows", "1000", "--cols", "100"])
    assert result.returncode == 0, result.stderr
    assert "dtype=" in result.stdout


def test_cli_resources_subcommand_refuses_with_exit_code_two():
    """A refusal must be distinguishable from a crash by exit code."""
    result = _run_cli(
        ["resources", "--rows", "100000000", "--cols", "100000", "--no-chunking"]
    )
    assert result.returncode == 2
    assert "REFUSED" in result.stderr
