"""Command-line interface.

    foldrescue predict --gene GLB1 --variants variants.csv \
                       --compounds compounds.smi --out ranked.csv

Output columns, exactly as specified:

    variant, compound, p_amenable, calibrated_p, ci_low, ci_high,
    out_of_domain, nearest_training_neighbour, top_shap_features

THE RANKING REFUSAL
===================

By default the CLI **will not** write out-of-domain predictions interleaved
with in-domain ones. It writes the in-domain rows, reports how many were
withheld and why, and tells the user about ``--include-ood``.

This is the one piece of deliberate friction in the package. A ranked CSV is
read by someone deciding what to put in an assay, usually by looking at the
top of the file. Sorting an extrapolation into that list by probability makes
it indistinguishable from a supported prediction -- the number looks the same.
Requiring a flag means including them is a choice someone made, which is
recoverable in a methods section; a silent merge is not.

With ``--include-ood``, OOD rows are appended *after* all in-domain rows,
still flagged, and never sorted among them.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .utils import ResourceRefusal, get_logger

_log = get_logger("foldrescue.cli")

OUTPUT_COLUMNS = [
    "variant",
    "compound",
    "p_amenable",
    "calibrated_p",
    "ci_low",
    "ci_high",
    "out_of_domain",
    "nearest_training_neighbour",
    "top_shap_features",
]


# ---------------------------------------------------------------------------
# Input parsing
# ---------------------------------------------------------------------------


def read_variants(path: Path, gene: str | None) -> pd.DataFrame:
    """Read a variants CSV, accepting either full schema or a minimal one.

    The minimal form is what a bench scientist actually has: a gene and a
    substitution like ``R201C``. Missing structural columns are filled with
    NaN and the affected feature blocks are dropped from the model with an
    explicit message -- rather than imputing zeros, which would silently claim
    "this variant is 0 angstrom from the pocket".
    """
    frame = pd.read_csv(path)
    cols = {c.lower(): c for c in frame.columns}

    if "variant" in cols and "position" not in cols:
        # Parse "R201C" / "p.Arg201Cys"-style shorthand.
        import re

        parsed = []
        for raw in frame[cols["variant"]].astype(str):
            m = re.fullmatch(r"\s*p?\.?\s*([A-Z])\s*(\d+)\s*([A-Z])\s*", raw.strip(), re.I)
            if not m:
                raise ValueError(
                    f"could not parse variant {raw!r}. Use one-letter form like "
                    f"'R201C', or supply explicit wt_aa/position/mut_aa columns."
                )
            parsed.append((m.group(1).upper(), int(m.group(2)), m.group(3).upper()))
        frame["wt_aa"] = [p[0] for p in parsed]
        frame["position"] = [p[1] for p in parsed]
        frame["mut_aa"] = [p[2] for p in parsed]

    for required in ("wt_aa", "position", "mut_aa"):
        if required not in frame.columns:
            raise ValueError(
                f"variants file {path} needs a '{required}' column (or a 'variant' "
                f"column in R201C form). Columns found: {list(frame.columns)}"
            )
    if "gene" not in frame.columns:
        if gene is None:
            raise ValueError(
                f"{path} has no 'gene' column; pass --gene to apply one to every row."
            )
        frame["gene"] = gene
    elif gene is not None:
        before = len(frame)
        frame = frame[frame["gene"].astype(str).str.upper() == gene.upper()]
        if frame.empty:
            raise ValueError(
                f"--gene {gene} matched none of the {before} rows in {path}; "
                f"genes present: {sorted(pd.read_csv(path)['gene'].unique())}"
            )
    return frame.reset_index(drop=True)


def read_compounds(path: Path) -> pd.DataFrame:
    """Read a ``.smi`` (SMILES [whitespace] id) or CSV compound file."""
    text_suffixes = {".smi", ".smiles", ".txt", ".ism"}
    if path.suffix.lower() in text_suffixes:
        rows = []
        for n, line in enumerate(path.read_text().splitlines(), start=1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            smiles = parts[0]
            cid = parts[1] if len(parts) > 1 else f"CMP_{n:04d}"
            rows.append({"compound_id": cid, "smiles": smiles, "mechanism": ""})
        if not rows:
            raise ValueError(f"no SMILES found in {path}")
        frame = pd.DataFrame(rows)
    else:
        frame = pd.read_csv(path)
        if "smiles" not in frame.columns:
            raise ValueError(f"{path} needs a 'smiles' column; found {list(frame.columns)}")
        if "compound_id" not in frame.columns:
            frame["compound_id"] = [f"CMP_{i:04d}" for i in range(len(frame))]
        if "mechanism" not in frame.columns:
            frame["mechanism"] = ""
    frame.index = pd.Index(frame["compound_id"].astype(str), name="compound_id")
    return frame


# ---------------------------------------------------------------------------
# predict
# ---------------------------------------------------------------------------


def _fill_missing_structural(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    """Add absent optional columns as NaN; report whether structure is usable."""
    optional = {
        "uniprot": "",
        "ddg_pred": np.nan,
        "rsa": np.nan,
        "sec_struct": "L",
        "conservation": np.nan,
        "residue_depth": np.nan,
        "dist_to_pocket": np.nan,
        "pocket_vol_delta": np.nan,
        "contacts_ligand": 0,
    }
    for col, default in optional.items():
        if col not in frame.columns:
            frame[col] = default
    have_structure = frame["dist_to_pocket"].notna().any()
    return frame, bool(have_structure)


def cmd_predict(args: argparse.Namespace) -> int:
    from .calibration import ConformalPredictor, VennAbersPredictor, fit_calibrator
    from .data import Cohort, load_cohort
    from .domain import ApplicabilityDomain
    from .evaluate import choose_threshold
    from .features import FeatureSpec, build_features
    from .models import build_model

    variants = read_variants(Path(args.variants), args.gene)
    compounds = read_compounds(Path(args.compounds))
    variants, have_structure = _fill_missing_structural(variants)
    variants.index = pd.Index(
        [
            f"{r.gene}_{r.wt_aa}{r.position}{r.mut_aa}"
            for r in variants.itertuples()
        ],
        name="variant_id",
    )

    # --- training cohort -------------------------------------------------
    train_cohort = load_cohort(
        args.cohort, benchmark_root=args.benchmark_root, seed=args.seed
    )
    if train_cohort.synthetic:
        print(
            "\n"
            + "!" * 78
            + f"\n{train_cohort.label_banner()}\n"
            "The model behind these predictions was trained on GENERATED data.\n"
            "Every probability below is a demonstration of the software pipeline.\n"
            "It carries NO experimental or biological information whatsoever.\n"
            "Supply a curated benchmark with --cohort <path> for real predictions.\n"
            + "!" * 78
            + "\n",
            file=sys.stderr,
        )

    blocks: list[str] = ["stability", "ligand", "interaction"]
    if "structural" and have_structure:
        blocks.insert(1, "structural")
    elif not have_structure:
        _log.warning(
            "no dist_to_pocket values supplied, so the structural (Tier 3) block is "
            "disabled for this query. Pocket-occupancy features are left out rather "
            "than imputed: imputing zero would assert the variant sits inside the "
            "pocket. Supply dist_to_pocket / pocket_vol_delta / contacts_ligand to "
            "enable Tier 3."
        )
    if args.no_ligand_block:
        blocks = [b for b in blocks if b not in ("ligand", "interaction")]
        _log.warning("--no-ligand-block: the model cannot distinguish compounds.")
    spec = FeatureSpec(blocks=tuple(blocks), ecfp_bits=args.ecfp_bits)

    splits = train_cohort.splits["random"]
    fm_train = build_features(
        train_cohort, spec=spec, rows=splits["train"], memory_fraction=args.memory_fraction
    )
    fm_calib = build_features(
        train_cohort, spec=spec, rows=splits["calib"], memory_fraction=args.memory_fraction
    )

    model = build_model(args.tier, random_state=args.seed)
    model.fit(
        fm_train.X,
        fm_train.y,
        eval_set=(fm_calib.X, fm_calib.y),
        feature_names=fm_train.feature_names,
    )
    p_calib = model.predict_proba(fm_calib.X)

    calibrator, cal_report = fit_calibrator(args.calibration, p_calib, fm_calib.y)
    # Venn-Abers supplies ci_low/ci_high: a probability interval that is
    # actually informative for a binary outcome. Split-conformal regression on
    # |y - p| is also fitted, but its useful output here is the LABEL SET, not
    # an interval -- on binary labels its interval degenerates to [0, 1].
    venn_abers = VennAbersPredictor().fit(p_calib, fm_calib.y)
    conformal = ConformalPredictor(alpha=args.alpha).fit(p_calib, fm_calib.y)
    threshold = choose_threshold(fm_calib.y, p_calib)

    # --- query pairs: the cross product of variants x compounds ----------
    query_pairs = pd.DataFrame(
        [
            {"variant_id": v, "compound_id": c}
            for v in variants.index
            for c in compounds.index
        ]
    )
    query_cohort = Cohort(
        variants=pd.concat([train_cohort.variants, variants[train_cohort.variants.columns]]),
        compounds=pd.concat(
            [train_cohort.compounds, compounds[train_cohort.compounds.columns]]
        ).pipe(lambda d: d[~d.index.duplicated(keep="last")]),
        pairs=query_pairs.assign(label=0, assay="query", source="cli"),
        provenance="CLI query",
    )
    fm_query = build_features(
        query_cohort,
        spec=spec,
        pairs=query_pairs,
        memory_fraction=args.memory_fraction,
    )

    p_raw = model.predict_proba(fm_query.X)
    p_cal = calibrator.transform(p_raw)
    ci_low, ci_high = venn_abers.interval(p_raw)
    label_sets = [
        "/".join(str(v) for v in ls) if ls else "(none)"
        for ls in conformal.label_set(p_cal)
    ]

    # --- applicability domain -------------------------------------------
    ad = ApplicabilityDomain(
        tanimoto_distance=args.tanimoto_distance, residue_window=args.residue_window
    ).fit(train_cohort, splits["train"])
    verdict = ad.judge(query_pairs, query_cohort, memory_fraction=args.memory_fraction)

    # --- attribution -----------------------------------------------------
    shap_col = ["" for _ in range(len(query_pairs))]
    if not args.no_shap and args.tier in ("tier1_gbm", "tier3_structure"):
        try:
            from .attribution import shap_values_for_tree_model, top_shap_features

            sv = shap_values_for_tree_model(model, fm_query, max_rows=len(fm_query.X))
            shap_col = top_shap_features(sv, fm_query.feature_names, k=args.shap_top_k)
        except Exception as exc:  # attribution is diagnostic, never fatal
            _log.warning("SHAP attribution failed (%s); leaving the column blank", exc)

    out = pd.DataFrame(
        {
            "variant": query_pairs["variant_id"].to_numpy(),
            "compound": query_pairs["compound_id"].to_numpy(),
            "p_amenable": p_raw,
            "calibrated_p": p_cal,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "out_of_domain": verdict.out_of_domain,
            "nearest_training_neighbour": verdict.nearest_compound,
            "top_shap_features": shap_col,
        }
    )
    out["conformal_label_set"] = label_sets
    out["ood_reason"] = verdict.reasons()

    # --- the ranking refusal ---------------------------------------------
    in_domain = out[~out["out_of_domain"]].sort_values("calibrated_p", ascending=False)
    ood = out[out["out_of_domain"]].sort_values("calibrated_p", ascending=False)

    if len(ood) and not args.include_ood:
        ranked = in_domain
        print(
            f"\nWITHHELD {len(ood)} of {len(out)} pairs as out-of-domain; they are not "
            f"ranked alongside in-domain predictions.\n"
            f"  compound axis (Tanimoto distance > {args.tanimoto_distance}): "
            f"{int(verdict.compound_ood.sum())}\n"
            f"  protein axis (no training variant within {args.residue_window} residues): "
            f"{int(verdict.protein_ood.sum())}\n"
            f"  gene absent from training data: {int(verdict.gene_unseen.sum())}\n"
            f"Pass --include-ood to append them (flagged, after all in-domain rows).\n",
            file=sys.stderr,
        )
    else:
        # Concatenated, never co-sorted: OOD rows follow every in-domain row.
        ranked = pd.concat([in_domain, ood], ignore_index=True)

    if ranked.empty:
        print(
            "No in-domain pairs to report. Every query fell outside the training "
            "distribution -- which is itself the finding, and is what you would "
            "expect when predicting for a gene the model has never seen. Use "
            "--include-ood to see the (unsupported) numbers anyway.",
            file=sys.stderr,
        )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    header = [
        f"# foldrescue {__version__} -- {'SYNTHETIC TRAINING DATA' if train_cohort.synthetic else 'curated training data'}",
        f"# provenance: {train_cohort.provenance}",
        f"# tier={args.tier} calibration={args.calibration} ({cal_report.summary()})",
        "# ci_low/ci_high: Venn-Abers probability interval (validity: one endpoint is calibrated)",
        f"# conformal_label_set: split-conformal labels not rejected at alpha={args.alpha}; '0/1' = abstention",
        f"# decision threshold chosen on calibration split: {threshold:.4f}",
        f"# applicability domain: Tanimoto distance <= {args.tanimoto_distance}, residue window <= {args.residue_window}",
        "# NOT CLINICAL DECISION SUPPORT. Predictions are hypotheses for assay prioritisation only.",
    ]
    with open(out_path, "w") as fh:
        fh.write("\n".join(header) + "\n")
        ranked[OUTPUT_COLUMNS + ["conformal_label_set", "ood_reason"]].to_csv(fh, index=False)

    print(f"wrote {len(ranked)} rows to {out_path}", file=sys.stderr)
    if args.json_summary:
        Path(args.json_summary).write_text(
            json.dumps(
                {
                    "n_query": int(len(out)),
                    "n_in_domain": int(len(in_domain)),
                    "n_out_of_domain": int(len(ood)),
                    "synthetic_training_data": bool(train_cohort.synthetic),
                    "tier": args.tier,
                    "calibration": cal_report.summary(),
                    "threshold": float(threshold),
                },
                indent=2,
            )
        )
    return 0


# ---------------------------------------------------------------------------
# resources
# ---------------------------------------------------------------------------


def cmd_resources(args: argparse.Namespace) -> int:
    """Report the guard's verdict for a hypothetical allocation."""
    from .utils import check_resources

    try:
        plan = check_resources(
            args.rows,
            args.cols,
            memory_fraction=args.memory_fraction,
            allow_downcast=not args.no_downcast,
            allow_chunking=not args.no_chunking,
            label="requested allocation",
        )
    except ResourceRefusal as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2
    print(plan.describe())
    return 0


# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="foldrescue",
        description=(
            "Predict whether a pharmacological chaperone rescues a destabilising "
            "missense variant. Pair-level, calibrated, and explicit about its "
            "applicability domain. NOT clinical decision support."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--version", action="version", version=f"foldrescue {__version__}")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("predict", help="rank (variant, compound) pairs")
    pr.add_argument("--gene", help="restrict/assign a gene symbol, e.g. GLB1")
    pr.add_argument("--variants", required=True, help="CSV of variants")
    pr.add_argument("--compounds", required=True, help=".smi or CSV of compounds")
    pr.add_argument("--out", required=True, help="output CSV path")
    pr.add_argument(
        "--cohort",
        default="synthetic",
        help="'synthetic' or a path to a curated benchmark root",
    )
    pr.add_argument("--benchmark-root", default=None, help="explicit benchmark directory")
    pr.add_argument("--tier", default="tier3_structure", choices=["tier1_gbm", "tier2_fusion", "tier3_structure"])
    pr.add_argument("--calibration", default="temperature", choices=["temperature", "isotonic"])
    pr.add_argument("--alpha", type=float, default=0.1, help="conformal miscoverage level")
    pr.add_argument("--tanimoto-distance", type=float, default=0.60)
    pr.add_argument("--residue-window", type=int, default=25)
    pr.add_argument(
        "--include-ood",
        action="store_true",
        help="append out-of-domain pairs (flagged) instead of withholding them",
    )
    pr.add_argument("--ecfp-bits", type=int, default=1024)
    pr.add_argument("--no-ligand-block", action="store_true", help="ablation: drop chemistry")
    pr.add_argument("--no-shap", action="store_true")
    pr.add_argument("--shap-top-k", type=int, default=3)
    pr.add_argument("--memory-fraction", type=float, default=0.5)
    pr.add_argument("--seed", type=int, default=0)
    pr.add_argument("--json-summary", default=None, help="also write a JSON run summary")
    pr.set_defaults(func=cmd_predict)

    rs = sub.add_parser("resources", help="ask the resource guard about an allocation")
    rs.add_argument("--rows", type=int, required=True)
    rs.add_argument("--cols", type=int, required=True)
    rs.add_argument("--memory-fraction", type=float, default=0.5)
    rs.add_argument("--no-downcast", action="store_true")
    rs.add_argument("--no-chunking", action="store_true")
    rs.set_defaults(func=cmd_resources)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except ResourceRefusal as exc:
        print(f"\nRESOURCE REFUSAL\n{exc}\n", file=sys.stderr)
        return 2
    except (ValueError, KeyError, FileNotFoundError, ImportError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
