"""Command-line interface for the UnfavorSeg pipeline.

Examples
--------
    segment install-trainers
    segment coarse-train  --dataset Dataset005_Hardness --out models/rf.joblib
    segment pseudolabel   --dataset Dataset005_Hardness --model models/rf.joblib \
                             --out nnUNet_raw/Dataset011_HardnessCC_pl
    segment build-cc      --src Dataset005_Hardness --pseudolabels <dir> \
                             --dst Dataset011_HardnessCC
    segment make-splits   --dataset Dataset011_HardnessCC --protocol leave_one_tunnel_out
    segment e1            --dataset Dataset005_Hardness
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List

import numpy as np

from .config import Config, load_config
from .io import nnunet_dirs, read_case, read_dataset_json, list_cases, read_volume, write_json


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _resolve_dataset(name_or_path: str, base: str = "nnUNet_raw") -> Path:
    p = Path(name_or_path)
    if p.exists():
        return p
    root = nnunet_dirs().get(base)
    if root is not None and (root / name_or_path).exists():
        return root / name_or_path
    raise FileNotFoundError(f"Dataset not found: {name_or_path} (base={base})")


def effective_config(
    dataset_dir: Path, config_path: str | None, active_class: str | None = None
) -> Config:
    """Base YAML config with channels/classes taken from the dataset.json.

    Each geology type is an INDEPENDENT 0/1 target. ``geology_classes`` (all
    available types) come from the dataset.json field of the same name, else
    from the standard ``labels``, else from the YAML config. ``active_class``
    narrows ``classes`` to a single ``[<type>]`` so the rest of the pipeline runs
    as a binary problem; it is validated against the available types.
    """
    cfg = load_config(config_path)
    ds = read_dataset_json(dataset_dir)
    channels = [ds["channel_names"][str(i)] for i in range(len(ds["channel_names"]))]
    # available geology types: explicit field > standard labels > YAML config
    if "geology_classes" in ds:
        geology_classes = list(ds["geology_classes"])
    else:
        geology_classes = [name for name in ds.get("labels", {}) if name != "background"]
        geology_classes = sorted(
            geology_classes,
            key=lambda n: ds["labels"][n] if isinstance(ds["labels"][n], int) else 0,
        )
    if not geology_classes:
        geology_classes = cfg.classes

    cfg.raw["channels"] = channels
    cfg.raw["geology_classes"] = geology_classes

    if active_class is not None:
        if active_class not in geology_classes:
            raise ValueError(
                f"--class {active_class!r} is not among the dataset's geology types "
                f"{geology_classes}"
            )
        cfg.raw["classes"] = [active_class]
    elif len(geology_classes) > 1:
        raise ValueError(
            f"Dataset declares multiple geology types {geology_classes}; pass "
            f"--class <type> to select one (each type is an independent binary run)."
        )
    else:
        cfg.raw["classes"] = list(geology_classes)
    return cfg


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_install_trainers(args):
    from .register import install_trainer_shim

    path = install_trainer_shim()
    print(f"Installed trainer shim -> {path}")


def cmd_uninstall_trainers(args):
    from .register import uninstall_trainer_shim

    uninstall_trainer_shim()
    print("Removed trainer shim.")


def cmd_coarse_train(args):
    from .coarse.rf_classifier import CoarseClassifier
    from .experiments.records import build_records
    from .process_contract import feature_columns

    dataset_dir = _resolve_dataset(args.dataset)
    cfg = effective_config(dataset_dir, args.config, args.geology_class)
    # soft-target carrier channels (if present) are not coarse physical features
    channels = [c for c in cfg.channels if c not in {"confidence", "probfg"}]
    rec = build_records(dataset_dir, channels, cfg.half_window, cfg.statistics,
                        cfg.mode_decimals, n_per_class=args.n_per_class, seed=42,
                        class_name=cfg.classes[0],
                        strict_per_class=len(cfg.geology_classes) > 1,
                        process_feature_mode=cfg.process_feature_mode,
                        process_depth=cfg.process_depth,
                        process_size=cfg.process_size)
    clf = CoarseClassifier(model=args.model_type, num_classes=len(cfg.classes),
                           params=cfg.get("coarse", "random_forest", default={}))
    clf.fit(rec["X"], rec["y"])
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    clf.save(args.out)
    if cfg.process_feature_mode:
        sidecar = Path(args.out).with_suffix(".features.json")
        cols = feature_columns(cfg.process_feature_mode)
        with open(sidecar, "w", encoding="utf-8") as fh:
            json.dump(
                {
                    "schema_version": 1,
                    "feature_mode": cfg.process_feature_mode,
                    "depth": cfg.process_depth,
                    "size": cfg.process_size,
                    "feature_columns": cols,
                    "label_columns": cfg.classes,
                    "feature_count": len(cols),
                    "model_out": str(Path(args.out)),
                    "recommended_thresholds": {
                        name: {"threshold": 0.5} for name in cfg.classes
                    },
                },
                fh,
                indent=2,
            )
        print(f"Wrote process feature contract -> {sidecar}")
    print(f"Trained {args.model_type} (binary, class={cfg.classes[0]}) on "
          f"{rec['X'].shape[0]} records -> {args.out}")


def cmd_pseudolabel(args):
    from .coarse.pseudolabel import generate_pseudolabels, write_pseudolabels
    from .coarse.rf_classifier import CoarseClassifier

    dataset_dir = _resolve_dataset(args.dataset)
    cfg = effective_config(dataset_dir, args.config, args.geology_class)
    channels = [c for c in cfg.channels if c not in {"confidence", "probfg"}]
    clf = CoarseClassifier.load(args.model)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    ds = read_dataset_json(dataset_dir)
    fe = ds.get("file_ending", ".nii.gz")
    cases = list_cases(dataset_dir / "imagesTr", fe)
    for case in cases:
        vol, geom = read_case(dataset_dir / "imagesTr", case, len(channels), fe)
        pl = generate_pseudolabels(list(vol), clf, cfg.half_window, cfg.statistics,
                                   cfg.mode_decimals,
                                   process_feature_mode=cfg.process_feature_mode,
                                   process_depth=cfg.process_depth,
                                   process_size=cfg.process_size)
        write_pseudolabels(pl, geom, out_dir, case, fe)
        print(f"  pseudo-labelled {case}")
    print(f"Wrote pseudo-labels + foreground probabilities/probfg to {out_dir}")


def cmd_build_cc(args):
    from .fine.dataset import build_cc_dataset

    src = _resolve_dataset(args.src)
    cfg = effective_config(src, args.config, args.geology_class)
    dst_root = nnunet_dirs().get("nnUNet_raw")
    dst = Path(args.dst) if Path(args.dst).is_absolute() else (dst_root / args.dst)
    build_cc_dataset(src, args.pseudolabels, dst, cfg.channels, cfg.classes)
    print(f"Built probability-augmented dataset -> {dst}")


def cmd_make_splits(args):
    from .data.splits import (
        default_case_to_tunnel,
        kfold_cases,
        leave_one_tunnel_out,
        stratified_kfold_cases,
    )

    pre = nnunet_dirs().get("nnUNet_preprocessed")
    dataset_dir = pre / args.dataset if pre else Path(args.dataset)
    ds = read_dataset_json(_resolve_dataset(args.dataset))
    fe = ds.get("file_ending", ".nii.gz")
    raw = _resolve_dataset(args.dataset)
    cases = list_cases(raw / "imagesTr", fe)
    mapping = default_case_to_tunnel(cases)
    if args.protocol == "kfold":
        folds = kfold_cases(cases, n_splits=args.folds)
    elif args.protocol == "stratified_kfold":
        ratios = {}
        for case in cases:
            label_path = raw / "labelsTr" / f"{case}{fe}"
            label, _ = read_volume(label_path)
            ratios[case] = float(np.count_nonzero(label) / label.size)
        folds = stratified_kfold_cases(cases, ratios, n_splits=args.folds)
    elif args.protocol == "leave_one_tunnel_out":
        folds = leave_one_tunnel_out(mapping)
    elif len(set(mapping.values())) < 2:
        folds = kfold_cases(cases, n_splits=args.folds)
    else:
        folds = leave_one_tunnel_out(mapping)
    out = dataset_dir / "splits_final.json"
    write_json(folds, out)
    print(f"Wrote {len(folds)} folds -> {out}")
    for i, fold in enumerate(folds):
        print(f"  fold {i}: train={len(fold['train'])}, val={len(fold['val'])}")


def _coarse_experiment(args, module, **kwargs):
    dataset_dir = _resolve_dataset(args.dataset)
    cfg = effective_config(dataset_dir, args.config, args.geology_class)
    rows = module.run(dataset_dir, config=cfg, out_dir=args.out, **kwargs)
    for r in rows:
        print(r)


def cmd_e1(args):
    from .experiments import e1_coarse_heldout
    _coarse_experiment(args, e1_coarse_heldout, n_per_class=args.n_per_class)


def cmd_ab_burial(args):
    from .experiments import ab_burial_depth
    _coarse_experiment(args, ab_burial_depth, n_per_class=args.n_per_class)


def cmd_ab_window(args):
    from .experiments import ab_window_size
    _coarse_experiment(args, ab_window_size, n_per_class=args.n_per_class)


def cmd_ab_tsp(args):
    from .experiments import ab_tsp_perturbation
    _coarse_experiment(args, ab_tsp_perturbation, n_per_class=args.n_per_class)


def cmd_plot_training(args):
    from .experiments.training_curves import run

    paths = run(args.run_dir, out_dir=args.out, smooth=args.smooth)
    print(f"Wrote training curves CSV -> {paths['csv']}")
    print(f"Wrote training curves plot -> {paths['png']}")


def _fine_experiment_config(args) -> Config:
    dataset_dir = _resolve_dataset(args.dataset)
    return effective_config(dataset_dir, args.config, args.geology_class)


def cmd_e2(args):
    from .experiments import e2_tfr_finegrained

    rows = e2_tfr_finegrained.run(
        args.pred,
        args.reference,
        config=_fine_experiment_config(args),
        out_dir=args.out,
    )
    for r in rows:
        print(r)


def cmd_e3(args):
    from .experiments import e3_borehole

    rows = e3_borehole.run(
        args.pred,
        args.reference,
        config=_fine_experiment_config(args),
        out_dir=args.out,
    )
    for r in rows:
        print(r)


def cmd_e4(args):
    from .experiments import e4_pseudo_vs_refined

    rows = e4_pseudo_vs_refined.run(
        args.pred,
        args.pseudolabels,
        args.reference,
        config=_fine_experiment_config(args),
        out_dir=args.out,
    )
    for r in rows:
        print(r)


def _parse_key_value_pairs(items: List[str], key_type=str):
    out = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"expected KEY=VALUE, got {item!r}")
        key, value = item.split("=", 1)
        out[key_type(key)] = value
    return out


def cmd_e5(args):
    from .experiments import e5_lambda_sensitivity

    rows = e5_lambda_sensitivity.run(
        _parse_key_value_pairs(args.lambda_pred, float),
        args.reference,
        config=_fine_experiment_config(args),
        out_dir=args.out,
    )
    for r in rows:
        print(r)


def cmd_uncertainty(args):
    from .experiments import uncertainty

    rows = uncertainty.run(
        _parse_key_value_pairs(args.method_pred),
        classes_config=_fine_experiment_config(args),
        pseudolabel_dir=args.pseudolabels,
        out_dir=args.out,
    )
    for r in rows:
        print(r)


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="segment", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp):
        sp.add_argument("--config", default=None, help="path to YAML config")

    def add_class(sp):
        sp.add_argument(
            "--class", dest="geology_class", default=None,
            help="independent geology type to run as a 0/1 binary problem; "
                 "required when the dataset declares multiple types",
        )

    sp = sub.add_parser("install-trainers"); sp.set_defaults(func=cmd_install_trainers)
    sp = sub.add_parser("uninstall-trainers"); sp.set_defaults(func=cmd_uninstall_trainers)

    sp = sub.add_parser("coarse-train"); add_common(sp); add_class(sp)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--out", required=True)
    sp.add_argument("--model-type", default="random_forest")
    sp.add_argument("--n-per-class", type=int, default=2000)
    sp.set_defaults(func=cmd_coarse_train)

    sp = sub.add_parser("pseudolabel"); add_common(sp); add_class(sp)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--model", required=True)
    sp.add_argument("--out", required=True)
    sp.set_defaults(func=cmd_pseudolabel)

    sp = sub.add_parser("build-cc"); add_common(sp); add_class(sp)
    sp.add_argument("--src", required=True)
    sp.add_argument("--pseudolabels", required=True)
    sp.add_argument("--dst", required=True)
    sp.set_defaults(func=cmd_build_cc)

    sp = sub.add_parser("make-splits"); add_common(sp)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--protocol", choices=("auto", "kfold", "stratified_kfold", "leave_one_tunnel_out"),
                    default="auto")
    sp.add_argument("--folds", type=int, default=5)
    sp.set_defaults(func=cmd_make_splits)

    for name, fn in (("e1", cmd_e1), ("ab-burial", cmd_ab_burial),
                     ("ab-window", cmd_ab_window), ("ab-tsp", cmd_ab_tsp)):
        sp = sub.add_parser(name); add_common(sp); add_class(sp)
        sp.add_argument("--dataset", required=True)
        sp.add_argument("--out", default=f"results/{name}")
        sp.add_argument("--n-per-class", type=int, default=1500)
        sp.set_defaults(func=fn)

    sp = sub.add_parser("plot-training")
    sp.add_argument("run_dir", help="nnU-Net fold output directory")
    sp.add_argument("--out", default=None, help="output directory; defaults to run_dir")
    sp.add_argument("--smooth", type=int, default=5, help="moving-average window")
    sp.set_defaults(func=cmd_plot_training)

    def add_fine_common(sp):
        add_common(sp)
        add_class(sp)
        sp.add_argument("--dataset", required=True,
                        help="process-exported/fine nnU-Net raw dataset")
        sp.add_argument("--pred", required=True,
                        help="nnU-Net prediction directory")
        sp.add_argument("--reference", required=True,
                        help="reference label directory, normally labelsTr or labelsTs")

    sp = sub.add_parser("e2"); add_fine_common(sp)
    sp.add_argument("--out", default="results/e2_tfr")
    sp.set_defaults(func=cmd_e2)

    sp = sub.add_parser("e3"); add_fine_common(sp)
    sp.add_argument("--out", default="results/e3_borehole")
    sp.set_defaults(func=cmd_e3)

    sp = sub.add_parser("e4"); add_fine_common(sp)
    sp.add_argument("--pseudolabels", required=True,
                    help="process labelsTr/labelsTs or legacy pseudolabel dir")
    sp.add_argument("--out", default="results/e4_pseudo_vs_refined")
    sp.set_defaults(func=cmd_e4)

    sp = sub.add_parser("e5"); add_common(sp); add_class(sp)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--reference", required=True)
    sp.add_argument("--lambda-pred", action="append", required=True,
                    help="lambda=prediction_dir; repeat for each lambda")
    sp.add_argument("--out", default="results/e5_lambda")
    sp.set_defaults(func=cmd_e5)

    sp = sub.add_parser("uncertainty"); add_common(sp); add_class(sp)
    sp.add_argument("--dataset", required=True)
    sp.add_argument("--method-pred", action="append", required=True,
                    help="name=prediction_dir; repeat for each method")
    sp.add_argument("--pseudolabels", default=None,
                    help="process labelsTr/labelsTs or legacy pseudolabel dir")
    sp.add_argument("--out", default="results/uncertainty")
    sp.set_defaults(func=cmd_uncertainty)

    return p


def main(argv: List[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
