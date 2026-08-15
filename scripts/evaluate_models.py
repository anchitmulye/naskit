"""
evaluate.py — Config-driven final model evaluation.

Reads eval_config.yaml and runs run_analysis() on each (base, nas) pair
using the dataloader class specified in the config. No hardcoded paths,
no hardcoded class names — everything lives in the YAML.

Usage (from naskit/ root)
-----
    # Evaluate all models in eval_config.yaml
    conda run -n pointmlp python evaluate.py

    # Specific models only
    conda run -n pointmlp python evaluate.py --models pointnet pointnet_c40

    # Override which config file to use
    conda run -n pointmlp python evaluate.py --config my_eval_config.yaml

    # Preview without running (dry run)
    conda run -n pointmlp python evaluate.py --dry-run
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

_NASKIT = Path(__file__).resolve().parent.parent
_H3DNAS = _NASKIT / "submodules" / "h3dnas"
sys.path.insert(0, str(_H3DNAS))
sys.path.insert(0, str(_NASKIT))

import yaml
import numpy as np
from h3dnas.utils.logger import get_logger

logger = get_logger(__name__)


# ── Dataloader loading ────────────────────────────────────────────────────────

def load_datamodule(dl_cfg: dict, root: Path):
    """
    Import the dataloader class from config and instantiate it.

    dl_cfg example:
      module: submodules/h3dnas/examples/pointnet/datamodule.py
      class:  ModelNetDataModule
      args:
        data_root:   submodules/.../data/modelnet40_normal_resampled
        model_repo:  submodules/.../pointnet_pointnet2
        num_classes: 40
        seed:        42

    All path values in args are resolved relative to naskit/ root.
    """
    module_path = root / dl_cfg["module"]
    if not module_path.exists():
        raise FileNotFoundError(f"Dataloader module not found: {module_path}")

    # Load as uniquely named module to avoid sys.modules collision
    unique_name = f"_dm_{module_path.stem}_{id(dl_cfg)}"
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    cls = getattr(mod, dl_cfg["class"])

    # Resolve path-valued args relative to root
    args = dl_cfg.get("args", {})
    resolved = {}
    for k, v in args.items():
        if isinstance(v, str) and ("/" in v or "\\" in v) and not v.startswith("/"):
            # Looks like a relative path — resolve to absolute
            candidate = root / v
            if candidate.exists():
                resolved[k] = str(candidate)
                continue
        resolved[k] = v

    return cls(**resolved)


# ── Single model evaluation ───────────────────────────────────────────────────

def evaluate_one(
    name:         str,
    model_cfg:    dict,
    eval_cfg:     dict,
    root:         Path,
    dry_run:      bool,
    no_save:      bool,
) -> bool:

    base_onnx = root / model_cfg["base_onnx"]
    nas_onnx  = root / model_cfg["nas_onnx"]

    logger.info(f"\n{'='*70}")
    logger.info(f"  Model   : {name}")
    logger.info(f"  Base    : {base_onnx.name}")
    logger.info(f"  NAS     : {nas_onnx.name}")

    # Validate paths
    if not base_onnx.exists():
        logger.warning(f"  SKIP — base_onnx not found: {base_onnx}")
        return False
    if not nas_onnx.exists():
        logger.warning(f"  SKIP — nas_onnx not found: {nas_onnx}")
        return False

    if dry_run:
        dl = model_cfg.get("dataloader", {})
        logger.info(f"  Loader  : {dl.get('class')} from {dl.get('module')}")
        logger.info(f"  [DRY RUN] — skipping actual evaluation")
        return True

    # Build dataloader
    try:
        dm = load_datamodule(model_cfg["dataloader"], root)
    except Exception as exc:
        logger.error(f"  FAIL — could not build dataloader: {exc}")
        return False

    # Output path: next to the NAS ONNX's run
    run_dir     = nas_onnx.parent.parent    # artifacts/<model>/<run>/
    output_path = run_dir / "results" / "analysis.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info(f"  Output  : {output_path}")

    # Run analysis
    from h3dnas.analysis.model_analyser import run_analysis
    try:
        run_analysis(
            base_onnx    = str(base_onnx),
            nas_onnx     = str(nas_onnx),
            loader       = dm.full_loader(),
            sample_input = dm.sample_input(),
            class_names  = model_cfg.get("class_names"),
            output       = str(output_path),
            warmup       = eval_cfg.get("warmup",      10),
            runs         = eval_cfg.get("runs",         100),
            num_threads  = eval_cfg.get("num_threads",  4),
        )
    except Exception as exc:
        import traceback
        logger.error(f"  FAIL — run_analysis raised: {exc}")
        logger.error(traceback.format_exc())
        return False

    # Save to committed results store
    if not no_save:
        results_name = model_cfg.get("results_name", "final_eval")
        notes        = model_cfg.get("notes")
        results_dir  = root / "results" / name / results_name
        results_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(output_path, results_dir / "analysis.json")
        # Copy config.yaml from run if present
        for cfg_src in (run_dir / "results" / "config.yaml", run_dir / "config.yaml"):
            if cfg_src.exists():
                shutil.copy2(cfg_src, results_dir / "config.yaml")
                break
        if notes:
            (results_dir / "notes.md").write_text(notes + "\n")
        logger.info(f"  Saved   : {results_dir}")

    return True


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(
        description="Config-driven final model evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python evaluate.py\n"
            "  python evaluate.py --models pointnet pointnet_c40\n"
            "  python evaluate.py --config custom.yaml --dry-run\n"
        ),
    )
    p.add_argument("--config",   default=_NASKIT/"scripts/eval_config.yml",
                   help="YAML config file (default: eval_config.yaml)")
    p.add_argument("--models",   nargs="+", default=None,
                   help="Model keys to evaluate (default: all in config)")
    p.add_argument("--dry-run",  action="store_true",
                   help="Show what would be evaluated without running")
    p.add_argument("--no-save",  action="store_true",
                   help="Run evaluation but skip saving to results/ store")
    args = p.parse_args()

    config_path = _NASKIT / args.config
    if not config_path.exists():
        logger.error(f"Config not found: {config_path}")
        sys.exit(1)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    eval_cfg   = cfg.get("eval", {})
    models_cfg = cfg.get("models", {})

    # Filter to requested models
    keys = args.models or list(models_cfg.keys())
    missing = [k for k in keys if k not in models_cfg]
    if missing:
        logger.error(f"Models not found in config: {missing}")
        sys.exit(1)

    logger.info(f"Config  : {config_path}")
    logger.info(f"Models  : {keys}")
    logger.info(f"Eval    : warmup={eval_cfg.get('warmup',10)}  "
                f"runs={eval_cfg.get('runs',100)}  "
                f"threads={eval_cfg.get('num_threads',4)}")

    passed = failed = skipped = 0
    for name in keys:
        ok = evaluate_one(
            name      = name,
            model_cfg = models_cfg[name],
            eval_cfg  = eval_cfg,
            root      = _NASKIT,
            dry_run   = args.dry_run,
            no_save   = args.no_save,
        )
        if ok:
            passed += 1
        else:
            failed += 1

    logger.info(f"\n{'='*70}")
    logger.info(f"  Done: {passed} passed, {failed} failed")

    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
