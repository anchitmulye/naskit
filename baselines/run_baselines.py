"""
Unified baseline runner — produces paper comparison table.

Runs both baselines at the same prune_ratio as h3dnas and compares:
  1. Uniform L1 Pruning         (simplest possible baseline)
  2. Screening Methods Pruning  (Wang et al., arXiv:2502.07189, Feb 2025)
  3. h3dnas-Base two-stage NAS  (ours — from existing finetuned ONNX)

All methods:
  - Same model (PointNet on ModelNet40)
  - Same prune ratio target (~32% param reduction matching h3dnas stage1)
  - Same fine-tuning setup (AdamW, lr=3e-4, 50 epochs, warmup+cosine LR)
  - Same ORT evaluation (2468 test samples, intra_op_num_threads=1)

Usage:
    conda run -n pointmlp python baselines/run_baselines.py
    conda run -n pointmlp python baselines/run_baselines.py --skip-finetune
    conda run -n pointmlp python baselines/run_baselines.py --prune-ratio 0.32
"""

from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path

NASKIT_ROOT   = Path(__file__).parent.parent          # naskit/
H3DNAS_ROOT   = NASKIT_ROOT / "submodules/h3dnas"
POINTNET_REPO = NASKIT_ROOT / "submodules/vision-onnx-models/submodules/pointnet_pointnet2"
MODELNET_DATA = POINTNET_REPO / "data/modelnet40_normal_resampled"

sys.path.insert(0, str(H3DNAS_ROOT))
sys.path.insert(0, str(POINTNET_REPO))
sys.path.insert(0, str(Path(__file__).parent))

from h3dnas.utils.logger import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuration — update these paths for your setup
# ---------------------------------------------------------------------------

# Base PointNet ONNX (opset 13, no constant folding)
BASE_ONNX = NASKIT_ROOT / "models/pointnet/pointnet_cls_c40_n1024.onnx"

# ModelNet40 dataset (normal-resampled format)
MODELNET_DATA = POINTNET_REPO / "data/modelnet40_normal_resampled"

# h3dnas finetuned ONNX (from your NAS + finetune run)
# Set to None to skip h3dnas comparison and run only baselines
H3DNAS_ONNX = NASKIT_ROOT / "models/pointnet/pointnet_cls_c40_n1024_h3dnas.onnx"
# H3DNAS_ONNX = None   # ← uncomment to run baselines only

NUM_CLASSES  = 40
EVAL_SAMPLES = 2468   # full ModelNet40 test set
OUT_DIR      = Path(__file__).parent / "outputs"


# ---------------------------------------------------------------------------
# h3dnas result reader (from existing finetuned model)
# ---------------------------------------------------------------------------

def get_h3dnas_result(onnx_path: Path) -> dict:
    """Evaluate the already-finetuned h3dnas model on full test set."""
    import argparse as _ap
    import numpy as np
    import onnx as _onnx
    from torch.utils.data import DataLoader
    from data_utils.ModelNetDataLoader import ModelNetDataLoader
    from h3dnas.parser.onnx_parser import ONNXParser
    from h3dnas.core.nas_pipeline import _eval_fn
    from h3dnas.core.types import ONNXModel

    logger.info(f"  Evaluating h3dnas model: {onnx_path.name}")
    parser = ONNXParser(); parser.initialize()
    model  = parser.execute(str(onnx_path))

    # Base model for params reference
    base   = parser.execute(str(BASE_ONNX))

    args = _ap.Namespace(num_point=1024, use_uniform_sample=False,
                         use_normals=False, num_category=40)
    ds = ModelNetDataLoader(root=str(MODELNET_DATA), args=args,
                            split="test", process_data=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    base_acc = _eval_fn(base, loader)
    nas_acc  = _eval_fn(model, loader)

    param_red = (1 - model.parameters / base.parameters) * 100
    flops_red = (1 - model.flops / base.flops) * 100 if base.flops else 0

    return {
        "method":    "h3dnas two-stage NAS (ours)",
        "reference": "this work",
        "base_acc":  base_acc * 100,
        "zero_shot": None,   # not applicable for finetuned
        "finetuned": nas_acc * 100,
        "ort_acc":   nas_acc * 100,
        "params":    model.parameters,
        "param_red": param_red,
        "flops_red": flops_red,
        "prune_ratio": None,
    }


# ---------------------------------------------------------------------------
# Comparison table printer
# ---------------------------------------------------------------------------

def print_table(results: list, base_acc: float) -> None:
    print()
    print("=" * 100)
    print("COMPARISON TABLE — PointNet on ModelNet40 (2468 test samples)")
    print("=" * 100)
    print(f"{'Method':<35s}  {'Base%':>6}  {'ZeroShot%':>10}  {'Finetuned%':>11}  "
          f"{'Δ(pp)':>7}  {'Params':>10}  {'Param↓%':>8}  {'Reference'}")
    print("-" * 100)
    for r in results:
        zs_str = f"{r['zero_shot']:>9.2f}%" if r.get('zero_shot') else "        —"
        delta  = r['finetuned'] - base_acc
        print(
            f"  {r['method']:<33s}  "
            f"{r['base_acc']:>5.2f}%  "
            f"{zs_str}  "
            f"{r['finetuned']:>10.2f}%  "
            f"{delta:>+6.2f}pp  "
            f"{r['params']:>10,}  "
            f"{r['param_red']:>7.1f}%  "
            f"{r['reference']}"
        )
    print("=" * 100)
    print()
    # Highlight winner
    best = max(results, key=lambda r: r['finetuned'])
    print(f"  Winner: {best['method']}  ({best['finetuned']:.2f}%)")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(prune_ratio: float, epochs: int, lr: float, batch_size: int,
         n_samples: int, skip_finetune: bool, seed: int = 42) -> list:
    """
    Run all 3 baselines + h3dnas and produce the paper comparison table.

    Baselines (all source-free, ONNX-native, same compression target):
      1. Random Pruning         — lower bound, random channel selection
      2. Uniform L1 Pruning     — standard baseline, L1 weight importance
      3. Activation Rank        — inspired by HRank (Lin et al., CVPR 2020)
      4. h3dnas (ours)          — NAS-guided, SWAP-ranked, two-stage search
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    all_results = []

    # ── 0. h3dnas result — commented out, add manually from your NAS run ────────
    # Uncomment and set H3DNAS_ONNX above once your finetune is complete.
    # The baseline methods below are independent of h3dnas and can run first.
    #
    # if H3DNAS_ONNX and Path(H3DNAS_ONNX).exists():
    #     logger.info("\n[0] h3dnas two-stage NAS result (ours)")
    #     h3dnas_result = get_h3dnas_result(Path(H3DNAS_ONNX))
    #     base_acc = h3dnas_result["base_acc"]
    #     logger.info(f"  h3dnas: {h3dnas_result['finetuned']:.2f}%  "
    #                 f"Δ={h3dnas_result['finetuned']-base_acc:+.2f}pp")
    #     all_results.append(h3dnas_result)
    # else:
    #     base_acc = 90.32
    base_acc = 90.32  # PointNet ModelNet40 baseline — update after running baselines

    # ── 1. Random Pruning ─────────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("[1] Random Channel Pruning  (lower bound baseline)")
    logger.info("=" * 70)
    from random_pruning import main as random_main
    t0 = time.time()
    random_result = random_main(
        base_onnx   = BASE_ONNX,
        prune_ratio = prune_ratio,
        epochs      = 0 if skip_finetune else epochs,
        lr          = lr,
        batch_size  = batch_size,
        out_dir     = OUT_DIR,
        seed        = seed,
    )
    logger.info(f"  Random done: {random_result['finetuned']:.2f}%  ({(time.time()-t0)/60:.1f} min)")
    all_results.append(random_result)

    # ── 2. Uniform L1 Pruning ─────────────────────────────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("[2] Uniform L1 Pruning  (standard structured pruning baseline)")
    logger.info("=" * 70)
    from uniform_pruning import main as uniform_main
    t0 = time.time()
    uniform_result = uniform_main(
        base_onnx   = BASE_ONNX,
        prune_ratio = prune_ratio,
        epochs      = 0 if skip_finetune else epochs,
        lr          = lr,
        batch_size  = batch_size,
        out_dir     = OUT_DIR,
    )
    logger.info(f"  Uniform done: {uniform_result['finetuned']:.2f}%  ({(time.time()-t0)/60:.1f} min)")
    all_results.append(uniform_result)

    # ── 3. Activation Rank Pruning (HRank-inspired) ───────────────────────────
    logger.info("\n" + "=" * 70)
    logger.info("[3] Activation Rank Pruning  (inspired by HRank, Lin et al. CVPR 2020)")
    logger.info("    Source-free adaptation: random inputs, ONNX-native rank scoring")
    logger.info("=" * 70)
    try:
        from activation_rank_pruning import main as rank_main
        t0 = time.time()
        rank_result = rank_main(
            base_onnx   = BASE_ONNX,
            prune_ratio = prune_ratio,
            epochs      = 0 if skip_finetune else epochs,
            lr          = lr,
            batch_size  = batch_size,
            n_samples   = n_samples,
            out_dir     = OUT_DIR,
        )
        logger.info(f"  Activation Rank done: {rank_result['finetuned']:.2f}%  ({(time.time()-t0)/60:.1f} min)")
        all_results.append(rank_result)
    except Exception as exc:
        logger.warning(f"  Activation Rank failed: {exc}")

    # ── Print comparison table ────────────────────────────────────────────────
    print_table(all_results, base_acc)

    # ── Save results ──────────────────────────────────────────────────────────
    out_json = OUT_DIR / "comparison_results.json"
    with open(out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Results saved → {out_json}")
    return all_results


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Baseline comparison for h3dnas paper — PointNet ModelNet40"
    )
    p.add_argument("--prune-ratio",   type=float, default=0.32,
                   help="Compression target matching h3dnas stage1 (default: 32%)")
    p.add_argument("--epochs",        type=int,   default=50,
                   help="Finetune epochs per baseline (default: 50)")
    p.add_argument("--lr",            type=float, default=3e-4)
    p.add_argument("--batch-size",    type=int,   default=24)
    p.add_argument("--n-samples",     type=int,   default=64,
                   help="Random inputs for activation rank scoring")
    p.add_argument("--seed",          type=int,   default=42)
    p.add_argument("--skip-finetune", action="store_true",
                   help="Prune only, no finetuning (quick sanity check)")
    args = p.parse_args()

    main(
        prune_ratio   = args.prune_ratio,
        epochs        = args.epochs,
        lr            = args.lr,
        batch_size    = args.batch_size,
        n_samples     = args.n_samples,
        skip_finetune = args.skip_finetune,
        seed          = args.seed,
    )
