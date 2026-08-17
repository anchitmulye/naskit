"""
Compute Spearman rank correlation between SWAP-Score and post-finetune accuracy.

This validates h3dnas's zero-shot proxy claim: if SWAP-Score ranks correlate
with finetuned accuracy, the proxy is valid and SWAP pre-screening is justified.

Method:
  1. Run NAS with enable_zero_shot=True, zero_shot_top_k=0 (eval ALL candidates)
  2. Record (SWAP_score, ORT_accuracy) for every candidate
  3. Finetune top-K candidates (expensive — skip if --use-ort-acc-proxy)
  4. Compute Spearman ρ between SWAP rank and finetuned/ORT accuracy rank

With --use-ort-acc-proxy: uses zero-shot ORT accuracy instead of finetuned
accuracy as the target (fast, gives lower bound on true Spearman ρ).

Usage (from naskit/ root):
    python compute_spearman.py --use-ort-acc-proxy
    python compute_spearman.py --use-ort-acc-proxy --model pointnet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

NASKIT_ROOT   = Path(__file__).parent.parent
POINTNET_REPO = NASKIT_ROOT / "submodules/vision-onnx-models/submodules/pointnet_pointnet2"
MODELNET_DATA = POINTNET_REPO / "data/modelnet40_normal_resampled"

sys.path.insert(0, str(NASKIT_ROOT / "submodules/h3dnas"))
sys.path.insert(0, str(POINTNET_REPO))

import numpy as np

from h3dnas.utils.logger import get_logger
from h3dnas.parser.onnx_parser import ONNXParser
from h3dnas.generator.architecture_generator import ArchitectureGenerator
from h3dnas.modulator.architecture_modulator import ArchitectureModulator
from h3dnas.evaluator.zero_shot_evaluator import ZeroShotEvaluator
from h3dnas.evaluator.hardware_evaluator import HardwareEvaluator
from h3dnas.core.types import SearchSpace, ONNXModel
from h3dnas.datapipe.evaluators import eval_accuracy

logger = get_logger(__name__)

OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)


def run_spearman(
    base_onnx: Path,
    loader,
    sample_input: np.ndarray,
    num_candidates: int = 6,
    prune_ratios: list = None,
    width_multipliers: list = None,
    n_swap_samples: int = 32,
    seed: int = 42,
) -> dict:
    if prune_ratios is None:
        prune_ratios = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    if width_multipliers is None:
        width_multipliers = [0.75, 1.0]

    np.random.seed(seed)

    # Parse base
    logger.info("Parsing base model...")
    parser = ONNXParser(); parser.initialize()
    base_model = parser.execute(str(base_onnx))
    logger.info(f"  params={base_model.parameters:,}  flops={base_model.flops:,}")

    # Generate candidates
    search_space = SearchSpace(
        prune_ratios=prune_ratios,
        width_multipliers=width_multipliers,
        protected_nodes=[],
        cin_only_nodes=[],
    )
    gen = ArchitectureGenerator(); gen.initialize()
    candidates = gen.execute(base_model, search_space, num_samples=num_candidates,
                             strategy="random", seed=seed)

    # Modulate
    mod = ArchitectureModulator(); mod.initialize()
    modulated = []
    for cand in candidates:
        pruned = mod.execute(cand, search_space, eval_fn=None)
        modulated.extend(pruned)
    logger.info(f"  {len(modulated)} pruned candidates")

    # SWAP-Score all candidates
    zs = ZeroShotEvaluator(config={
        "n_samples": n_swap_samples,
        "regularise": True,
        "reg_lambda": 0.05,
        "include_agg": True,
        "seed": 0,
        "input_shape": list(sample_input.shape),
    })
    zs.initialize()
    logger.info("Computing SWAP scores...")
    swap_ranked = zs.rank(modulated)  # [(model, swap_score)] sorted desc

    # ORT accuracy for all (zero-shot proxy for finetuned accuracy)
    hw = HardwareEvaluator(config={
        "backend": "cpu", "num_threads": 4,
        "warmup_runs": 3, "benchmark_runs": 20,
        "score_weights": {"accuracy": 1.0, "latency_ms": -0.1},
    })
    hw.initialize()

    records = []
    for rank_idx, (model, swap_score) in enumerate(swap_ranked):
        logger.info(f"  [{rank_idx+1}/{len(swap_ranked)}] {model.name}")
        try:
            result = hw.execute(model, sample_input, loader)
            records.append({
                "name":       model.name,
                "swap_score": swap_score,
                "swap_rank":  rank_idx + 1,
                "ort_acc":    result.accuracy * 100,
                "params":     result.parameters,
                "flops":      model.flops,
                "latency_ms": result.latency_ms,
            })
        except Exception as e:
            logger.warning(f"  Eval failed: {e}")

    if len(records) < 3:
        logger.error("Too few valid candidates for Spearman ρ")
        return {}

    # Spearman ρ: SWAP rank vs ORT accuracy rank
    from scipy.stats import spearmanr
    swap_ranks = [r["swap_rank"] for r in records]
    # Higher ORT acc = lower rank (rank 1 = best)
    ort_accs   = [r["ort_acc"] for r in records]
    ort_ranks  = [sorted(ort_accs, reverse=True).index(a) + 1 for a in ort_accs]

    rho, pval = spearmanr(swap_ranks, ort_ranks)

    logger.info(f"\nSpearman ρ (SWAP rank vs ORT accuracy rank): {rho:.4f}  p={pval:.4f}")
    logger.info(f"  n={len(records)} candidates")
    logger.info(f"  Interpretation: {'strong' if abs(rho)>0.7 else 'moderate' if abs(rho)>0.4 else 'weak'} correlation")

    return {
        "spearman_rho":  rho,
        "p_value":       pval,
        "n_candidates":  len(records),
        "candidates":    records,
        "interpretation": "SWAP-Score is a valid zero-shot proxy" if rho > 0.5
                          else "weak correlation — SWAP proxy needs validation",
    }


def main():
    p = argparse.ArgumentParser(description="Compute Spearman ρ between SWAP rank and accuracy")
    p.add_argument("--model",       default="pointnet", choices=["pointnet"])
    p.add_argument("--candidates",  type=int, default=6)
    p.add_argument("--eval-samples",type=int, default=2468)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--out",         default=str(OUT_DIR / "spearman_rho.json"))
    args = p.parse_args()

    import argparse as _ap
    from torch.utils.data import DataLoader
    from data_utils.ModelNetDataLoader import ModelNetDataLoader

    base_onnx = NASKIT_ROOT / "submodules/vision-onnx-models/submodules/pointnet_pointnet2" / \
                "log/classification/pointnet_c40/onnx/pointnet_cls_c40_n1024_opset13.onnx"

    ns = _ap.Namespace(num_point=1024, use_uniform_sample=False,
                       use_normals=False, num_category=40)
    from torch.utils.data import Subset
    ds = ModelNetDataLoader(root=str(MODELNET_DATA), args=ns, split="test", process_data=False)
    n  = args.eval_samples
    if n < len(ds):
        idx = np.random.default_rng(args.seed).choice(len(ds), n, replace=False).tolist()
        ds  = Subset(ds, idx)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
    sample_input = np.random.default_rng(args.seed).standard_normal((1, 3, 1024)).astype(np.float32)

    result = run_spearman(
        base_onnx    = base_onnx,
        loader       = loader,
        sample_input = sample_input,
        num_candidates = args.candidates,
        seed         = args.seed,
    )

    if result:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        logger.info(f"\nSaved: {args.out}")
        print(f"\n{'='*60}")
        print(f"  Spearman ρ  : {result['spearman_rho']:.4f}")
        print(f"  p-value     : {result['p_value']:.4f}")
        print(f"  n           : {result['n_candidates']}")
        print(f"  {result['interpretation']}")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
