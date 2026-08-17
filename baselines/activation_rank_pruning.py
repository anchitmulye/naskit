"""
Baseline 3 — Activation Rank Pruning (inspired by HRank, Lin et al., CVPR 2020).

Ranks channels by the average numerical rank of their output activation maps
across N random inputs. Low-rank activations carry redundant information.

    rank(channel_j) = mean_over_N( matrix_rank(F_j(x_i)) )

This is a SOURCE-FREE adaptation of the HRank principle [Lin et al., CVPR 2020]:
  - Original HRank: requires PyTorch source + real training data
  - This implementation: ONNX-native, random inputs only, no source code needed

Adaptation notes (for paper transparency):
  - Random inputs used instead of training data (source-free requirement)
  - 1D Conv outputs [C, L] reshaped to [sqrt(L), sqrt(L)] for rank computation
  - Gemm/Linear layers: L1 norm used as proxy (no spatial dims for rank)

Reference: HRank — https://arxiv.org/abs/2002.10179 (Lin et al., CVPR 2020)
"""

from __future__ import annotations
import argparse, copy, sys, time, tempfile, os
from pathlib import Path
from typing import Dict, List, Tuple

NASKIT_ROOT   = Path(__file__).parent.parent
H3DNAS_ROOT   = NASKIT_ROOT / "submodules/h3dnas"
POINTNET_REPO = NASKIT_ROOT / "submodules/vision-onnx-models/submodules/pointnet_pointnet2"
MODELNET_DATA = POINTNET_REPO / "data/modelnet40_normal_resampled"

sys.path.insert(0, str(H3DNAS_ROOT))
sys.path.insert(0, str(POINTNET_REPO))

import numpy as np
import onnx
from onnx import numpy_helper
import onnxruntime as ort

from h3dnas.parser.onnx_parser import ONNXParser
from h3dnas.modulator.topology_analyser import TopologyAnalyser
from h3dnas.core.types import ONNXModel
from h3dnas.utils.logger import get_logger

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# HRank: feature-map rank importance
# ---------------------------------------------------------------------------

def _feature_map_rank(feature_map: np.ndarray) -> float:
    """
    Compute the numerical rank of a single channel's feature map.

    feature_map : shape [H, W] for 2D conv, [L] for 1D conv
                  (one channel from one sample, spatial dims only)

    For 1D feature maps (PointNet Conv1d outputs shape [L]):
      Reshape [L] → [sqrt(L), sqrt(L)] patch matrix and compute matrix rank.
      This allows rank to vary across channels: a near-constant channel
      (all values ≈ same) will have rank ≈ 1, while an active channel with
      diverse values will have higher rank — the key HRank discriminator.

    For 2D feature maps shape [H, W]:
      Compute matrix rank directly via numpy.linalg.matrix_rank.

    Returns a non-negative float (rank value).
    """
    fm = feature_map.astype(np.float64)
    if fm.ndim == 1:
        L = fm.shape[0]
        # Reshape to a square-ish matrix to get a meaningful rank measure.
        # For L=1024: reshape to [32, 32]; rank ranges from 1 (flat) to 32 (diverse).
        side = max(2, int(np.sqrt(L)))
        # Trim to side*side and reshape
        trimmed = fm[:side * side].reshape(side, side)
        try:
            return float(np.linalg.matrix_rank(trimmed))
        except np.linalg.LinAlgError:
            return float(np.abs(fm).mean())
    elif fm.ndim == 2:
        try:
            return float(np.linalg.matrix_rank(fm))
        except np.linalg.LinAlgError:
            return float(min(fm.shape))
    else:
        # Flatten spatial dims, treat as matrix [H, W*D...]
        h = fm.shape[0]
        fm2d = fm.reshape(h, -1)
        try:
            return float(np.linalg.matrix_rank(fm2d))
        except np.linalg.LinAlgError:
            return float(h)


def _hrank_importance(activations: np.ndarray) -> np.ndarray:
    """
    Compute HRank importance scores for each channel.

    activations : shape [N, C, *spatial] — N samples, C channels, spatial dims
                  For Gemm/linear: shape [N, C] — spatial already pooled

    Returns hrank_score per channel, shape [C].
    Higher rank = more informative = keep (same direction as L1, F-stat).

    Implementation note: for PointNet Conv1d, the activations collected from
    ORT have shape [1, C, L] per sample (batch=1). After squeeze we get [C, L].
    We compute rank per channel (each [L] vector) then average over N samples.
    """
    if activations.ndim == 2:
        # Already pooled: [N, C] — use L1 as rank proxy (no spatial dims)
        return np.abs(activations).mean(axis=0)

    N = activations.shape[0]
    C = activations.shape[1]
    scores = np.zeros(C, dtype=np.float64)

    for n in range(N):
        for c in range(C):
            fm = activations[n, c]   # [*spatial]
            scores[c] += _feature_map_rank(fm)

    return (scores / max(N, 1)).astype(np.float32)


def _collect_conv_activations(
    model: ONNXModel,
    free_node_names: List[str],
    inputs: List[np.ndarray],
) -> Dict[str, np.ndarray]:
    """
    Run each input through ORT and collect raw channel activations
    (with spatial dims preserved) at the output of each free Conv node.

    Returns {node_name: activations [N, C, *spatial]}
    """
    name_to_output: Dict[str, str] = {}
    for node in model.proto.graph.node:
        if node.name in free_node_names and node.output:
            name_to_output[node.name] = node.output[0]

    if not name_to_output:
        return {}

    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=False) as f:
        tmp = f.name
    try:
        model.save(tmp)
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
        sess = ort.InferenceSession(tmp, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name

        node_acts: Dict[str, List[np.ndarray]] = {n: [] for n in name_to_output}

        for x in inputs:
            try:
                outputs = sess.run(list(name_to_output.values()), {inp_name: x})
                for node_name, act in zip(name_to_output.keys(), outputs):
                    # act shape: [1, C, *spatial] — drop batch dim
                    act_np = act.squeeze(0)   # [C, *spatial]
                    node_acts[node_name].append(act_np)
            except Exception as exc:
                logger.debug(f"  Activation collection failed: {exc}")
                continue

        result = {}
        for node_name, acts_list in node_acts.items():
            if acts_list:
                result[node_name] = np.stack(acts_list)   # [N, C, *spatial]
        return result
    finally:
        try:
            os.unlink(tmp)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# HRank pruning — apply rank importance to free nodes
# ---------------------------------------------------------------------------

def hrank_prune_onnx(
    base_onnx: Path,
    prune_ratio: float,
    n_samples: int = 64,
    seed: int = 42,
) -> Tuple[ONNXModel, float, int, int]:
    """
    Apply HRank importance pruning to the ONNX model.

    1. Identify free nodes via TopologyAnalyser
    2. Feed n_samples random inputs through ORT, collect channel activations
       (with spatial dims preserved)
    3. Compute average feature-map rank per channel per free node
    4. Select top-(1-prune_ratio) channels by HRank score
    5. Apply same graph surgery as h3dnas (Cin propagation, BN alignment)
    """
    import argparse as _ap
    from torch.utils.data import DataLoader
    from data_utils.ModelNetDataLoader import ModelNetDataLoader
    from h3dnas.core.nas_pipeline import run_nas, NASConfig, _eval_fn

    logger.info(f"HRank pruning: {base_onnx.name}  ratio={prune_ratio}  "
                f"n_samples={n_samples}")

    parser = ONNXParser(); parser.initialize()
    model  = parser.execute(str(base_onnx))

    ta = TopologyAnalyser()
    protected, cin_only, free_nodes = ta.analyse(model)
    logger.info(f"  Free nodes: {len(free_nodes)}  Protected: {len(protected)}")

    # Random inputs (no labels needed — key advantage over F-stat)
    rng    = np.random.default_rng(seed)
    inputs = [rng.standard_normal((1, 3, 1024)).astype(np.float32)
              for _ in range(n_samples)]

    # Collect raw activations (with spatial dims)
    logger.info("  Collecting channel activations at free nodes...")
    node_acts = _collect_conv_activations(model, free_nodes, inputs)
    logger.info(f"  Collected activations for {len(node_acts)} nodes")

    # Compute HRank scores
    hrank_scores: Dict[str, np.ndarray] = {}
    for node_name in free_nodes:
        if node_name in node_acts:
            acts = node_acts[node_name]   # [N, C, *spatial]
            logger.debug(f"  HRank {node_name}: acts shape={acts.shape}")
            hrank_scores[node_name] = _hrank_importance(acts)
        else:
            logger.debug(f"  No activations for {node_name} — L1 fallback")

    # Inject HRank scores by patching the pruner's importance function
    import h3dnas.modulator.pruner_ops as pruner_ops
    original_l1 = pruner_ops._l1_importance_idx

    def hrank_importance_idx(W: np.ndarray, k: int,
                              node_name: str = "") -> np.ndarray:
        if node_name and node_name in hrank_scores:
            scores = hrank_scores[node_name]
            if len(scores) >= k:
                idx = np.argsort(scores)[::-1][:k]
                return np.sort(idx)
        return original_l1(W, k)

    pruner_ops._l1_importance_idx = hrank_importance_idx

    try:
        args = _ap.Namespace(num_point=1024, use_uniform_sample=False,
                             use_normals=False, num_category=10)
        ds = ModelNetDataLoader(root=str(MODELNET_DATA), args=args,
                                split="test", process_data=False)
        loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)
        import random as _random, torch as _torch
        _random.seed(seed); np.random.seed(seed); _torch.manual_seed(seed)

        config = NASConfig(
            base_onnx         = base_onnx,
            sample_input      = np.random.randn(1, 3, 1024).astype(np.float32),
            data_module       = _make_dm(loader, np.random.randn(1, 3, 1024).astype(np.float32)),
            num_candidates    = 1,
            prune_ratios      = [prune_ratio],
            width_multipliers = [1.0],
            strategy          = "random",
            use_sensitivity   = False,
            seed              = seed,
            enable_graph_mutations = False,
            enable_zero_shot  = False,
        )
        result = run_nas(config)
    finally:
        pruner_ops._l1_importance_idx = original_l1

    if not result.candidates:
        raise RuntimeError("No candidates from HRank pruning")

    pruned = result.candidates[0]["model"]
    logger.info(
        f"  Pruned: params={pruned.parameters:,}  "
        f"param_red={(1-pruned.parameters/result.base_params)*100:.1f}%"
    )
    return pruned, result.base_acc, result.base_params, result.base_flops


def _make_dm(loader, sample_input):
    _si = sample_input
    class _DM:
        def eval_loader(self): return loader
        def full_loader(self): return loader
        def sample_input(self): return _si
    return _DM()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(base_onnx: Path, prune_ratio: float, epochs: int,
         lr: float, batch_size: int, n_samples: int, out_dir: Path) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BASELINE 3: HRank Pruning (Lin et al., CVPR 2020)")
    logger.info("As used in CP³ (Huang et al., CVPR 2023) — SOTA for PointNet++")
    logger.info("=" * 70)
    logger.info(f"  Base ONNX   : {base_onnx}")
    logger.info(f"  Prune ratio : {prune_ratio}")
    logger.info(f"  n_samples   : {n_samples}  (for rank computation, no labels)")
    logger.info(f"  Fine-tune   : {epochs} epochs, lr={lr}")

    pruned_model, base_acc, base_params, base_flops = hrank_prune_onnx(
        base_onnx, prune_ratio, n_samples=n_samples, seed=42
    )
    pruned_path = out_dir / f"hrank_pruned_pr{int(prune_ratio*100):02d}.onnx"
    pruned_model.save(str(pruned_path))
    logger.info(f"  Pruned model saved: {pruned_path}")

    sys.path.insert(0, str(Path(__file__).parent))
    from uniform_pruning import finetune
    ft_path = out_dir / f"hrank_finetuned_pr{int(prune_ratio*100):02d}.onnx"
    metrics = finetune(pruned_path, prune_ratio, epochs, lr, batch_size, ft_path)

    param_red = (1 - pruned_model.parameters / base_params) * 100
    flops_red = (1 - pruned_model.flops / base_flops) * 100 if base_flops else 0

    result = {
        "method":      "HRank Pruning (Lin et al., CVPR 2020)",
        "reference":   "Lin et al., CVPR 2020 / used in CP3 CVPR 2023",
        "base_acc":    base_acc * 100,
        "zero_shot":   metrics["pre_acc"],
        "finetuned":   metrics["best_acc"],
        "ort_acc":     metrics["ort_acc"],
        "params":      pruned_model.parameters,
        "param_red":   param_red,
        "flops_red":   flops_red,
        "prune_ratio": prune_ratio,
        "labels_needed": "None (random inputs only)",
    }

    logger.info("\n" + "=" * 70)
    logger.info("HRANK PRUNING SUMMARY")
    logger.info("=" * 70)
    logger.info(f"  Base accuracy   : {result['base_acc']:.2f}%")
    logger.info(f"  Zero-shot       : {result['zero_shot']:.2f}%")
    logger.info(f"  Fine-tuned      : {result['finetuned']:.2f}%")
    logger.info(f"  ORT verified    : {result['ort_acc']:.2f}%")
    logger.info(f"  Param reduction : {param_red:.1f}%  ({pruned_model.parameters:,} params)")
    logger.info(f"  FLOPs reduction : {flops_red:.1f}%")
    logger.info("=" * 70)
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="HRank pruning baseline (CVPR 2020 / CP3 CVPR 2023)")
    p.add_argument("--onnx",        type=Path,  required=True)
    p.add_argument("--prune-ratio", type=float, default=0.1)
    p.add_argument("--epochs",      type=int,   default=15)
    p.add_argument("--lr",          type=float, default=5e-4)
    p.add_argument("--batch-size",  type=int,   default=24)
    p.add_argument("--n-samples",   type=int,   default=64)
    p.add_argument("--out-dir",     type=Path,  default=Path("baselines/outputs"))
    args = p.parse_args()
    main(args.onnx, args.prune_ratio, args.epochs, args.lr,
         args.batch_size, args.n_samples, args.out_dir)
