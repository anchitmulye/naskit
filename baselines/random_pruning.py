"""
Baseline 1 — Random Channel Pruning.

Randomly selects which channels to prune with no importance scoring.
This is the weakest possible structured pruning baseline — if h3dnas
cannot beat random channel selection, the NAS search adds no value.

Why include this:
  - Establishes the lower bound of the accuracy-compression tradeoff
  - Requires no source code, no data, no importance scoring
  - 100% ONNX-native, fully source-free
  - Standard sanity check baseline in pruning literature

Usage:
    python baselines/random_pruning.py --onnx <path> --prune-ratio 0.32
"""

from __future__ import annotations
import argparse, sys, time
from pathlib import Path

NASKIT_ROOT   = Path(__file__).parent.parent
H3DNAS_ROOT   = NASKIT_ROOT / "submodules/h3dnas"
POINTNET_REPO = NASKIT_ROOT / "submodules/vision-onnx-models/submodules/pointnet_pointnet2"
MODELNET_DATA = POINTNET_REPO / "data/modelnet40_normal_resampled"

sys.path.insert(0, str(H3DNAS_ROOT))
sys.path.insert(0, str(POINTNET_REPO))

import numpy as np
from h3dnas.utils.logger import get_logger

logger = get_logger(__name__)


def random_prune_onnx(base_onnx: Path, prune_ratio: float, seed: int = 42):
    """
    TRUE random channel pruning — randomly selects which channels to remove
    with NO importance scoring whatsoever.

    Technical difference from Uniform L1:
      - Uniform L1: ranks channels by L1 norm of weights, prunes lowest
      - Random:     shuffles channel indices randomly, prunes first N%
      This is the absolute lower bound — no signal, pure chance.

    Implementation: monkey-patches h3dnas's _l1_importance_idx with a random
    version so the graph pruner handles all structural propagation correctly
    (BN, residuals, coupled groups) while using random channel selection.
    """
    import h3dnas.modulator.pruner_ops as _pruner_ops
    from h3dnas.parser.onnx_parser import ONNXParser
    from h3dnas.modulator.architecture_modulator import ArchitectureModulator
    from h3dnas.core.types import SearchSpace

    rng = np.random.default_rng(seed)
    logger.info(f"Random pruning: {base_onnx.name}  ratio={prune_ratio}  seed={seed}")

    # Monkey-patch L1 importance → random importance.
    # ArchitectureModulator calls _l1_importance_idx for channel selection;
    # all structural propagation (BN, residuals, coupled groups) is handled
    # by the modulator itself — we only replace the scoring signal.
    _orig_l1 = _pruner_ops._l1_importance_idx

    def _random_importance_idx(w: np.ndarray, keep_k: int, axis: int = 0) -> np.ndarray:
        n = w.shape[axis]
        return np.sort(rng.choice(n, keep_k, replace=False))

    _pruner_ops._l1_importance_idx = _random_importance_idx

    try:
        parser = ONNXParser(); parser.initialize()
        base_model = parser.execute(str(base_onnx))
        base_params = base_model.parameters
        base_flops  = base_model.flops

        search_space = SearchSpace(
            prune_ratios      = [prune_ratio],
            width_multipliers = [1.0],
            use_sensitivity   = False,
        )
        modulator = ArchitectureModulator()
        candidates = modulator.execute(base_model, search_space, eval_fn=None)
        pruned_model = candidates[0]
    finally:
        _pruner_ops._l1_importance_idx = _orig_l1

    param_red = (1 - pruned_model.parameters / base_params) * 100
    logger.info(f"  Pruned: params={pruned_model.parameters:,}  param_red={param_red:.1f}%")
    return pruned_model, 0.0, base_params, base_flops


def finetune_and_eval(pruned_path: Path, epochs: int, lr: float,
                      batch_size: int, out_path: Path, seed: int = 42) -> dict:
    """Finetune pruned ONNX via onnx2torch and evaluate."""
    import torch
    import torch.nn.functional as F
    import onnx
    import onnx2torch
    import onnxruntime as ort
    from data_utils.ModelNetDataLoader import ModelNetDataLoader
    from torch.utils.data import DataLoader
    import argparse as _ap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load pruned model via onnx2torch
    proto = onnx.load(str(pruned_path))
    model = onnx2torch.convert(proto).to(device)
    n_params = sum(p.numel() for p in model.parameters())

    # Data
    args = _ap.Namespace(num_point=1024, use_uniform_sample=False,
                         use_normals=False, num_category=40)
    train_ds = ModelNetDataLoader(root=str(MODELNET_DATA), args=args,
                                  split="train", process_data=False)
    test_ds  = ModelNetDataLoader(root=str(MODELNET_DATA), args=args,
                                  split="test", process_data=False)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=g)
    test_loader  = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                              num_workers=0)

    def eval_pt(m, loader):
        m.eval(); correct = total = 0
        with torch.no_grad():
            for pts, labels in loader:
                if pts.ndim == 3 and pts.shape[2] == 3:
                    pts = pts.permute(0, 2, 1)
                out = m(pts.float().to(device))
                if isinstance(out, (list, tuple)): out = out[0]
                correct += out.argmax(1).eq(labels.view(-1).long().to(device)).sum().item()
                total += labels.size(0)
        return 100. * correct / total

    zero_shot = eval_pt(model, test_loader)
    logger.info(f"  Zero-shot: {zero_shot:.2f}%")

    # Warmup + cosine LR
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    warmup = min(3, epochs // 5)
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(epochs - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_acc = zero_shot
    best_state = {k: v.clone() for k, v in model.state_dict().items()}

    for ep in range(1, epochs + 1):
        model.train(); t0 = time.time()
        for pts, labels in train_loader:
            if pts.ndim == 3 and pts.shape[2] == 3:
                pts = pts.permute(0, 2, 1)
            out = model(pts.float().to(device))
            if isinstance(out, (list, tuple)): out = out[0]
            loss = F.cross_entropy(out, labels.view(-1).long().to(device))
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        sched.step()
        val = eval_pt(model, test_loader)
        if val > best_acc:
            best_acc = val
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        logger.info(f"  Ep {ep:3d}/{epochs} | val={val:.2f}% | best={best_acc:.2f}% | ({time.time()-t0:.0f}s)")

    model.load_state_dict(best_state)
    model.eval()

    # Export finetuned ONNX
    dummy = torch.zeros(1, 3, 1024)
    torch.onnx.export(model.cpu(), dummy, str(out_path),
                      export_params=True, opset_version=13,
                      do_constant_folding=False, dynamo=False,
                      input_names=["point_cloud"], output_names=["logits"],
                      dynamic_axes={"point_cloud": {0: "batch"}, "logits": {0: "batch"}})

    # ORT eval
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    sess = ort.InferenceSession(str(out_path), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    inp = sess.get_inputs()[0].name
    correct = total = 0
    for pts, labels in test_loader:
        pts_np = pts.numpy().astype(np.float32)
        if pts_np.ndim == 3 and pts_np.shape[2] == 3:
            pts_np = pts_np.transpose(0, 2, 1)
        preds = np.argmax(sess.run(None, {inp: pts_np})[0], axis=1)
        correct += int((preds == labels.numpy().ravel()).sum())
        total += len(labels)
    ort_acc = 100. * correct / total

    return {"zero_shot": zero_shot, "best_acc": best_acc, "ort_acc": ort_acc,
            "params": n_params}


def main(base_onnx: Path, prune_ratio: float, epochs: int,
         lr: float, batch_size: int, out_dir: Path, seed: int = 42) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)
    logger.info("=" * 70)
    logger.info("BASELINE 1: Random Channel Pruning")
    logger.info("=" * 70)
    logger.info(f"  Base ONNX   : {base_onnx}")
    logger.info(f"  Prune ratio : {prune_ratio}  (random channel selection)")
    logger.info(f"  Fine-tune   : {epochs} epochs, lr={lr}")

    pruned, _, base_params, base_flops = random_prune_onnx(base_onnx, prune_ratio, seed)
    base_acc = 90.32  # PointNet ModelNet40 — set from actual ORT eval in run_baselines.py
    pruned_path = out_dir / f"random_pruned_pr{int(prune_ratio*100):02d}.onnx"
    pruned.save(str(pruned_path))

    ft_path = out_dir / f"random_finetuned_pr{int(prune_ratio*100):02d}.onnx"
    metrics = finetune_and_eval(pruned_path, epochs, lr, batch_size, ft_path, seed)

    param_red = (1 - metrics["params"] / base_params) * 100
    flops_red = (1 - pruned.flops / base_flops) * 100 if base_flops else 0

    result = {
        "method":      "Random Pruning",
        "reference":   "baseline (this work)",
        "base_acc":    base_acc * 100,
        "zero_shot":   metrics["zero_shot"],
        "finetuned":   metrics["ort_acc"],
        "delta_pp":    metrics["ort_acc"] - base_acc * 100,
        "params":      metrics["params"],
        "param_red":   param_red,
        "flops_red":   flops_red,
        "prune_ratio": prune_ratio,
    }
    logger.info(f"\n  Base: {result['base_acc']:.2f}%  "
                f"Finetuned: {result['finetuned']:.2f}%  "
                f"Δ={result['delta_pp']:+.2f}pp  "
                f"Param↓={param_red:.1f}%")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Random channel pruning baseline")
    p.add_argument("--onnx",        type=Path, required=True)
    p.add_argument("--prune-ratio", type=float, default=0.32)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--batch-size",  type=int,   default=24)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--out-dir",     type=Path,  default=Path("baselines/outputs"))
    args = p.parse_args()
    main(args.onnx, args.prune_ratio, args.epochs, args.lr,
         args.batch_size, args.out_dir, args.seed)
