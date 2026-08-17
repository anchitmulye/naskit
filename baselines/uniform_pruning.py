"""
Baseline 1 — Uniform L1 Channel Pruning.

Applies the same L1 importance pruning ratio to every free Conv/Gemm node.
No sensitivity analysis, no width multipliers, no NAS search.
This is the simplest possible structured pruning baseline:
  - identify free nodes via TopologyAnalyser (same as h3dnas)
  - select top-(1-prune_ratio) channels by L1 norm of weights
  - propagate Cin indices downstream (same graph surgery as h3dnas)
  - fine-tune for N epochs
  - report accuracy

Reference comparison point: shows what plain uniform pruning achieves at
the same parameter budget as h3dnas. If h3dnas beats it, the NAS search
and sensitivity-guided allocation are justified.

This is a fair comparison because:
  - same topology constraint detection (TopologyAnalyser)
  - same ORT evaluation
  - same fine-tuning setup (AdamW, warmup+cosine LR, 50 epochs)
  - same dataset (ModelNet40, 2468 test samples)
"""

from __future__ import annotations
import argparse, sys, time, math
from pathlib import Path

NASKIT_ROOT   = Path(__file__).parent.parent
H3DNAS_ROOT   = NASKIT_ROOT / "submodules/h3dnas"
POINTNET_REPO = NASKIT_ROOT / "submodules/vision-onnx-models/submodules/pointnet_pointnet2"
MODELNET_DATA = POINTNET_REPO / "data/modelnet40_normal_resampled"

sys.path.insert(0, str(H3DNAS_ROOT))
sys.path.insert(0, str(POINTNET_REPO))

import numpy as np
import onnx, onnxruntime as ort
from onnx import numpy_helper

from h3dnas.parser.onnx_parser import ONNXParser
from h3dnas.modulator.topology_analyser import TopologyAnalyser
from h3dnas.core.nas_pipeline import _eval_fn
from h3dnas.utils.logger import get_logger


def _wrap_loader(loader, sample_input):
    _si = sample_input
    class _DM:
        def eval_loader(self): return loader
        def full_loader(self): return loader
        def sample_input(self): return _si
    return _DM()

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pure ONNX uniform pruning — no PyTorch needed for the pruning step
# ---------------------------------------------------------------------------

def uniform_prune_onnx(base_onnx: Path, prune_ratio: float) -> "ONNXModel":
    """
    Apply uniform L1 channel pruning to all free nodes in the ONNX model.

    Steps:
      1. Parse ONNX → ONNXModel with shape inference
      2. TopologyAnalyser → identify free nodes
      3. For each free node: select top-(1-prune_ratio) channels by L1 norm
      4. Propagate Cin indices downstream (exact same logic as h3dnas)
      5. Return pruned ONNXModel

    This reuses h3dnas's own modulator with use_sensitivity=False and
    a single prune_ratio — giving a fair apples-to-apples comparison.
    """
    from h3dnas.core.nas_pipeline import run_nas, NASConfig
    from h3dnas.core.types import SearchSpace
    import argparse as _ap
    from torch.utils.data import DataLoader, Subset
    from data_utils.ModelNetDataLoader import ModelNetDataLoader

    logger.info(f"Uniform pruning: {base_onnx.name}  ratio={prune_ratio}")

    # Use h3dnas pipeline with single candidate, no width scaling, no sensitivity
    args = _ap.Namespace(num_point=1024, use_uniform_sample=False,
                         use_normals=False, num_category=10)
    ds = ModelNetDataLoader(root=str(MODELNET_DATA), args=args,
                            split="test", process_data=False)
    loader = DataLoader(ds, batch_size=16, shuffle=False, num_workers=0)

    import random as _random, torch as _torch
    _random.seed(42); np.random.seed(42); _torch.manual_seed(42)
    config = NASConfig(
        base_onnx         = base_onnx,
        sample_input      = np.random.randn(1, 3, 1024).astype(np.float32),
        data_module       = _wrap_loader(loader, np.random.randn(1, 3, 1024).astype(np.float32)),
        num_candidates    = 1,
        prune_ratios      = [prune_ratio],
        width_multipliers = [1.0],          # no width scaling
        strategy          = "random",
        use_sensitivity   = False,          # UNIFORM — same ratio everywhere
        seed              = 42,
        enable_graph_mutations = False,
        enable_zero_shot  = False,
    )

    from h3dnas.core.nas_pipeline import run_nas
    result = run_nas(config)

    if not result.candidates:
        raise RuntimeError("No candidates produced by uniform pruning")

    # Return the single pruned candidate
    pruned = result.candidates[0]["model"]
    logger.info(
        f"  Pruned: params={pruned.parameters:,}  "
        f"flops={pruned.flops:,}  "
        f"param_red={(1-pruned.parameters/result.base_params)*100:.1f}%"
    )
    return pruned, result.base_acc, result.base_params, result.base_flops


# ---------------------------------------------------------------------------
# Fine-tuning (shared with other baselines)
# ---------------------------------------------------------------------------


def finetune(pruned_model_path: Path, epochs: int = 50,
             lr: float = 3e-4, batch_size: int = 24,
             out_path: Path = None, seed: int = 42) -> dict:
    """Fine-tune a pruned ONNX via onnx2torch — no source code needed."""
    import torch
    import torch.nn.functional as F
    import onnx
    import onnx2torch
    import onnxruntime as ort
    from data_utils.ModelNetDataLoader import ModelNetDataLoader
    from torch.utils.data import DataLoader
    import argparse as _ap

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    proto = onnx.load(str(pruned_model_path))
    model = onnx2torch.convert(proto).to(device)

    ns = _ap.Namespace(num_point=1024, use_uniform_sample=False,
                       use_normals=False, num_category=40)
    train_ds = ModelNetDataLoader(root=str(MODELNET_DATA), args=ns,
                                  split="train", process_data=False)
    test_ds  = ModelNetDataLoader(root=str(MODELNET_DATA), args=ns,
                                  split="test",  process_data=False)
    g = torch.Generator(); g.manual_seed(seed)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=0, generator=g)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
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
                total   += labels.size(0)
        return 100. * correct / total

    pre_acc = eval_pt(model, test_loader)
    logger.info(f"  Zero-shot: {pre_acc:.2f}%")

    # Warmup + cosine LR
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    warmup = min(3, epochs // 5)
    def lr_lambda(ep):
        if ep < warmup: return (ep + 1) / max(warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * (ep - warmup) / max(epochs - warmup, 1)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    best_acc = pre_acc
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

    ort_acc = best_acc
    if out_path:
        dummy = torch.zeros(1, 3, 1024)
        torch.onnx.export(model.cpu(), dummy, str(out_path),
                          export_params=True, opset_version=13,
                          do_constant_folding=False, dynamo=False,
                          input_names=["point_cloud"], output_names=["logits"],
                          dynamic_axes={"point_cloud": {0: "batch"}, "logits": {0: "batch"}})
        opts = ort.SessionOptions(); opts.intra_op_num_threads = 1
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
            total   += len(labels)
        ort_acc = 100. * correct / total

    return {"pre_acc": pre_acc, "best_acc": best_acc, "ort_acc": ort_acc}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(base_onnx: Path, prune_ratio: float, epochs: int,
         lr: float, batch_size: int, out_dir: Path) -> dict:

    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("BASELINE 1: Uniform L1 Channel Pruning")
    logger.info("=" * 70)
    logger.info(f"  Base ONNX   : {base_onnx}")
    logger.info(f"  Prune ratio : {prune_ratio}  (uniform, all free nodes)")
    logger.info(f"  Fine-tune   : {epochs} epochs, lr={lr}")

    # Step 1 — prune
    pruned_model, base_acc, base_params, base_flops = uniform_prune_onnx(
        base_onnx, prune_ratio
    )
    pruned_path = out_dir / f"uniform_pruned_pr{int(prune_ratio*100):02d}.onnx"
    pruned_model.save(str(pruned_path))
    logger.info(f"  Pruned model saved: {pruned_path}")

    # Step 2 — fine-tune
    ft_path = out_dir / f"uniform_finetuned_pr{int(prune_ratio*100):02d}.onnx"
    metrics = finetune(pruned_path, epochs=epochs, lr=lr,
                       batch_size=batch_size, out_path=ft_path)
    param_red = (1 - pruned_model.parameters / base_params) * 100
    flops_red = (1 - pruned_model.flops / base_flops) * 100 if base_flops else 0

    result = {
        "method":      "Uniform L1 Pruning",
        "reference":   "baseline (this work)",
        "base_acc":    base_acc * 100,
        "zero_shot":   metrics["pre_acc"],
        "finetuned":   metrics["best_acc"],
        "ort_acc":     metrics["ort_acc"],
        "params":      pruned_model.parameters,
        "param_red":   param_red,
        "flops_red":   flops_red,
        "prune_ratio": prune_ratio,
    }

    logger.info("\n" + "=" * 70)
    logger.info("UNIFORM PRUNING SUMMARY")
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
    p = argparse.ArgumentParser(description="Uniform L1 pruning baseline")
    p.add_argument("--onnx",        type=Path, required=True)
    p.add_argument("--prune-ratio", type=float, default=0.32)
    p.add_argument("--epochs",      type=int,   default=50)
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--batch-size",  type=int,   default=24)
    p.add_argument("--out-dir",     type=Path,  default=Path("baselines/outputs"))
    args = p.parse_args()
    main(args.onnx, args.prune_ratio, args.epochs, args.lr, args.batch_size, args.out_dir)
