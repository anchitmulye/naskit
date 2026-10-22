"""
Test CIFAR-10 models — PyTorch checkpoint vs ONNX model accuracy.

Uses the chenyaofo/pytorch-cifar-models weights (downloaded automatically).
CIFAR-10 test set is loaded via torchvision (downloaded to --data-root if absent).

Usage (from naskit/)
-----
    # ONNX only:
    python models/scripts/test_cifar.py --model resnet56 --onnx models/base/cifar_resnet56.onnx

    # PyTorch only:
    python models/scripts/test_cifar.py --model resnet56 --pytorch

    # Both (comparison):
    python models/scripts/test_cifar.py --model resnet56 \
        --pytorch --onnx models/base/cifar_resnet56.onnx

    # VGG-16-BN:
    python models/scripts/test_cifar.py --model vgg16_bn \
        --pytorch --onnx models/base/cifar_vgg16.onnx
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent           # models/scripts/
sys.path.insert(0, str(HERE))
SCRIPTS = Path(__file__).resolve().parent             # models/scripts
MODELS = SCRIPTS.parent                               # models
ROOT = MODELS.parent    
DATA = ROOT / "data/cifar10"       # set by --data

WEIGHTS_DIR = HERE / "weights"
CLASS_NAMES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


# ── Data loader ───────────────────────────────────────────────────────────────

def build_loader(data_root: str, batch_size: int):
    import torchvision
    import torchvision.transforms as T
    import torch.utils.data

    # Normalisation matching chenyaofo training
    mean = (0.4914, 0.4822, 0.4465)
    std  = (0.2023, 0.1994, 0.2010)
    transform = T.Compose([T.ToTensor(), T.Normalize(mean, std)])
    ds = torchvision.datasets.CIFAR10(root=data_root, train=False,
                                      download=True, transform=transform)
    import torch.utils.data as D
    return D.DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=0, drop_last=False)


# ── PyTorch evaluation ────────────────────────────────────────────────────────

def eval_pytorch(model_name: str, data_root: str, batch_size: int) -> dict:
    import torch
    from cifar_models import load

    net = load(model_name, WEIGHTS_DIR).eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net = net.to(device)
    print(f"  Device: {device}")

    loader = build_loader(data_root, batch_size)
    print(f"  Test samples: {len(loader.dataset)}")

    correct = total = 0
    class_correct = np.zeros(10)
    class_total   = np.zeros(10)
    t0 = time.time()

    with torch.no_grad():
        for imgs, labels in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            preds  = net(imgs).argmax(1)
            correct += preds.eq(labels).sum().item()
            total   += labels.size(0)
            for p, g in zip(preds.cpu().numpy(), labels.cpu().numpy()):
                class_correct[g] += int(p == g)
                class_total[g]   += 1

    elapsed   = time.time() - t0
    top1      = 100.0 * correct / total
    class_avg = 100.0 * np.mean(class_correct / class_total)
    return {"top1": top1, "class_avg": class_avg, "total": total,
            "elapsed_s": elapsed,
            "class_correct": class_correct, "class_total": class_total}


# ── ONNX evaluation ───────────────────────────────────────────────────────────

def eval_onnx(onnx_path: str, data_root: str, batch_size: int) -> dict:
    import onnxruntime as ort

    print(f"\n  Loading ONNX: {onnx_path}")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 4
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess     = ort.InferenceSession(onnx_path, sess_options=opts,
                                    providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name
    print(f"  Input : {inp_name}  shape: {sess.get_inputs()[0].shape}")
    print(f"  Output: {sess.get_outputs()[0].name}")

    loader = build_loader(data_root, batch_size)
    print(f"  Test samples: {len(loader.dataset)}")

    correct = total = 0
    class_correct = np.zeros(10)
    class_total   = np.zeros(10)
    t0 = time.time()

    for imgs, labels in loader:
        imgs_np = imgs.numpy().astype(np.float32)
        lbl_np  = labels.numpy()

        logits = sess.run(None, {inp_name: imgs_np})[0]
        preds  = np.argmax(logits, axis=1)

        correct += int((preds == lbl_np).sum())
        total   += len(lbl_np)
        for p, g in zip(preds, lbl_np):
            class_correct[g] += int(p == g)
            class_total[g]   += 1

    elapsed   = time.time() - t0
    top1      = 100.0 * correct / total
    class_avg = 100.0 * np.mean(class_correct / class_total)
    return {"top1": top1, "class_avg": class_avg, "total": total,
            "elapsed_s": elapsed,
            "class_correct": class_correct, "class_total": class_total}


# ── Report ────────────────────────────────────────────────────────────────────

def print_report(model_name: str, pt_res, onnx_res) -> None:
    from cifar10_models import PUBLISHED_TOP1
    published = PUBLISHED_TOP1.get(model_name)
    W = 72
    print("\n" + "=" * W)
    print(f"  CIFAR-10 {model_name.upper()} — EVALUATION REPORT".center(W))
    print("=" * W)

    print(f"\n  {'Metric':<30} {'PyTorch':>12} {'ONNX':>12}  {'Delta':>8}")
    print("  " + "-" * 66)

    def row(label, pv, ov):
        ps = f"{pv:.2f}%" if pv is not None else "       —"
        os = f"{ov:.2f}%" if ov is not None else "       —"
        ds = f"{ov-pv:+.2f}pp" if (pv is not None and ov is not None) else "       —"
        print(f"  {label:<30} {ps:>12} {os:>12}  {ds:>8}")

    row("Top-1 Accuracy",
        pt_res["top1"]      if pt_res  else None,
        onnx_res["top1"]    if onnx_res else None)
    row("Class Average Accuracy",
        pt_res["class_avg"] if pt_res  else None,
        onnx_res["class_avg"] if onnx_res else None)
    if published is not None:
        ref = pt_res["top1"] if pt_res else (onnx_res["top1"] if onnx_res else None)
        pub_delta = f"{ref - published:+.2f}pp" if ref is not None else "—"
        print(f"  {'Published (chenyaofo)':<30} {published:.2f}%{'':>24}  {pub_delta:>8}")

    if pt_res:
        print(f"  {'Eval time (PyTorch)':<30} {pt_res['elapsed_s']:>11.1f}s")
    if onnx_res:
        print(f"  {'Eval time (ONNX)':<30} {onnx_res['elapsed_s']:>11.1f}s")

    print(f"\n  {'Class':<20} {'PT acc':>8} {'ONNX acc':>10} {'Delta':>8}")
    print("  " + "-" * 50)
    res = pt_res or onnx_res
    for i, name in enumerate(CLASS_NAMES):
        pt_a = (100.0 * pt_res["class_correct"][i] / pt_res["class_total"][i]
                if pt_res else None)
        ox_a = (100.0 * onnx_res["class_correct"][i] / onnx_res["class_total"][i]
                if onnx_res else None)
        if pt_a is not None and ox_a is not None:
            print(f"  {name:<20} {pt_a:>7.1f}% {ox_a:>9.1f}% {ox_a-pt_a:>+7.1f}pp")
        elif pt_a is not None:
            print(f"  {name:<20} {pt_a:>7.1f}%          —")
        elif ox_a is not None:
            print(f"  {name:<20}          — {ox_a:>9.1f}%")

    print("\n" + "=" * W)


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description="CIFAR-10: PyTorch vs ONNX accuracy")
    p.add_argument("--model",      required=True, choices=["resnet56", "vgg16_bn"])
    p.add_argument("--pytorch",    action="store_true",
                   help="evaluate the PyTorch checkpoint (downloaded automatically)")
    p.add_argument("--onnx",       default=None,
                   help="ONNX model path — omit to skip")
    p.add_argument("--data-root",  default=DATA,
                   help="CIFAR-10 root (torchvision downloads if absent)")
    p.add_argument("--batch-size", type=int, default=256)
    args = p.parse_args()

    if not args.pytorch and not args.onnx:
        p.error("Provide at least one of --pytorch or --onnx")

    data_root = str(Path(args.data_root).expanduser())
    pt_res = onnx_res = None

    if args.pytorch:
        print(f"\n[1/2] Evaluating PyTorch ({args.model})")
        pt_res = eval_pytorch(args.model, data_root, args.batch_size)
        print(f"  PyTorch Top-1: {pt_res['top1']:.2f}%  "
              f"Class avg: {pt_res['class_avg']:.2f}%  ({pt_res['elapsed_s']:.0f}s)")

    if args.onnx:
        step = "2/2" if args.pytorch else "1/1"
        print(f"\n[{step}] Evaluating ONNX: {args.onnx}")
        onnx_res = eval_onnx(args.onnx, data_root, args.batch_size)
        print(f"  ONNX Top-1: {onnx_res['top1']:.2f}%  "
              f"Class avg: {onnx_res['class_avg']:.2f}%  ({onnx_res['elapsed_s']:.0f}s)")

    print_report(args.model, pt_res, onnx_res)


if __name__ == "__main__":
    main()
