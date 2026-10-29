"""CIFAR-10: h3dnas (ONNX only) vs source-code pruners at matched MACs.

Methods, all starting from the same pretrained weights:
  h3dnas            uniform group ratio (ONNX graph only)
  h3dnas_search     per-group search, fitness = validation accuracy (run_nas
                    default)
  h3dnas_search_agree per-group search, fitness = top-1 agreement with the base
                    model on 2000 unlabeled training images (label-free)
  h3dnas_labelfree  per-group search, fitness = output fidelity (cosine) on
                    unlabeled calibration images
  depgraph          Torch-Pruning, group L2 importance, uniform per layer
  depgraph_global   Torch-Pruning, group L2 importance, global ranking
  l1                Torch-Pruning, filter L1 (Li et al. 2017), uniform

Protocol (identical for every method):
  - MAC target set by h3dnas-uniform at --ratio; source-code pruners are
    bisected to within 0.5% of it; MACs counted by h3dnas.core.metrics on the
    exported ONNX of every method
  - BatchNorm re-estimated on the same unlabeled calibration images
  - fine-tuning by h3dnas.finetune (same code, recipe and seed for all;
    torch models are exported to ONNX first), last epoch kept
  - validation split for every search decision; test split read once at the end
  - latency of every final model measured in one interleaved session

    python -m scripts.bench_cifar cifar_resnet56 --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import io
import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import onnx
import torch

from experiments.registry import get, paths
from experiments.pipeline import parse as _parse, write_json as _write
from scripts import MODEL_SCRIPTS

METHODS = ["h3dnas", "h3dnas_search", "h3dnas_search_agree", "h3dnas_labelfree", "depgraph",
           "depgraph_global", "l1"]


def _onnx_of(model: torch.nn.Module):
    from cifar_models import export          # models/scripts
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / "m.onnx"
        export(model, p)
        return _parse(p)


def _tp(model, ratio: float, method: str):
    import torch_pruning as tp
    imp, glob = {"depgraph": (tp.importance.GroupMagnitudeImportance(p=2), False),
                 "depgraph_global": (tp.importance.GroupMagnitudeImportance(p=2), True),
                 "l1": (tp.importance.MagnitudeImportance(p=1, group_reduction="first"), False)}[method]
    last = [m for m in model.modules() if isinstance(m, torch.nn.Linear)][-1]
    tp.pruner.MetaPruner(model, torch.zeros(1, 3, 32, 32), importance=imp, pruning_ratio=ratio,
                         global_pruning=glob, ignored_layers=[last]).step()
    return model


def _matched_tp(arch: str, method: str, target_macs: int):
    from cifar_models import load
    wdir = MODEL_SCRIPTS / "weights"
    lo, hi, best = 0.0, 0.95, None
    for _ in range(16):
        mid = (lo + hi) / 2
        m = _onnx_of(_tp(load(arch, wdir), mid, method))
        best = (mid, m)
        if abs(m.macs - target_macs) / target_macs < 0.005:
            break
        lo, hi = (mid, hi) if m.macs > target_macs else (lo, mid)
    return best


def run(model_name: str, seeds: List[int], ratio: float, epochs: int = None,
        methods: List[str] = None) -> None:
    from h3dnas.core.latency import measure
    from h3dnas.core.metrics import per_sample_inputs
    from h3dnas.core.inputs import resolve_input_spec
    from h3dnas.core.types import SearchSpace
    from h3dnas.datapipe import eval_accuracy, feeds_from_loader
    from h3dnas.finetune import finetune
    from h3dnas.generator.group_evolution import GroupSearchConfig, search_group_ratios
    from h3dnas.modulator.architecture_modulator import ArchitectureModulator
    from h3dnas.modulator.bn_recalib import recalibrate_bn

    arches = {"cifar_resnet56": "resnet56", "cifar_vgg16": "vgg16_bn"}
    if model_name not in arches:
        raise SystemExit(f"bench_cifar supports {sorted(arches)} (it rebuilds the PyTorch model for "
                         f"the source-code pruners), not {model_name!r}")
    spec = get(model_name)
    arch = arches[model_name]
    methods = methods or METHODS
    device = spec.resolved()["finetune"]["device"]          # fine-tune where h3dnas.yaml says
    recipe = dict(spec.recipe)
    if epochs:
        recipe["epochs"] = epochs
    # distillation as in the pipeline (defaults.finetune.distill, a recipe's own value wins),
    # applied to every method alike with the unpruned model as teacher
    recipe["distill"] = bool(recipe.get("distill", spec.resolved()["finetune"].get("distill", False)))
    for seed in seeds:
        tag = "" if methods == METHODS else "_" + "-".join(methods)
        out = paths().results / "bench_cifar" / f"{model_name}_r{int(ratio * 100)}_seed{seed}{tag}.json"
        if out.exists():
            print(f"[bench] {out.name}: done")
            continue
        torch.manual_seed(seed)
        base = _parse(spec.onnx)
        dm = spec.data(seed)
        calib = feeds_from_loader(base, dm.calib_loader(), 4)
        val = dm.val_loader()
        t0 = time.time()

        mod = ArchitectureModulator()
        mod.initialize()
        uni = mod.execute(base, SearchSpace(prune_ratios=[ratio]), calib_feeds=calib)[0]
        target = uni.macs
        pruned = {"h3dnas": uni}
        cfg = GroupSearchConfig(population=16, generations=10, seed=seed)   # = run_nas defaults
        if "h3dnas_search" in methods:
            best, _ = search_group_ratios(base, max_macs=target, config=cfg, calib_feeds=calib,
                                          eval_fn=lambda m: eval_accuracy(m, val))
            pruned["h3dnas_search"] = best.model
        if "h3dnas_search_agree" in methods:
            from h3dnas.core.fidelity import AgreementScorer
            agree = AgreementScorer(base, feeds_from_loader(base, dm.pool_loader(), n_samples=2000))
            best, _ = search_group_ratios(base, max_macs=target, config=cfg, calib_feeds=calib,
                                          eval_fn=agree.score)
            pruned["h3dnas_search_agree"] = best.model
        if "h3dnas_labelfree" in methods:
            best, _ = search_group_ratios(base, max_macs=target, config=cfg, calib_feeds=calib)
            pruned["h3dnas_labelfree"] = best.model
        tp_ratio = {}
        for meth in [m for m in ("depgraph", "depgraph_global", "l1") if m in methods]:
            r, m = _matched_tp(arch, meth, target)
            recalibrate_bn(m, calib)
            pruned[meth], tp_ratio[meth] = m, r

        rec = {"model": model_name, "seed": seed, "ratio": ratio, "recipe": recipe,
               "finetune_device": device,
               "base": {"params": base.parameters, "macs": base.macs,
                        "val": eval_accuracy(base, val)},
               "data": dm.describe(), "tp_ratio": tp_ratio, "methods": {}}
        tuned = {}
        for meth in methods:
            m = pruned[meth]
            kw = dict(recipe)
            if kw.pop("distill"):
                kw["distill"] = base
            r = finetune(m, dm, seed=seed, device=device, **kw)
            tuned[meth] = r.model
            rec["methods"][meth] = {"params": m.parameters, "macs": m.macs,
                                    "val_before_ft": eval_accuracy(m, val),
                                    "val_after_ft": r.val_accuracy, "curve": r.curve}
            print(f"[bench] {model_name} seed {seed} {meth}: MACs {m.macs:,} "
                  f"val {rec['methods'][meth]['val_before_ft']*100:.2f} -> {r.val_accuracy*100:.2f}")

        _, feed = per_sample_inputs(resolve_input_spec(base.proto))
        lat = measure({"base": base, **tuned}, feed, threads=4, rounds=5, runs_per_round=20)
        rec["latency"] = {k: {"median_ms": v.median_ms, "iqr_ms": v.iqr_ms}
                          for k, v in lat.results.items()}
        rec["latency_device"] = lat.device
        # Test split: read once, after every decision above.
        rec["base"]["test"] = eval_accuracy(base, dm.test_loader())
        for meth, m in tuned.items():
            rec["methods"][meth]["test"] = eval_accuracy(m, dm.test_loader())
            rec["methods"][meth]["speedup"] = (lat.results["base"].median_ms /
                                               lat.results[meth].median_ms)
        rec["seconds"] = round(time.time() - t0, 1)
        _write(out, rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--ratio", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--methods", nargs="+", default=None, choices=METHODS)
    a = ap.parse_args()
    run(a.model, a.seeds, a.ratio, a.epochs, a.methods)
