"""Structural graph mutation on a distributed ONNX file, measured rather than assumed.

    python -m scripts.mutate_study fashion_mnist --seed 105

Width search changes how many channels a layer has. This script changes *what the layers are*:
operator substitution (activation, pooling, aggregation), kernel size, and the two structural
decompositions GraphMutator implements (GhostConv, depthwise+pointwise). Each mutation family
is applied on its own so the effect is attributable, then all of them together.

Every variant is scored the same way a search candidate is: BatchNorm statistics are
re-estimated on training images, then validation accuracy is read from one forward pass. The
test split is read once, at the end, for the variants that are reported.

Why this matters: Level-1 mutations keep every trained weight (Relu->Gelu moves no tensor), so
they are free to try. Level-2 decompositions introduce weights that were never trained --
GhostConv's cheap branch, the factorised DW/PW pair -- so a zero-shot drop is expected and the
question is whether a short fine-tune recovers it. The table separates the two.
"""

from __future__ import annotations

import argparse
import json
import time
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
H3DNAS = ROOT / "submodules" / "h3dnas"
H3DNAS_MODELS = ROOT / "models" / "h3dnas"
sys.path.insert(0, str(H3DNAS))

from experiments.registry import get, paths
from experiments.pipeline import parse as _parse


FAMILIES = {
    "activation": {"activation_search": True},
    "pooling": {"pooling_search": True},
    "aggregation": {"aggregation_search": True},
    "kernel": {"kernel_search": True},
    "ghost": {"ghost_conv_search": True},
    "dw_decomp": {"dw_decomposition": True},
    "all_level1": {"activation_search": True, "pooling_search": True,
                   "aggregation_search": True, "kernel_search": True},
    "all_level2": {"ghost_conv_search": True, "dw_decomposition": True},
    "everything": {"activation_search": True, "pooling_search": True,
                   "aggregation_search": True, "kernel_search": True,
                   "ghost_conv_search": True, "dw_decomposition": True},
}
OFF = {k: False for k in ("activation_search", "pooling_search", "aggregation_search",
                          "kernel_search", "ghost_conv_search", "dw_decomposition")}


def run(name: str, seed: int, n_variants: int, calib_samples: int, threads: int,
        ft_epochs: int, ft_lr: float, device: str,
        families: list | None) -> Dict:
    from h3dnas.core.types import SearchSpace
    from h3dnas.generator.graph_mutator import GraphMutator
    from h3dnas.modulator.bn_recalib import recalibrate_bn
    from h3dnas.datapipe import eval_accuracy, feeds_from_loader
    from h3dnas.finetune import finetune

    spec = get(name)
    base = _parse(spec.onnx)
    dm = spec.data(seed)
    val, test = dm.val_loader(), dm.test_loader()
    calib = feeds_from_loader(base, dm.pool_loader(), n_samples=calib_samples)

    base_val = eval_accuracy(base, val)
    base_test = eval_accuracy(base, test)
    rows: List[Dict] = [{"family": "base", "variant": 0, "ok": True,
                         "params": base.parameters, "macs": base.macs,
                         "val_zeroshot": base_val, "val_ft": base_val,
                         "test_ft": base_test}]
    print(f"base  params {base.parameters/1e6:.3f}M  MACs {base.macs/1e6:.1f}M  "
          f"val {100*base_val:.2f}%  test {100*base_test:.2f}%", flush=True)

    # CONTROL. Every mutant below is fine-tuned, so comparing them against the *distributed*
    # base measures mutation + training against mutation-free + no training. The honest
    # reference is the base put through the identical recipe; without this row a mutation that
    # merely benefits from 10 more epochs looks like a free architectural win.
    if ft_epochs > 0:
        t0 = time.perf_counter()
        res = finetune(base, dm, epochs=ft_epochs, lr=ft_lr, optimizer="sgd", seed=seed,
                       threads=threads, distill=base, distill_temperature=4.0, device=device,
                       log=lambda *a, **k: None)
        cv, ct = eval_accuracy(res.model, val), eval_accuracy(res.model, test)
        rows.append({"family": "base+ft (control)", "variant": 0, "ok": True,
                     "params": base.parameters, "macs": base.macs,
                     "val_zeroshot": base_val, "val_ft": cv, "test_ft": ct,
                     "ft_s": round(time.perf_counter() - t0, 1)})
        print(f"base+ft (control)        val {100*cv:6.2f}%  test {100*ct:6.2f}%"
              f"  ({100*(cv - base_val):+6.2f} val vs base)", flush=True)

    ss = SearchSpace()
    picked = {k: v for k, v in FAMILIES.items() if not families or k in families}
    for fam, flags in picked.items():
        cfg = dict(OFF)
        cfg.update(flags)
        cfg.update({"mutation_prob": 1.0, "seed": seed})
        mut = GraphMutator(config=cfg)
        mut.initialize()
        t0 = time.perf_counter()
        try:
            variants = mut.execute(base, ss, n_variants=n_variants)
        except Exception as e:
            print(f"{fam:12s} GENERATION FAILED {type(e).__name__}: {str(e)[:90]}", flush=True)
            rows.append({"family": fam, "ok": False, "error": f"{type(e).__name__}: {e}"})
            continue
        gen_s = time.perf_counter() - t0

        for i, g in enumerate(variants):
            try:
                m = _parse_proto(g)
                recalibrate_bn(g, calib)
                v = eval_accuracy(g, val)
                row = {"family": fam, "variant": i, "ok": True,
                       "params": m.parameters, "macs": m.macs, "val_zeroshot": v,
                       "name": g.name, "gen_s": round(gen_s, 2)}
                print(f"{fam:12s} v{i}  params {m.parameters/1e6:.3f}M  "
                      f"MACs {m.macs/1e6:7.1f}M  val0 {100*v:6.2f}%"
                      f"  ({100*(v - base_val):+6.2f})", flush=True, end="")

                # Recovery, with the ORIGINAL graph as teacher -- the same self-distillation
                # the method uses, so a mutated graph is given exactly the budget a pruned one
                # gets. Zero-shot collapse means nothing until this has run.
                if ft_epochs > 0:
                    t0 = time.perf_counter()
                    res = finetune(g, dm, epochs=ft_epochs, lr=ft_lr, optimizer="sgd",
                                   seed=seed, threads=threads, distill=base,
                                   distill_temperature=4.0, device=device,
                                   log=lambda *a, **k: None)
                    ft_s = time.perf_counter() - t0
                    vt = eval_accuracy(res.model, val)
                    tt = eval_accuracy(res.model, test)
                    row.update({"val_ft": vt, "test_ft": tt, "ft_s": round(ft_s, 1)})
                    print(f"  ->  val {100*vt:6.2f}%  test {100*tt:6.2f}%"
                          f"  ({100*(vt - base_val):+6.2f} val vs base)  [{ft_s:.0f}s]",
                          flush=True)
                else:
                    print("", flush=True)
                rows.append(row)
            except Exception as e:
                rows.append({"family": fam, "variant": i, "ok": False,
                             "error": f"{type(e).__name__}: {e}"})
                print(f"\n{fam:12s} v{i}  FAILED {type(e).__name__}: {str(e)[:80]}", flush=True)
    return {"model": name, "seed": seed, "base_val": base_val, "base_test": base_test,
            "ft_epochs": ft_epochs, "rows": rows}


def _parse_proto(g):
    """Parse an in-memory ONNXModel/ModelProto for its parameter and MAC counts."""
    from h3dnas.parser import ONNXParser
    p = ONNXParser()
    return p.parse(g) if hasattr(p, "parse") else g


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--seed", type=int, default=105)
    ap.add_argument("--variants", type=int, default=2)
    ap.add_argument("--calib-samples", type=int, default=512)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--ft-epochs", type=int, default=10)
    ap.add_argument("--ft-lr", type=float, default=0.01)
    ap.add_argument("--device", default="cpu", help="cpu or cuda")
    ap.add_argument("--families", nargs="*", default=None,
                    help=f"subset of {list(FAMILIES)}; default all")
    a = ap.parse_args()
    rec = run(a.model, a.seed, a.variants, a.calib_samples, a.threads,
              a.ft_epochs, a.ft_lr, a.device, a.families)
    out = Path(paths().results) / "mutate_study"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{a.model}_seed{a.seed}.json").write_text(json.dumps(rec, indent=2),
                                                      encoding="utf-8")
    print(f"\nwrote {out / f'{a.model}_seed{a.seed}.json'}")


if __name__ == "__main__":
    main()
