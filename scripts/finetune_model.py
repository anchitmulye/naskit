"""Fine-tune ONE saved candidate, including the ones the search did not select.

    python -m scripts.finetune_model vit-base-patch16-224 \
        artifacts/vit-base-patch16-224/2026-11-02_08-57-29/models/vit-base-patch16-224_gs_50pct.onnx

`python -m examples <name> finetune` only fine-tunes what search.json recorded as
`selected` (plus more with settings.finetune.candidates / all_budgets), and changing
those settings makes the run's identity mismatch, so an existing search cannot be
reused for a different candidate. This script takes the candidate's ONNX file
directly instead: same data module, same recipe and same protocol as the pipeline,
but its own output folder, so the run's records stay untouched.

Protocol, as in experiments.pipeline.finetune:
  - the model's recipe from h3dnas.yaml (--epochs / --lr / --device override it)
  - distillation from the *base* ONNX when the recipe or settings ask for it
  - validation for every reading during training; the test split is read once at
    the end, for the fine-tuned model and the base

Output: results/finetune_model/<model>_<candidate>_seed<N>.json and the fine-tuned
ONNX next to it. An existing result file is kept (the run is skipped) unless --force.
"""

from __future__ import annotations

import argparse
import platform
import time
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
H3DNAS = ROOT / "submodules" / "h3dnas"
sys.path.insert(0, str(H3DNAS))

from experiments.registry import get, paths
from experiments.pipeline import apply_runtime, datamodule_for, parse, scores, write_json


def run(name: str, candidate: Path, seed: int | None = None, epochs: int | None = None,
        lr: float | None = None, device: str | None = None, distill: bool | None = None,
        provider: str | None = None, force: bool = False) -> dict:
    from h3dnas.finetune import finetune as ft

    spec = get(name)
    if not spec.recipe:
        raise SystemExit(f"{name}: no fine-tuning recipe in h3dnas.yaml")
    if not candidate.exists():
        raise SystemExit(f"{candidate} not found")
    over = {} if seed is None else {"seed": seed}
    if provider:                                 # ONNX Runtime for evaluation (not the torch device)
        over["runtime"] = {k: {"provider": provider} for k in ("search", "evaluate", "latency")}
    settings = spec.resolved(over)
    seed = settings["seed"]
    apply_runtime(settings)

    out = paths().results / "finetune_model" / f"{name}_{candidate.stem}_seed{seed}.json"
    if out.exists() and not force:
        print(f"[finetune_model] {out.name}: done (--force to redo)")
        return {}

    recipe = dict(spec.recipe)
    recipe["distill"] = bool(recipe.get("distill", settings["finetune"].get("distill", False)))
    if epochs is not None:
        recipe["epochs"] = epochs
    if lr is not None:
        recipe["lr"] = lr
    if distill is not None:
        recipe["distill"] = distill
    dev = device or settings["finetune"]["device"]

    base = parse(spec.onnx)                      # the teacher, and the base for the comparison
    model = parse(candidate)
    dm = datamodule_for(spec, base, seed, False, settings)
    print(f"[finetune_model] {name} / {candidate.name}: {model.parameters:,} params, "
          f"{model.macs:,} MACs ({100 * (1 - model.macs / base.macs):.1f}% fewer than base) "
          f"| recipe {recipe} on {dev}")

    t0 = time.time()
    kw = dict(recipe)
    if kw.pop("distill", False):                 # the unpruned graph teaches, as in the pipeline
        kw["distill"] = base
    r = ft(model, dm, seed=seed, device=dev, **kw)
    out.parent.mkdir(parents=True, exist_ok=True)
    r.model.save(str(out.with_suffix("")) + "_ft.onnx")

    rec = {"model": name, "candidate": candidate.stem, "candidate_path": str(candidate),
           "seed": seed, "recipe": recipe, "finetune_device": dev, "data": dm.describe(),
           "host": platform.node(), "seconds": round(time.time() - t0, 1),
           "base": {"params": base.parameters, "macs": base.macs},
           "pruned": {"params": model.parameters, "macs": model.macs,
                      "val_accuracy": r.val_accuracy, "curve": r.curve}}
    # Test split: read once, after training and every validation reading above.
    test = dm.test_loader()
    rec["base"]["test"] = scores(spec, base, test, dm)
    rec["pruned"]["test"] = scores(spec, r.model, dm.test_loader(), dm)
    write_json(out, rec)
    print(f"[finetune_model] base test {rec['base']['test']} -> pruned {rec['pruned']['test']}")
    print(f"[finetune_model] written {out}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model", help="model name in h3dnas.yaml (gives the data module and recipe)")
    ap.add_argument("candidate", type=Path, help="the candidate's ONNX file (any saved model)")
    ap.add_argument("--seed", type=int, help="default: the model's resolved seed")
    ap.add_argument("--epochs", type=int, help="override the recipe's epochs")
    ap.add_argument("--lr", type=float, help="override the recipe's learning rate")
    ap.add_argument("--device", help="torch device for training: cpu | cuda | mps "
                                     "(default: settings.finetune.device)")
    ap.add_argument("--provider", help="ONNX Runtime provider for evaluating val/test: cpu | cuda | "
                                       "tensorrt (default: settings.runtime). Set this to cpu on a "
                                       "machine whose onnxruntime has no GPU build.")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--distill", dest="distill", action="store_true", help="force distillation on")
    g.add_argument("--no-distill", dest="distill", action="store_false", help="force it off")
    ap.set_defaults(distill=None)
    ap.add_argument("--force", action="store_true", help="redo an existing result")
    a = ap.parse_args()
    run(a.model, a.candidate, a.seed, a.epochs, a.lr, a.device, a.distill, a.provider, a.force)
