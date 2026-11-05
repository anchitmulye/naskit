"""Evaluate every fine-tuned model sitting in the run folders: accuracy and compression.

    python -m scripts.eval_artifacts                        # every model, every run
    python -m scripts.eval_artifacts pointnet pointnet2     # only these models
    python -m scripts.eval_artifacts --run 2026-11-04_11-08-56
    python -m scripts.eval_artifacts --provider cpu         # on a machine without a GPU build
    python -m scripts.eval_artifacts --splits test          # test only (default: val and test)

Walks paths.runs (artifacts/) for <model>/<run>/models/*_ft.onnx - the models the
pipeline fine-tuned, plus any written by scripts.finetune_model - and reports each
against its run's base model: parameters, MACs, their reductions, and the task's
metrics on the validation and test splits.

Every run is evaluated with the seed recorded in its own debug/run.json, so each
model is read on the split it was selected on; a model evaluated with another
seed would have part of its selection data inside its test split. Runs whose
run.json is missing are skipped (the seed is unknown).

The base model is evaluated once per (model, seed) and reused.

Output: results/eval_artifacts/<model>.json and a table per model on screen.
Existing results are reused unless --force.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional

from experiments.registry import get, paths, specs
from experiments.pipeline import apply_runtime, datamodule_for, parse, scores, write_json


def _runs(model_dir: Path, pick: Optional[str]) -> List[Path]:
    out = [d for d in sorted(model_dir.iterdir()) if d.is_dir() and (d / "models").is_dir()]
    return [d for d in out if pick is None or d.name == pick]


def _metric(d: Dict, key="1") -> Optional[float]:
    """Top-1 (classification) or the first metric of the task."""
    if not d:
        return None
    for k in (key, 1, "miou", "macc"):
        if k in d:
            return d[k]
    return next(iter(d.values()), None)


def evaluate_model(name: str, runs_root: Path, pick: Optional[str], splits: List[str],
                   provider: Optional[str], force: bool) -> List[Dict]:
    model_dir = runs_root / name
    if not model_dir.is_dir():
        return []
    try:
        spec = get(name)
    except KeyError as e:                       # not in sources.yaml
        print(f"[eval] {name}: {e.args[0] if e.args else e}")
        return []

    rows: List[Dict] = []
    base_cache: Dict[int, Dict] = {}            # seed -> base metrics (evaluated once)
    for run in _runs(model_dir, pick):
        rj = run / "debug" / "run.json"
        if not rj.exists():
            print(f"[eval] {name}/{run.name}: no debug/run.json (seed unknown), skipped")
            continue
        prev = json.loads(rj.read_text())
        seed = prev.get("seed")
        fts = sorted((run / "models").glob("*_ft.onnx"))
        if not fts:
            continue

        over = {"seed": seed}
        if provider:
            over["runtime"] = {k: {"provider": provider} for k in ("search", "evaluate", "latency")}
        settings = spec.resolved(over)
        apply_runtime(settings)
        base = parse(spec.onnx)
        dm = None

        if seed not in base_cache:
            dm = datamodule_for(spec, base, seed, False, settings)
            b = {"params": base.parameters, "macs": base.macs}
            if "val" in splits:
                b["val"] = scores(spec, base, dm.val_loader(), dm)
            if "test" in splits:
                b["test"] = scores(spec, base, dm.test_loader(), dm)
            base_cache[seed] = b
        b = base_cache[seed]

        for f in fts:
            done = next((r for r in rows if r["file"] == str(f)), None)
            if done and not force:
                continue
            if dm is None:
                dm = datamodule_for(spec, base, seed, False, settings)
            t0 = time.time()
            m = parse(f)
            row = {"model": name, "run": run.name, "candidate": f.stem, "file": str(f),
                   "seed": seed, "params": m.parameters, "macs": m.macs,
                   "params_red": 1 - m.parameters / b["params"], "macs_red": 1 - m.macs / b["macs"],
                   "speedup_macs": b["macs"] / m.macs, "base": b}
            if "val" in splits:
                row["val"] = scores(spec, m, dm.val_loader(), dm)
            if "test" in splits:
                row["test"] = scores(spec, m, dm.test_loader(), dm)
            row["seconds"] = round(time.time() - t0, 1)
            rows.append(row)
            print(f"[eval] {name}/{run.name}/{f.stem}: MACs -{100*row['macs_red']:.1f}% "
                  f"params -{100*row['params_red']:.1f}%"
                  + (f"  test {100*_metric(row.get('test')):.2f}" if row.get("test") else ""))
    return rows


def table(name: str, rows: List[Dict]) -> str:
    if not rows:
        return ""
    out = [f"## {name}", "",
           "| Run | Candidate | Seed | Params | MACs | Params ↓ | MACs ↓ | MAC speed-up | Val | Test | Δ test |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in sorted(rows, key=lambda r: (r["run"], r["macs"])):
        bv, bt = _metric(r["base"].get("val")), _metric(r["base"].get("test"))
        v, t = _metric(r.get("val")), _metric(r.get("test"))
        d = f"{100*(t-bt):+.2f}" if (t is not None and bt is not None) else "--"
        out.append(f"| {r['run']} | {r['candidate']} | {r['seed']} | {r['params']:,} | {r['macs']:,} | "
                   f"{100*r['params_red']:.1f}% | {100*r['macs_red']:.1f}% | {r['speedup_macs']:.2f}× | "
                   f"{100*v:.2f} | {100*t:.2f} | {d} |" if (v is not None and t is not None) else
                   f"| {r['run']} | {r['candidate']} | {r['seed']} | {r['params']:,} | {r['macs']:,} | "
                   f"{100*r['params_red']:.1f}% | {100*r['macs_red']:.1f}% | {r['speedup_macs']:.2f}× | "
                   f"{'--' if v is None else f'{100*v:.2f}'} | {'--' if t is None else f'{100*t:.2f}'} | {d} |")
    b = rows[0]["base"]
    bv, bt = _metric(b.get("val")), _metric(b.get("test"))
    out += ["", f"base: {b['params']:,} params, {b['macs']:,} MACs"
                + (f", val {100*bv:.2f}" if bv is not None else "")
                + (f", test {100*bt:.2f}" if bt is not None else ""), ""]
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="models to evaluate (default: every one with runs)")
    ap.add_argument("--run", help="only this run folder name")
    ap.add_argument("--splits", nargs="+", default=["val", "test"], choices=["val", "test"])
    ap.add_argument("--provider", help="ONNX Runtime provider: cpu | cuda | tensorrt "
                                       "(default: the run's settings)")
    ap.add_argument("--force", action="store_true", help="re-evaluate models already in the result file")
    a = ap.parse_args()

    import sys
    sys.stdout.reconfigure(encoding="utf-8")
    runs_root = paths().runs
    names = a.models or sorted(d.name for d in runs_root.iterdir()
                               if d.is_dir() and d.name in specs())
    out_dir = paths().results / "eval_artifacts"
    tables = []
    for n in names:
        rows = evaluate_model(n, runs_root, a.run, a.splits, a.provider, a.force)
        if rows:
            write_json(out_dir / f"{n}.json", {"model": n, "rows": rows})
            tables.append(table(n, rows))
    print()
    print("\n".join(tables) if tables else "no *_ft.onnx found under " + str(runs_root))
