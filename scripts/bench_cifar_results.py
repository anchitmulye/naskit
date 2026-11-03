"""Read the bench_cifar results (results/bench_cifar/*.json) as one table per model and budget.

    python -m scripts.bench_report                       # every file in results/bench_cifar
    python -m scripts.bench_report cifar_resnet56        # files of these models only
    python -m scripts.bench_report --files a.json b.json # these files
    python -m scripts.bench_report --per-seed            # also one row per seed

Per method: parameter and MAC reductions against the base, MACs against the budget target,
validation accuracy before fine-tuning (the pruned architecture with BatchNorm re-estimated),
after fine-tuning, test accuracy and its change against the base on the same split, and the
latency speed-up measured by the run. Seeds are pooled as mean +- sample std (n shown); the
change against the base is taken per seed first, on that seed's split. The run settings
(recipe, search size, fine-tuning device) are printed above each table: check them.
"""

from __future__ import annotations

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
H3DNAS = ROOT / "submodules" / "h3dnas"
sys.path.insert(0, str(H3DNAS))

from experiments.registry import paths

NEEDS_SOURCE = {"depgraph", "depgraph_global", "l1"}


def _ms(xs: List[float], scale: float = 100.0, digits: int = 2, sign: bool = False) -> str:
    xs = [scale * x for x in xs]
    m = statistics.fmean(xs)
    s = f"{m:+.{digits}f}" if sign else f"{m:.{digits}f}"
    return s + (f" ± {statistics.stdev(xs):.{digits}f}" if len(xs) > 1 else "")


def _point(r: Dict) -> str:
    return f"budget {r['budget']:.0%} of base MACs" if r.get("budget") is not None else f"ratio {r['ratio']}"


def report(runs: List[Dict], per_seed: bool = False) -> str:
    groups = defaultdict(list)
    for r in runs:
        groups[(r["model"], _point(r))].append(r)
    out = []
    for (model, point), rs in sorted(groups.items()):
        rs.sort(key=lambda r: r["seed"])
        r0 = rs[0]
        settings = {json.dumps([r.get("recipe"), r.get("search"), r.get("finetune_device")], sort_keys=True)
                    for r in rs}
        out += [f"## {model}, {point} (seeds {', '.join(str(r['seed']) for r in rs)})", "",
                f"- target MACs {r0.get('target_macs', 'not recorded (older bench_cifar)')}; base {r0['base']['macs']:,} MACs, "
                f"{r0['base']['params']:,} params; base test {_ms([r['base']['test'] for r in rs])}",
                f"- recipe {r0.get('recipe')}; search {r0.get('search')}; fine-tune on "
                f"{r0.get('finetune_device')}"
                + ("" if len(settings) == 1 else "  **(settings differ between seeds)**"),
                f"- data: {(r0.get('data') or {}).get('split', 'not recorded')} "
                f"(val {(r0.get('data') or {}).get('val')}, test {(r0.get('data') or {}).get('test')})", ""]
        out += ["| Method | Source | Params ↓ % | MACs ↓ % | MACs / target | Val pre-FT | Val post-FT "
                "| Test | Δ test | Speed-up |",
                "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
        methods = list(dict.fromkeys(m for r in rs for m in r["methods"]))
        rows = []
        for m in methods:
            have = [r for r in rs if m in r["methods"]]
            g = lambda k: [r["methods"][m][k] for r in have]
            d = [r["methods"][m]["test"] - r["base"]["test"] for r in have]
            rows.append((statistics.fmean(d), m, have, d, g))
        best = max(rows, key=lambda t: t[0])[1] if rows else None
        for _, m, have, d, g in rows:
            name = f"**{m}**" if m == best else m
            out.append(
                f"| {name}{'' if len(have) == len(rs) else f' (n={len(have)})'} | "
                f"{'yes' if m in NEEDS_SOURCE else 'no'} | "
                f"{_ms([1 - r['methods'][m]['params'] / r['base']['params'] for r in have], digits=1)} | "
                f"{_ms([1 - r['methods'][m]['macs'] / r['base']['macs'] for r in have], digits=1)} | "
                f"{_ms([r['methods'][m]['macs'] / r['target_macs'] for r in have], 1.0, 3) if all(r.get('target_macs') for r in have) else '-'} | "
                f"{_ms(g('val_before_ft'))} | {_ms(g('val_after_ft'))} | {_ms(g('test'))} | "
                f"{_ms(d, sign=True)} | {_ms(g('speedup'), 1.0)}× |")
        if per_seed:
            out += ["", "| Seed | Method | MACs | Val pre-FT | Val post-FT | Test | Δ test |",
                    "|---|---|---:|---:|---:|---:|---:|"]
            for r in rs:
                for m, x in r["methods"].items():
                    out.append(f"| {r['seed']} | {m} | {x['macs']:,} | {100 * x['val_before_ft']:.2f} | "
                               f"{100 * x['val_after_ft']:.2f} | {100 * x['test']:.2f} | "
                               f"{100 * (x['test'] - r['base']['test']):+.2f} |")
        out.append("")
    return "\n".join(out) if out else "no bench_cifar results"


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="*", help="only these models (default: all)")
    ap.add_argument("--files", type=Path, nargs="+", help="read these files instead of results/bench_cifar")
    ap.add_argument("--per-seed", action="store_true")
    a = ap.parse_args()
    files = a.files or sorted((paths().results / "bench_cifar").glob("*.json"))
    runs = [json.loads(Path(f).read_text()) for f in files]
    runs = [r for r in runs if not a.models or r["model"] in a.models]
    import sys
    sys.stdout.reconfigure(encoding="utf-8")          # the table's arrows and +- on a Windows console
    print(report(runs, a.per_seed))
