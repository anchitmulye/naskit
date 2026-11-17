"""Print everything the paper needs about a model's runs, from the artifacts alone.

    python -m scripts.dump_runs                      # the three CIFAR models
    python -m scripts.dump_runs cifar_vgg16          # one model
    python -m scripts.dump_runs --all                # every model under artifacts/
    python -m scripts.dump_runs --raw                # also dump each JSON's key structure

Walks artifacts/<model>/<run>/ and reports, per run: the recipe actually used, which
candidates exist, which of them were fine-tuned (a *_ft.onnx beside the candidate), and
every params / MACs / val / test number it can find for them.

Deliberately stdlib-only and layout-agnostic: it searches every JSON in the run for
candidate-shaped records rather than assuming a filename, because the layout differs
between runs. Anything it cannot interpret is listed under UNPARSED with its keys, so a
missing number is visible instead of silently absent.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ART = ROOT / "artifacts"
CIFAR = ["cifar_resnet56", "cifar_resnet18", "cifar_vgg16"]
RECIPE_KEYS = ("epochs", "lr", "optimizer", "weight_decay", "distill", "distill_weight",
               "distill_temperature", "label_smoothing", "keep", "warmup_steps")
# a record is "candidate-shaped" if it carries at least one of these
METRIC_KEYS = ("params", "macs", "val_accuracy", "val", "test", "top1", "accuracy", "perplexity")


def jsons(run: Path):
    return sorted(p for p in run.rglob("*.json") if p.is_file())


def num(x):
    """Accuracies are stored as fractions in some files and percentages in others."""
    if isinstance(x, dict):
        for k in ("1", "top1", "accuracy", "acc", "perplexity"):
            if k in x:
                return num(x[k])
        return None
    if isinstance(x, (int, float)):
        return 100 * x if 0 < x <= 1 else float(x)
    return None


def fmt(x, nd=2):
    return "--" if x is None else (f"{x:,}" if isinstance(x, int) and abs(x) > 9999 else f"{x:.{nd}f}")


def candidate_records(obj, known, under_candidates=False):
    """Yield (name, record) for candidate-shaped dicts.

    A record counts only if its key is a file in models/ or sits inside an explicit
    "candidates" block. Without that test, bookkeeping dicts such as `prunability`
    (which has a `params` field) and `reported` are reported as if they were models.
    """
    if not isinstance(obj, dict):
        return
    for key, val in obj.items():
        if not isinstance(val, dict):
            continue
        if under_candidates or key in known:
            yield key, val
        else:
            yield from candidate_records(val, known, under_candidates=(key == "candidates"))


def report(model: str, raw: bool) -> None:
    mdir = ART / model
    if not mdir.is_dir():
        print(f"\n### {model}: no artifacts/{model}/ directory")
        return
    runs = sorted(p for p in mdir.iterdir() if p.is_dir())
    print(f"\n{'#' * 78}\n### {model}   ({len(runs)} run(s))\n{'#' * 78}")
    for run in runs:
        models_dir = run / "models"
        onnx = sorted(p.name for p in models_dir.glob("*.onnx")) if models_dir.is_dir() else []
        ft = {n[:-len("_ft.onnx")] for n in onnx if n.endswith("_ft.onnx")}
        cands = sorted({n[:-len(".onnx")] for n in onnx
                        if n.endswith(".onnx") and not n.endswith("_ft.onnx")})
        print(f"\n-- {run.name}")
        print(f"   candidates on disk : {', '.join(cands) if cands else '(none)'}")
        print(f"   FINE-TUNED (_ft)   : {', '.join(sorted(ft)) if ft else '(none)'}")

        known = set(cands) | set(ft)
        seen, unparsed = {}, []
        for jf in jsons(run):
            try:
                d = json.load(open(jf, encoding="utf-8"))
            except Exception as e:                      # noqa: BLE001 - report, never abort
                unparsed.append(f"{jf.relative_to(run)}: unreadable ({e})")
                continue
            if not isinstance(d, dict):
                continue
            recipe = d.get("recipe") or {k: d[k] for k in RECIPE_KEYS if k in d}
            if recipe:
                print(f"   recipe [{jf.relative_to(run)}]: "
                      + ", ".join(f"{k}={recipe[k]}" for k in RECIPE_KEYS if k in recipe))
            for k in ("seed", "selection_rule", "selected_after_ft", "selected"):
                if k in d:
                    print(f"   {k}: {d[k]}")
            if d.get("base_test") is not None:
                print(f"   base_test: {fmt(num(d['base_test']))}")
            # A file called test.json / val.json holds that split's metrics directly, so the
            # record *is* the measurement; structured files name the split inside the record.
            kind = {"test": "test", "val": "val"}.get(jf.stem)
            found = list(candidate_records(d, known))
            for name, rec in found:
                e = seen.setdefault(name, {})
                for k in ("params", "macs"):
                    if rec.get(k) is not None:
                        e[k] = rec[k]
                if rec.get("val_accuracy") is not None:
                    e["val"] = num(rec["val_accuracy"])
                for k in ("test", "top1", "accuracy"):
                    if rec.get(k) is not None:
                        e["test"] = num(rec[k])
                        break
                if rec.get("perplexity") is not None:
                    e["ppl_" + (kind or "test")] = num(rec["perplexity"])
                elif kind and "test" not in e and "val" not in e:
                    v = num(rec)                      # flat split file: the record is the metric
                    if v is not None:
                        e[kind] = v
                e.setdefault("_src", str(jf.relative_to(run)))
            if not found and not recipe:
                unparsed.append(f"{jf.relative_to(run)}: keys={list(d)[:12]}")

        if seen:
            has_ppl = any(k.startswith("ppl_") for e in seen.values() for k in e)
            head = f"   {'candidate':<30}{'ft?':<5}{'params':>12}{'MACs':>16}{'val':>9}{'test':>9}"
            print(head + (f"{'ppl val':>9}{'ppl test':>10}" if has_ppl else ""))
            for name in sorted(seen):
                e = seen[name]
                row = (f"   {name:<30}{'yes' if name in ft else '':<5}"
                       f"{fmt(e.get('params')):>12}{fmt(e.get('macs')):>16}"
                       f"{fmt(e.get('val')):>9}{fmt(e.get('test')):>9}")
                if has_ppl:
                    row += f"{fmt(e.get('ppl_val')):>9}{fmt(e.get('ppl_test')):>10}"
                print(row)
        else:
            print("   (no candidate metrics found in any JSON)")
        if unparsed and raw:
            print("   UNPARSED:")
            for u in unparsed:
                print(f"     {u}")
        elif unparsed:
            print(f"   UNPARSED: {len(unparsed)} file(s) -- rerun with --raw to see their keys")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("models", nargs="*", help="model names (default: the three CIFAR models)")
    ap.add_argument("--all", action="store_true", help="every model under artifacts/")
    ap.add_argument("--raw", action="store_true", help="also list JSONs that could not be parsed")
    a = ap.parse_args(argv)
    if a.all:
        names = sorted(p.name for p in ART.iterdir() if p.is_dir()) if ART.is_dir() else []
    else:
        names = a.models or CIFAR
    print(f"artifacts root: {ART}")
    for m in names:
        report(m, a.raw)
    return 0


if __name__ == "__main__":
    sys.exit(main())