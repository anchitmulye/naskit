"""Compare the accuracy of the base models of models/sources.yaml with their h3dnas
versions in models/h3dnas, next to parameters, MACs and file size.

The counterpart of scripts.compare_models (which has latency and needs no data):
this one evaluates every model on the validation and test split of the dataset
h3dnas.yaml gives it, with the pipeline's own data modules and metrics, so the
numbers are the ones of `python -m examples <name> all`.

    python -m scripts.compare_accuracy --seed 105                  # every model with an h3dnas version
    python -m scripts.compare_accuracy pointnet pointmlp --seed 105 # only these
    python -m scripts.compare_accuracy --all                       # also base models without a version
    python -m scripts.compare_accuracy --splits test               # test only (val, test, imagenetv2)
    python -m scripts.compare_accuracy --provider cuda --precision fp16   # accuracy as deployed
    python -m scripts.compare_accuracy --providers cpu cuda tensorrt:fp16 # every runtime, one table
    python -m scripts.compare_accuracy --quick                     # a few batches per split (smoke run)

Files are matched to base models as in compare_models: models/h3dnas/<name>.onnx,
<name>_<tag>.onnx and <name>/<file>.onnx. Files that match no base model are
skipped (their dataset is unknown), as are base models without a `models:` entry
in h3dnas.yaml.

--seed matters. The validation split (and therefore the test split, the rest of
the official test set) is drawn with the seed. A model chosen on one seed's validation
split was never trained or selected on that seed's test split; evaluated with another
seed, part of its test split is the validation data it was selected on. Pass the
seed of the run that produced the h3dnas models (the run's `results.json` names it);
the default is `defaults.seed` of h3dnas.yaml. The whole validation split is used
(no search/selection halves), so Val here is every validation sample.

Metrics per task: classification top-1 (ImageNet also top-5) and mean class accuracy
(test); segmentation mIoU or instance mIoU; language models perplexity (lower is
better). Accuracy columns are percent, "change" columns percentage points
(perplexity: relative change). Output: the tables on screen and
results/compare/accuracy_<host>_<provider>[_fp16].md and .json (--out for a file stem).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from scripts.compare_models import H3DNAS, _change, _num, versions

ROOT = Path(__file__).resolve().parents[1]
H3DNAS = ROOT / "submodules" / "h3dnas"
H3DNAS_MODELS = ROOT / "models" / "h3dnas"
sys.path.insert(0, str(H3DNAS))

SPLITS = ("val", "test", "imagenetv2")
LABEL = {"top1": "top-1 %", "top5": "top-5 %", "macc": "mean class acc. %", "miou": "mIoU %",
         "pixel_accuracy": "pixel acc. %", "instance_miou": "instance mIoU %",
         "class_miou": "class mIoU %", "perplexity": "perplexity"}
EXTRA = {"top1": ("macc", "top5"), "miou": ("pixel_accuracy",), "instance_miou": ("class_miou",),
         "perplexity": ()}                    # extra metrics, shown for the test split
TITLE = {"top1": "Classification", "miou": "Segmentation (mean IoU)",
         "instance_miou": "Part segmentation (instance mIoU)", "perplexity": "Language modelling"}


# -- measurement ------------------------------------------------------------------

def evaluate_group(name: str, files: Dict[str, Path], seed: int, splits, quick: bool) -> Dict:
    """Counts and metrics of one base model and its versions on the model's data."""
    import onnx
    from experiments import pipeline, registry
    from h3dnas.core.inputs import resolve_input_spec
    from h3dnas.core.metrics import count
    from h3dnas.core.types import ONNXModel
    from h3dnas.datapipe import evaluate

    spec = registry.get(name)
    settings = spec.resolved({"search": {"fitness_split": None}})        # all of validation
    models = {label: ONNXModel(proto=onnx.load(str(path)), name=label) for label, path in files.items()}
    dm = pipeline.datamodule_for(spec, models["base"], seed, quick, settings)
    task = getattr(dm, "task", None)
    classification = task is None or task.type == "classification"
    primary = "top1" if classification else task.metric
    ks = pipeline._ks(spec)

    loaders = {"val": dm.val_loader, "test": dm.test_loader}
    if "imagenetv2" in splits and not quick and getattr(dm, "test_root2", None):
        loaders["imagenetv2"] = dm.test2_loader
    loaders = {s: f for s, f in loaders.items() if s in splits}

    rows: Dict[str, Dict[str, Any]] = {}
    for label, m in models.items():
        row: Dict[str, Any] = {"file": (str(files[label].relative_to(ROOT))
                                        if files[label].is_relative_to(ROOT) else str(files[label])),
                               "size_mb": files[label].stat().st_size / 2**20}
        try:
            c = count(m.proto, spec=resolve_input_spec(m.proto, dm.sample_input()))
            row.update(params=c.params, macs=c.macs, macs_complete=c.complete)
        except Exception as exc:
            row["errors"] = {"count": f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"}
        for split, loader in loaders.items():
            t0 = time.time()
            try:
                row[split] = evaluate(m, loader(), None if classification else task, ks)
            except Exception as exc:
                row.setdefault("errors", {})[split] = f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}"
            print(f"    {name}/{label} {split}: "
                  + (f"{primary} {row[split][primary]:.4f}" if split in row else "FAILED")
                  + f" ({time.time() - t0:.0f}s)", flush=True)
        rows[label] = row
    return {"model": name, "dataset": spec.dataset, "primary": primary, "seed": seed,
            "published_top1": spec.published, "data": dm.describe(), "rows": rows}


# -- tables -----------------------------------------------------------------------

def _value(key: str, v: Optional[float]) -> str:
    return "-" if v is None else f"{v:.2f}" if key == "perplexity" else f"{v * 100:.2f}"


def _delta(key: str, v: Optional[float], b: Optional[float]) -> str:
    if v is None or b is None:
        return "-"
    return f"{(v / b - 1) * 100:+.1f}%" if key == "perplexity" else f"{(v - b) * 100:+.2f}"


def _columns(groups: List[Dict], primary: str, splits) -> List[tuple]:
    cols = [(s, primary) for s in splits if any(s in r for g in groups for r in g["rows"].values())]
    if "test" in splits:
        cols += [("test", k) for k in EXTRA.get(primary, ())
                 if any(k in r.get("test", {}) for g in groups for r in g["rows"].values())]
    return cols


def markdown(groups: List[Dict], splits, settings: Dict, quick: bool) -> str:
    runtime = ", ".join(f"{k} {v}" for k, v in settings.items())
    lines = ["# ONNX models: accuracy of base vs h3dnas", "",
             f"Evaluated with: {runtime}" + (" (quick: a few batches per split)" if quick else ""), "",
             "Data (split seed in brackets; Val = the whole validation split):", ""]
    for g in groups:
        d = g["data"]
        rep = f", reported top-1 {g['published_top1']}" if g.get("published_top1") else ""
        lines.append(f"- {g['model']}: {g['dataset']} [seed {g['seed']}], validation {d.get('val', '?')}, "
                     f"test {d.get('test', '?')}{rep}")
    for primary in dict.fromkeys(g["primary"] for g in groups):
        sub = [g for g in groups if g["primary"] == primary]
        cols = _columns(sub, primary, splits)
        head = (["model", "version", "params", "MACs", "size MB", "params vs base", "MACs vs base"]
                + [f"{s} {LABEL[k]}" for s, k in cols] + [f"{s} {LABEL[k]} change" for s, k in cols])
        lines += ["", f"## {TITLE.get(primary, primary)}", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for g in sub:
            b = g["rows"].get("base", {})
            for label, r in g["rows"].items():
                rel = label != "base" and "base" in g["rows"]
                macs = _num(r.get("macs")) + ("" if r.get("macs_complete", True) else "*")
                cells = [g["model"] if label == "base" or "base" not in g["rows"] else "", label,
                         _num(r.get("params")), macs, f"{r['size_mb']:.2f}",
                         _change(r.get("params"), b.get("params")) if rel else "-",
                         _change(r.get("macs"), b.get("macs")) if rel else "-"]
                cells += [_value(k, r.get(s, {}).get(k)) if s in r else r.get("errors", {}).get(s, "-")
                          for s, k in cols]
                cells += [_delta(k, r.get(s, {}).get(k), b.get(s, {}).get(k)) if rel else "-"
                          for s, k in cols]
                lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "change: percentage points against the base model (perplexity: relative change, "
              "lower is better). params / MACs vs base: relative change (-50% = half). MACs are per "
              "sample at the data's input shape; `*`: some nodes could not be sized."]
    return "\n".join(lines) + "\n"


EP = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
      "tensorrt": "TensorrtExecutionProvider", "coreml": "CoreMLExecutionProvider"}


def run_providers(plan, providers: List[str], base_rt: Dict, seed: int, splits, quick: bool):
    """Every group evaluated on each provider ('name[:precision]'); one skeleton per model
    (counts, data) with the metrics of each provider under rows[label]['by_provider'][spec]."""
    import copy
    import onnxruntime as ort
    from h3dnas.core import runtime
    have = set(ort.get_available_providers())
    groups: Dict[str, Dict] = {}
    devices: Dict[str, Dict] = {}
    failed: List[tuple] = []
    for spec in providers:
        prov, _, prec = spec.partition(":")
        rt = {**base_rt, "provider": prov, "precision": prec or "fp32"}
        if EP[prov] not in have:
            devices[spec] = {"error": f"{EP[prov]} is not in this onnxruntime build "
                                      f"(onnxruntime {ort.__version__}: {', '.join(sorted(have))})"}
            print(f"[{spec}] not available: {devices[spec]['error']}", flush=True)
            continue
        devices[spec] = {"provider": prov, "precision": rt["precision"], "onnxruntime": ort.__version__}
        with runtime.using(evaluate=rt):
            for i, (name, files) in enumerate(plan, 1):
                t0 = time.time()
                print(f"[{spec} {i}/{len(plan)}] {name}: {len(files)} file(s), seed {seed}", flush=True)
                try:
                    res = evaluate_group(name, files, seed, splits, quick)
                except Exception as exc:
                    failed.append((f"{name} on {spec}", f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"))
                    print(f"    FAILED: {failed[-1][1]}", flush=True)
                    continue
                if name not in groups:
                    g = copy.deepcopy(res)
                    for r in g["rows"].values():
                        for s in SPLITS:
                            r.pop(s, None)
                        r.pop("errors", None)
                    groups[name] = g
                for label, row in res["rows"].items():
                    bp = {s: row[s] for s in splits if s in row}
                    if row.get("errors"):
                        bp["errors"] = {k: v for k, v in row["errors"].items() if k in SPLITS}
                    groups[name]["rows"].setdefault(label, {}).setdefault("by_provider", {})[spec] = bp
                print(f"    {time.time() - t0:.0f}s", flush=True)
    return [groups[n] for n, _ in plan if n in groups], devices, failed


def markdown_multi(groups: List[Dict], splits, providers: List[str], devices: Dict, seed: int,
                   quick: bool) -> str:
    lines = ["# ONNX models: accuracy of base vs h3dnas, per runtime", "",
             f"Host {platform.node()}, split seed {seed}" + (" (quick: a few batches per split)" if quick else ""), ""]
    for p in providers:
        d = devices.get(p) or {}
        lines.append(f"- {p}: " + (f"not evaluated - {d['error']}" if "error" in d else
                                   ", ".join(f"{k} {v}" for k, v in d.items())))
    lines += ["", "Data (Val = the whole validation split):", ""]
    for g in groups:
        d = g["data"]
        rep = f", reported top-1 {g['published_top1']}" if g.get("published_top1") else ""
        lines.append(f"- {g['model']}: {g['dataset']}, validation {d.get('val', '?')}, test {d.get('test', '?')}{rep}")
    ran = [p for p in providers if "error" not in (devices.get(p) or {})]
    for primary in dict.fromkeys(g["primary"] for g in groups):
        sub = [g for g in groups if g["primary"] == primary]
        seen = lambda s, k: any(k in bp.get(s, {}) for g in sub for r in g["rows"].values()
                                for bp in r.get("by_provider", {}).values())
        cols = [(s, primary) for s in splits if seen(s, primary)]
        if "test" in splits:
            cols += [("test", k) for k in EXTRA.get(primary, ()) if seen("test", k)]
        head = (["model", "version", "params", "MACs", "size MB", "params vs base", "MACs vs base"]
                + [f"{p} {s} {LABEL[k]} (change)" for p in ran for s, k in cols])
        lines += ["", f"## {TITLE.get(primary, primary)}", "",
                  "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
        for g in sub:
            b = g["rows"].get("base", {})
            for label, r in g["rows"].items():
                rel = label != "base" and "base" in g["rows"]
                cells = [g["model"] if label == "base" or "base" not in g["rows"] else "", label,
                         _num(r.get("params")), _num(r.get("macs")) + ("" if r.get("macs_complete", True) else "*"),
                         f"{r['size_mb']:.2f}",
                         _change(r.get("params"), b.get("params")) if rel else "-",
                         _change(r.get("macs"), b.get("macs")) if rel else "-"]
                for p in ran:
                    x = r.get("by_provider", {}).get(p, {})
                    bx = b.get("by_provider", {}).get(p, {})
                    for s, k in cols:
                        if s not in x:
                            cells.append((x.get("errors") or {}).get(s, "-")[:60])
                            continue
                        v = _value(k, x[s].get(k))
                        cells.append(f"{v} ({_delta(k, x[s].get(k), bx.get(s, {}).get(k))})" if rel else v)
                lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "(change): percentage points against the base model on the same runtime (perplexity: "
              "relative change, lower is better). params / MACs vs base: relative change (-50% = half)."]
    return "\n".join(lines) + "\n"


# -- main -------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.compare_accuracy",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="base model names (default: every one with a version)")
    ap.add_argument("--all", action="store_true", help="also base models without an h3dnas version")
    ap.add_argument("--h3dnas", type=Path, default=H3DNAS_MODELS)
    ap.add_argument("--seed", type=int, help="data split seed (default: defaults.seed of h3dnas.yaml); "
                                             "use the seed of the run that made the h3dnas models")
    ap.add_argument("--splits", nargs="+", choices=SPLITS, default=["val", "test"])
    ap.add_argument("--quick", action="store_true", help="a few batches per split (smoke run)")
    ap.add_argument("--provider", choices=["cpu", "cuda", "tensorrt", "coreml"])
    ap.add_argument("--providers", nargs="+", metavar="PROVIDER[:PRECISION]",
                    help="evaluate on every listed runtime, one combined table, e.g. "
                         "--providers cpu cuda tensorrt tensorrt:fp16")
    ap.add_argument("--precision", choices=["fp32", "fp16"])
    ap.add_argument("--device-id", type=int)
    ap.add_argument("--out", type=Path, help="output file stem (default: results/compare/accuracy_<host>_<provider>)")
    a = ap.parse_args(argv)

    from experiments.registry import paths
    from h3dnas.core import runtime
    ws = paths()
    s = ws.settings()
    seed = s["seed"] if a.seed is None else a.seed
    rt = {**s["runtime"]["evaluate"], **{k: v for k, v in {
        "provider": a.provider, "precision": a.precision, "device_id": a.device_id}.items()
        if v is not None}}

    files = {n: f for n, f in ws.model_files.items() if f.exists()}
    found = versions(files, a.h3dnas)
    chosen = a.names or (sorted(files) if a.all else sorted(n for n in found if n))
    unknown = [n for n in chosen if n not in files]
    if unknown:
        raise SystemExit(f"not in {ws.sources} (or file missing): {unknown}")
    no_data = [n for n in chosen if n not in ws.models]
    if no_data:
        print(f"skipped, no `models:` entry in {ws.file.name} (dataset unknown): {', '.join(no_data)}")
    chosen = [n for n in chosen if n in ws.models]
    stray = [p.name for p in found.get(None, [])]
    if stray and not a.names:
        print(f"skipped, matching no base model: {', '.join(stray)}")
    plan = [(n, {"base": files[n],
                 **{str(p.relative_to(a.h3dnas).with_suffix("")).replace("\\", "/"): p
                    for p in found.get(n, [])}}) for n in chosen]
    if not plan:
        raise SystemExit(f"nothing to compare: no ONNX files in {a.h3dnas} "
                         f"(name them <base name>[_<tag>].onnx or put them in <base name>/)")

    if a.providers:
        bad = [p for p in a.providers if p.partition(":")[0] not in EP
               or p.partition(":")[2] not in ("", "fp32", "fp16")]
        if bad or a.provider or a.precision:
            raise SystemExit("--providers takes cpu|cuda|tensorrt|coreml[:fp32|fp16] and replaces "
                             "--provider / --precision" + (f" (got {bad})" if bad else ""))
        groups, devices, failed = run_providers(plan, a.providers, rt, seed, a.splits, a.quick)
        if not groups:
            raise SystemExit("no model could be evaluated: "
                             + "; ".join(f"{n}: {e}" for n, e in failed) if failed else "no runtime available")
        md = markdown_multi(groups, a.splits, a.providers, devices, seed, a.quick)
        if failed:
            md += "\nNot evaluated: " + "; ".join(f"{n} ({e})" for n, e in failed) + "\n"
        print("\n" + md)
        stem = a.out or ws.results / "compare" / (
            f"accuracy_{platform.node()}_" + "_".join(p.replace(":", "-") for p in a.providers))
        stem.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{stem}.md").write_text(md, encoding="utf-8")
        Path(f"{stem}.json").write_text(json.dumps(
            {"host": platform.node(), "seed": seed, "providers": devices, "splits": a.splits,
             "quick": a.quick, "models": groups, "failed": [{"model": n, "error": e} for n, e in failed]},
            indent=2, default=str), encoding="utf-8")
        print(f"written: {stem}.md, {stem}.json")
        return

    groups, failed = [], []
    with runtime.using(evaluate=rt):
        for i, (name, group) in enumerate(plan, 1):
            t0 = time.time()
            print(f"[{i}/{len(plan)}] {name}: {len(group)} file(s), seed {seed}", flush=True)
            try:
                groups.append(evaluate_group(name, group, seed, a.splits, a.quick))
            except Exception as exc:
                failed.append((name, f"{type(exc).__name__}: {str(exc).splitlines()[0][:200]}"))
                print(f"    FAILED: {failed[-1][1]}", flush=True)
            print(f"    {time.time() - t0:.0f}s", flush=True)
    if not groups:
        raise SystemExit("no model could be evaluated: " + "; ".join(f"{n}: {e}" for n, e in failed))

    settings = {"provider": rt["provider"], "precision": rt["precision"], "seed": seed,
                "host": platform.node()}
    md = markdown(groups, a.splits, settings, a.quick)
    if failed:
        md += "\nNot evaluated: " + "; ".join(f"{n} ({e})" for n, e in failed) + "\n"
    print("\n" + md)
    tag = f"{platform.node()}_{rt['provider']}" + ("_fp16" if rt["precision"] == "fp16" else "")
    stem = a.out or ws.results / "compare" / f"accuracy_{tag}"
    stem.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{stem}.md").write_text(md, encoding="utf-8")
    Path(f"{stem}.json").write_text(json.dumps(
        {"settings": settings, "splits": a.splits, "quick": a.quick, "models": groups,
         "failed": [{"model": n, "error": e} for n, e in failed]}, indent=2, default=str),
        encoding="utf-8")
    print(f"written: {stem}.md, {stem}.json")


if __name__ == "__main__":
    main()
