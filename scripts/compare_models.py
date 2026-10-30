"""Compare the base models of models/sources.yaml (each at its `path`) with
their h3dnas versions in models/h3dnas: parameters, MACs, file size and latency on this machine.

No dataset and no accuracy: parameters, MACs and size are properties of the
file, and latency is timed on a synthesized input, so the script runs on any
machine with the h3dnas library - run it on the target device (e.g. the
Jetson) for that device's latency.

    python -m scripts.compare_models                          # every model with an h3dnas version
    python -m scripts.compare_models cifar_resnet56 pointnet  # only these
    python -m scripts.compare_models --all                    # also base models without a version
    python -m scripts.compare_models --provider tensorrt --precision fp16
    python -m scripts.compare_models --provider coreml --compute-units ALL   # on a Mac
    python -m scripts.compare_models --no-latency             # counts only (fast)
    python -m scripts.compare_models --providers cpu cuda tensorrt tensorrt:fp16   # all runtimes, one table

The h3dnas versions of base model <name> are models/h3dnas/<name>.onnx,
models/h3dnas/<name>_<tag>.onnx and models/h3dnas/<name>/<file>.onnx; the
longest matching base name wins (pointnet2_s3dis_x.onnx belongs to
pointnet2_s3dis, not pointnet2). A file that matches no base model is listed
on its own.

Latency: batch 1, the same input for a base model and its versions, timed
interleaved after warm-up (h3dnas.core.latency: median and IQR over rounds).
Threads, rounds and the provider default to `defaults.latency` and
`defaults.runtime.latency` of h3dnas.yaml. MACs are per sample; a `*` marks a
count with nodes whose shape could not be resolved.

Output: the table on screen and results/compare/<host>_<provider>[_fp16].md
and .json (paths.results; --out to choose another file stem).
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import sys
from pathlib import Path
from typing import Dict, List, Optional


ROOT = Path(__file__).resolve().parents[1]
H3DNAS = ROOT / "submodules" / "h3dnas"
H3DNAS_MODELS = ROOT / "models" / "h3dnas"
sys.path.insert(0, str(H3DNAS))


# -- which files to compare -------------------------------------------------------

def versions(names, h3dnas_dir: Path) -> Dict[Optional[str], List[Path]]:
    """{base model name: [h3dnas files]}; key None holds files with no base model."""
    names = set(names)
    out: Dict[Optional[str], List[Path]] = {}
    for p in sorted(h3dnas_dir.rglob("*.onnx")) if h3dnas_dir.exists() else []:
        rel = p.relative_to(h3dnas_dir)
        if len(rel.parts) > 1 and rel.parts[0] in names:
            base: Optional[str] = rel.parts[0]
        else:
            hits = [n for n in names if p.stem == n or p.stem.startswith(n + "_")]
            base = max(hits, key=len) if hits else None
        out.setdefault(base, []).append(p)
    return out


# -- measurements -----------------------------------------------------------------

def _counts(proto, shapes, feed) -> Dict:
    from h3dnas.core.metrics import count
    c = count(proto, input_shapes=shapes, feed=feed)
    return {"params": c.params, "macs": c.macs, "macs_complete": c.complete}


def measure_group(files: Dict[str, Path], latency: Optional[Dict], counts: bool = True) -> Dict[str, Dict]:
    """Counts (and latency) of one base model and its versions, on one input."""
    import onnx
    from h3dnas.core.inputs import resolve_input_spec
    from h3dnas.core.latency import measure
    from h3dnas.core.metrics import per_sample_inputs

    protos, rows = {}, {}
    for label, path in files.items():
        rows[label] = {"file": str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path),
                       "size_mb": path.stat().st_size / 2**20}
        try:
            protos[label] = onnx.load(str(path))
        except Exception as exc:
            rows[label]["error"] = f"load: {exc}"

    first = next(iter(protos.values()), None)
    if first is None:
        return rows
    shapes, feed = per_sample_inputs(resolve_input_spec(first))
    for label, proto in protos.items() if counts else ():
        try:
            rows[label].update(_counts(proto, shapes, feed))
        except Exception as exc:
            rows[label]["error"] = f"count: {exc}"

    if latency is not None:
        timed = {k: v for k, v in protos.items() if "error" not in rows[k]}
        try:
            report = measure(timed, feed, **latency)
            for k, lat in report.results.items():
                rows[k].update(latency_ms=lat.median_ms, latency_iqr_ms=lat.iqr_ms)
        except Exception:                      # time one by one to find the model that fails
            for k, proto in timed.items():
                try:
                    lat = measure({k: proto}, feed, **latency).results[k]
                    rows[k].update(latency_ms=lat.median_ms, latency_iqr_ms=lat.iqr_ms)
                except Exception as exc:
                    rows[k]["error"] = f"latency: {str(exc).splitlines()[0][:120]}"
    return rows


# -- table ------------------------------------------------------------------------

def _num(x: Optional[float]) -> str:
    if x is None:
        return "-"
    for div, unit in ((1e9, "G"), (1e6, "M"), (1e3, "K")):
        if abs(x) >= div:
            return f"{x / div:.2f}{unit}"
    return f"{x:.0f}"


def _change(v: Optional[float], b: Optional[float]) -> str:
    return "-" if v is None or not b else f"{(v / b - 1) * 100:+.1f}%"


def markdown(groups: List[Dict], device: Dict, settings: Optional[Dict]) -> str:
    head = ["model", "version", "params", "MACs", "size MB", "latency ms (IQR)",
            "params vs base", "MACs vs base", "speedup"]
    timing = ("batch 1, " + ", ".join(f"{k} {v}" for k, v in settings.items())
              if settings else "not measured (--no-latency)")
    lines = ["# ONNX models: base vs h3dnas", "",
             "Device: " + ", ".join(f"{k} {v}" for k, v in device.items()), "",
             "Latency: " + timing, "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in groups:
        b = g["rows"].get("base", {})
        for label, r in g["rows"].items():
            macs = _num(r.get("macs")) + ("" if r.get("macs_complete", True) else "*")
            lat = (f"{r['latency_ms']:.3f} ({r['latency_iqr_ms']:.3f})"
                   if "latency_ms" in r else r.get("error", "-") if "error" in r else "-")
            rel = label != "base" and "base" in g["rows"]
            speed = (f"{b['latency_ms'] / r['latency_ms']:.2f}x"
                     if rel and "latency_ms" in r and "latency_ms" in b else "-")
            lines.append("| " + " | ".join([
                g["model"] if label == "base" or "base" not in g["rows"] else "", label,
                _num(r.get("params")), macs, f"{r['size_mb']:.2f}", lat,
                _change(r.get("params"), b.get("params")) if rel else "-",
                _change(r.get("macs"), b.get("macs")) if rel else "-", speed]) + " |")
    lines += ["", "params / MACs vs base: change against the base model (-50% = half). "
              "speedup: base latency / version latency. `*`: MACs of some nodes not resolved."]
    return "\n".join(lines) + "\n"


def markdown_multi(groups: List[Dict], providers: List[str], devices: Dict[str, Dict],
                   lat: Dict) -> str:
    """One table: counts once, latency and speed-up per provider."""
    head = (["model", "version", "params", "MACs", "size MB", "params vs base", "MACs vs base"]
            + [h for p in providers for h in (f"{p} ms (IQR)", f"{p} speedup")])
    lines = ["# ONNX models: base vs h3dnas, latency per runtime", "",
             f"Host: {platform.node()}. Latency: batch 1, " + ", ".join(f"{k} {v}" for k, v in lat.items()), ""]
    for p in providers:
        d = devices.get(p) or {}
        lines.append(f"- {p}: " + (f"not measured - {d['error']}" if "error" in d else
                                   ", ".join(f"{k} {v}" for k, v in d.items() if k != "host") or "-"))
    lines += ["", "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for g in groups:
        b = g["rows"].get("base", {})
        for label, r in g["rows"].items():
            rel = label != "base" and "base" in g["rows"]
            cells = [g["model"] if label == "base" or "base" not in g["rows"] else "", label,
                     _num(r.get("params")), _num(r.get("macs")) + ("" if r.get("macs_complete", True) else "*"),
                     f"{r['size_mb']:.2f}",
                     _change(r.get("params"), b.get("params")) if rel else "-",
                     _change(r.get("macs"), b.get("macs")) if rel else "-"]
            for p in providers:
                x, bx = (r.get("latency") or {}).get(p, {}), (b.get("latency") or {}).get(p, {})
                cells.append(f"{x['latency_ms']:.3f} ({x['latency_iqr_ms']:.3f})" if "latency_ms" in x
                             else x.get("error", "-")[:60])
                cells.append(f"{bx['latency_ms'] / x['latency_ms']:.2f}x"
                             if rel and "latency_ms" in x and "latency_ms" in bx else "-")
            lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "params / MACs vs base: change against the base model (-50% = half). "
              "speedup: base latency / version latency on the same runtime. `*`: MACs of some nodes "
              "not resolved. A runtime that is not available here shows its error instead of a time."]
    return "\n".join(lines) + "\n"


def run_providers(plan, providers: List[str], base_rt: Dict, lat: Dict) -> tuple:
    """Counts once per file, then latency of every group on each provider ('name[:precision]')."""
    import onnxruntime as ort
    from h3dnas.core import runtime
    from h3dnas.core.latency import device_info
    EP = {"cpu": "CPUExecutionProvider", "cuda": "CUDAExecutionProvider",
          "tensorrt": "TensorrtExecutionProvider", "coreml": "CoreMLExecutionProvider"}
    have = set(ort.get_available_providers())
    groups = []
    for i, (name, files) in enumerate(plan, 1):
        groups.append({"model": name, "rows": measure_group(files, None)})
        print(f"[counts {i}/{len(plan)}] {name}", flush=True)
    devices: Dict[str, Dict] = {}
    for spec in providers:
        prov, _, prec = spec.partition(":")
        rt = {**base_rt, "provider": prov, "precision": prec or "fp32"}
        if EP[prov] not in have:
            devices[spec] = {"error": f"{EP[prov]} is not in this onnxruntime build "
                                      f"(onnxruntime {ort.__version__}: {', '.join(sorted(have))})"}
            print(f"[{spec}] not available: {devices[spec]['error']}", flush=True)
            for g in groups:
                for r in g["rows"].values():
                    r.setdefault("latency", {})[spec] = {"error": "n/a"}
            continue
        try:
            with runtime.using(latency=rt):
                devices[spec] = device_info(lat["threads"], runtime.current("latency"))
                for (name, files), g in zip(plan, groups):
                    t0 = time.time()
                    timed = measure_group(files, lat, counts=False)
                    for label, r in timed.items():
                        x = ({"latency_ms": r["latency_ms"], "latency_iqr_ms": r["latency_iqr_ms"]}
                             if "latency_ms" in r else {"error": r.get("error", "not measured")})
                        g["rows"][label].setdefault("latency", {})[spec] = x
                    print(f"[{spec}] {name}: {time.time() - t0:.0f}s", flush=True)
        except Exception as exc:                       # provider unusable here: record, go on
            msg = f"{type(exc).__name__}: {str(exc).splitlines()[0][:100]}"
            print(f"[{spec}] not available: {msg}", flush=True)
            devices[spec] = {"error": msg}
            for g in groups:
                for r in g["rows"].values():
                    r.setdefault("latency", {}).setdefault(spec, {"error": "runtime not available"})
    return groups, devices


# -- main -------------------------------------------------------------------------

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.compare_models",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="base model names (default: every one with a version)")
    ap.add_argument("--all", action="store_true", help="also base models without an h3dnas version")
    ap.add_argument("--h3dnas", type=Path, default=H3DNAS_MODELS)
    ap.add_argument("--no-latency", action="store_true")
    ap.add_argument("--provider", choices=["cpu", "cuda", "tensorrt", "coreml"])
    ap.add_argument("--providers", nargs="+", metavar="PROVIDER[:PRECISION]",
                    help="time every listed runtime in one run, one combined table, e.g. "
                         "--providers cpu cuda tensorrt tensorrt:fp16")
    ap.add_argument("--compute-units", choices=["ALL", "CPUAndNeuralEngine", "CPUAndGPU", "CPUOnly"],
                    help="CoreML: where it may run the model (default ALL)")
    ap.add_argument("--precision", choices=["fp32", "fp16"])
    ap.add_argument("--device-id", type=int)
    ap.add_argument("--threads", type=int)
    ap.add_argument("--rounds", type=int)
    ap.add_argument("--runs-per-round", type=int)
    ap.add_argument("--warmup", type=int)
    ap.add_argument("--out", type=Path, help="output file stem (default: results/compare/<host>_<provider>)")
    a = ap.parse_args(argv)

    from experiments.registry import paths
    from h3dnas.core import runtime
    from h3dnas.core.latency import device_info
    ws = paths()
    s = ws.settings()
    rt = {**s["runtime"]["latency"], **{k: v for k, v in {
        "provider": a.provider, "precision": a.precision, "device_id": a.device_id,
        "compute_units": a.compute_units}.items()
        if v is not None}}
    lat = {k: getattr(a, k) if getattr(a, k) is not None else s["latency"][k]
           for k in ("threads", "rounds", "runs_per_round", "warmup")}
    if rt.get("threads") and a.threads is None:
        lat["threads"] = rt["threads"]

    files = {n: f for n, f in ws.model_files.items() if f.exists()}   # models/sources.yaml paths
    found = versions(files, a.h3dnas)
    bases = sorted(files)
    chosen = a.names or (bases if a.all else sorted(n for n in found if n))
    unknown = [n for n in chosen if n not in files]
    if unknown:
        raise SystemExit(f"not in {ws.sources} (or file missing): {unknown}")
    plan = [(n, {"base": files[n],
                 **{str(p.relative_to(a.h3dnas).with_suffix("")).replace("\\", "/"): p
                    for p in found.get(n, [])}}) for n in chosen]
    if not a.names:
        plan += [(p.stem, {str(p.relative_to(a.h3dnas).with_suffix("")).replace("\\", "/"): p})
                 for p in found.get(None, [])]
    if not plan:
        raise SystemExit(f"nothing to compare: no ONNX files in {a.h3dnas} "
                         f"(name them <base name>[_<tag>].onnx or put them in <base name>/)")

    if a.providers:
        bad = [p for p in a.providers if p.partition(":")[0] not in ("cpu", "cuda", "tensorrt", "coreml")
               or p.partition(":")[2] not in ("", "fp32", "fp16")]
        if bad or a.provider or a.no_latency:
            raise SystemExit(f"--providers takes cpu|cuda|tensorrt|coreml[:fp32|fp16] and replaces "
                             f"--provider / --no-latency" + (f" (got {bad})" if bad else ""))
        groups, devices = run_providers(plan, a.providers, rt, lat)
        md = markdown_multi(groups, a.providers, devices, lat)
        print("\n" + md)
        stem = a.out or ws.results / "compare" / f"{platform.node()}_{'_'.join(p.replace(':', '-') for p in a.providers)}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        Path(f"{stem}.md").write_text(md, encoding="utf-8")
        Path(f"{stem}.json").write_text(json.dumps(
            {"host": platform.node(), "latency": lat, "providers": devices, "models": groups},
            indent=2, default=str), encoding="utf-8")
        print(f"written: {stem}.md, {stem}.json")
        return

    groups = []
    with runtime.using(latency=rt):
        for i, (name, files) in enumerate(plan, 1):
            t0 = time.time()
            rows = measure_group(files, None if a.no_latency else lat)
            groups.append({"model": name, "rows": rows})
            print(f"[{i}/{len(plan)}] {name}: {len(files)} file(s), {time.time() - t0:.0f}s", flush=True)
        device = {"host": platform.node(), **device_info(lat["threads"], runtime.current("latency"))}

    settings = None if a.no_latency else {**lat, "provider": rt["provider"],
                                          "precision": rt["precision"],
                                          **({"compute_units": rt["compute_units"]}
                                             if rt["provider"] == "coreml" else {})}
    md = markdown(groups, device, settings)
    print("\n" + md)
    tag = (f"{platform.node()}_{rt['provider']}" + ("_fp16" if rt["precision"] == "fp16" else "")
           + (f"_{rt['compute_units']}" if rt["provider"] == "coreml" else ""))
    stem = a.out or ws.results / "compare" / tag
    stem.parent.mkdir(parents=True, exist_ok=True)
    Path(f"{stem}.md").write_text(md, encoding="utf-8")
    Path(f"{stem}.json").write_text(json.dumps(
        {"device": device, "latency": settings, "models": groups},
        indent=2), encoding="utf-8")
    print(f"written: {stem}.md, {stem}.json")


if __name__ == "__main__":
    main()
