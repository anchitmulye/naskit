"""swan models vs the INT8-quantized base model (and swan + INT8), on one data split.

    python -m scripts.compare_quant cifar_resnet56 --seed 105 \
        --ours runs/.../models/cifar_resnet56_gs_70pct_ft.onnx runs/.../models/cifar_resnet56_gs_50pct_ft.onnx

Rows: the base model (fp32), the base quantized to INT8, every --ours model (fp32) and, unless
--no-int8-ours, every --ours model quantized the same way. Quantization is ONNX Runtime static
quantization (QDQ, signed 8-bit activations and per-channel weights, MinMax calibration unless
--calibrate says otherwise) after its pre-processing, calibrated on --calib-samples training
images without augmentation, so neither validation nor test data is read for it.

Columns: parameters (an INT8 model: those of its source) and MACs (an INT8 model has the same
MACs, in 8-bit), file size, CPU latency (batch 1, interleaved, swan.core.latency, the `latency` runtime of
swan.yaml), validation and test top-1. The split is the model's data module at --seed: pass the
seed of the run the --ours models come from, so the test images are the run's test images.
The test split is read once, after everything else.
INT8 latency depends on the CPU: without int8 dot-product instructions (VNNI; e.g. AMD Zen 2) a
QDQ model can be slower than fp32 at batch 1; run it on the deployment device.

Output: the table on screen and results/compare_quant/<model>_seed<seed>.md / .json; the
quantized files in results/compare_quant/<model>/.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
SWAN = ROOT / "submodules" / "swan"
SWAN_MODELS = ROOT / "models" / "swan"
sys.path.insert(0, str(SWAN))

from experiments.registry import get, paths
from experiments.pipeline import parse as _parse, write_json as _write


def _fit_batch(calib: List[Dict], want: int) -> List[Dict]:
    """Re-chunk calibration feeds to `want` samples each.

    The pool loader yields batches of 32. A point-network export declares a symbolic batch on
    its input but carries concrete batch-1 shapes inside its set-abstraction reshapes, so the
    declared dimension cannot be trusted and ORT fails during calibration with
    "input shape {32}, requested shape {1,1}". Calibration only collects activation ranges, so
    feeding one sample at a time is equivalent; it is simply slower, hence the retry rather
    than doing it always.
    """
    have = next(iter(calib[0].values())).shape[0]
    if have == want:
        return calib
    out = []
    for feed in calib:
        n = next(iter(feed.values())).shape[0]
        for i in range(0, n - want + 1, want):
            out.append({k: v[i:i + want] for k, v in feed.items()})
    return out


def _upgraded(src: Path, tmpdir: Path, min_opset: int = 13) -> Path:
    """Raise the default opset to `min_opset` if the graph is below it, else return `src`.

    Per-channel quantization writes an `axis` attribute on QuantizeLinear/DequantizeLinear,
    which only exists from opset 13. Quantizing an older graph therefore produces a model the
    runtime refuses to load ("Unrecognized attribute: axis for operator DequantizeLinear") --
    the quantization itself succeeds, so the failure only shows up later, at session creation.
    """
    import onnx
    m = onnx.load(str(src))
    cur = max((o.version for o in m.opset_import if o.domain in ("", "ai.onnx")), default=0)
    if cur >= min_opset:
        return src
    out = tmpdir / "opset13.onnx"
    try:
        from onnx import version_converter
        onnx.save(version_converter.convert_version(m, min_opset), str(out))
        print(f"    [quantize] {src.name}: opset {cur} -> {min_opset} for per-channel QDQ",
              flush=True)
        return out
    except Exception as e:
        print(f"    [quantize] {src.name}: opset {cur} < {min_opset} and conversion failed "
              f"({type(e).__name__}); will fall back to per-tensor", flush=True)
        return src


def quantize(src: Path, dst: Path, calib: List[Dict], method: str) -> Path:
    from onnxruntime.quantization import (CalibrationDataReader, CalibrationMethod, QuantFormat,
                                          QuantType, quantize_static)
    from onnxruntime.quantization.shape_inference import quant_pre_process

    class Reader(CalibrationDataReader):
        def __init__(self):
            self.it = iter(calib)

        def get_next(self):
            return next(self.it, None)

    import tempfile
    import onnx
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as d:
        # pre-processing folds BatchNorm into the convolutions; with its output, ORT's calibration
        # graph only serializes with external data, so quantize here and save one file below
        pre, q = Path(d) / "pre.onnx", Path(d) / "q.onnx"

        # Per-channel QDQ needs opset >= 13 (the `axis` attribute); raise it when we can.
        src = _upgraded(src, Path(d))
        per_channel_ok = max(
            (o.version for o in onnx.load(str(src)).opset_import
             if o.domain in ("", "ai.onnx")), default=0) >= 13

        def _quantize_from(model_path, feeds, per_channel=None):
            class R(CalibrationDataReader):
                def __init__(self):
                    self.it = iter(feeds)

                def get_next(self):
                    return next(self.it, None)

            pc = per_channel_ok if per_channel is None else per_channel
            quantize_static(str(model_path), str(q), R(), quant_format=QuantFormat.QDQ,
                            per_channel=pc, activation_type=QuantType.QInt8,
                            weight_type=QuantType.QInt8, use_external_data_format=True,
                            calibrate_method={"minmax": CalibrationMethod.MinMax,
                                              "entropy": CalibrationMethod.Entropy,
                                              "percentile": CalibrationMethod.Percentile}[method])
            # quantize_static succeeds on a graph the runtime will later refuse, so load the
            # result here: a failure must be attributed to this attempt, not to evaluation.
            import onnxruntime as _ort
            _ort.InferenceSession(str(q), providers=["CPUExecutionProvider"])

        # Pre-processing folds BatchNorm into the convolutions, but it is an optimisation, not a
        # requirement -- quantize_static runs its own shape inference. Point networks defeat it
        # in two ways: symbolic shape inference raises on their Tile operands, and with it
        # skipped the remaining passes rewrite a data-dependent Reshape in PointNet++'s
        # set-abstraction layers into one that fails *during calibration*, i.e. inside
        # quantize_static rather than in pre-processing. So each stage is judged by whether the
        # whole quantization succeeds, not by whether pre-processing returned.
        attempts = [
            ("full", lambda: (quant_pre_process(str(src), str(pre)), pre)[1]),
            ("skip_symbolic", lambda: (quant_pre_process(str(src), str(pre),
                                                         skip_symbolic_shape=True,
                                                         skip_optimization=True), pre)[1]),
            ("none", lambda: src),
        ]
        # Calibration batch is the second axis of the retry: a graph that only runs at batch 1
        # fails the same way at every pre-processing stage, so trying batch 1 is what rescues it.
        batches = [("loader", calib), ("one", _fit_batch(calib, 1))]
        if not per_channel_ok:
            print(f"    [quantize] {Path(src).name}: opset < 13, using per-tensor weights",
                  flush=True)
        last, done = None, False
        for bname, feeds in batches:
            for stage, build in attempts:
                try:
                    _quantize_from(build(), feeds)
                    if stage != "full" or bname != "loader":
                        print(f"    [quantize] {Path(src).name}: pre-process={stage}, "
                              f"calibration batch={bname}", flush=True)
                    done = True
                    break
                except Exception as e:
                    last = e
                    print(f"    [quantize] {Path(src).name}: pre-process={stage}, batch={bname} "
                          f"failed ({type(e).__name__}: {str(e)[:110]})", flush=True)
            if done:
                break
        if not done:
            raise RuntimeError(
                f"every quantization path failed for {Path(src).name}; last error: {last}")
        onnx.save(onnx.load(str(q)), str(dst))
    return dst


def run(name: str, ours: List[Path], seed: int, calib_samples: int, method: str,
        int8_ours: bool, threads: int) -> Dict:
    from swan.core.inputs import resolve_input_spec
    from swan.core.latency import measure
    from swan.core.metrics import per_sample_inputs
    from swan.datapipe import eval_accuracy, feeds_from_loader

    spec = get(name)
    base = _parse(spec.onnx)
    dm = spec.data(seed)
    qdir = paths().results / "compare_quant" / name
    calib = feeds_from_loader(base, dm.pool_loader(), n_samples=calib_samples)   # training images

    files = {"base fp32": Path(spec.onnx)}
    files["base int8"] = quantize(Path(spec.onnx), qdir / f"{name}_int8.onnx", calib, method)
    for p in ours:
        files[f"{p.stem} fp32"] = p
        if int8_ours:
            files[f"{p.stem} int8"] = quantize(p, qdir / f"{p.stem}_int8.onnx", calib, method)

    models = {k: _parse(p) for k, p in files.items()}
    val = dm.val_loader()
    # an INT8 file holds the same weights as its source (the counter sees only float ones)
    src = lambda k: k.rsplit(" ", 1)[0] + " fp32"
    rows = {k: {"file": str(p), "precision": k.rsplit(" ", 1)[1], "params": models[src(k)].parameters,
                "macs": models[k].macs, "size_mb": round(p.stat().st_size / 2**20, 3),
                "val": eval_accuracy(models[k], val)} for k, p in files.items()}
    _, feed = per_sample_inputs(resolve_input_spec(base.proto))
    lat = measure(models, feed, threads=threads, rounds=5, runs_per_round=20)
    for k, v in lat.results.items():
        rows[k].update(latency_ms=v.median_ms, latency_iqr_ms=v.iqr_ms)
    # Test split: read once, after every measurement above.
    test = dm.test_loader()
    for k in rows:
        rows[k]["test"] = eval_accuracy(models[k], test)
    return {"model": name, "seed": seed, "data": dm.describe(), "host": platform.node(),
            "latency_device": lat.device, "threads": threads,
            "quantization": {"tool": "onnxruntime.quantization.quantize_static", "format": "QDQ",
                             "activations": "int8", "weights": "int8 per channel",
                             "calibration": f"{method}, {calib_samples} training images"},
            "rows": rows}


def table(rec: Dict) -> str:
    b = rec["rows"]["base fp32"]
    out = [f"{rec['model']} (seed {rec['seed']}, {rec['latency_device']}, {rec['threads']} threads); "
           f"quantization: {rec['quantization']['calibration']}", "",
           "| Model | Prec. | Params | MACs | Size MB | Latency ms | Speed-up | Val | Test | dTest |",
           "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for k, r in rec["rows"].items():
        out.append(f"| {k.rsplit(' ', 1)[0]} | {r['precision']} | {r['params'] / 1e6:.3f}M | "
                   f"{r['macs'] / 1e6:.1f}M | {r['size_mb']:.2f} | {r['latency_ms']:.3f} | "
                   f"{b['latency_ms'] / r['latency_ms']:.2f}x | {100 * r['val']:.2f} | "
                   f"{100 * r['test']:.2f} | {100 * (r['test'] - b['test']):+.2f} |")
    return "\n".join(out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("model")
    ap.add_argument("--ours", type=Path, nargs="+", required=True, help="swan ONNX files of this model")
    ap.add_argument("--seed", type=int, required=True, help="seed of the run the --ours models come from")
    ap.add_argument("--calib-samples", type=int, default=1024)
    ap.add_argument("--calibrate", choices=["minmax", "entropy", "percentile"], default="minmax")
    ap.add_argument("--no-int8-ours", action="store_true", help="do not also quantize the --ours models")
    ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args()
    t0 = time.time()
    rec = run(a.model, a.ours, a.seed, a.calib_samples, a.calibrate, not a.no_int8_ours, a.threads)
    rec["seconds"] = round(time.time() - t0, 1)
    stem = paths().results / "compare_quant" / f"{a.model}_seed{a.seed}"
    _write(stem.with_suffix(".json"), rec)
    md = table(rec)
    stem.with_suffix(".md").write_text(md + "\n", encoding="utf-8")
    print(md)
