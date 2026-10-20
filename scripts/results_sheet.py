"""The paper's results workbook: models, datasets, and base vs h3dnas per run.

    python -m scripts.results_sheet                       # models in models/models.yaml
    python -m scripts.results_sheet --all                 # every model of h3dnas.yaml
    python -m scripts.results_sheet --out ../paper/h3dnas_results.xlsx

Sheets
  README    conventions (MACs, splits, latency, colours)
  Models    one row per model: task, dataset, source, published and measured base
            metric, params, MACs, FLOPs, size, opset, nodes, prunability,
            fine-tuning recipe, file, sha256
  Datasets  one row per dataset: classes, official and used split sizes, the
            split rule, the data module arguments, the models using it
  Results   one row per run: settings, base, h3dnas, the change (formulas) and
            on-device columns to fill in. Every full run in paths.runs
            (artifacts/<model>/<run>/results.json) is filled in; each model
            gets empty rows up to --rows-per-model for the runs still to do
  Summary   per model, formulas over the Results rows marked `done`

Facts come from the files (ONNX, h3dnas.yaml, models.yaml, run folders);
nothing is typed by hand. Yellow cells are for you to fill, grey cells are
formulas. Re-running rewrites the file: keep hand-filled numbers in a copy,
or in the run folders, from which they are read again.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]
H3DNAS = ROOT / "submodules" / "h3dnas"
sys.path.insert(0, str(H3DNAS))

# -- collect ----------------------------------------------------------------------


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _runs(runs_dir: Path) -> List[Dict[str, Any]]:
    """Every full run with a results.json: results + run.json + audit.json, oldest first."""
    out = []
    for res in sorted(runs_dir.glob("*/*/results.json")):
        d = res.parent
        r = json.loads(res.read_text(encoding="utf-8"))
        if r.get("quick"):
            continue
        dbg = d / "debug"
        run = json.loads((dbg / "run.json").read_text()) if (dbg / "run.json").exists() else {}
        audit = json.loads((dbg / "audit.json").read_text()) if (dbg / "audit.json").exists() else {}
        out.append({"dir": d, "res": r, "run": run, "audit": audit})
    return out


def _model_facts(spec, entry: Dict[str, Any], source: Dict[str, Any]) -> Dict[str, Any]:
    from experiments.pipeline import parse
    from h3dnas.analysis.prunability import analyse_prunability
    from h3dnas.core.inputs import resolve_input_spec
    f: Dict[str, Any] = {"onnx": spec.onnx, "exists": spec.onnx.exists(),
                         "note": source.get("note", ""), "sha256": source.get("sha256")}
    if not f["exists"]:
        return f
    m = parse(spec.onnx)
    f.update(params=m.parameters, macs=m.macs, size_mb=spec.onnx.stat().st_size / 2**20,
             opset=max((o.version for o in m.proto.opset_import if o.domain in ("", "ai.onnx")),
                       default=None),
             nodes=len(m.proto.graph.node),
             inputs="; ".join(f"{s.name} {list(s.shape)}" for s in resolve_input_spec(m.proto)))
    if not f["sha256"]:
        f["sha256"] = _sha256(spec.onnx)
    try:
        p = analyse_prunability(m)
        f.update(groups=p.groups, prunable_groups=p.prunable_groups,
                 pins=", ".join(f"{k}: {v}" for k, v in p.pins_by_reason.items()),
                 prunable_share=100 * p.prunable_param_fraction,
                 ffn=p.ffn_blocks, attention=p.attention_blocks)
    except Exception as exc:                       # report, do not stop the sheet
        f["note"] = (f["note"] + f"; prunability failed: {exc}").strip("; ")
    return f


def _dataset_facts(spec) -> Dict[str, Any]:
    dm = spec.data(0)
    d = dict(dm.describe())
    task = getattr(dm, "task", None)
    d["classes"] = getattr(task, "num_classes", None) or None
    if not d["classes"] and hasattr(dm, "splits"):            # classification: distinct labels
        import numpy as np
        d["classes"] = int(len(np.unique(np.asarray(dm.splits()["train"][1]))))
    d["metric"] = getattr(task, "metric", "top1")
    return d


def _recipe(spec) -> Dict[str, Any]:
    """The fine-tuning recipe with h3dnas.finetune's defaults filled in."""
    ft = dict(spec.recipe or {})
    if ft:
        ft.setdefault("optimizer", "sgd")
        if ft.get("weight_decay") is None:
            ft["weight_decay"] = 0.01 if ft["optimizer"] == "adamw" else 5e-4
    return ft


# -- workbook ---------------------------------------------------------------------

def build(out: Path, include_all: bool, rows_per_model: int, log=print) -> None:
    import yaml
    from openpyxl import Workbook
    from openpyxl.comments import Comment
    from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
    from openpyxl.utils import get_column_letter
    from openpyxl.worksheet.datavalidation import DataValidation
    from experiments.registry import get, paths, specs, unlisted

    ws_cfg = paths()
    names = sorted(specs()) + (sorted(unlisted()) if include_all else [])
    sources = {}
    for cat, entries in (yaml.safe_load(ws_cfg.sources.read_text(encoding="utf-8")) or {}).items():
        sources.update(entries or {})
    runs = _runs(ws_cfg.runs)
    latest_audit = {}
    for r in runs:                                   # newest wins
        if r["audit"]:
            latest_audit[r["res"]["model"]] = r["audit"]
    for n in names:                                  # fall back to the paper audits
        p = ws_cfg.results / "audit" / f"{n}.json"
        if n not in latest_audit and p.exists():
            latest_audit[n] = json.loads(p.read_text())

    HEAD = PatternFill("solid", fgColor="1F3864")
    GROUP = PatternFill("solid", fgColor="D9E1F2")
    FILL_ME = PatternFill("solid", fgColor="FFF2CC")
    FORMULA = PatternFill("solid", fgColor="EDEDED")
    thin = Side(style="thin", color="BFBFBF")
    BOX = Border(left=thin, right=thin, top=thin, bottom=thin)
    wb = Workbook()

    def sheet(title, cols, tab, groups=False):
        """cols: [(group, header, key, width, kind)], kind: value|fill|formula."""
        ws = wb.create_sheet(title)
        ws.sheet_properties.tabColor = tab
        hr = 2 if groups else 1
        if groups:
            start = 1
            for i in range(1, len(cols) + 1):
                if i == len(cols) or cols[i][0] != cols[start - 1][0]:
                    ws.cell(1, start, cols[start - 1][0]).fill = GROUP
                    ws.cell(1, start).font = Font(bold=True)
                    ws.cell(1, start).alignment = Alignment(horizontal="center")
                    if i > start:
                        ws.merge_cells(start_row=1, start_column=start, end_row=1, end_column=i)
                    start = i + 1
        for i, (_, h, _, w, _) in enumerate(cols, 1):
            c = ws.cell(hr, i, h)
            c.fill, c.font = HEAD, Font(bold=True, color="FFFFFF")
            c.alignment = Alignment(wrap_text=True, vertical="center")
            c.border = BOX
            ws.column_dimensions[get_column_letter(i)].width = w
        ws.row_dimensions[hr].height = 42
        ws.freeze_panes = ws.cell(hr + 1, 3)
        col = {k: get_column_letter(i) for i, (_, _, k, _, _) in enumerate(cols, 1)}
        return ws, col, hr

    def put(ws, cols, r, values: Dict[str, Any], fmt: Dict[str, str]):
        for i, (_, _, k, _, kind) in enumerate(cols, 1):
            c = ws.cell(r, i, values.get(k))
            c.border = BOX
            if kind == "fill":
                c.fill = FILL_ME
            elif kind == "formula":
                c.fill = FORMULA
            if k in fmt:
                c.number_format = fmt[k]

    INT, P2, P1, MS = "#,##0", "0.00", "0.0", "0.000"

    # README ------------------------------------------------------------------
    ws = wb.active
    ws.title = "README"
    lines = [
        ("h3dnas results workbook", ""),
        ("Generated by", "python -m scripts.results_sheet (naskit); facts from the ONNX files, "
                         "h3dnas.yaml, models/sources.yaml and the run folders (paths.runs)"),
        ("MACs", "multiply-accumulates per sample (batch 1), the shared h3dnas counter. "
                 "FLOPs = 2 x MACs. Torch-Pruning / DepGraph 'FLOPs' and thop counts are MACs"),
        ("Params", "learned float tensors (BatchNorm running statistics excluded)"),
        ("Validation", "held out of the official test set (stratified per class, seeded); used for "
                       "search and selection only"),
        ("Test", "the rest of the official test set, read once after selection"),
        ("Full official test", "(val x n_val + test x n_test) / (n_val + n_test): comparable with "
                               "published numbers. For the h3dnas model it includes the validation "
                               "samples it was selected on, so it is optimistic: use it for the base"),
        ("Latency (server)", "median ms, batch 1, interleaved rounds, on the 'Latency hardware' of "
                             "the run (not the deployment device)"),
        ("Latency (device)", "fill from bench.py results.json on the device (python -m examples "
                             "<name> deploy, then bench.py on the board)"),
        ("Change", "params / MACs / size: (h3dnas / base - 1) x 100; speedup: base / h3dnas latency; "
                   "metric change: h3dnas - base in points"),
        ("Colours", "yellow = fill in; grey = formula; white = from files"),
        ("Status", "done rows are averaged in Summary; todo rows are ignored"),
    ]
    for i, (a, b) in enumerate(lines, 1):
        ws.cell(i, 1, a).font = Font(bold=True, size=14 if i == 1 else 11)
        ws.cell(i, 2, b).alignment = Alignment(wrap_text=True, vertical="top")
    ws.column_dimensions["A"].width = 22
    ws.column_dimensions["B"].width = 110

    # Models ------------------------------------------------------------------
    MC = [("Model", "Model", "model", 22, "value"),
          ("Model", "Task", "task", 14, "value"),
          ("Model", "Modality", "modality", 9, "value"),
          ("Model", "Description", "description", 40, "value"),
          ("Model", "Dataset", "dataset", 15, "value"),
          ("Model", "Metric", "metric", 11, "value"),
          ("Model", "Input (batch 1)", "inputs", 26, "value"),
          ("Source", "Source checkpoint", "source", 40, "value"),
          ("Source", "Published metric (%)", "published", 11, "value"),
          ("Base, measured", "Val (%)", "val", 9, "value"),
          ("Base, measured", "Test (%)", "test", 9, "value"),
          ("Base, measured", "Test mean class acc. (%)", "test_macc", 10, "value"),
          ("Base, measured", "Full official test (%)", "full", 11, "value"),
          ("Base, measured", "Audit seed", "audit_seed", 8, "value"),
          ("Size", "Params", "params", 13, "value"),
          ("Size", "MACs / sample", "macs", 15, "value"),
          ("Size", "FLOPs / sample (2 x MACs)", "flops", 15, "formula"),
          ("Size", "ONNX size (MB)", "size_mb", 10, "value"),
          ("Graph", "Opset", "opset", 7, "value"),
          ("Graph", "Nodes", "nodes", 8, "value"),
          ("Prunability", "Channel groups", "groups", 9, "value"),
          ("Prunability", "Prunable groups", "prunable_groups", 9, "value"),
          ("Prunability", "Pinned groups (reason: n)", "pins", 34, "value"),
          ("Prunability", "Prunable param share (%)", "prunable_share", 10, "value"),
          ("Prunability", "FFN blocks", "ffn", 7, "value"),
          ("Prunability", "Attention blocks", "attention", 9, "value"),
          ("Fine-tuning", "Epochs", "ft_epochs", 8, "value"),
          ("Fine-tuning", "LR", "ft_lr", 9, "value"),
          ("Fine-tuning", "Optimizer", "ft_opt", 10, "value"),
          ("Fine-tuning", "Weight decay", "ft_wd", 9, "value"),
          ("File", "ONNX file", "file", 34, "value"),
          ("File", "sha256", "sha256", 20, "value"),
          ("File", "Notes", "note", 60, "value")]
    ws, col, hr = sheet("Models", MC, "4472C4", groups=True)
    ds_facts: Dict[str, Dict[str, Any]] = {}
    ds_users: Dict[str, List[str]] = defaultdict(list)
    ds_key_of: Dict[str, str] = {}
    for i, n in enumerate(names):
        log(f"[models] {n}")
        spec = get(n)
        entry = ws_cfg.model(n)
        key = f"{spec.data_type} {json.dumps(spec.data_params, sort_keys=True, default=str)}"
        ds_key_of[n] = key
        ds_users[key].append(n)
        if key not in ds_facts:
            try:
                ds_facts[key] = {**_dataset_facts(spec), "spec": spec}
            except Exception as exc:
                ds_facts[key] = {"error": str(exc).splitlines()[0][:160], "spec": spec}
        f = _model_facts(spec, entry, sources.get(n, {}))
        a = latest_audit.get(n, {})
        v, t = (a.get("val") or {}), (a.get("test") or {})
        pick = lambda d: next((d[k] for k in ("1", "top1", "miou", "instance_miou") if k in d), None)
        val, test = pick(v), pick(t)
        dd = a.get("data") or {}
        full = (100 * (val * dd["val"] + test * dd["test"]) / (dd["val"] + dd["test"])
                if None not in (val, test) and dd.get("val") and dd.get("test")
                and ds_facts[key].get("metric", "top1") == "top1" else None)
        r = hr + 1 + i
        ft = _recipe(spec)
        put(ws, MC, r, {
            "model": n, "task": spec.task, "modality": spec.modality,
            "description": entry.get("description"), "dataset": spec.data_type,
            "metric": ds_facts[key].get("metric", "top1"), "inputs": f.get("inputs"),
            "source": spec.source, "published": spec.published,
            "val": round(100 * val, 2) if val is not None else None,
            "test": round(100 * test, 2) if test is not None else None,
            "test_macc": round(100 * t["macc"], 2) if t.get("macc") is not None else None,
            "full": round(full, 2) if full is not None else None, "audit_seed": dd.get("seed"),
            "params": f.get("params"), "macs": f.get("macs"),
            "flops": f'=IF({col["macs"]}{r}="","",2*{col["macs"]}{r})',
            "size_mb": round(f["size_mb"], 2) if f.get("size_mb") else None,
            "opset": f.get("opset"), "nodes": f.get("nodes"), "groups": f.get("groups"),
            "prunable_groups": f.get("prunable_groups"), "pins": f.get("pins"),
            "prunable_share": round(f["prunable_share"], 1) if f.get("prunable_share") is not None else None,
            "ffn": f.get("ffn"), "attention": f.get("attention"),
            "ft_epochs": ft.get("epochs"), "ft_lr": ft.get("lr"), "ft_opt": ft.get("optimizer"),
            "ft_wd": ft.get("weight_decay"),
            "file": str(Path(spec.onnx).relative_to(ROOT)) if Path(spec.onnx).is_relative_to(ROOT)
            else str(spec.onnx),
            "sha256": f.get("sha256"),
            "note": f.get("note") if f["exists"] else f"ONNX missing; {f.get('note', '')}"},
            {"params": INT, "macs": INT, "flops": INT, "size_mb": P2, "published": P2, "val": P2,
             "test": P2, "test_macc": P2, "full": P2})
    ws.auto_filter.ref = f"A{hr}:{get_column_letter(len(MC))}{ws.max_row}"

    # Datasets ----------------------------------------------------------------
    DC = [("Dataset", "Dataset", "dataset", 16, "value"),
          ("Dataset", "Task", "task", 14, "value"),
          ("Dataset", "Modality", "modality", 9, "value"),
          ("Dataset", "Classes", "classes", 8, "value"),
          ("Dataset", "Metric", "metric", 11, "value"),
          ("Official", "Train", "off_train", 10, "value"),
          ("Official", "Test", "off_test", 10, "value"),
          ("Used", "Train", "train", 10, "value"),
          ("Used", "Validation", "val", 10, "value"),
          ("Used", "Test", "test", 10, "value"),
          ("Used", "Validation per class", "vpc", 10, "value"),
          ("Used", "Validation fraction (proportional)", "vfrac", 11, "value"),
          ("Used", "Validation taken from", "val_from", 11, "value"),
          ("Used", "Split rule", "split", 60, "value"),
          ("Data module", "Arguments (h3dnas.yaml)", "args", 50, "value"),
          ("Data module", "Used by models", "users", 34, "value"),
          ("Data module", "Notes", "note", 40, "value")]
    ws, col, hr = sheet("Datasets", DC, "70AD47", groups=True)
    for i, (key, d) in enumerate(ds_facts.items()):
        spec = d["spec"]
        tr, va, te, vf = d.get("train"), d.get("val"), d.get("test"), d.get("val_from")
        args = {k: v for k, v in spec.data_params.items()}
        put(ws, DC, hr + 1 + i, {
            "dataset": spec.data_type, "task": spec.task, "modality": spec.modality,
            "classes": d.get("classes"), "metric": d.get("metric"),
            "off_train": (tr + va) if vf == "train" and None not in (tr, va) else tr,
            "off_test": (te + va) if vf == "test" and None not in (te, va) else te,
            "train": tr, "val": va, "test": te, "vpc": d.get("val_per_class"), "vfrac": d.get("val_fraction"), "val_from": vf,
            "split": d.get("split"),
            "args": json.dumps(args, default=str), "users": ", ".join(ds_users[key]),
            "note": d.get("error", "")},
            {k: INT for k in ("off_train", "off_test", "train", "val", "test")})

    # Results -----------------------------------------------------------------
    RC = [("Run", "Model", "model", 20, "value"),
          ("Run", "Dataset", "dataset", 13, "value"),
          ("Run", "Seed", "seed", 6, "fill"),
          ("Run", "Run folder", "run", 21, "fill"),
          ("Run", "Status", "status", 8, "fill"),
          ("Settings", "Strategy", "strategy", 18, "fill"),
          ("Settings", "Fitness", "fitness", 12, "fill"),
          ("Settings", "Fitness split (of validation)", "fit_split", 9, "fill"),
          ("Settings", "Budgets (uniform ratios / MAC fractions)", "budgets", 16, "fill"),
          ("Settings", "Max drop (pts)", "max_drop", 8, "fill"),
          ("Settings", "Device budget", "device", 12, "fill"),
          ("Settings", "FT epochs", "ft_epochs", 7, "fill"),
          ("Settings", "FT lr", "ft_lr", 8, "fill"),
          ("Settings", "FT optimizer", "ft_opt", 9, "fill"),
          ("Settings", "FT weight decay", "ft_wd", 8, "fill"),
          ("Settings", "Selection rule", "rule", 30, "fill"),
          ("Split", "Val n", "n_val", 7, "fill"),
          ("Split", "Test n", "n_test", 7, "fill"),
          ("Base", "Params", "b_params", 12, "fill"),
          ("Base", "MACs", "b_macs", 14, "fill"),
          ("Base", "Size (MB)", "b_size", 8, "fill"),
          ("Base", "Latency (ms)", "b_lat", 9, "fill"),
          ("Base", "Val (%)", "b_val", 8, "fill"),
          ("Base", "Test (%)", "b_test", 8, "fill"),
          ("Base", "Test mean class acc. (%)", "b_macc", 9, "fill"),
          ("Base", "Full official test (%)", "b_full", 10, "formula"),
          ("h3dnas", "Model", "x_name", 20, "fill"),
          ("h3dnas", "Fine-tuned", "x_ft", 8, "fill"),
          ("h3dnas", "Params", "x_params", 12, "fill"),
          ("h3dnas", "MACs", "x_macs", 14, "fill"),
          ("h3dnas", "Size (MB)", "x_size", 8, "fill"),
          ("h3dnas", "Latency (ms)", "x_lat", 9, "fill"),
          ("h3dnas", "Val (%)", "x_val", 8, "fill"),
          ("h3dnas", "Test (%)", "x_test", 8, "fill"),
          ("h3dnas", "Test mean class acc. (%)", "x_macc", 9, "fill"),
          ("h3dnas", "Full official test (%, includes val)", "x_full", 10, "formula"),
          ("Change vs base", "Params (%)", "c_params", 9, "formula"),
          ("Change vs base", "MACs (%)", "c_macs", 9, "formula"),
          ("Change vs base", "Size (%)", "c_size", 9, "formula"),
          ("Change vs base", "Speedup (x)", "c_speed", 9, "formula"),
          ("Change vs base", "Val (pts)", "c_val", 8, "formula"),
          ("Change vs base", "Test (pts)", "c_test", 8, "formula"),
          ("Change vs base", "Test mean class acc. (pts)", "c_macc", 9, "formula"),
          ("Change vs base", "Latency hardware", "lat_hw", 30, "fill"),
          ("On device", "Device", "d_name", 18, "fill"),
          ("On device", "Provider / precision", "d_prov", 14, "fill"),
          ("On device", "Base latency (ms)", "d_b_lat", 9, "fill"),
          ("On device", "h3dnas latency (ms)", "d_x_lat", 9, "fill"),
          ("On device", "Speedup (x)", "d_speed", 9, "formula"),
          ("On device", "Base metric (%)", "d_b_metric", 9, "fill"),
          ("On device", "h3dnas metric (%)", "d_x_metric", 9, "fill"),
          ("Notes", "Notes", "notes", 40, "fill")]
    ws, col, hr = sheet("Results", RC, "FFC000", groups=True)
    by_model = defaultdict(list)
    for r in runs:
        by_model[r["res"]["model"]].append(r)
    row = hr + 1
    fmt = {**{k: INT for k in ("b_params", "b_macs", "x_params", "x_macs", "n_val", "n_test")},
           **{k: P2 for k in ("b_size", "x_size", "b_val", "b_test", "b_full", "x_val", "x_test",
                              "b_macc", "x_macc", "c_macc",
                              "x_full", "c_params", "c_macs", "c_size", "c_speed", "c_val", "c_test",
                              "d_speed", "d_b_metric", "d_x_metric")},
           **{k: MS for k in ("b_lat", "x_lat", "d_b_lat", "d_x_lat")}}

    def formulas(r):
        C = {k: f"{v}{r}" for k, v in col.items()}

        def ratio(a, b):
            return f'=IF(OR({a}="",{b}=""),"",ROUND(({a}/{b}-1)*100,2))'

        def full(v, t):
            return (f'=IF(OR({v}="",{t}="",{C["n_val"]}="",{C["n_test"]}=""),"",ROUND(({v}*{C["n_val"]}'
                    f'+{t}*{C["n_test"]})/({C["n_val"]}+{C["n_test"]}),2))')
        return {
            "b_full": full(C["b_val"], C["b_test"]), "x_full": full(C["x_val"], C["x_test"]),
            "c_params": ratio(C["x_params"], C["b_params"]), "c_macs": ratio(C["x_macs"], C["b_macs"]),
            "c_size": ratio(C["x_size"], C["b_size"]),
            "c_speed": f'=IF(OR({C["b_lat"]}="",{C["x_lat"]}=""),"",ROUND({C["b_lat"]}/{C["x_lat"]},2))',
            "c_val": f'=IF(OR({C["x_val"]}="",{C["b_val"]}=""),"",ROUND({C["x_val"]}-{C["b_val"]},2))',
            "c_test": f'=IF(OR({C["x_test"]}="",{C["b_test"]}=""),"",ROUND({C["x_test"]}-{C["b_test"]},2))',
            "c_macc": f'=IF(OR({C["x_macc"]}="",{C["b_macc"]}=""),"",ROUND({C["x_macc"]}-{C["b_macc"]},2))',
            "d_speed": f'=IF(OR({C["d_b_lat"]}="",{C["d_x_lat"]}=""),"",ROUND({C["d_b_lat"]}/{C["d_x_lat"]},2))',
        }

    for n in names:
        spec = get(n)
        done = by_model.get(n, [])
        for rec in done:
            res, run = rec["res"], rec["run"]
            st = run.get("settings") or {}
            srch = st.get("search") or {}
            ft = dict((res.get("settings") or {}).get("finetune") or run.get("finetune") or {})
            if ft and ft.get("weight_decay") is None:            # h3dnas.finetune's default
                ft["weight_decay"] = 0.01 if ft.get("optimizer") == "adamw" else 5e-4
            b, x = res["base"], res["best"]
            put(ws, RC, row, {
                "model": n, "dataset": spec.data_type, "seed": res.get("seed"),
                "run": rec["dir"].name, "status": "done",
                "strategy": srch.get("strategy"), "fitness": srch.get("fitness"),
                "fit_split": srch.get("fitness_split"),
                "budgets": json.dumps(srch.get("budgets") or srch.get("ratios")),
                "max_drop": (st.get("objective") or {}).get("max_drop"), "device": st.get("device"),
                "ft_epochs": ft.get("epochs"), "ft_lr": ft.get("lr"), "ft_opt": ft.get("optimizer"),
                "ft_wd": ft.get("weight_decay"), "rule": res.get("selection"),
                "n_val": (res.get("data") or {}).get("val"), "n_test": (res.get("data") or {}).get("test"),
                "b_params": b.get("params"), "b_macs": b.get("macs"), "b_size": b.get("size_mb"),
                "b_lat": b.get("latency_ms"), "b_val": b.get("val"), "b_test": b.get("test"),
                "b_macc": b.get("test_macc"),
                "x_name": x.get("name"), "x_ft": "yes" if x.get("fine_tuned") else "no",
                "x_params": x.get("params"), "x_macs": x.get("macs"), "x_size": x.get("size_mb"),
                "x_lat": x.get("latency_ms"), "x_val": x.get("val"), "x_test": x.get("test"),
                "x_macc": x.get("test_macc"),
                "lat_hw": res.get("latency_device"), **formulas(row)}, fmt)
            row += 1
        for _ in range(max(0, rows_per_model - len(done))):
            put(ws, RC, row, {"model": n, "dataset": spec.data_type, "status": "todo",
                              **formulas(row)}, fmt)
            row += 1
    last = row - 1
    dv = DataValidation(type="list", formula1='"done,todo,running,failed"', allow_blank=True)
    ws.add_data_validation(dv)
    dv.add(f"{col['status']}{hr + 1}:{col['status']}{max(last, hr + 1)}")
    ws.auto_filter.ref = f"A{hr}:{get_column_letter(len(RC))}{max(last, hr + 1)}"
    ws.cell(hr, RC.index(next(c for c in RC if c[2] == "x_full")) + 1).comment = Comment(
        "Includes the validation samples the model was selected on: optimistic. "
        "Report the held-out Test column for h3dnas.", "h3dnas")
    rcol, rfirst, rlast = col, hr + 1, max(last, hr + 1)

    # Summary -----------------------------------------------------------------
    SC = [("Model", "Model", "model", 20, "value"),
          ("Model", "Dataset", "dataset", 13, "value"),
          ("Runs", "Runs done", "n", 8, "formula"),
          ("Size", "Params change mean (%)", "params", 11, "formula"),
          ("Size", "MACs change mean (%)", "macs", 11, "formula"),
          ("Server", "Speedup mean (x)", "speed", 10, "formula"),
          ("Accuracy", "Base test mean (%)", "b_test", 10, "formula"),
          ("Accuracy", "h3dnas test mean (%)", "x_test", 10, "formula"),
          ("Accuracy", "Test change mean (pts)", "c_test", 10, "formula"),
          ("Accuracy", "Test change min (pts)", "c_min", 10, "formula"),
          ("Accuracy", "Test change max (pts)", "c_max", 10, "formula"),
          ("Accuracy", "Test mean class acc. change mean (pts)", "c_macc", 11, "formula"),
          ("Device", "Device speedup mean (x)", "d_speed", 10, "formula")]
    ws, col, hr = sheet("Summary", SC, "A5A5A5", groups=True)
    def rng(k):
        return f"Results!${rcol[k]}${rfirst}:${rcol[k]}${rlast}"

    for i, n in enumerate(names):
        r = hr + 1 + i
        c = f'{rng("model")},$A{r},{rng("status")},"done"'

        def avg(k):
            return f'=IFERROR(ROUND(AVERAGEIFS({rng(k)},{c}),2),"")'
        put(ws, SC, r, {
            "model": n, "dataset": get(n).data_type, "n": f"=COUNTIFS({c})",
            "params": avg("c_params"), "macs": avg("c_macs"), "speed": avg("c_speed"),
            "b_test": avg("b_test"), "x_test": avg("x_test"), "c_test": avg("c_test"),
            "c_min": f'=IF(C{r}=0,"",MINIFS({rng("c_test")},{c}))',
            "c_max": f'=IF(C{r}=0,"",MAXIFS({rng("c_test")},{c}))',
            "c_macc": avg("c_macc"), "d_speed": avg("d_speed")},
            {k: P2 for k in ("params", "macs", "speed", "b_test", "x_test", "c_test", "c_min",
                             "c_max", "c_macc", "d_speed")})

    out.parent.mkdir(parents=True, exist_ok=True)
    wb.save(out)
    log(f"wrote {out}: {len(names)} models, {len(ds_facts)} datasets, {len(runs)} full runs "
        f"from {ws_cfg.runs}")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="python -m scripts.results_sheet",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, default=ROOT / "results" / "h3dnas_results.xlsx")
    ap.add_argument("--all", action="store_true",
                    help="every model of h3dnas.yaml, not only those in models/sources.yaml")
    ap.add_argument("--rows-per-model", type=int, default=3,
                    help="Results rows per model, filled runs included (default 3 = 3 seeds)")
    a = ap.parse_args(argv)
    build(a.out, a.all, a.rows_per_model)


if __name__ == "__main__":
    main()