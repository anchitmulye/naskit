"""
Generate results/README.md from all analysis.json files in results/.

Scans results/<model>/<run>/analysis.json, extracts key metrics for
base and NAS models, and writes a formatted markdown table.

Usage (from naskit/ root):
    python generate_results_readme.py
    python generate_results_readme.py --out results/RESULTS.md
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
DEFAULT_OUT  = RESULTS_DIR / "RESULTS.md"


def load_analysis(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_notes(run_dir: Path) -> str:
    notes_file = run_dir / "notes.md"
    if notes_file.exists():
        return notes_file.read_text().strip()
    return ""


def fmt(val, decimals=2, suffix=""):
    if val is None:
        return "—"
    return f"{val:.{decimals}f}{suffix}"


def collect_rows() -> list[dict]:
    rows = []
    for model_dir in sorted(RESULTS_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        model = model_dir.name

        for run_dir in sorted(model_dir.iterdir()):
            if not run_dir.is_dir():
                continue
            run = run_dir.name

            analysis_path = run_dir / "analysis.json"
            if not analysis_path.exists():
                continue

            data = load_analysis(analysis_path)
            if not data:
                continue

            b = data.get("base", {})
            n = data.get("nas", {})
            notes = load_notes(run_dir)

            rows.append({
                "model":        model,
                "run":          run,
                "notes":        notes,
                "base_acc":     b.get("top1_accuracy"),
                "nas_acc":      n.get("top1_accuracy"),
                "delta_pp":     data.get("accuracy_delta_pp"),
                "base_params":  b.get("parameters"),
                "nas_params":   n.get("parameters"),
                "param_red":    data.get("param_reduction_pct"),
                "flops_red":    data.get("flops_reduction_pct"),
                "base_lat":     b.get("latency_ms_p50"),
                "nas_lat":      n.get("latency_ms_p50"),
                "speedup":      data.get("latency_speedup_x"),
                "size_red":     data.get("size_reduction_pct"),
                "jetson":       data.get("jetson_feasible"),
            })

    return rows


def render_table(rows: list[dict]) -> str:
    if not rows:
        return "_No analysis.json results found._\n"

    lines = []
    header = (
        "| Model | Run | Base acc | NAS acc | Δ (pp) | "
        "Param↓ | FLOPs↓ | Base lat | NAS lat | Speedup | Jetson | Notes |"
    )
    sep = (
        "|---|---|---|---|---|---|---|---|---|---|---|---|"
    )
    lines.append(header)
    lines.append(sep)

    for r in rows:
        delta     = r["delta_pp"]
        delta_str = f"**{delta:+.2f}**" if delta is not None else "—"
        if delta is not None and delta > 0:
            delta_str = f"**+{delta:.2f}** ✓"
        elif delta is not None and delta < -1:
            delta_str = f"{delta:.2f}"

        jetson = "✓" if r["jetson"] else ("✗" if r["jetson"] is False else "—")
        notes  = r["notes"][:50] + "..." if len(r["notes"]) > 50 else r["notes"]

        line = (
            f"| {r['model']} "
            f"| {r['run']} "
            f"| {fmt(r['base_acc'])}% "
            f"| {fmt(r['nas_acc'])}% "
            f"| {delta_str} "
            f"| {fmt(r['param_red'], 1)}% "
            f"| {fmt(r['flops_red'], 1)}% "
            f"| {fmt(r['base_lat'], 3)}ms "
            f"| {fmt(r['nas_lat'], 3)}ms "
            f"| {fmt(r['speedup'], 2)}× "
            f"| {jetson} "
            f"| {notes} |"
        )
        lines.append(line)

    return "\n".join(lines) + "\n"


def render_per_model(rows: list[dict]) -> str:
    """Detailed per-model sections with per-class accuracy if available."""
    sections = []
    models = sorted({r["model"] for r in rows})

    for model in models:
        model_rows = [r for r in rows if r["model"] == model]
        sections.append(f"### {model}\n")

        for r in model_rows:
            sections.append(f"**{r['run']}**")
            if r["notes"]:
                sections.append(f"_{r['notes']}_")
            sections.append("")
            sections.append(
                f"- Base: {fmt(r['base_acc'])}%  "
                f"| {r['base_params']:,} params  "
                f"| {fmt(r['base_lat'], 3)}ms latency"
            )
            sections.append(
                f"- NAS:  {fmt(r['nas_acc'])}%  "
                f"| {r['nas_params']:,} params  "
                f"| {fmt(r['nas_lat'], 3)}ms latency"
            )
            sections.append(
                f"- Δ accuracy: **{r['delta_pp']:+.2f}pp**  "
                f"| Param↓ {fmt(r['param_red'], 1)}%  "
                f"| FLOPs↓ {fmt(r['flops_red'], 1)}%  "
                f"| {fmt(r['speedup'], 2)}× speedup"
            )
            if r["jetson"] is True:
                sections.append("- ✓ Jetson Orin Nano 8GB feasible")
            sections.append("")

    return "\n".join(sections)


def main():
    p = argparse.ArgumentParser(description="Generate results README from analysis JSONs")
    p.add_argument("--out", default=str(DEFAULT_OUT), help="Output markdown file")
    args = p.parse_args()

    rows = collect_rows()
    out  = Path(args.out)

    content = f"""# h3dnas — Experimental Results

Auto-generated from `results/*/analysis.json`. Run `python generate_results_readme.py` to update.

---

## Summary Table

{render_table(rows)}

> Δ (pp) = accuracy of NAS finetuned model minus base model accuracy.
> Param↓, FLOPs↓ = reduction relative to base.
> Latency measured on CPU (ORT, 100 runs, P50).

---

## Per-Model Detail

{render_per_model(rows)}
---
_Generated by `generate_results_readme.py`_
"""

    out.write_text(content, encoding="utf-8")
    print(f"Written: {out}")
    print(f"  {len(rows)} result(s) across {len({r['model'] for r in rows})} model(s)")


if __name__ == "__main__":
    main()
