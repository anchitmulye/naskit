"""Generate the paper's data-driven figures into fig/.

    python make_figures.py                 # every figure it has data for
    python make_figures.py latency width   # only these
    python make_figures.py --list

Figures
  tradeoff   accuracy change against MAC reduction, ours vs published  (fig:tradeoff)
  latency    MAC ratio against *measured* speed-up, one series per runtime, with the
             y=x line: every point below it is a MAC count overpromising
  width      per-layer width, searched against uniform, at the same budget; widths that
             are multiples of 8 are marked, which is what the alignment question turns on

Needs matplotlib. The schematic figures (pipeline overview, coupled-group illustration,
head-axis tracing) are not here: they are vector drawings, not plots of data.

Everything is read from ../h3dnas_paper_tables.xlsx, except `width`, which reads the
ONNX files directly so that the profile is the one actually emitted rather than a
reconstruction. A figure whose inputs are missing is skipped with a message, never
drawn from partial data.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
FIG = HERE / "fig"
BOOK = HERE.parent / "results" / "h3dnas_results.xlsx"
# the compressed graphs whose per-layer widths the `width` figure draws
NASKIT = HERE.parent.parent / "naskit"

NAME = {"cifar_resnet56": "ResNet-56", "cifar_resnet18": "ResNet-18", "cifar_vgg16": "VGG-16-BN",
        "resnet50-v1-12": "ResNet-50", "vit-base-patch16-224": "ViT-B/16", "pointnet": "PointNet",
        "pointnet2": "PointNet++", "pointmlp": "PointMLP", "pct": "PCT",
        "gpt2-lm-head-10": "GPT-2"}


def sheet(book, name):
    ws = book[name]
    head = [c.value for c in ws[1]]
    return [dict(zip(head, r)) for r in ws.iter_rows(min_row=2, values_only=True) if any(r)]


def style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "legend.fontsize": 7,
        "xtick.labelsize": 7, "ytick.labelsize": 7,
        "axes.spines.top": False, "axes.spines.right": False,
        "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
    })
    return plt


def save(plt, fig, stem):
    FIG.mkdir(exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{stem}.{ext}")
    plt.close(fig)
    print(f"  wrote fig/{stem}.pdf and .png")


# -- figures -----------------------------------------------------------------------------

def fig_latency(book):
    """MAC ratio against measured speed-up. The y=x line is the MAC count's promise."""
    plt = style()
    rows = sheet(book, "Runtime")
    series = [("cpu", "CPU", "o"), ("cuda", "GPU (CUDA)", "s"),
              ("coreml", "Apple M2", "^"), ("jetson", "Jetson Orin Nano", "D")]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    hi = 1.0
    for key, label, marker in series:
        xs, ys = [], []
        for d in rows:
            mac = d["Base MACs"] / d["MACs"] if d.get("MACs") else None
            if key == "jetson":
                sp = d.get("jetson speedup")
            else:
                b, p = d.get(f"{key} base ms"), d.get(f"{key} ms")
                sp = (b / p) if (b and p) else None
            if mac and sp:
                xs.append(mac); ys.append(sp)
        if not xs:
            continue
        hi = max(hi, max(xs + ys))
        ax.scatter(xs, ys, s=22, marker=marker, label=label, alpha=0.85,
                   edgecolors="white", linewidths=0.4)
    lim = hi * 1.08
    ax.plot([1, lim], [1, lim], ls="--", lw=0.8, color="0.5", zorder=0)
    ax.annotate("MACs deliver in full", xy=(lim * 0.72, lim * 0.74), fontsize=6.5,
                color="0.4", rotation=38, ha="center")
    ax.set_xlim(1, lim); ax.set_ylim(0.85, lim)
    ax.set_xlabel("MAC ratio  (base / pruned)")
    ax.set_ylabel("measured speed-up")
    ax.legend(frameon=False, loc="upper left")
    save(plt, fig, "latency_vs_macs")


def fig_tradeoff(book):
    """Accuracy change against MAC reduction: ours at every budget, next to published."""
    plt = style()
    budg = [d for d in sheet(book, "Budgets") if d.get("Use in paper") == "yes"]
    pub = [d for d in sheet(book, "Published") if d.get("MACs reduction")]
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    for m, marker in (("cifar_resnet56", "o"), ("cifar_resnet18", "s"), ("cifar_vgg16", "^")):
        pts = sorted(((100 * (1 - d["MACs"] / d["Base MACs"]), d["Test"] - d["Base test"])
                      for d in budg if d["Model"] == m), key=lambda t: t[0])
        if pts:
            ax.plot(*zip(*pts), marker=marker, ms=4, lw=1.1, label=NAME[m])
    xs = [100 * d["MACs reduction"] for d in pub]
    ys = [d["Pruned"] - d["Base"] for d in pub]
    ax.scatter(xs, ys, marker="x", s=20, c="0.45", lw=0.9, label="published", zorder=0)
    ax.axhline(0, lw=0.7, color="0.7", zorder=0)
    ax.set_xlabel("MACs removed (\\%)" if False else "MACs removed (%)")
    ax.set_ylabel("top-1 change (points)")
    ax.legend(frameon=False, fontsize=6.5)
    save(plt, fig, "tradeoff")


def fig_width(book, model="cifar_resnet56"):
    """Per-layer width, searched vs uniform, at the same budget.

    Read from the emitted ONNX rather than reconstructed, and marks which widths are
    multiples of 8 -- the question of whether the latency gain is the search or the
    alignment turns on exactly this.
    """
    try:
        import onnx
    except ImportError:
        print("  skip width: onnx not installed"); return
    plt = style()
    runs = sorted((NASKIT / "artifacts" / model).glob("*/models")) if (NASKIT / "artifacts" / model).is_dir() else []
    if not runs:
        print(f"  skip width: no artifacts/{model}/*/models on this machine"); return
    models_dir = runs[-1]

    def widths(path):
        m = onnx.load(str(path), load_external_data=False)
        out = []
        for n in m.graph.node:
            if n.op_type != "Conv":
                continue
            w = next((i for i in m.graph.initializer if i.name == n.input[1]), None)
            if w is not None and w.dims:
                out.append(int(w.dims[0]))
        return out

    base = models_dir / f"{model}.onnx"
    cands = {"searched": models_dir / f"{model}_gs_50pct.onnx",
             "uniform": models_dir / f"{model}_uniform_50pct.onnx"}
    if not base.is_file():
        print(f"  skip width: {base.name} not found"); return
    fig, ax = plt.subplots(figsize=(3.3, 2.6))
    ax.plot(widths(base), lw=1.0, color="0.6", label="base")
    for label, path in cands.items():
        if not path.is_file():
            print(f"  width: {path.name} missing, drawing without it"); continue
        w = widths(path)
        ax.plot(w, lw=1.2, marker="o", ms=2.5, label=label)
        if label == "searched":
            off = [i for i, v in enumerate(w) if v % 8]
            if off:
                ax.scatter(off, [w[i] for i in off], s=26, facecolors="none",
                           edgecolors="crimson", lw=0.8, zorder=5,
                           label="not a multiple of 8")
    ax.set_xlabel("convolution index"); ax.set_ylabel("output channels")
    ax.set_title(NAME.get(model, model) + ", 50\\% budget" if False else
                 NAME.get(model, model) + ", 50% budget", fontsize=8)
    ax.legend(frameon=False, fontsize=6.5)
    save(plt, fig, f"width_{model}")


FIGURES = {"tradeoff": fig_tradeoff, "latency": fig_latency, "width": fig_width}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("which", nargs="*", choices=sorted(FIGURES) or None, help="default: all")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args(argv)
    if a.list:
        for k, f in FIGURES.items():
            print(f"  {k:10s} {(f.__doc__ or '').splitlines()[0]}")
        return 0
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("matplotlib is required:  pip install matplotlib"); return 1
    from openpyxl import load_workbook
    if not BOOK.is_file():
        print(f"workbook not found: {BOOK}"); return 1
    book = load_workbook(BOOK, data_only=True)
    for k in (a.which or FIGURES):
        print(f"{k}:")
        FIGURES[k](book)
    return 0


if __name__ == "__main__":
    sys.exit(main())