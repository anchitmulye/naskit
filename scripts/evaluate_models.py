"""
h3dnas CLI - dataset-agnostic subcommands.

Registered as a console script via pyproject.toml:
    h3dnas = "h3dnas.cli:main"

Subcommands
-----------
  analyse  Compare base vs NAS model with paper-grade metrics
  prove    Execute Theorem 1 CDG proof on any ONNX model
  info     Print version and environment info

NAS runs are launched directly from example entry points:
    python examples/fashion_mnist/run_nas.py
    python examples/resnet/run_nas.py
    python examples/pointnet/run_nas.py

Usage (after pip/poetry install):
    h3dnas analyse --base model.onnx --nas nas_best.onnx
    h3dnas prove   --model examples/pointnet/pointnet_base.onnx
    h3dnas info
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE    = Path(__file__).parent
NAS_ROOT = _HERE.parent

if str(NAS_ROOT) not in sys.path:
    sys.path.insert(0, str(NAS_ROOT))

from h3dnas.utils.logger import get_logger

logger = get_logger(__name__)


# Subcommand: analyse

def _cmd_analyse(args) -> None:
    """
    Dataset loading is owned by example scripts - run analyse.py from your example:

        python examples/fashion_mnist/analyse.py
        python examples/pointnet/analyse.py

    These scripts pass the loader and sample_input directly to run_analysis()
    in h3dnas.analysis.model_analyser, keeping the library dataset-agnostic.
    """
    print(
        "\n  h3dnas analyse is not a standalone command - dataset loading lives\n"
        "  in each example. Run the example's analyse.py instead:\n\n"
        "      python examples/fashion_mnist/analyse.py\n"
        "      python examples/pointnet/analyse.py\n"
    )
    sys.exit(0)


# Subcommand: prove

def _cmd_prove(args) -> None:
    """Execute Theorem 1 CDG proof on an ONNX model."""
    from h3dnas.parser.onnx_parser import ONNXParser
    from h3dnas.parser.channel_dependency_graph import validate_theorem1

    parser = ONNXParser()
    parser.initialize()
    model  = parser.execute(args.model)
    result = validate_theorem1(model, Path(args.model).stem)

    if args.output:
        out = {
            "model":              args.model,
            "rho_f":              result.rho_f,
            "n_free":             len(result.free),
            "n_constrained":      len(result.constrained),
            "n_total":            len(result.nodes),
            "free_nodes":         [n.name for n in result.free],
            "constrained_nodes":  [
                {"name": n.name, "reason": n.reason}
                for n in result.constrained
            ],
            "op_class_counts":    result.op_class_counts,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        logger.info(f"Proof saved: {args.output}")


# Subcommand: info

def _cmd_info(_args) -> None:
    """Print version and environment info."""
    import h3dnas
    print(f"h3dnas version : {h3dnas.__version__}")
    print(f"Python         : {sys.version.split()[0]}")
    for pkg, attr in [
        ("onnxruntime", "__version__"),
        ("onnx",        "__version__"),
        ("torch",       "__version__"),
        ("numpy",       "__version__"),
    ]:
        try:
            import importlib
            m = importlib.import_module(pkg)
            print(f"{pkg:<15}: {getattr(m, attr)}")
        except ImportError:
            print(f"{pkg:<15}: not installed")

    print("\nExample entry points:")
    examples_dir = NAS_ROOT / "examples"
    if examples_dir.exists():
        for ep in sorted(examples_dir.glob("*/run_nas.py")):
            print(f"  python {ep.relative_to(NAS_ROOT)}")

    print("\nUsage:")
    print("  h3dnas analyse --base model.onnx --nas nas_best.onnx")
    print("  h3dnas prove   --model examples/pointnet/pointnet_base.onnx")


# Main - argument parsing and dispatch

def main() -> None:
    p = argparse.ArgumentParser(
        prog="h3dnas",
        description="Hardware-Aware ONNX-Native Neural Architecture Search",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "NAS runs:\n"
            "  python examples/fashion_mnist/run_nas.py\n"
            "  python examples/resnet/run_nas.py\n"
            "  python examples/pointnet/run_nas.py\n\n"
            "Analysis / proof:\n"
            "  h3dnas analyse --base model.onnx --nas best.onnx\n"
            "  h3dnas prove   --model examples/pointnet/pointnet_base.onnx\n"
            "  h3dnas info\n"
        ),
    )
    sub = p.add_subparsers(dest="command", metavar="COMMAND")
    sub.required = True

    # -- analyse --------------------------------------------------------------
    an_p = sub.add_parser("analyse",
                          help="Redirect - use examples/<model>/analyse.py instead")

    # -- prove ----------------------------------------------------------------
    pr_p = sub.add_parser("prove", help="Execute Theorem 1 CDG proof on an ONNX model")
    pr_p.add_argument("--model",  required=True, help="Path to ONNX model")
    pr_p.add_argument("--output", default=None,
                      help="Save proof results to JSON (optional)")

    # -- info -----------------------------------------------------------------
    sub.add_parser("info", help="Print version and environment info")

    args = p.parse_args()
    dispatch = {
        "analyse": _cmd_analyse,
        "prove":   _cmd_prove,
        "info":    _cmd_info,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
