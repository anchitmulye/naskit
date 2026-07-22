"""
Workspace configuration loader for naskit.

Resolves all paths from workspace.yaml so no script ever hardcodes
absolute paths. The workspace root is always the directory containing
workspace.yaml (i.e. the naskit/ repo root).

Usage
-----
    from utils.workspace import load_workspace

    ws = load_workspace()

    base   = ws.base_onnx("pointnet")      # Path to base .onnx
    data   = ws.dataset("pointnet")        # Path to dataset root
    repo   = ws.model_repo("pointnet")     # Path to paper repo (for data_utils)
    arts   = ws.artifacts("pointnet")      # Path to artifacts dir (created if missing)
    shape  = ws.input_shape("pointnet")    # list[int] e.g. [1, 3, 1024]
    n_cls  = ws.num_classes("pointnet")    # int

    ws.activate("pointnet")                # add model_repo to sys.path
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def _find_workspace_yaml() -> Path:
    """Walk up from cwd until workspace.yaml is found."""
    for parent in [Path.cwd()] + list(Path.cwd().parents):
        candidate = parent / "workspace.yaml"
        if candidate.exists():
            return candidate
    # Fallback: relative to this file (utils/ → naskit/)
    candidate = Path(__file__).parent.parent / "workspace.yaml"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        "workspace.yaml not found.\n"
        "Copy workspace.yaml.template → workspace.yaml and fill in your paths.\n"
        "Run scripts from within the naskit/ directory."
    )


# ---------------------------------------------------------------------------
# Workspace
# ---------------------------------------------------------------------------

class Workspace:
    def __init__(self, cfg: dict, yaml_path: Path):
        self._cfg  = cfg
        self._root = yaml_path.parent   # naskit/ repo root

    # ── Internal ─────────────────────────────────────────────────────────────

    def _resolve(self, rel: str | Path) -> Path:
        """Resolve a path relative to the workspace root, or return as-is if absolute."""
        p = Path(rel)
        return p if p.is_absolute() else (self._root / p).resolve()

    def _model(self, model: str) -> dict:
        try:
            return self._cfg["models"][model]
        except KeyError:
            available = list(self._cfg.get("models", {}).keys())
            raise KeyError(
                f"Model '{model}' not in workspace.yaml.\n"
                f"Available: {available}"
            )

    # ── Model paths ───────────────────────────────────────────────────────────

    def base_onnx(self, model: str) -> Path:
        """Path to the base (pre-NAS) .onnx file."""
        return self._resolve(self._model(model)["base_onnx"])

    def nas_onnx(self, model: str) -> Path:
        """Path to the best NAS candidate .onnx (output of run_nas)."""
        return self._resolve(self._model(model)["nas_onnx"])

    def finetuned_onnx(self, model: str) -> Path:
        """Path to the fine-tuned .onnx (output of finetune)."""
        return self._resolve(self._model(model)["finetuned"])

    def dataset(self, model: str) -> Path:
        """Path to the dataset root directory."""
        return self._resolve(self._model(model)["dataset"])

    def model_repo(self, model: str) -> Path:
        """
        Path to the paper's source repo (for importing data_utils etc.).
        Only present for models backed by a vision-onnx-models submodule.
        """
        key = self._model(model).get("model_repo")
        if not key:
            raise KeyError(
                f"model_repo not defined for '{model}' in workspace.yaml."
            )
        return self._resolve(key)

    def artifacts(self, model: str) -> Path:
        """Path to the artifacts directory for this model. Created if missing."""
        p = self._resolve(self._model(model)["artifacts"])
        p.mkdir(parents=True, exist_ok=True)
        return p

    def results_dir(self, model: str, name: str) -> Path:
        """
        Path to a named results directory under naskit/results/<model>/<name>/.
        Created if missing. This is the committed results store for paper runs.

        Example: ws.results_dir("pointnet", "two_stage_paper")
        → naskit/results/pointnet/two_stage_paper/
        """
        p = self._root / "results" / model / name
        p.mkdir(parents=True, exist_ok=True)
        return p
        return p

    # ── Model metadata ────────────────────────────────────────────────────────

    def num_classes(self, model: str) -> int:
        return int(self._model(model)["num_classes"])

    def input_shape(self, model: str) -> list:
        return list(self._model(model)["input_shape"])

    def task(self, model: str) -> str:
        return self._model(model).get("task", "classification")

    # ── Convenience ───────────────────────────────────────────────────────────

    def activate(self, model: str) -> None:
        """
        Add model_repo to sys.path so paper-specific imports work.

        Example: after ws.activate("pointnet"), you can do:
            from data_utils.ModelNetDataLoader import ModelNetDataLoader
        """
        repo = str(self.model_repo(model))
        if repo not in sys.path:
            sys.path.insert(0, repo)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def models(self) -> list[str]:
        return list(self._cfg.get("models", {}).keys())


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def load_workspace(yaml_path: str | Path | None = None) -> Workspace:
    """
    Load workspace.yaml and return a Workspace instance.

    Parameters
    ----------
    yaml_path : optional explicit path to workspace.yaml.
                Falls back to WORKSPACE_YAML env var, then auto-discovery.
    """
    if yaml_path is None:
        yaml_path = os.environ.get("WORKSPACE_YAML")
    if yaml_path is None:
        yaml_path = _find_workspace_yaml()

    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)
    return Workspace(cfg, yaml_path)
