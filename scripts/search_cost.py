"""What the search costs: units, scored genomes, screened genomes, wall-clock.

    python -m scripts.search_cost                     # every full run in paths.runs
    python -m scripts.search_cost --model pointnet    # one model
    python -m scripts.search_cost --tsv               # paste-ready rows

Reads three files already written by every run, so nothing has to be re-run:
  debug/audit.json   prunability.groups / .prunable_groups  -> |U|
  debug/run.json     settings.search.population / .generations
  debug/search.json  search_seconds, protocol.runtime

Genome counts are derived from the search's own rules (h3dnas.generator.
group_evolution): each generation keeps `elites` and opens
`slots = population - elites`; with screening on it breeds up to
`screen x slots` children, keeps the ones the ridge surrogate ranks highest
(plus `explore` at random) and scores only `slots` of them. The first
generation scores the whole initial population, and the last one only
evaluates, so

    scored   = population + (generations - 1) * slots
    screened = (generations - 1) * screen * slots

`scored` is the number of forward passes over the validation split; the
screened-but-not-scored genomes cost a cost-model check and nothing else.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[1]

# Defaults of GroupSearchConfig; overridden per run where the settings say so.
ELITES, SCREEN, EXPLORE = 2, 8, 0.25


def _load(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return None


def collect(run: Path) -> Optional[Dict[str, Any]]:
    """One row per run folder, or None if it has no finished search."""
    dbg = run / "debug"
    search, audit, meta = (_load(dbg / n) for n in ("search.json", "audit.json", "run.json"))
    if not search or search.get("search_seconds") is None:
        return None
    s = ((meta or {}).get("settings") or {}).get("search") or {}
    pop, gens = int(s.get("population") or 0), int(s.get("generations") or 0)
    slots = max(0, pop - ELITES)
    prun = (audit or {}).get("prunability") or {}
    secs = float(search["search_seconds"])
    scored = pop + (gens - 1) * slots if pop and gens else None
    rt = ((search.get("protocol") or {}).get("runtime") or {})
    return {
        "model": (meta or {}).get("example") or run.parent.name,
        "run": run.name,
        "units": prun.get("prunable_groups"),
        "groups": prun.get("groups"),
        "population": pop or None,
        "generations": gens or None,
        "scored": scored,
        "screened": (gens - 1) * SCREEN * slots if pop and gens else None,
        "seconds": round(secs, 1),
        "minutes": round(secs / 60, 1),
        "sec_per_scored": round(secs / scored, 2) if scored else None,
        "provider": (rt.get("search") or {}).get("provider"),
    }


def runs_under(root: Path, model: Optional[str]) -> List[Path]:
    globbed = root.glob(f"{model}/*" if model else "*/*")
    return sorted(p for p in globbed if p.is_dir() and not p.name.endswith("_quick"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--runs", type=Path, help="run root (default: the workspace's paths.runs)")
    ap.add_argument("--model", help="only this model's runs")
    ap.add_argument("--tsv", action="store_true", help="tab-separated, for pasting into a sheet")
    a = ap.parse_args()

    root = a.runs
    if root is None:
        try:                                  # the workspace knows where runs live
            import sys
            sys.path.insert(0, str(ROOT / "submodules" / "h3dnas"))
            from experiments.registry import paths
            root = paths().runs
        except Exception as e:                # a broken h3dnas.yaml must not stop a read-only report
            root = ROOT / "artifacts"
            print(f"[search_cost] workspace not loadable ({type(e).__name__}); using {root}\n")
    rows = [r for r in (collect(p) for p in runs_under(Path(root), a.model)) if r]
    if not rows:
        raise SystemExit(f"no finished searches under {root}")

    cols = ["model", "run", "units", "population", "generations", "scored", "screened",
            "minutes", "sec_per_scored", "provider"]
    if a.tsv:
        print("\t".join(cols))
        for r in rows:
            print("\t".join("" if r[c] is None else str(r[c]) for c in cols))
        return
    w = {c: max(len(c), *(len(str(r[c] if r[c] is not None else "--")) for r in rows)) for c in cols}
    print("  ".join(c.ljust(w[c]) for c in cols))
    print("  ".join("-" * w[c] for c in cols))
    for r in rows:
        print("  ".join(str(r[c] if r[c] is not None else "--").ljust(w[c]) for c in cols))
    tot = sum(r["seconds"] for r in rows)
    print(f"\n{len(rows)} run(s); {tot / 3600:.1f} GPU/CPU-hours of search in total.")
    print(f"Screening: each generation breeds up to {SCREEN}x the open slots and scores "
          f"{1 / SCREEN:.0%} of them ({EXPLORE:.0%} of those taken at random).")


if __name__ == "__main__":
    main()
