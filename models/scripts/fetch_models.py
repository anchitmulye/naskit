"""Fetch the base ONNX models listed in models.yaml into base/."""

from __future__ import annotations

import argparse
import hashlib
import os
import subprocess
import sys
import urllib.request
from pathlib import Path

import yaml

SCRIPTS = Path(__file__).resolve().parent             # models/scripts
MODELS = SCRIPTS.parent                               # models
ROOT = MODELS.parent                                  # the naskit folder
BASE = MODELS / "base"
H3DNAS = ROOT / "submodules" / "h3dnas"


def sources() -> dict:
    """{name: entry} with each entry's category added."""
    cfg = yaml.safe_load((MODELS / "models.yaml").read_text(encoding="utf-8"))
    out = {}
    for cat, entries in cfg.items():
        for name, e in entries.items():
            if name in out:
                raise SystemExit(f"models.yaml: {name} listed twice")
            out[name] = {**e, "category": cat}
    return out


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def target(name: str) -> Path:
    return BASE / f"{name}.onnx"


def download(name: str, e: dict) -> None:
    out = target(name)
    part = out.with_suffix(".onnx.part")
    req = urllib.request.Request(e["url"], headers={"User-Agent": "h3dnas-fetch"})
    h = hashlib.sha256()
    done = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(part, "wb") as f:
        total = int(r.headers.get("Content-Length") or e.get("size") or 0)
        for chunk in iter(lambda: r.read(1 << 20), b""):
            f.write(chunk)
            h.update(chunk)
            done += len(chunk)
            if total:
                print(f"\r  {name}: {done / 2**20:7.1f} / {total / 2**20:.1f} MB", end="", flush=True)
    print()
    if e.get("sha256") and h.hexdigest() != e["sha256"]:
        part.unlink()
        raise SystemExit(f"{name}: downloaded file has sha256 {h.hexdigest()}, expected {e['sha256']}")
    part.replace(out)
    print(f"  {name}: saved {out}")


def build(name: str, e: dict) -> None:
    out = target(name)
    subst = {"{out}": str(out), "{scripts}": str(SCRIPTS), "{root}": str(ROOT)}
    args = []
    for a in e["run"]:
        for k, v in subst.items():
            a = a.replace(k, v)
        args.append(a)
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(filter(None, [str(H3DNAS),
                                                                    os.environ.get("PYTHONPATH")]))}
    print(f"  {name}: {' '.join([Path(sys.executable).name, *args])}")
    subprocess.check_call([sys.executable, *args], cwd=H3DNAS, env=env)
    if not out.exists():
        raise SystemExit(f"{name}: the script did not write {out}")
    got = sha256(out)
    if e.get("sha256") and got != e["sha256"]:
        print(f"  {name}: rebuilt file differs from the recorded one (sha256 {got[:12]} vs "
              f"{e['sha256'][:12]}); results recorded with the old file are tied to the old hash")


def record(names: list, src: dict) -> None:
    """Write the current sha256 and size of base/<name>.onnx into models.yaml,
    editing only those two lines of each entry (comments and layout stay)."""
    path = MODELS / "models.yaml"
    lines = path.read_text(encoding="utf-8").splitlines()
    for name in names:
        e, f = src[name], target(name)
        if "url" in e:
            raise SystemExit(f"{name}: a url entry's sha256 is the published file's; not changed")
        if not f.exists():
            raise SystemExit(f"{name}: {f} not found")
        sha, size = sha256(f), f.stat().st_size
        start = next((i for i, l in enumerate(lines) if l == f"  {name}:"), None)
        if start is None:
            raise SystemExit(f"{name}: entry not found in {path}")
        end = next((i for i in range(start + 1, len(lines))
                    if lines[i].strip() and not lines[i].startswith("    ")), len(lines))
        block = [l for l in lines[start + 1:end]
                 if not l.lstrip().startswith(("sha256:", "size:"))]
        block[1:1] = [f"    sha256: {sha}", f"    size: {size}"]      # after the source line
        lines[start + 1:end] = block
        old = e.get("sha256")
        print(f"  {name}: sha256 {(old or 'none')[:12]} -> {sha[:12]}, size {size:,}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    now = sources()                                   # the file still parses, values are in
    for name in names:
        assert now[name]["sha256"] == sha256(target(name)), name


def status(name: str, e: dict, rehash: bool) -> str:
    p = target(name)
    if not p.exists():
        return "missing"
    if e.get("sha256") is None:
        return "present (no recorded sha256)"
    if not rehash:
        return "present" if p.stat().st_size == e.get("size", p.stat().st_size) else "SIZE MISMATCH"
    return "verified" if sha256(p) == e["sha256"] else "SHA256 MISMATCH"


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("names", nargs="*", help="model names or categories (default: every url entry)")
    ap.add_argument("--list", action="store_true", help="show every entry and its state")
    ap.add_argument("--verify", action="store_true", help="re-hash every present file")
    ap.add_argument("--record", action="store_true",
                    help="write the present files' sha256 and size into models.yaml (needs names)")
    a = ap.parse_args(argv)
    src = sources()
    BASE.mkdir(exist_ok=True)
    if a.record and not a.names:
        raise SystemExit("--record needs model names or categories")

    if a.list or a.verify:
        bad = 0
        for name, e in sorted(src.items(), key=lambda kv: (kv[1]["category"], kv[0])):
            kind = "url" if "url" in e else ("recipe" if "run" in e else "local")
            st = status(name, e, rehash=a.verify)
            bad += st.isupper() or "MISMATCH" in st
            print(f"{e['category']:<32} {name:<28} {kind:<7} {st}")
        if a.verify and bad:
            raise SystemExit(f"{bad} file(s) do not match models.yaml")
        return

    if a.names:
        chosen = [n for n in src if n in a.names or src[n]["category"] in a.names
                  or src[n]["category"].startswith(tuple(f"{x}/" for x in a.names))]
        unknown = [x for x in a.names if x not in src and not any(
            e["category"] == x or e["category"].startswith(f"{x}/") for e in src.values())]
        if unknown:
            raise SystemExit(f"unknown model or category: {unknown}")
    else:
        chosen = [n for n, e in src.items() if "url" in e]

    if a.record:
        record(chosen, src)
        return

    for name in chosen:
        e = src[name]
        if status(name, e, rehash=True) == "verified":
            print(f"  {name}: up to date")
        elif "url" in e:
            download(name, e)
        elif "run" in e:
            build(name, e)
        elif not target(name).exists():
            print(f"  {name}: no public source ({e.get('note', '')}); copy the file to {target(name)}")
        else:
            print(f"  {name}: local file differs from models.yaml sha256")


if __name__ == "__main__":
    main()
