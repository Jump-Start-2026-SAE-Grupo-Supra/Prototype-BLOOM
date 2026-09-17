"""Gera data/processed/cycles.parquet a partir dos CSVs brutos (~3,2 GB).

Uso:
    python scripts/build_dataset.py [--raw battery_alt_dataset/battery_alt_dataset] [--jobs 4]
"""
from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloom.extract import PACKS, build_pack_table  # noqa: E402


CACHE = ROOT / "data" / "interim"


def _one(args):
    raw, pack, force = args
    cache = CACHE / f"pack_{pack}.parquet"
    if cache.exists() and not force:
        return pd.read_parquet(cache)
    t0 = time.time()
    tab = build_pack_table(Path(raw), pack)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tab.to_parquet(cache, index=False)
    print(f"pack {pack}: {len(tab)} ciclos em {time.time() - t0:.0f}s", flush=True)
    return tab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(ROOT / "battery_alt_dataset" / "battery_alt_dataset"))
    ap.add_argument("--out", default=str(ROOT / "data" / "processed" / "cycles.parquet"))
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--packs", nargs="*", default=sorted(PACKS))
    ap.add_argument("--force", action="store_true", help="ignora o cache em data/interim")
    a = ap.parse_args()

    with ProcessPoolExecutor(max_workers=a.jobs) as ex:
        tables = list(ex.map(_one, [(a.raw, p, a.force) for p in a.packs]))
    out = pd.concat(tables, ignore_index=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(a.out, index=False)
    print(f"{len(out)} ciclos, {out['pack'].nunique()} packs -> {a.out}")


if __name__ == "__main__":
    main()
