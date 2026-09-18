"""Calcula as duas análises novas (janela de decisão e valor do reparo) e as funde no bloom.json.

Separado de export_site.py para não recomputar o que já está publicado: o engine é o custo
(~4 min) e os demais resultados são determinísticos (sementes fixas em Settings e nos rng).
`export_site.py` também produz estas chaves, para que um build completo seja reprodutível.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloom import pipeline as P  # noqa: E402
from scripts.export_site import SCENARIOS, clean, write_html  # noqa: E402

OUT = ROOT / "docs" / "data" / "bloom.json"


def windows_payload(engine) -> dict:
    janela, reparo = {}, {}
    for lm in SCENARIOS:
        t = time.time()
        w = P.decision_windows(engine, life_model=lm)
        janela[lm] = w.to_dict(orient="records")
        print(f"janela {lm} {time.time() - t:.0f}s", flush=True)
        t = time.time()
        rf = P.repair_frontier(engine, life_model=lm)
        reparo[lm] = rf.to_dict(orient="records")
        print(f"reparo {lm} {time.time() - t:.0f}s", flush=True)
    return {"janela": janela, "reparo": reparo}


def main():
    t0 = time.time()
    d = P.load_cycles()
    engine = P.Engine(d)
    print(f"engine {time.time() - t0:.0f}s", flush=True)
    extra = clean(windows_payload(engine))
    data = json.loads(OUT.read_text(encoding="utf-8"))
    data.update(extra)
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    OUT.write_text(payload, encoding="utf-8")
    write_html(payload)
    print(f"ok {time.time() - t0:.0f}s · {len(payload) / 1e6:.2f} MB")


if __name__ == "__main__":
    main()
