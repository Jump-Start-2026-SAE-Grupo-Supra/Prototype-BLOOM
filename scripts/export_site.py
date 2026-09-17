"""Gera o site estático em docs/ (GitHub Pages) com os resultados embutidos.

    python scripts/export_site.py              # recalcula tudo (~6 min)
    python scripts/export_site.py --html-only  # só reaplica site/template.html ao JSON existente

Saídas:
    docs/data/bloom.json         resultados (também embutidos no HTML)
    docs/index.html              página completa para o GitHub Pages
    site/build/laudo_bloom.html  mesma página sem o esqueleto <html>, para publicar como artifact
"""
from __future__ import annotations

import json
import math
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloom import pipeline as P  # noqa: E402
from bloom.config import DESTINATIONS, Settings  # noqa: E402
from bloom.extract import PACKS, SECOND_LIFE_ORIGIN  # noqa: E402
from bloom.health import metrics  # noqa: E402

SCENARIOS = ["conservador", "fade"]
FRACS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.97)
OUT_JSON = ROOT / "docs" / "data" / "bloom.json"


def clean(o):
    """Arredonda e converte tipos numpy para JSON compacto."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        f = float(o)
        if not math.isfinite(f):
            return None
        return round(f, 4) if abs(f) < 100 else round(f, 1)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def slim_laudo(r: dict) -> dict:
    keep_dest = ["destino", "elegivel", "motivos_eliminação", "vpl_medio", "vpl_p10", "vpl_p90", "vida_anos_p10",
                 "vida_anos_p50", "p_falha_garantia", "premio_garantia", "p_melhor", "receita_p50", "custos"]
    return {
        "ciclo": r["ciclo"], "fracao": r["fracao_vida"], "soh": r["soh"], "rul": r["vida_residual_efc"],
        "soh_real": r["soh_real"], "rotulo_extrapolado": r["rotulo_extrapolado"], "corrente": r["corrente_a"],
        "gates": [{k: g[k] for k in ("gate", "status", "evidence", "source")} for g in r["gates"]],
        "destinos": [{k: dd.get(k) for k in keep_dest} for dd in r["destinos"]],
        "recomendacao": r["recomendacao"], "confianca": r["confianca"], "voi": r["valor_da_informacao"],
        "pede_ensaio": r["recomenda_ensaio_completo"], "vetos": r["vetos"],
    }


def write_html(payload: str) -> None:
    template = (ROOT / "site" / "template.html").read_text(encoding="utf-8")
    body = template.replace("/*__BLOOM_DATA__*/null", payload.replace("</", "<\\/"))
    build = ROOT / "site" / "build" / "laudo_bloom.html"
    build.parent.mkdir(parents=True, exist_ok=True)
    build.write_text(body, encoding="utf-8")
    cut = body.index("</style>") + len("</style>")
    full = ("<!doctype html>\n<html lang=\"pt-BR\">\n<head>\n<meta charset=\"utf-8\">\n"
            "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1, viewport-fit=cover\">\n"
            + body[:cut] + "\n</head>\n<body style=\"margin:0\">\n" + body[cut:] + "\n</body>\n</html>\n")
    (ROOT / "docs" / "index.html").write_text(full, encoding="utf-8")
    (ROOT / "docs" / ".nojekyll").write_text("", encoding="utf-8")


def build_data() -> dict:
    t0 = time.time()
    d = P.load_cycles()
    engine = P.Engine(d)
    print(f"engine {time.time() - t0:.0f}s", flush=True)

    bench, _ = P.benchmark_soh(d)
    print(f"benchmark {time.time() - t0:.0f}s", flush=True)

    rows = d.loc[engine.soh.index]
    lab = (~rows["label_extrapolated"]) & rows["soh"].notna()
    sl = lab & (rows["group"] == "second_life") & engine.soh["anchored"]
    anchor = {"puro": metrics(rows.loc[sl, "soh"].to_numpy(), engine.soh_raw.loc[sl]),
              "ancorado": metrics(rows.loc[sl, "soh"].to_numpy(), engine.soh.loc[sl])}
    rul_m = metrics(rows["efc_to_eol"].to_numpy(), engine.rul, scale=1)

    packs = {}
    for pack in sorted(d["pack"].unique()):
        g = rows[rows["pack"] == pack]
        pr = engine.soh.loc[g.index]
        idx = g.index[::max(1, len(g) // 220)]
        refs = d[(d["pack"] == pack) & d["is_ref"] & (d["dis_ah"] > 1)]
        group, currents = PACKS[pack]
        packs[pack] = {
            "grupo": group, "correntes": currents, "ciclos": int(d.loc[d["pack"] == pack, "cycle"].max()),
            "efc_total": float(g["cum_efc"].max()), "origem": SECOND_LIFE_ORIGIN.get(pack),
            "serie": {"ciclo": g.loc[idx, "cycle"].tolist(), "p10": pr.loc[idx, "p10"].tolist(),
                      "p50": pr.loc[idx, "p50"].tolist(), "p90": pr.loc[idx, "p90"].tolist()},
            "referencias": {"ciclo": refs["cycle"].tolist(), "soh": refs["soh"].tolist()},
            "decisoes": {},
        }

    fleet = {}
    for lm in SCENARIOS:
        res = engine.fleet_decisions(Settings(life_model=lm, n_samples=2000), FRACS)
        for r in res:
            packs[r["pack"]]["decisoes"].setdefault(lm, []).append(slim_laudo(r))
        tab = P.decisions_table(res)
        tab["faixa"] = pd.cut(tab["fração da vida"], [0, .2, .4, .6, .8, 1.0],
                              labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
        ct = pd.crosstab(tab["faixa"], tab["recomendação"])
        fleet[lm] = {"faixas": [str(x) for x in ct.index], "contagens": {c: ct[c].tolist() for c in ct.columns},
                     "vetos": int((tab["vetos"] != "").sum()), "pede_ensaio": int(tab["pede ensaio"].sum()),
                     "n": int(len(tab))}
        print(f"decisões {lm} {time.time() - t0:.0f}s", flush=True)

    grids = []
    for moment, frac in [("30% da vida", 0.3), ("aposentadoria (90%)", 0.9)]:
        pts = [(p, engine.decision_points(p, (frac,))[0]) for p in sorted(d["pack"].unique())]
        for lm in SCENARIOS:
            gr = P.reuse_share_grid(engine, pts, life_model=lm)
            grids.append({"momento": moment, "cenario": lm,
                          "frete": sorted(gr["frete R$/kWh"].unique().tolist()),
                          "indice": sorted(gr["índice de preço do novo"].unique().tolist()),
                          "valores": gr.pivot(index="frete R$/kWh", columns="índice de preço do novo",
                                              values="fração para reuso").values.tolist()})
    print(f"grades {time.time() - t0:.0f}s", flush=True)

    st = engine.stress
    return clean({
        "gerado_em": date.today().isoformat(),
        "dataset": {"ciclos": int(len(d)), "packs": int(d["pack"].nunique()),
                    "referencias": int((d["is_ref"] & (d["dis_ah"] > 1)).sum())},
        "benchmark": bench.to_dict(orient="records"),
        "ancora_segunda_vida": anchor,
        "vida_residual": rul_m,
        "gates_ref": {"deriva_termica_limite": engine.ref.thermal_drift_limit, "r_novo": engine.ref.r_new,
                      "relaxacao_limite": engine.ref.self_discharge_limit, "novidade_limite": engine.ref.novelty_limit},
        "estresse": {"a": st.a, "b": st.b, "sigma": st.sigma, "pontos": st.points.to_dict(orient="records")},
        "destinos": [dict(vars(x)) for x in DESTINATIONS],
        "economia": Settings().economics.as_dict(),
        "calendario": Settings().calendar_fade_per_year,
        "packs": packs, "frota": fleet, "grades": grids,
        "laco": P.closed_loop(engine).to_dict(orient="records"),
    })


def main():
    if "--html-only" in sys.argv:
        write_html(OUT_JSON.read_text(encoding="utf-8"))
        print("html regenerado a partir de", OUT_JSON)
        return
    t0 = time.time()
    payload = json.dumps(build_data(), ensure_ascii=False, separators=(",", ":"))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(payload, encoding="utf-8")
    write_html(payload)
    print(f"ok {time.time() - t0:.0f}s · {len(payload) / 1e6:.2f} MB de dados")


if __name__ == "__main__":
    main()
