"""Reexecuta as análises de decisão sob os perfis de produto da revisão de premissas (18/09/2026).

    python scripts/rerun_premissas.py        # ~5 min (o ajuste do engine é o custo; o resto é rápido)

Só a camada econômica muda entre perfis: o modelo de saúde, os gates e o estresse são os mesmos, então
o engine é ajustado uma vez. Saída: docs/data/revisao_premissas.json. Ver PREMISSAS.md.
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloom import pipeline as P  # noqa: E402
from bloom.config import profile_settings  # noqa: E402
from scripts.export_site import SCENARIOS, clean  # noqa: E402

OUT = ROOT / "docs" / "data" / "revisao_premissas.json"
PROFILES = ["v0_ilustrativo", "bev_revisado", "hev_nimh_corolla"]
FRACS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.97)


def fleet_counts(engine, st):
    res = engine.fleet_decisions(replace(st, n_samples=2000), FRACS)
    tab = P.decisions_table(res)
    tab["faixa"] = pd.cut(tab["fração da vida"], [0, .2, .4, .6, .8, 1.0],
                          labels=["0–20%", "20–40%", "40–60%", "60–80%", "80–100%"])
    ct = pd.crosstab(tab["faixa"], tab["recomendação"])
    return {"faixas": [str(x) for x in ct.index], "contagens": {c: ct[c].tolist() for c in ct.columns},
            "vetos": int((tab["vetos"] != "").sum()), "pede_ensaio": int(tab["pede ensaio"].sum()), "n": int(len(tab))}


def grids(engine, st):
    out = []
    for moment, frac in [("30% da vida", 0.3), ("aposentadoria (90%)", 0.9)]:
        pts = [(p, engine.decision_points(p, (frac,))[0]) for p in sorted(engine.d["pack"].unique())]
        gr = P.reuse_share_grid(engine, pts, life_model=st.life_model, base=st)
        out.append({"momento": moment, "frete": sorted(gr["frete R$/kWh"].unique().tolist()),
                    "indice": sorted(gr["índice de preço do novo"].unique().tolist()),
                    "valores": gr.pivot(index="frete R$/kWh", columns="índice de preço do novo",
                                        values="fração para reuso").values.tolist()})
    return out


def value_vs_recycling(engine, st):
    """Ganho (R$/pack) de seguir a recomendação em vez de reciclar no mesmo momento.

    `ganho = VPL(destino recomendado) - VPL(reciclagem)`, ambos no mesmo ponto da vida. É a versão honesta
    do "valor por pack decidido": compara duas decisões alternativas no MESMO instante, sem misturar
    valor de instantes diferentes (o pack ainda rodaria na 1ª vida entre um instante e outro).
    """
    rows = []
    for pack in sorted(engine.d["pack"].unique()):
        n = engine.d.loc[engine.d["pack"] == pack, "cycle"].max()
        for f in (0.3, 0.6, 0.9):
            c = engine.decision_points(pack, (f,))[0]
            r = engine.laudo(pack, c, replace(st, n_samples=2000))
            d = {x["destino"]: x for x in r["destinos"]}
            best = d[r["recomendacao"]]
            rows.append({"pack": pack, "momento": f"{int(f * 100)}% da vida", "fracao": float(c / n),
                         "recomendacao": r["recomendacao"], "vpl_recomendado": float(best["vpl_medio"]),
                         "vpl_reciclagem": float(d["reciclagem"]["vpl_medio"]),
                         "ganho_brl_pack": float(best["vpl_medio"] - d["reciclagem"]["vpl_medio"])})
    return rows


def main():
    t0 = time.time()
    d = P.load_cycles()
    engine = P.Engine(d)
    print(f"engine {time.time() - t0:.0f}s", flush=True)
    out = {"gerado_em": date.today().isoformat(), "perfis": {}}
    for prof in PROFILES:
        node = {"economia": profile_settings(prof).economics.as_dict(),
                "destinos": [dict(vars(x)) for x in profile_settings(prof).destinations],
                "janela": {}, "frota": {}, "grades": {}, "reparo": {}}
        for lm in SCENARIOS:
            st = profile_settings(prof, life_model=lm)
            node["janela"][lm] = P.decision_windows(engine, life_model=lm, base=st).to_dict(orient="records")
            node["frota"][lm] = fleet_counts(engine, st)
            node["grades"][lm] = grids(engine, st)
            node["reparo"][lm] = P.repair_frontier(engine, life_model=lm, base=st).to_dict(orient="records")
            node.setdefault("valor_vs_reciclagem", {})[lm] = value_vs_recycling(engine, st)
            print(f"{prof} {lm} {time.time() - t0:.0f}s", flush=True)
        if prof == "hev_nimh_corolla":
            node["sensibilidade_remanufatura"] = {
                lm: P.remanufacture_breakeven(engine, profile_settings(prof, life_model=lm)).to_dict(orient="records")
                for lm in SCENARIOS}
        out["perfis"][prof] = node
    OUT.write_text(json.dumps(clean(out), ensure_ascii=False, separators=(",", ":")), encoding="utf-8")
    print(f"ok {time.time() - t0:.0f}s -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
