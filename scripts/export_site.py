"""Gera o site estático em docs/ (GitHub Pages) com os resultados embutidos.

    python scripts/export_site.py                        # perfil bev_revisado (padrão), ~6-10 min
    python scripts/export_site.py --perfil v0_ilustrativo
    python scripts/export_site.py --html-only            # só reaplica site/template.html ao JSON existente

O perfil escolhe as premissas econômicas (`bloom.config.profile_settings`). O padrão é
`bev_revisado`, a correção de preços de 18/09/2026 documentada em PREMISSAS.md, que é a que o
Business Case cita; `v0_ilustrativo` reproduz a versão anterior da página. Seja qual for o perfil da
página, a seção do Corolla é sempre calculada com `hev_nimh_corolla`, porque é um produto diferente
(pack de 1,3 kWh de NiMH) e não uma variante de premissa do mesmo pack.

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
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bloom import pipeline as P  # noqa: E402
from bloom.config import profile_settings  # noqa: E402
from bloom.extract import PACKS, SECOND_LIFE_ORIGIN  # noqa: E402
from bloom.health import metrics  # noqa: E402

SCENARIOS = ["conservador", "fade"]
FRACS = (0.1, 0.3, 0.5, 0.7, 0.9, 0.97)
OUT_JSON = ROOT / "docs" / "data" / "bloom.json"
DEFAULT_PROFILE = "bev_revisado"
PAGE_PROFILES = ("bev_revisado", "v0_ilustrativo")  # perfis que a página inteira aceita
HEV_PROFILE = "hev_nimh_corolla"

# Texto que a página mostra sobre o perfil escolhido. A conta é a mesma; só as premissas mudam.
PROFILE_NOTES = {
    "v0_ilustrativo": "Premissas originais do protótipo (produto novo a R$ 900/kWh, sistema estacionário a "
                      "R$ 1.800/kWh). Mantido para reproduzir a versão anterior da página.",
    "bev_revisado": "Premissas corrigidas em 18/09/2026 (PREMISSAS.md): pack de reposição de elétrico a "
                    "R$ 1.250/kWh, apurado de R$ 1.167–1.336/kWh, e sistema estacionário instalado a "
                    "R$ 3.100/kWh, ponto médio de R$ 2.700–3.500/kWh. Preços de imprensa, confiança média.",
    HEV_PROFILE: "Pack de híbrido (Corolla, NiMH, 1,3 kWh): reposição a R$ 13.077/kWh, remanufaturado a 53% "
                 "do novo, sem destino estacionário. Sensibilidade econômica, não simulação do pack real.",
}


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


def corolla_section(engine, t0: float) -> dict:
    """Perfil híbrido: remanufatura contra reciclagem, e o custo de equilíbrio da remanufatura.

    Roda sempre com `hev_nimh_corolla`, independentemente do perfil da página: é outro produto, não
    outra premissa do mesmo pack. Só a economia é de NiMH — a dinâmica de envelhecimento continua
    sendo a de íon-lítio do dataset, o que faz desta seção uma sensibilidade, não uma simulação.
    """
    st = profile_settings(HEV_PROFILE)
    out = {"perfil": HEV_PROFILE, "nota": PROFILE_NOTES[HEV_PROFILE],
           "economia": st.economics.as_dict(), "destinos": [dict(vars(x)) for x in st.destinations],
           "breakeven": {}}
    for lm in SCENARIOS:
        df = P.remanufacture_breakeven(engine, profile_settings(HEV_PROFILE, life_model=lm))
        out["breakeven"][lm] = df.to_dict(orient="records")
        print(f"corolla {lm} {time.time() - t0:.0f}s", flush=True)
    return out


def build_data(profile: str = DEFAULT_PROFILE) -> dict:
    t0 = time.time()
    base = profile_settings(profile)
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
        res = engine.fleet_decisions(replace(base, life_model=lm, n_samples=2000), FRACS)
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
            gr = P.reuse_share_grid(engine, pts, life_model=lm, base=base)
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
        "perfil": {"nome": profile, "nota": PROFILE_NOTES[profile]},
        "destinos": [dict(vars(x)) for x in base.destinations],
        "economia": base.economics.as_dict(),
        "calendario": base.calendar_fade_per_year,
        "packs": packs, "frota": fleet, "grades": grids,
        "laco": P.closed_loop(engine).to_dict(orient="records"),
        "janela": {lm: P.decision_windows(engine, life_model=lm, base=base).to_dict(orient="records")
                   for lm in SCENARIOS},
        "reparo": {lm: P.repair_frontier(engine, life_model=lm, base=base).to_dict(orient="records")
                   for lm in SCENARIOS},
        "corolla": corolla_section(engine, t0),
    })


def parse_profile(argv: list[str]) -> str:
    """--perfil <nome> ou --perfil=<nome>; o padrão é DEFAULT_PROFILE."""
    name = DEFAULT_PROFILE
    for i, a in enumerate(argv):
        if a == "--perfil" and i + 1 < len(argv):
            name = argv[i + 1]
        elif a.startswith("--perfil="):
            name = a.split("=", 1)[1]
    if name not in PAGE_PROFILES:
        raise SystemExit(f"perfil desconhecido: {name!r} · use um de {', '.join(PAGE_PROFILES)}"
                         f" ({HEV_PROFILE} é outro produto: sai na seção do Corolla, não na página toda)")
    return name


def main():
    if "--html-only" in sys.argv:
        write_html(OUT_JSON.read_text(encoding="utf-8"))
        print("html regenerado a partir de", OUT_JSON)
        return
    profile = parse_profile(sys.argv[1:])
    print(f"perfil: {profile}", flush=True)
    t0 = time.time()
    payload = json.dumps(build_data(profile), ensure_ascii=False, separators=(",", ":"))
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(payload, encoding="utf-8")
    write_html(payload)
    print(f"ok {time.time() - t0:.0f}s · {len(payload) / 1e6:.2f} MB de dados")


if __name__ == "__main__":
    main()
