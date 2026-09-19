"""Lifecycle Router: escolhe o destino de maior valor esperado entre os destinos seguros.

Ordem (Lai et al., 2021; Guan et al., 2025):
 1. gates de segurança eliminam destinos (safety.py);
 2. requisitos de cada destino (SOH mínimo, resistência) eliminam mais;
 3. entre os restantes, compara o VPL esperado sob incerteza (Monte Carlo sobre SOH e vida),
    incluindo o preço da garantia que a incerteza implica (Dossiê §3.4) e o valor da informação
    de um ensaio completo.

A vida em cada destino combina:
 - fade de capacidade por ciclo equivalente, que cresce com a corrente (modelo log-linear
   ajustado nos packs do dataset e extrapolado para a taxa C do destino) + fade calendárico;
 - no cenário "conservador", também a vida residual até a falha abrupta observada no ensaio
   acelerado, transportada para o destino sem creditar vida abaixo da menor corrente ensaiada.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .config import DEST, Settings
from .health import sample_from_quantiles
from .safety import GateResult, blocked_destinations

CALENDAR_CAP_YEARS = 10.0  # o dataset não mede envelhecimento calendárico; limitamos a vida
NOMINAL_AH = 2.5


@dataclass
class StressModel:
    """fade(I) = exp(a + b·I) em fração de SOH por ciclo equivalente completo (I em A por pack)."""
    a: float
    b: float
    sigma: float  # desvio (log) entre packs
    points: pd.DataFrame
    min_current_a: float

    def fade(self, current_a):
        return np.exp(self.a + self.b * np.asarray(current_a, dtype=float))

    def acceleration(self, from_a: float, to_a: float, clip_to_data: bool = False) -> float:
        """Quantas vezes a vida (em ciclos) se alonga ao passar da corrente `from_a` para `to_a`."""
        if clip_to_data:
            to_a = max(to_a, self.min_current_a)
        return float(self.fade(from_a) / self.fade(to_a))


def fit_stress_model(df: pd.DataFrame) -> StressModel:
    rows = []
    for pack, g in df[df["group"].isin(["regular_cc", "regular_random", "second_life"])].groupby("pack"):
        ref = g[g["is_ref"] & (g["dis_ah"] > 1.0)]
        reg = g[~g["is_ref"]]
        if len(ref) < 2:
            continue
        slope = -np.polyfit(ref["cum_ah"] / 2.5, ref["dis_ah"] / 2.5, 1)[0]
        if slope <= 0:
            continue
        rows.append({"pack": pack, "group": g["group"].iloc[0], "i_mean": reg["i_mean"].median(),
                     "fade_per_efc": slope, "n_ref": len(ref)})
    pts = pd.DataFrame(rows)
    b, a = np.polyfit(pts["i_mean"], np.log(pts["fade_per_efc"]), 1)
    resid = np.log(pts["fade_per_efc"]) - (a + b * pts["i_mean"])
    return StressModel(a=float(a), b=float(b), sigma=float(resid.std(ddof=1)), points=pts,
                       min_current_a=float(pts["i_mean"].min()))


@dataclass
class Snapshot:
    """Tudo o que o roteador precisa sobre um pack no instante da decisão."""
    pack: str
    cycle: int
    soh_q: tuple[float, float, float]
    rul_efc_q: tuple[float, float, float]   # vida residual (ciclos eq.) na severidade atual
    current_a: float                        # severidade do uso atual (A)
    gates: list[GateResult]


def _life_years(snap: Snapshot, dest, soh: np.ndarray, rul: np.ndarray, stress: StressModel,
                rng: np.random.Generator, settings: Settings) -> np.ndarray:
    """Anos até o SOH cair ao fim de vida do destino (fade por ciclo + fade calendárico)."""
    n = len(soh)
    noise = np.exp(rng.normal(0, stress.sigma, n))
    dest_a = dest.c_rate * NOMINAL_AH
    fade_per_year = dest.cycles_per_year * stress.fade(dest_a) * noise + settings.calendar_fade_per_year
    years = np.clip(soh - dest.soh_eol, 0, None) / fade_per_year
    if settings.life_model == "conservador":
        by_failure = rul * stress.acceleration(snap.current_a, dest_a, clip_to_data=True) * noise
        years = np.minimum(years, by_failure / dest.cycles_per_year)
    return np.minimum(years, CALENDAR_CAP_YEARS)


def route(snap: Snapshot, stress: StressModel, settings: Settings) -> dict:
    eco = settings.economics
    rng = np.random.default_rng(settings.seed)
    n = settings.n_samples
    soh = np.clip(sample_from_quantiles(*snap.soh_q, n, rng), 0.3, 1.05)
    rul = np.clip(sample_from_quantiles(*snap.rul_efc_q, n, rng), 0, None)
    kwh = eco.pack_kwh
    blocked = blocked_destinations(snap.gates)
    recycle_net = kwh * (eco.material_value_brl_kwh - eco.recycling_process_brl_kwh - eco.logistics_brl_kwh)

    table, samples = [], {}
    for d in settings.destinations:
        row = {"destino": d.key, "rótulo": d.label, "critério": d.criterion, "motivos_eliminação": []}
        if d.kind == "reciclagem":
            v = np.full(n, recycle_net)
            row.update(vida_anos_p50=None, p_falha_garantia=None, premio_garantia=0.0)
        else:
            reasons = list(blocked.get(d.key, []))
            if snap.soh_q[1] < d.soh_min:
                reasons.append(f"SOH P50 {snap.soh_q[1]:.0%} abaixo do mínimo do destino ({d.soh_min:.0%})")
            row["motivos_eliminação"] = reasons
            years = _life_years(snap, d, soh, rul, stress, rng, settings)
            usable = kwh * soh
            unit_price = d.new_product_brl_kwh * eco.new_price_index * d.price_factor
            sale = usable * unit_price * np.minimum(1, years / d.ref_life_years)
            costs = kwh * (d.repack_brl_kwh + ((eco.logistics_brl_kwh + eco.certification_brl_kwh) if d.moves else 0))
            eol = recycle_net / (1 + eco.discount_rate) ** years
            p_fail = float(np.mean(years < d.warranty_years))
            premium = eco.warranty_loading * p_fail * float(np.median(usable)) * unit_price
            v = sale - costs + eol - premium
            row.update(vida_anos_p10=float(np.quantile(years, 0.1)), vida_anos_p50=float(np.median(years)),
                       p_falha_garantia=p_fail, premio_garantia=premium,
                       receita_p50=float(np.median(sale)), custos=float(costs))
        row.update(elegivel=not row["motivos_eliminação"], vpl_medio=float(v.mean()),
                   vpl_p10=float(np.quantile(v, 0.1)), vpl_p90=float(np.quantile(v, 0.9)))
        samples[d.key] = v
        table.append(row)

    eligible = [r["destino"] for r in table if r["elegivel"]]
    V = np.column_stack([samples[k] for k in eligible])
    best_idx = np.argmax(V, axis=1)
    p_best = {k: float(np.mean(best_idx == i)) for i, k in enumerate(eligible)}
    means = V.mean(axis=0)
    choice = eligible[int(np.argmax(means))]
    evpi = float(np.mean(V.max(axis=1)) - means.max())
    for r in table:
        r["p_melhor"] = p_best.get(r["destino"], 0.0)

    gates_veto = [g for g in snap.gates if g.status == "veta"]
    inconclusive = any(g.status == "inconclusivo" for g in snap.gates)
    need_test = evpi > eco.reference_test_brl_pack
    return {
        "pack": snap.pack, "ciclo": snap.cycle,
        "soh": dict(zip(("p10", "p50", "p90"), snap.soh_q)),
        "vida_residual_efc": dict(zip(("p10", "p50", "p90"), snap.rul_efc_q)),
        "gates": [g.__dict__ for g in snap.gates],
        "destinos": table,
        "recomendacao": choice,
        "confianca": p_best[choice],
        "valor_da_informacao": evpi,
        "recomenda_ensaio_completo": bool(need_test or inconclusive),
        "vetos": [g.gate for g in gates_veto],
        "premissas": eco.as_dict(),
        "modelo_de_vida": settings.life_model,
    }


def laudo_markdown(res: dict) -> str:
    """Laudo auditável: a conta de cada destino e o motivo de cada eliminação."""
    s = res["soh"]
    L = [f"## Laudo BLOOM · pack {res['pack']} · ciclo {res['ciclo']}", "",
         f"**SOH** P50 {s['p50']:.1%} (intervalo 80%: {s['p10']:.1%} – {s['p90']:.1%})  ",
         f"**Vida residual no uso atual**: {res['vida_residual_efc']['p50']:.0f} ciclos equivalentes "
         f"(P10 {res['vida_residual_efc']['p10']:.0f})", "", "### 1. Gates de segurança", "",
         "| gate | status | evidência | fonte |", "|---|---|---|---|"]
    for g in res["gates"]:
        L.append(f"| {g['gate']} | **{g['status']}** | {g['evidence']} | {g['source']} |")
    L += ["", "### 2. Destinos", "",
          "| destino | elegível | VPL médio (R$) | VPL P10 | vida P50 (anos) | P(falha na garantia) | prêmio garantia | P(melhor) |",
          "|---|---|---|---|---|---|---|---|"]
    for r in res["destinos"]:
        vida = "—" if r.get("vida_anos_p50") is None else f"{r['vida_anos_p50']:.1f}"
        pf = "—" if r.get("p_falha_garantia") is None else f"{r['p_falha_garantia']:.0%}"
        L.append(f"| {r['rótulo']} | {'sim' if r['elegivel'] else 'não'} | {r['vpl_medio']:,.0f} | "
                 f"{r['vpl_p10']:,.0f} | {vida} | {pf} | {r['premio_garantia']:,.0f} | {r['p_melhor']:.0%} |")
    elim = [r for r in res["destinos"] if not r["elegivel"]]
    if elim:
        L += ["", "**Destinos eliminados e por quê:**"]
        for r in elim:
            L += [f"- *{r['rótulo']}*: " + "; ".join(r["motivos_eliminação"])]
    best = DEST[res["recomendacao"]]
    L += ["", "### 3. Recomendação", "",
          f"**{best.label}** — escolhido em {res['confianca']:.0%} dos cenários simulados.",
          f"Valor da informação de um ensaio completo: R$ {res['valor_da_informacao']:,.0f} "
          f"(ensaio custa R$ {res['premissas']['reference_test_brl_pack']:,.0f}) → "
          + ("**recomenda-se ensaio antes de decidir**." if res["recomenda_ensaio_completo"]
             else "decisão pode ser tomada com o laudo rápido.")]
    return "\n".join(L)
