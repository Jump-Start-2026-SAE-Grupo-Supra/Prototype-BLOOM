"""Gates de segurança: eliminam destinos inseguros antes de qualquer otimização.

Lai et al. (2021, seção 3.2.3): "side reaction is the most important one [...] a one-dimensional
classification problem". Aqui, sem EIS nem sensores de gás, cada gate usa o sinal disponível
no dataset e é explícito sobre o que mede:

- G1 térmico: aquecimento não explicado por I²R subindo em relação à linha de base do próprio
  pack (Li et al., 2024: célula mais envelhecida gera mais calor irreversível) -> VETO de reuso.
  A comparação é com o próprio pack porque o offset do termopar varia ±11 °C entre packs do
  dataset. Geng et al. (2026): a triagem usual considera o risco térmico só implicitamente.
- G2 resistência: crescimento de R aparente vs. pack novo -> restringe destinos de potência.
- G3 relaxação pós-carga (proxy de autodescarga): queda anômala no repouso -> restringe backup.
- G4 joelho: fade recente muito acima do fade típico do próprio pack (Guan et al., 2025: packs
  aposentados podem estar no "knee point") -> VETO de segunda vida.
- G5 incerteza/novidade: intervalo largo ou pack fora da distribuição de treino -> inconclusivo.
- G6 procedência: pack remontado com células de outros packs sem ensaio de recomissão -> inconclusivo
  (o modelo superestima o SOH desses packs: a capacidade segue a pior célula, a resistência a média).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

from .config import DESTINATIONS, SafetyThresholds

REUSE = [d.key for d in DESTINATIONS if d.kind == "reuso"]


def _thermal_design(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({"i_mean": df["i_mean"], "i2r": df["i_mean"] ** 2 * df["r_app_120s"],
                         "temp_start": df["temp_start"]})


@dataclass
class FleetReference:
    thermal_model: HuberRegressor
    thermal_drift_limit: float = np.nan
    self_discharge_limit: float = np.nan
    r_new: float = np.nan
    novelty_limit: float = np.nan
    details: dict = field(default_factory=dict)

    def thermal_residual(self, df: pd.DataFrame) -> pd.Series:
        X = _thermal_design(df)
        ok = X.notna().all(axis=1) & df["dT_120s"].notna()
        out = pd.Series(np.nan, index=df.index)
        out[ok] = df.loc[ok, "dT_120s"] - self.thermal_model.predict(X[ok])
        return out


def thermal_drift(history: pd.DataFrame, baseline_frac: float = 0.1, recent: int = 5) -> float:
    """Resíduo térmico recente menos o resíduo do início de vida do mesmo pack (°C)."""
    h = history.dropna(subset=["thermal_resid"])
    nb = max(5, int(len(h) * baseline_frac))
    if len(h) < nb + recent:
        return np.nan
    return float(h["thermal_resid"].iloc[-recent:].median() - h["thermal_resid"].iloc[:nb].median())


def fit_reference(df: pd.DataFrame, th: SafetyThresholds, novelty_limit: float = np.nan,
                  early_frac: float = 0.25) -> FleetReference:
    """Estatísticas da frota saudável: primeiros 25% da vida dos packs de primeira vida."""
    first = df[(df["group"] != "second_life") & (~df["is_ref"])].copy()
    first["life_frac"] = first["cycle"] / first.groupby("pack")["cycle"].transform("max")
    healthy = first[first["life_frac"] <= early_frac].dropna(subset=["dT_120s", "r_app_120s", "temp_start"])
    model = HuberRegressor().fit(_thermal_design(healthy), healthy["dT_120s"])
    ref = FleetReference(model, novelty_limit=novelty_limit)

    # deriva térmica "normal": pontos ao longo da primeira metade da vida dos packs de primeira vida
    first["thermal_resid"] = ref.thermal_residual(first)
    drifts = []
    for _, g in first.groupby("pack"):
        g = g[g["life_frac"] <= 0.5]
        for i in range(10, len(g), 5):
            v = thermal_drift(g.iloc[:i])
            if np.isfinite(v):
                drifts.append(v)
    sd = healthy["post_chg_drop_540s"].dropna()
    ref.thermal_drift_limit = float(np.quantile(drifts, th.thermal_quantile) + th.thermal_margin_c)
    ref.self_discharge_limit = float(sd.quantile(th.self_discharge_quantile) * th.self_discharge_margin)
    ref.r_new = float(healthy.groupby("pack")["r_app_10s"].median().median())
    ref.details = {"n_healthy_cycles": int(len(healthy)),
                   "thermal_coef_[I, I2R, T0]": [float(c) for c in model.coef_],
                   "thermal_drift_p99": float(np.quantile(drifts, 0.99)),
                   "self_discharge_p99": float(sd.quantile(0.99))}
    return ref


@dataclass
class GateResult:
    gate: str
    status: str               # "ok" | "restringe" | "veta" | "inconclusivo" | "sem dado"
    value: float | None
    limit: float | None
    blocks: list[str]
    evidence: str
    source: str


def recent_fade_ratio(history: pd.DataFrame, window: int) -> float:
    """Inclinação da capacidade de missão nos últimos `window` ciclos / inclinação da vida toda."""
    h = history[~history["is_ref"]].dropna(subset=["dis_ah"])
    if len(h) < 2 * window:
        return np.nan
    base = h["dis_ah"].iloc[:5].median()
    full = -np.polyfit(h["cum_efc"], h["dis_ah"] / base, 1)[0]
    tail = h.iloc[-window:]
    recent = -np.polyfit(tail["cum_efc"], tail["dis_ah"] / base, 1)[0]
    return float(recent / full) if full > 1e-6 else np.nan


def evaluate(history: pd.DataFrame, soh_p10: float, soh_p90: float, ref: FleetReference,
             th: SafetyThresholds, novelty: float = np.nan, rebuilt: bool = False,
             commissioned: bool = False) -> list[GateResult]:
    """Avalia os gates no último ciclo de `history` (histórico do pack até a decisão).

    `history` precisa da coluna `thermal_resid` (FleetReference.thermal_residual).
    """
    hist = history[~history["is_ref"]]
    last = hist.iloc[-5:]
    out: list[GateResult] = []

    drift = thermal_drift(hist)
    if np.isfinite(drift):
        veto = drift > ref.thermal_drift_limit
        out.append(GateResult(
            "G1 autoaquecimento", "veta" if veto else "ok", drift, ref.thermal_drift_limit,
            REUSE if veto else [],
            f"aquecimento em 2 min de pulso não explicado por I²R: {drift:+.1f} °C sobre a linha de base "
            f"do próprio pack (limite {ref.thermal_drift_limit:.1f} °C)",
            "Li et al. 2024; Geng et al. 2026 §6.1"))
    else:
        out.append(GateResult("G1 autoaquecimento", "sem dado", None, ref.thermal_drift_limit, [],
                              "sem histórico térmico suficiente para linha de base", ""))

    r = float(last["r_app_10s"].median())
    growth = r / ref.r_new
    blocks = [d.key for d in DESTINATIONS if d.kind == "reuso" and growth > d.r_growth_max]
    out.append(GateResult(
        "G2 resistência", "restringe" if blocks else "ok", growth, None, blocks,
        f"R aparente (pulso 10 s) = {r * 1000:.0f} mΩ = {growth:.2f}× o pack novo da frota",
        "Guan et al. 2025 §4.2.2; Seger et al. 2022"))

    sd = float(last["post_chg_drop_540s"].median())
    if np.isfinite(sd):
        bad = sd > ref.self_discharge_limit
        out.append(GateResult(
            "G3 relaxação pós-carga", "restringe" if bad else "ok", sd, ref.self_discharge_limit,
            [d.key for d in DESTINATIONS if d.needs_low_self_discharge] if bad else [],
            f"queda de tensão em 9 min de repouso pós-carga = {sd * 1000:.0f} mV "
            f"(limite {ref.self_discharge_limit * 1000:.0f} mV; proxy de autodescarga)",
            "Dossiê §3.5; Wang et al. 2023"))

    ratio = recent_fade_ratio(history, th.knee_window)
    if np.isfinite(ratio):
        knee = ratio > th.knee_ratio
        out.append(GateResult(
            "G4 joelho de degradação", "veta" if knee else "ok", ratio, th.knee_ratio,
            [k for k in REUSE if k != "original"] if knee else [],
            f"fade dos últimos {th.knee_window} ciclos = {ratio:.1f}× o fade médio do próprio pack",
            "Guan et al. 2025 §4.1.3"))
    else:
        out.append(GateResult("G4 joelho de degradação", "sem dado", None, th.knee_ratio, [],
                              "histórico curto demais para medir inclinação", ""))

    width = (soh_p90 - soh_p10) * 100
    novel = bool(np.isfinite(novelty) and np.isfinite(ref.novelty_limit) and novelty > ref.novelty_limit)
    msg = f"intervalo P10–P90 de SOH = {width:.1f} p.p."
    if np.isfinite(novelty):
        msg += f"; distância à frota de treino = {novelty:.1f} (limite {ref.novelty_limit:.1f})"
    out.append(GateResult(
        "G5 incerteza e novidade", "inconclusivo" if (width > th.uncertainty_max_pp or novel) else "ok",
        width, th.uncertainty_max_pp, [], msg, "Wei et al. 2022 (erro como distribuição)"))

    if rebuilt:
        out.append(GateResult(
            "G6 procedência", "ok" if commissioned else "inconclusivo", None, None, [],
            "pack remontado com células de outros packs; "
            + ("SOH ancorado no ensaio de recomissão" if commissioned
               else "sem ensaio de recomissão o laudo rápido tende a superestimar o SOH"),
            "Wang et al. 2023; UL 1974 (avaliação individual)"))
    return out


def blocked_destinations(gates: list[GateResult]) -> dict[str, list[str]]:
    reasons: dict[str, list[str]] = {}
    for g in gates:
        for k in g.blocks:
            reasons.setdefault(k, []).append(f"{g.gate}: {g.evidence}")
    return reasons
