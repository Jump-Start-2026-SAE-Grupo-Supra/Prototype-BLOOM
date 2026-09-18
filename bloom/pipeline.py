"""Orquestra o protótipo ponta a ponta: saúde -> gates -> roteador -> laudo.

Todas as previsões usadas nos laudos são fora da amostra: o pack avaliado nunca participou
do treino do modelo que o avalia (validação por grupos de packs).
"""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from .config import Settings
from .extract import SECOND_LIFE_ORIGIN
from .health import (FEATURE_SETS, QUICK_TEST, TELEMETRY, QuantileModel, _pack_balanced_quantile,
                     add_derived, age_baseline_cv, conformity_scores, cross_validate, metrics,
                     training_rows)
from .router import NOMINAL_AH, Snapshot, StressModel, fit_stress_model, route
from .safety import FleetReference, evaluate, fit_reference

ROOT = Path(__file__).resolve().parents[1]
CYCLES = ROOT / "data" / "processed" / "cycles.parquet"


def load_cycles(path: Path = CYCLES) -> pd.DataFrame:
    return add_derived(pd.read_parquet(path))


# --------------------------------------------------------------------------------------------
# 1. Saúde
# --------------------------------------------------------------------------------------------
def benchmark_soh(d: pd.DataFrame, n_splits: int = 13) -> tuple[pd.DataFrame, dict]:
    """Compara os conjuntos de features com validação por packs."""
    tr = training_rows(d)
    rows, oofs = [], {}
    for name, feats in FEATURE_SETS.items():
        oof = age_baseline_cv(tr, n_splits=n_splits) if name.startswith("idade") \
            else cross_validate(tr, feats, n_splits=n_splits)
        oofs[name] = oof
        rows.append({"conjunto": name, "n_features": len(feats), **metrics(tr["soh"].to_numpy(), oof)})
        for group, g in tr.groupby("group"):
            rows.append({"conjunto": name, "grupo": group, **metrics(g["soh"].to_numpy(), oof.loc[g.index])})
    table = pd.DataFrame(rows)
    table["grupo"] = table["grupo"].fillna("todos")
    return table, {"rows": tr, "oof": oofs}


def oof_all_rows(d: pd.DataFrame, features: list[str], target: str, n_splits: int = 13,
                 log_target: bool = False) -> pd.DataFrame:
    """Previsão fora da amostra para TODAS as missões (inclusive sem rótulo interpolado).

    O treino usa só as linhas rotuladas dos outros packs; o alargamento conformal de cada fold
    vem dos escores dos demais folds.
    """
    rows = d[~d["is_ref"]]
    if target == "soh":
        labeled = (~rows["label_extrapolated"]) & rows["soh"].notna()
    else:
        labeled = rows[target].notna()
    y = np.log1p(rows[target]) if log_target else rows[target]
    frame = rows.assign(_y=y)
    groups = rows["pack"].to_numpy()
    pred = pd.DataFrame(index=rows.index, columns=["p10", "p50", "p90", "fold"], dtype=float)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(rows, groups=groups)):
        train = frame.iloc[tr][labeled.iloc[tr].to_numpy()]
        m = QuantileModel(features, "_y").fit(train)
        pred.iloc[te, :3] = m.predict(frame.iloc[te]).to_numpy()
        pred.iloc[te, 3] = k
    s = conformity_scores(frame["_y"].to_numpy(), pred)
    lab = labeled.to_numpy()
    for k in np.unique(pred["fold"]):
        other = (pred["fold"].to_numpy() != k) & lab
        w = _pack_balanced_quantile(s[other], groups[other], 0.8)
        mask = pred["fold"].to_numpy() == k
        pred.loc[mask, "p10"] -= w
        pred.loc[mask, "p90"] += w
    if log_target:
        pred[["p10", "p50", "p90"]] = np.expm1(pred[["p10", "p50", "p90"]]).clip(lower=0)
    pred["pack"] = rows["pack"]
    return pred


def novelty_scores(d: pd.DataFrame, features: list[str] = QUICK_TEST, k: int = 10) -> pd.Series:
    """Distância média aos k vizinhos na frota de primeira vida, sem o próprio pack."""
    rows = d[~d["is_ref"]]
    X = rows[features].fillna(rows[features].median())
    first = (rows["group"] != "second_life").to_numpy()
    out = pd.Series(np.nan, index=rows.index)
    for p in rows["pack"].unique():
        own = (rows["pack"] == p).to_numpy()
        ref = first & ~own
        sc = StandardScaler().fit(X[ref])
        nn = NearestNeighbors(n_neighbors=k).fit(sc.transform(X[ref]))
        out[own] = nn.kneighbors(sc.transform(X[own]))[0].mean(axis=1)
    return out


def anchor_after_commissioning(d: pd.DataFrame, soh_pred: pd.DataFrame, n_anchor: int = 5) -> pd.DataFrame:
    """Packs remontados: um ensaio de referência na recomissão ancora o laudo rápido.

    O deslocamento é medido nas `n_anchor` primeiras missões (onde o ensaio de recomissão
    fornece o SOH) e aplicado dali em diante. O intervalo é recalibrado com os erros ancorados
    dos OUTROS packs remontados (só há três no dataset: a calibração é frágil e isso é declarado).
    """
    out = soh_pred.copy()
    out["anchored"] = False
    rows = d.loc[soh_pred.index]
    packs = list(rows.loc[rows["group"] == "second_life", "pack"].unique())
    scores = {}
    for p in packs:
        idx = rows.index[rows["pack"] == p]
        first = idx[:n_anchor]
        offset = float((rows.loc[first, "soh"] - out.loc[first, "p50"]).mean())
        out.loc[idx, ["p10", "p50", "p90"]] += offset
        out.loc[idx[n_anchor:], "anchored"] = True
        after = idx[n_anchor:]
        lab = after[(~rows.loc[after, "label_extrapolated"]) & rows.loc[after, "soh"].notna()]
        scores[p] = conformity_scores(rows.loc[lab, "soh"].to_numpy(), out.loc[lab])
    for p in packs:
        others = np.concatenate([scores[q] for q in packs if q != p])
        groups = np.concatenate([[q] * len(scores[q]) for q in packs if q != p])
        w = max(0.0, _pack_balanced_quantile(others, groups, 0.8))
        idx = rows.index[(rows["pack"] == p)][n_anchor:]
        out.loc[idx, "p10"] -= w
        out.loc[idx, "p90"] += w
    return out


# --------------------------------------------------------------------------------------------
# 2. Decisão
# --------------------------------------------------------------------------------------------
class Engine:
    """Estado ajustado uma vez e reutilizado nos laudos."""

    def __init__(self, d: pd.DataFrame, settings: Settings | None = None, n_splits: int = 13):
        self.settings = settings or Settings()
        self.d = d.copy()
        self.stress: StressModel = fit_stress_model(d)
        raw = oof_all_rows(d, QUICK_TEST, "soh", n_splits)
        self.soh_raw = raw
        self.soh = anchor_after_commissioning(d, raw)
        self.rul = oof_all_rows(d, TELEMETRY, "efc_to_eol", n_splits, log_target=True)
        self.novelty = novelty_scores(d)
        first = self.d.loc[self.novelty.index, "group"] != "second_life"
        limit = float(self.novelty[first].quantile(0.99))
        self.ref: FleetReference = fit_reference(d, self.settings.safety, novelty_limit=limit)
        self.d["thermal_resid"] = self.ref.thermal_residual(self.d)

    def snapshot(self, pack: str, cycle: int, window: int = 5, commissioned: bool = True) -> tuple[Snapshot, pd.DataFrame]:
        g = self.d[self.d["pack"] == pack]
        hist = g[g["cycle"] <= cycle]
        reg = hist[~hist["is_ref"]]
        recent = reg.index[-window:]
        soh_src = self.soh if commissioned else self.soh_raw
        soh_q = tuple(float(soh_src.loc[recent, q].median()) for q in ("p10", "p50", "p90"))
        rul_q = tuple(float(self.rul.loc[recent[-1], q]) for q in ("p10", "p50", "p90"))
        rebuilt = g["group"].iloc[0] == "second_life"
        anchored = bool(self.soh.loc[recent, "anchored"].all()) if rebuilt and commissioned else False
        gates = evaluate(hist, soh_q[0], soh_q[2], self.ref, self.settings.safety,
                         novelty=float(self.novelty.loc[recent].median()), rebuilt=rebuilt,
                         commissioned=anchored)
        snap = Snapshot(pack, int(cycle), soh_q, rul_q, float(reg["i_mean"].iloc[-10:].median()), gates)
        return snap, hist

    def laudo(self, pack: str, cycle: int, settings: Settings | None = None, **kw) -> dict:
        snap, hist = self.snapshot(pack, cycle, **kw)
        res = route(snap, self.stress, settings or self.settings)
        last = hist[~hist["is_ref"]].iloc[-1]
        res.update(grupo=str(last["group"]), soh_real=float(last["soh"]) if np.isfinite(last["soh"]) else None,
                   rotulo_extrapolado=bool(last["label_extrapolated"]),
                   efc_real_ate_falha=float(last["efc_to_eol"]), corrente_a=snap.current_a,
                   ciclos_totais=int(self.d.loc[self.d["pack"] == pack, "cycle"].max()))
        return res

    def decision_points(self, pack: str, fracs=(0.1, 0.3, 0.5, 0.7, 0.9, 0.97)) -> list[int]:
        cycles = self.d.loc[(self.d["pack"] == pack) & (~self.d["is_ref"]), "cycle"].to_numpy()
        return sorted({int(cycles[min(len(cycles) - 1, max(5, int(f * (len(cycles) - 1))))]) for f in fracs})

    def fleet_decisions(self, settings: Settings | None = None, fracs=(0.1, 0.3, 0.5, 0.7, 0.9, 0.97)) -> list[dict]:
        out = []
        for pack in sorted(self.d["pack"].unique()):
            n = self.d.loc[self.d["pack"] == pack, "cycle"].max()
            for c in self.decision_points(pack, fracs):
                r = self.laudo(pack, c, settings)
                r["fracao_vida"] = c / n
                out.append(r)
        return out


def decisions_table(results: list[dict]) -> pd.DataFrame:
    rows = []
    for r in results:
        row = {"pack": r["pack"], "grupo": r["grupo"], "ciclo": r["ciclo"], "fração da vida": r["fracao_vida"],
               "SOH real": r["soh_real"], "SOH P10": r["soh"]["p10"], "SOH P50": r["soh"]["p50"],
               "SOH P90": r["soh"]["p90"], "recomendação": r["recomendacao"], "confiança": r["confianca"],
               "vetos": ", ".join(r["vetos"]), "pede ensaio": r["recomenda_ensaio_completo"],
               "valor da informação": r["valor_da_informacao"]}
        for dst in r["destinos"]:
            row[f"VPL {dst['destino']}"] = dst["vpl_medio"] if dst["elegivel"] else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# 3. Laço fechado e sensibilidade
# --------------------------------------------------------------------------------------------
def closed_loop(engine: Engine) -> pd.DataFrame:
    """Packs de origem na aposentadoria vs. o que suas células entregaram na segunda vida.

    No dataset, os packs de origem foram ciclados até falhar e as células sobreviventes foram
    remontadas. Comparamos a vida prevista pelo roteador (na taxa de corrente real da segunda
    vida) com os ciclos equivalentes efetivamente entregues pelo pack remontado.
    """
    rows = []
    rng = np.random.default_rng(0)
    st = engine.stress
    for second, origins in SECOND_LIFE_ORIGIN.items():
        sl = engine.d[(engine.d["pack"] == second) & (~engine.d["is_ref"])]
        realized_efc = float(sl["cum_efc"].max() + sl["dis_ah"].iloc[-1] / NOMINAL_AH)
        i_second = float(sl["i_mean"].median())
        soh_start, soh_end = float(sl["soh"].dropna().iloc[0]), float(sl["soh"].dropna().iloc[-1])
        for o in origins:
            g = engine.d[(engine.d["pack"] == o) & (~engine.d["is_ref"])]
            c = int(g["cycle"].iloc[-2])  # último ciclo antes da missão que falhou
            snap, _ = engine.snapshot(o, c)
            soh = np.array(snap.soh_q)
            fade = st.fade(i_second)
            by_fade = (soh - soh_end) / fade
            acc = st.acceleration(snap.current_a, i_second, clip_to_data=True)
            by_failure = np.array(snap.rul_efc_q) * acc
            rows.append({"pack 2ª vida": second, "pack de origem": o, "corrente 1ª vida (A)": snap.current_a,
                         "corrente 2ª vida (A)": i_second, "SOH origem P50": soh[1],
                         "SOH início 2ª vida (real)": soh_start, "SOH fim 2ª vida (real)": soh_end,
                         "EFC previsto só fade (P50)": by_fade[1],
                         "EFC previsto conservador (P10–P90)": f"{min(by_fade[0], by_failure[0]):.0f}–"
                                                              f"{min(by_fade[2], by_failure[2]):.0f}",
                         "EFC conservador P50": min(by_fade[1], by_failure[1]),
                         "EFC entregue na 2ª vida": realized_efc})
    return pd.DataFrame(rows)


def reuse_share_grid(engine: Engine, results_cycles: list[tuple[str, int]], logistics=(20, 60, 120, 200),
                     price_index=(0.6, 0.8, 1.0, 1.2), life_model: str = "conservador") -> pd.DataFrame:
    """Teste de falsificação #2 do dossiê: o roteador manda o bastante para reuso ou a reciclagem domina?"""
    snaps = [engine.snapshot(p, c)[0] for p, c in results_cycles]
    out = []
    for lg in logistics:
        for price in price_index:
            s = Settings(economics=replace(Settings().economics, logistics_brl_kwh=lg, new_price_index=price),
                         n_samples=1000, life_model=life_model)
            recs = [route(sn, engine.stress, s)["recomendacao"] for sn in snaps]
            out.append({"frete R$/kWh": lg, "índice de preço do novo": price,
                        "fração para reuso": float(np.mean([r != "reciclagem" for r in recs]))})
    return pd.DataFrame(out)


# --------------------------------------------------------------------------------------------
# 4. Janela de decisão e valor do reparo
# --------------------------------------------------------------------------------------------
def decision_windows(engine: Engine, life_model: str = "conservador", n_points: int = 14,
                     n_samples: int = 1200) -> pd.DataFrame:
    """Até que ponto da vida um destino de REUSO ainda vence a reciclagem.

    Este é o teste que reorganiza o produto. A sensibilidade (`reuse_share_grid`) mostrou que na
    aposentadoria a reciclagem domina; logo, se a decisão de destino tem valor econômico, ele está
    ANTES. Aqui medimos, pack a pack, onde fica essa fronteira — a *janela de realocação* — em vez
    de assumir que a decisão acontece quando o pack chega ao pátio.
    """
    st = Settings(life_model=life_model, n_samples=n_samples)
    fracs = tuple(np.linspace(0.08, 0.97, n_points))
    rows = []
    for pack in sorted(engine.d["pack"].unique()):
        n = engine.d.loc[engine.d["pack"] == pack, "cycle"].max()
        traj = []
        for c in engine.decision_points(pack, fracs):
            r = engine.laudo(pack, c, st)
            traj.append({"fracao": float(c / n), "ciclo": int(c), "rec": r["recomendacao"],
                         "conf": float(r["confianca"]), "voi": float(r["valor_da_informacao"])})
        reuse = [t["fracao"] for t in traj if t["rec"] != "reciclagem"]
        rows.append({
            "pack": pack,
            "grupo": str(engine.d.loc[engine.d["pack"] == pack, "group"].iloc[0]),
            "ciclos": int(n),
            "fecha_em": float(max(reuse)) if reuse else float("nan"),
            "abre_em": float(min(reuse)) if reuse else float("nan"),
            "pontos_reuso": len(reuse),
            "pontos": len(traj),
            "sempre_reciclagem": not reuse,
            "trajetoria": traj,
        })
    return pd.DataFrame(rows)


def repair_value(engine: Engine, pack: str, cycle: int, uplift_pp=(0, 2, 4, 6, 8, 10, 15),
                 life_model: str = "conservador", n_samples: int = 2000) -> pd.DataFrame:
    """Quanto vale pagar para reparar o pack ANTES de decidir o destino.

    Reparo e remanufatura não são destinos: são AÇÕES que mudam o ativo, e só depois se decide
    para onde ele vai. Modelamos a ação como um ganho de SOH (`uplift`) aplicado aos três quantis
    — trocar o módulo pior puxa a capacidade do pack, que segue a pior célula ("efeito barril",
    Wang et al. 2023 — o mesmo mecanismo observado nos packs remontados deste dataset).

    Os gates permanecem: um veto de segurança não pode ser comprado com reparo.

    A saída é o **break-even** (o preço máximo defensável do reparo), não um preço inventado.
    O uplift também não é calibrado — é varrido. Calibrá-lo exige packs reparados, que este
    dataset não tem: é o dado que a fase F2 precisa gerar.
    """
    st = Settings(life_model=life_model, n_samples=n_samples)
    snap, _ = engine.snapshot(pack, cycle)
    base = route(snap, engine.stress, st)
    base_v = max(dd["vpl_medio"] for dd in base["destinos"] if dd["elegivel"])
    out = []
    for u in uplift_pp:
        s2 = replace(snap, soh_q=tuple(min(1.0, q + u / 100) for q in snap.soh_q))
        r2 = route(s2, engine.stress, st)
        v2 = max(dd["vpl_medio"] for dd in r2["destinos"] if dd["elegivel"])
        out.append({"pack": pack, "ciclo": int(cycle), "uplift_pp": u,
                    "destino_base": base["recomendacao"], "destino_reparado": r2["recomendacao"],
                    "vpl_base": base_v, "vpl_reparado": v2,
                    "break_even_reparo_brl": v2 - base_v,
                    "muda_destino": bool(r2["recomendacao"] != base["recomendacao"])})
    return pd.DataFrame(out)


def repair_frontier(engine: Engine, fracs=(0.3, 0.6, 0.9), uplift_pp=(0, 4, 8, 15),
                    life_model: str = "conservador") -> pd.DataFrame:
    """`repair_value` varrido por momento da vida, para toda a frota."""
    rows = []
    for pack in sorted(engine.d["pack"].unique()):
        n = engine.d.loc[engine.d["pack"] == pack, "cycle"].max()
        for f in fracs:
            c = engine.decision_points(pack, (f,))[0]
            df = repair_value(engine, pack, c, uplift_pp, life_model, n_samples=1200)
            df["fracao"] = c / n
            df["momento"] = f"{int(f * 100)}% da vida"
            rows.append(df)
    return pd.concat(rows, ignore_index=True)
