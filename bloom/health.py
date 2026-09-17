"""Health Intelligence: SOH e vida residual com incerteza declarada.

Desenho, a partir das referências:
- Wei, Han & Li (2022): o erro do modelo é tratado como distribuição (lá, cadeia de Markov
  sobre faixas de erro). Aqui usamos regressão quantílica + calibração conformal por pack,
  que entrega um intervalo P10–P90 com cobertura verificável fora da amostra.
- Kumar et al. (2023): a relaxação de tensão prediz capacidade em nível de pack.
- Liu et al. (2024): features de trecho parcial de carga evitam ciclo completo.
- Guan et al. (2025): poucos métodos operam em nível de pack e o histórico costuma faltar —
  por isso comparamos um "laudo rápido" (sem histórico) com "telemetria" (com histórico BMS).

Validação sempre deixando packs inteiros de fora (GroupKFold): um pack nunca está no treino
e no teste ao mesmo tempo.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import GroupKFold

QUANTILES = (0.1, 0.5, 0.9)

# Bancada de ~15–25 min sem histórico: pulso de descarga de 2 min, janela parcial de carga
# (7,6 → 8,4 V) e repouso curto. Nada exige descarregar o pack até o corte.
QUICK_TEST = [
    "i_mean", "v_ocv_pre", "r_app_2s", "r_app_10s", "r_app_60s", "r_app_120s",
    "pol_10_120", "dT_120s", "heat_index_120s", "temp_start",
    "chg_t_7v6_8v4", "chg_dv_300s", "post_chg_drop_540s",
]
# Telemetria que o BMS/carregador já registra no uso: acrescenta aquecimento da missão,
# relaxação pós-uso e o histórico acumulado — ainda sem medir a capacidade diretamente.
TELEMETRY = QUICK_TEST + [
    "dT_dis", "heat_index", "temp_max_dis",
    "v_relax_5s", "v_relax_60s", "v_relax_300s", "relax_amp", "cum_efc",
]
# Limite superior: usa a descarga completa até o corte e a carga a partir do vazio,
# ou seja, justamente a ciclagem completa que o BLOOM quer evitar.
FULL_CYCLE = TELEMETRY + ["dis_ah", "dis_wh", "dis_duration_s", "chg_duration_s"]
AGE_ONLY = ["cum_efc", "i_mean"]  # linha de base: só "idade" (throughput) e severidade de uso
FEATURE_SETS = {"idade (linha de base)": AGE_ONLY, "laudo rápido": QUICK_TEST,
                "telemetria BMS": TELEMETRY, "ciclo completo (limite superior)": FULL_CYCLE}


def add_derived(df: pd.DataFrame, nominal_ah: float = 2.5) -> pd.DataFrame:
    df = df.copy()
    # falhas esporádicas de aquisição geram quedas de tensão impossíveis (±8 V)
    bad = (df["post_chg_drop_540s"] < 0) | (df["post_chg_drop_540s"] > 1.5)
    df.loc[bad, "post_chg_drop_540s"] = np.nan
    df["pol_10_120"] = df["v_load_10s"] - df["v_load_120s"]
    df["cum_efc"] = df["cum_ah"] / nominal_ah               # ciclos equivalentes completos
    df["efc_to_eol"] = df["ah_to_eol"] / nominal_ah
    return df


def training_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Missões regulares com rótulo interpolado (sem extrapolar além das referências)."""
    m = (~df["is_ref"]) & (~df["label_extrapolated"]) & df["soh"].notna()
    return df[m].reset_index(drop=True)


def _hgb(q: float, seed: int = 0) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(loss="quantile", quantile=q, max_iter=150, learning_rate=0.08,
                                         max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=1.0,
                                         random_state=seed)


@dataclass
class QuantileModel:
    features: list[str]
    target: str = "soh"
    conformal_q: float = 0.0   # alargamento (CQR) para atingir 80% de cobertura
    models: dict | None = None

    def fit(self, df: pd.DataFrame) -> "QuantileModel":
        X, y = df[self.features], df[self.target]
        self.models = {q: _hgb(q).fit(X, y) for q in QUANTILES}
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = df[self.features]
        p = {q: self.models[q].predict(X) for q in QUANTILES}
        lo = np.minimum(p[0.1], p[0.5]) - self.conformal_q
        hi = np.maximum(p[0.9], p[0.5]) + self.conformal_q
        return pd.DataFrame({"p10": lo, "p50": p[0.5], "p90": hi}, index=df.index)


def conformity_scores(y: np.ndarray, pred: pd.DataFrame) -> np.ndarray:
    """Escore CQR (Romano et al., 2019): quanto o intervalo precisa crescer para conter y."""
    return np.maximum(pred["p10"].to_numpy() - y, y - pred["p90"].to_numpy())


def cross_validate(df: pd.DataFrame, features: list[str], target: str = "soh", n_splits: int = 13,
                   conformal: bool = True) -> pd.DataFrame:
    """Previsões fora da amostra, deixando packs inteiros de fora.

    A calibração conformal usa, para cada fold de teste, apenas os escores dos outros folds
    (nunca do próprio pack), preservando a honestidade da cobertura.
    """
    groups = df["pack"].to_numpy()
    oof = pd.DataFrame(index=df.index, columns=["p10", "p50", "p90", "fold"], dtype=float)
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(df, groups=groups)):
        m = QuantileModel(features, target).fit(df.iloc[tr])
        oof.iloc[te, :3] = m.predict(df.iloc[te]).to_numpy()
        oof.iloc[te, 3] = k
    if conformal:
        y = df[target].to_numpy()
        s = conformity_scores(y, oof)
        widen = np.zeros(len(df))
        for k in np.unique(oof["fold"]):
            other = oof["fold"].to_numpy() != k
            widen[~other] = _pack_balanced_quantile(s[other], groups[other], 0.8)
        oof["p10"] -= widen
        oof["p90"] += widen
        oof["conformal_widen"] = widen
    return oof


def _pack_balanced_quantile(scores: np.ndarray, groups: np.ndarray, level: float) -> float:
    """Quantil ponderado para que cada pack pese igual (packs longos não dominam)."""
    w = pd.Series(1.0, index=range(len(groups))).groupby(groups).transform(lambda s: 1 / len(s)).to_numpy()
    order = np.argsort(scores)
    cw = np.cumsum(w[order]) / w.sum()
    return float(scores[order][np.searchsorted(cw, level)])


def fit_final(df: pd.DataFrame, features: list[str], target: str = "soh") -> QuantileModel:
    oof = cross_validate(df, features, target)
    s = conformity_scores(df[target].to_numpy(), oof.assign(p10=oof["p10"] + oof["conformal_widen"],
                                                            p90=oof["p90"] - oof["conformal_widen"]))
    model = QuantileModel(features, target).fit(df)
    model.conformal_q = _pack_balanced_quantile(s, df["pack"].to_numpy(), 0.8)
    return model


def metrics(y: np.ndarray, pred: pd.DataFrame, scale: float = 100.0) -> dict:
    e = pred["p50"].to_numpy() - y
    inside = (y >= pred["p10"].to_numpy()) & (y <= pred["p90"].to_numpy())
    return {
        "MAE": float(np.mean(np.abs(e)) * scale),
        "RMSE": float(np.sqrt(np.mean(e ** 2)) * scale),
        "P90 |erro|": float(np.quantile(np.abs(e), 0.9) * scale),
        "cobertura P10–P90": float(inside.mean()),
        "largura média": float(np.mean(pred["p90"] - pred["p10"]) * scale),
    }


def age_baseline_cv(df: pd.DataFrame, target: str = "soh", n_splits: int = 13) -> pd.DataFrame:
    """Linha de base ingênua: regressão linear em throughput e corrente, com intervalo empírico."""
    groups = df["pack"].to_numpy()
    out = pd.DataFrame(index=df.index, columns=["p10", "p50", "p90"], dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(df, groups=groups):
        lr = LinearRegression().fit(df.iloc[tr][AGE_ONLY], df.iloc[tr][target])
        res = df.iloc[tr][target] - lr.predict(df.iloc[tr][AGE_ONLY])
        p = lr.predict(df.iloc[te][AGE_ONLY])
        out.iloc[te] = np.c_[p + res.quantile(0.1), p, p + res.quantile(0.9)]
    return out


def sample_from_quantiles(p10: float, p50: float, p90: float, n: int, rng: np.random.Generator) -> np.ndarray:
    """Amostra de uma distribuição normal-dividida que respeita P10, P50 e P90."""
    z = rng.standard_normal(n)
    s_lo = max(p50 - p10, 1e-6) / 1.2816
    s_hi = max(p90 - p50, 1e-6) / 1.2816
    return p50 + np.where(z < 0, z * s_lo, z * s_hi)
