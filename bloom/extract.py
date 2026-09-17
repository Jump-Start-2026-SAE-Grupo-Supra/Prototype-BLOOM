"""Extração de features por ciclo a partir do dataset NASA/UCF
"Randomized & Recommissioned" (Fricke, Nascimento & Viana, 2023).

Cada pack (2 células 18650 em série) é ciclado até falhar. Um ciclo é:
missão de descarga (mode=-1) -> repouso (0) -> carga CC-CV (1) -> repouso (0).
Algumas missões são descargas de referência a 2,5 A (mission_type=0): elas dão
a capacidade "verdadeira" e viram o rótulo de SOH. As demais features imitam o
que um BMS/carregador ou uma bancada rápida conseguem medir sem ciclagem completa.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

NOMINAL_AH = 2.5  # capacidade nominal de cada célula 18650 do pack

# Metadados dos packs, transcritos dos README do dataset (corrente média aproximada em A).
PACKS = {
    # regular, corrente constante
    "00": ("regular_cc", [16.0]), "10": ("regular_cc", [16.0]),
    "01": ("regular_cc", [9.3]), "11": ("regular_cc", [9.3]),
    "22": ("regular_cc", [12.9]), "31": ("regular_cc", [12.9]),
    "23": ("regular_cc", [14.3]), "52": ("regular_cc", [14.3]),
    "20": ("regular_cc", [19.0]), "30": ("regular_cc", [19.0]), "21": ("regular_cc", [19.0]),
    # regular, carga aleatória
    "41": ("regular_random", [14.3]), "51": ("regular_random", [14.3]),
    "40": ("regular_random", [17.0]), "50": ("regular_random", [17.0]),
    # recomissionados: nível de carga muda ao longo da vida
    "32": ("recommissioned", [16.0, 14.3]), "53": ("recommissioned", [14.3, 16.0]),
    "02": ("recommissioned", [16.0, 12.9]), "33": ("recommissioned", [16.0, 12.9]),
    "12": ("recommissioned", [16.0, 9.3]), "24": ("recommissioned", [16.0, 9.3]),
    "03": ("recommissioned", [16.0, 14.3, 7.5]), "25": ("recommissioned", [16.0, 7.5, 14.3]),
    # segunda vida: células sobreviventes de packs de primeira vida
    "13": ("second_life", [9.3]), "36": ("second_life", [7.5]), "54": ("second_life", [5.0]),
}
SECOND_LIFE_ORIGIN = {"13": ["00", "11"], "36": ["23", "52"], "54": ["22", "31"]}

FOLDER = {
    "regular_cc": "regular_alt_batteries", "regular_random": "regular_alt_batteries",
    "recommissioned": "recommissioned_batteries", "second_life": "second_life_batteries",
}

COLS = ["time", "mode", "voltage_charger", "temperature_battery",
        "voltage_load", "current_load", "mission_type"]


@dataclass
class Segment:
    mode: int
    i0: int
    i1: int  # inclusivo


def _segments(mode: np.ndarray) -> list[Segment]:
    change = np.flatnonzero(np.diff(mode) != 0) + 1
    starts = np.r_[0, change]
    ends = np.r_[change - 1, len(mode) - 1]
    return [Segment(int(mode[s]), int(s), int(e)) for s, e in zip(starts, ends)]


def _value_at(t: np.ndarray, y: np.ndarray, i0: int, i1: int, dt: float) -> float:
    """Valor de y no primeiro instante >= t[i0] + dt dentro do segmento."""
    j = i0 + int(np.searchsorted(t[i0:i1 + 1], t[i0] + dt))
    return float(y[j]) if j <= i1 else np.nan


def _nan_stats(x: np.ndarray) -> tuple[float, float]:
    x = x[np.isfinite(x)]
    return (float(x.mean()), float(x.std())) if len(x) else (np.nan, np.nan)


def load_pack(root: Path, pack: str) -> pd.DataFrame:
    group = PACKS[pack][0]
    path = root / FOLDER[group] / f"battery{pack}.csv"
    try:
        return pd.read_csv(path, usecols=COLS, dtype={c: "float32" for c in COLS if c != "time"} | {"time": "float64"})
    except ValueError:
        # alguns arquivos têm tokens não numéricos esporádicos: converte-os em NaN
        df = pd.read_csv(path, usecols=COLS, dtype=str)
        out = df.apply(pd.to_numeric, errors="coerce").astype("float32")
        out["time"] = pd.to_numeric(df["time"], errors="coerce").astype("float64")
        return out.dropna(subset=["time"]).reset_index(drop=True)


def discharge_features(df: pd.DataFrame) -> pd.DataFrame:
    t = df["time"].to_numpy()
    mode = np.nan_to_num(df["mode"].to_numpy(), nan=0).astype(int)
    v = df["voltage_charger"].to_numpy()
    temp = df["temperature_battery"].to_numpy()
    vl = df["voltage_load"].to_numpy()
    il = df["current_load"].to_numpy()
    mt = df["mission_type"].to_numpy()

    segs = _segments(mode)
    rows = []
    for k, s in enumerate(segs):
        if s.mode != -1:
            continue
        i0, i1 = s.i0, s.i1
        load = np.isfinite(il[i0:i1 + 1])
        tt, ii, vv = t[i0:i1 + 1][load], il[i0:i1 + 1][load], vl[i0:i1 + 1][load]
        dur = float(t[i1] - t[i0])
        ah = float(np.trapezoid(ii, tt) / 3600) if len(tt) > 1 else 0.0
        if dur < 60 or ah < 0.2:
            continue  # missão abortada / falha de aquisição
        row = {
            "t_start_h": t[i0] / 3600,
            "mission_type": int(np.nanmax(mt[i0:i1 + 1])) if np.isfinite(mt[i0:i1 + 1]).any() else -1,
            "dis_duration_s": dur,
            "dis_ah": ah,
            "dis_wh": float(np.trapezoid(ii * vv, tt) / 3600),
            "i_mean": float(ii.mean()), "i_std": float(ii.std()),
            "v_end": float(v[i1]),
        }
        # Repouso imediatamente anterior: tensão de circuito aberto antes do pulso.
        prev = segs[k - 1] if k > 0 else None
        v_ocv = float(v[i0 - 1]) if prev is not None and prev.mode == 0 else np.nan
        row["v_ocv_pre"] = v_ocv
        row["rest_pre_s"] = float(t[prev.i1] - t[prev.i0]) if prev is not None and prev.mode == 0 else np.nan

        # Pulso inicial (≈ HPPC de 10 s / 60 s): resistência aparente e polarização.
        for dt in (2, 10, 60, 120):
            vd = _value_at(t, v, i0, i1, dt)
            idt = _value_at(t, np.nan_to_num(il, nan=np.nan), i0, i1, dt)
            row[f"v_load_{dt}s"] = vd
            row[f"r_app_{dt}s"] = (v_ocv - vd) / idt if np.isfinite(idt) and idt > 0.5 else np.nan

        # Temperatura: aquecimento no início (bancada) e na missão inteira (telemetria).
        T0 = float(temp[i0])
        row["temp_start"] = T0
        row["temp_max_dis"] = float(np.nanmax(temp[i0:i1 + 1]))
        row["dT_dis"] = row["temp_max_dis"] - T0
        row["dT_120s"] = _value_at(t, temp, i0, i1, 120) - T0
        i2 = float(np.mean(ii ** 2))
        row["heat_index"] = row["dT_dis"] / (i2 * dur / 3600) if i2 > 0 else np.nan  # K / (A²·h)
        row["heat_index_120s"] = row["dT_120s"] / (i2 * 120 / 3600) if i2 > 0 else np.nan

        # Relaxação após a descarga (Kumar et al., 2023: forma da relaxação ~ capacidade).
        nxt = segs[k + 1] if k + 1 < len(segs) else None
        if nxt is not None and nxt.mode == 0:
            j0, j1 = nxt.i0, nxt.i1
            for dt in (5, 30, 60, 180, 300, 540):
                row[f"v_relax_{dt}s"] = _value_at(t, v, j0, j1, dt)
            row["relax_amp"] = row["v_relax_540s"] - row["v_relax_5s"]
            row["temp_relax_end"] = _value_at(t, temp, j0, j1, 540)
        # Carga seguinte (pode vir fatiada por microrrepousos) e repouso pós-carga.
        m = k + 2
        charge_idx = []
        while m < len(segs) and segs[m].mode in (0, 1):
            if segs[m].mode == 1:
                charge_idx.append(m)
            elif charge_idx and (t[segs[m].i1] - t[segs[m].i0]) > 120:
                break
            m += 1
        if charge_idx:
            c0, c1 = segs[charge_idx[0]].i0, segs[charge_idx[-1]].i1
            vc, tc = v[c0:c1 + 1], t[c0:c1 + 1]
            row["chg_duration_s"] = float(tc[-1] - tc[0])
            row["chg_v_start"] = float(vc[0])
            # Janela parcial de tensão (Liu et al., 2024: features de trecho de carga).
            a, b = np.searchsorted(np.maximum.accumulate(vc), [7.6, 8.4])
            row["chg_t_7v6_8v4"] = float(tc[b] - tc[a]) if b < len(tc) and a < b else np.nan
            row["chg_dv_300s"] = _value_at(t, v, c0, c1, 300) - float(vc[0])
            row["dT_chg"] = float(np.nanmax(temp[c0:c1 + 1]) - temp[c0])
            if m < len(segs) and segs[m].mode == 0:
                r0, r1 = segs[m].i0, segs[m].i1
                row["rest_post_s"] = float(t[r1] - t[r0])
                v540 = _value_at(t, v, r0, r1, 540)
                row["v_post_chg_540s"] = v540
                row["post_chg_drop_540s"] = float(v[r0]) - v540
        rows.append(row)
    return pd.DataFrame(rows)


def build_pack_table(root: Path, pack: str) -> pd.DataFrame:
    df = load_pack(root, pack)
    feat = discharge_features(df)
    group, currents = PACKS[pack]
    feat.insert(0, "pack", pack)
    feat.insert(1, "group", group)
    feat["cycle"] = np.arange(len(feat))
    feat["cum_ah"] = feat["dis_ah"].cumsum() - feat["dis_ah"]
    feat["is_ref"] = (feat["mission_type"] == 0) & (feat["i_mean"] < 4)

    # Rótulo: capacidade das descargas de referência, interpolada em Ah acumulado.
    ref = feat[feat["is_ref"] & (feat["dis_ah"] > 1.0)]
    if len(ref):
        feat["ref_ah"] = np.interp(feat["cum_ah"], ref["cum_ah"], ref["dis_ah"])
        # fora do intervalo das referências, não extrapolar
        outside = (feat["cum_ah"] < ref["cum_ah"].min()) | (feat["cum_ah"] > ref["cum_ah"].max())
        feat["label_extrapolated"] = outside
    else:
        feat["ref_ah"] = np.nan
        feat["label_extrapolated"] = True
    feat["soh"] = feat["ref_ah"] / NOMINAL_AH

    # Vida restante até o fim do ensaio (o pack falhou ou saiu do teste).
    feat["cycles_to_eol"] = len(feat) - 1 - feat["cycle"]
    feat["ah_to_eol"] = feat["dis_ah"][::-1].cumsum()[::-1] - feat["dis_ah"]
    feat["load_level_nominal"] = currents[0] if len(currents) == 1 else np.nan
    return feat
