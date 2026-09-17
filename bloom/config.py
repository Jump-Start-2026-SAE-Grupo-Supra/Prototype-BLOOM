"""Premissas explícitas do protótipo BLOOM.

Nenhum dataset público traz custo logístico, preço de material ou regra local
(Análise de Referências, seção 8.3). Por isso a metade econômica do motor entra
como constantes configuráveis, *ilustrativas*, que aparecem no laudo. Trocar
estes números muda a recomendação — e o laudo mostra a conta para que isso seja
auditável. Valores em R$ de 2026, por kWh nominal, salvo indicação.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict


@dataclass(frozen=True)
class Destination:
    key: str
    label: str
    kind: str                 # "reuso" | "reciclagem"
    criterion: str            # parâmetro de ordenação dominante (Guan et al., 2025, seção 4.2.2)
    soh_min: float            # piso de SOH para entrar no destino
    soh_eol: float            # SOH em que a bateria sai deste destino
    r_growth_max: float       # crescimento máximo de resistência aceito (R / R_novo)
    c_rate: float             # taxa típica de descarga no destino (C, relativa à capacidade nominal)
    cycles_per_year: float    # ciclos equivalentes completos por ano no destino
    ref_life_years: float     # vida de um produto novo equivalente
    new_product_brl_kwh: float  # preço do produto NOVO equivalente no destino (sistema, não célula)
    price_factor: float       # preço por kWh útil do produto de 2ª vida relativo ao novo equivalente
    repack_brl_kwh: float     # reembalagem, BMS, gabinete, integração (o que o produto novo já traz)
    warranty_years: float     # garantia exigida pelo comprador
    needs_low_self_discharge: bool = False
    moves: bool = True        # exige frete, certificação e troca de dono


DESTINATIONS = [
    Destination("original", "Continuar na aplicação de origem (tração)", "reuso", "SOH + resistência",
                soh_min=0.85, soh_eol=0.80, r_growth_max=1.30, c_rate=1.0,
                cycles_per_year=250, ref_life_years=8, new_product_brl_kwh=900, price_factor=0.75, repack_brl_kwh=60,
                warranty_years=1, moves=False),
    Destination("potencia", "2ª vida · potência (empilhadeira, troca de bateria de moto)", "reuso",
                "resistência + vida residual", soh_min=0.78, soh_eol=0.70, r_growth_max=1.50,
                c_rate=0.8, cycles_per_year=300, ref_life_years=6, new_product_brl_kwh=1400, price_factor=0.60,
                repack_brl_kwh=250, warranty_years=2),
    Destination("energia", "2ª vida · energia estacionária (BESS comercial/residencial)", "reuso",
                "capacidade + vida residual", soh_min=0.65, soh_eol=0.55, r_growth_max=2.20,
                c_rate=0.3, cycles_per_year=330, ref_life_years=10, new_product_brl_kwh=1800, price_factor=0.55,
                repack_brl_kwh=450, warranty_years=4),
    Destination("backup", "2ª vida · backup / nobreak (baixa ciclagem)", "reuso",
                "capacidade + autodescarga", soh_min=0.60, soh_eol=0.50, r_growth_max=2.50,
                c_rate=0.2, cycles_per_year=24, ref_life_years=6, new_product_brl_kwh=600, price_factor=0.50,
                repack_brl_kwh=200, warranty_years=3, needs_low_self_discharge=True),
    Destination("reciclagem", "Reciclagem (recuperação de material)", "reciclagem", "valor de material",
                soh_min=0.0, soh_eol=0.0, r_growth_max=float("inf"), c_rate=0.0,
                cycles_per_year=0, ref_life_years=0, new_product_brl_kwh=0, price_factor=0.0, repack_brl_kwh=0,
                warranty_years=0),
]
DEST = {d.key: d for d in DESTINATIONS}


@dataclass
class Economics:
    pack_kwh: float = 10.0                 # pack de referência ao qual o laudo é escalado
    new_price_index: float = 1.0           # multiplica o preço dos produtos novos (1.0 = premissa; <1 = bateria nova mais barata)
    logistics_brl_kwh: float = 60.0        # frete de carga perigosa classe 9 (Relectrify: US$ 20–40/kWh internacional)
    certification_brl_kwh: float = 40.0    # avaliação individual tipo UL 1974 / IEC 62619
    bloom_test_brl_pack: float = 150.0     # laudo rápido BLOOM (pulso + janela de carga + térmico)
    reference_test_brl_pack: float = 800.0 # ensaio de referência completo (ciclo lento), alternativa ao laudo
    material_value_brl_kwh: float = 110.0  # valor de material recuperável (NMC)
    recycling_process_brl_kwh: float = 80.0
    discount_rate: float = 0.12            # a.a.
    warranty_loading: float = 1.30         # carregamento sobre o custo esperado de sinistro (repor o produto)

    def as_dict(self):
        return asdict(self)


@dataclass
class SafetyThresholds:
    """Gates de segurança (Lai et al., 2021, seção 3.2.3: segurança como filtro 1-D antes da ordenação).

    Limiares são percentis da frota saudável (primeiros ciclos de cada pack), calculados em
    `bloom.safety.fit_reference`, multiplicados pelas margens abaixo.
    """
    thermal_quantile: float = 0.99          # deriva térmica acima do P99 da frota saudável -> veto
    thermal_margin_c: float = 1.0           # margem em °C somada ao P99
    self_discharge_quantile: float = 0.99   # queda pós-carga acima do P99 -> restringe backup
    self_discharge_margin: float = 1.25
    knee_ratio: float = 3.0                 # fade recente > 3x o fade típico do pack -> veto de 2ª vida
    knee_window: int = 15                   # ciclos usados para a inclinação recente
    uncertainty_max_pp: float = 12.0        # intervalo P10–P90 de SOH acima disso -> laudo inconclusivo


@dataclass
class Settings:
    economics: Economics = field(default_factory=Economics)
    safety: SafetyThresholds = field(default_factory=SafetyThresholds)
    n_samples: int = 4000
    seed: int = 7
    # "fade": vida no destino limitada só pelo fade de capacidade extrapolado para a taxa C do destino.
    # "conservador": também limita pela vida até a falha abrupta observada no ensaio acelerado,
    #  sem creditar vida extra abaixo da menor corrente ensaiada.
    life_model: str = "conservador"
    calendar_fade_per_year: float = 0.02  # perda de SOH por ano parado (o dataset não mede; premissa)
