# Revisão de premissas econômicas — 18/09/2026

Registro da revisão do preço do "produto novo equivalente" (`new_product_brl_kwh`), a premissa que o
destino *continuar na origem* usa para valorar a bateria. O que mudou, por quê, com que fonte, e o que
os resultados fazem depois da correção.

> **Resumo.** A premissa era única e ilustrativa. A pesquisa de preços mostrou que ela depende do
> produto. Foram criados **perfis de produto** (`bloom/config.py::profile_settings`). Resultado
> principal: a conclusão de que **decidir cedo importa** sobrevive nos dois modelos de vida, mas a
> **posição da janela é muito sensível ao modelo de vida** — de 28% a 90% da vida, contra os 28–49%
> reportados antes. Isso não é um refinamento: é a maior incerteza do protótipo e só se resolve com
> dado de degradação real em segunda vida.

## 1. O que estava errado

O destino `original` compara o pack usado com o preço do que ele substitui no mesmo uso. O valor único
de **R$ 900/kWh** vinha de preço de célula/pack de elétrico e foi aplicado a qualquer produto.

| Produto | Preço de reposição encontrado | Em R$/kWh | Contra R$ 900/kWh | Fonte (confiança) |
|---|---|---:|---|---|
| Pack de elétrico (BYD Dolphin, 44,9 kWh) | R$ 60 mil | 1.336 | +48% | imprensa (média) |
| Pack de elétrico (BYD Dolphin Plus, 60 kWh) | R$ 70 mil | 1.167 | +30% | imprensa (média) |
| Pack de híbrido (Corolla Hybrid, NiMH, 1,3 kWh) | ~R$ 17 mil (remanufaturado: R$ 8–10 mil) | 13.077 | 14,5× | imprensa (média) |

**Nota de correção.** Numa análise anterior a premissa foi descrita como "errada por uma ordem de
grandeza". Isso só vale para o pack **híbrido**. Para o pack de **elétrico**, que é o que o protótipo
modela (10 kWh de referência, íon-lítio), a defasagem é de 30–48%. A comparação tinha misturado dois
produtos; este documento corrige.

Segunda premissa afetada — **sistema estacionário instalado** (destino `energia`): R$ 1.800/kWh contra
**R$ 2.700–3.500/kWh** encontrados para 2026 (imprensa setorial; confiança média). Mesma unidade, mesmo
produto, defasagem de 50–94%.

## 2. O que foi alterado

| Premissa | Antes | Depois | Base |
|---|---:|---:|---|
| `original.new_product_brl_kwh` (perfil `bev_revisado`) | 900 | **1.250** | ponto médio de R$ 1.167–1.336/kWh |
| `energia.new_product_brl_kwh` (perfil `bev_revisado`) | 1.800 | **3.100** | ponto médio de R$ 2.700–3.500/kWh |
| Perfil `hev_nimh_corolla` | não existia | pack 1,3 kWh; reposição R$ 13.077/kWh; remanufaturado a 53% do novo; destinos: remanufatura e reciclagem; metal contido ~R$ 650/kWh | Corolla Hybrid (NiMH, 1,3 kWh, 80 kg) |

**Deliberadamente não alterado**

- `repack_brl_kwh` (R$ 200–450/kWh): a calculadora B2U do NREL (US$ 22/kWh ≈ R$ 113/kWh) cobre
  repropósito e teste, **não** gabinete, BMS e integração do produto novo. O escopo difere, então ela não
  valida nem refuta a premissa. Uma versão anterior deste raciocínio a chamou de "conservadora"; isso
  não se sustenta. Além disso, o NREL assume 10 mil t/ano, escala que o Brasil não atinge.
- `material_value_brl_kwh` (R$ 110/kWh, íon-lítio): cai dentro da faixa derivada de R$ 108–179/kWh. A
  fração paga pelo reciclador (60%) é suposição do time, não dado.
- `potencia` (R$ 1.400/kWh) e `backup` (R$ 600/kWh): nenhum preço encontrado.

**Mudanças de código**

- `Settings.destinations`: a lista de destinos deixou de ser global. Cada perfil tem os seus.
- `router.route` passou a ler `settings.destinations`.
- `pipeline.decision_windows`, `repair_value`, `repair_frontier` e `reuse_share_grid` aceitam `base=`.
- `config.profile_settings(nome)`: `v0_ilustrativo` (premissas originais, mantido para
  reprodutibilidade), `bev_revisado`, `hev_nimh_corolla`.
- `scripts/rerun_premissas.py` reexecuta tudo e grava `docs/data/revisao_premissas.json`.

**Verificação de neutralidade.** O perfil `v0_ilustrativo` reproduz **exatamente** a janela, o
break-even do reparo, as grades de sensibilidade e as contagens da frota já publicadas em
`docs/data/bloom.json`. Ou seja, a refatoração não alterou o comportamento; só a premissa alterada
muda o resultado.

## 3. Resultados

### 3.1 Onde a janela de reuso fecha (mediana da vida; quartis; packs que abrem janela)

| Perfil | Cenário conservador | Cenário só fade |
|---|---|---|
| `v0_ilustrativo` (antes) | 28% (22–33) · 22/26 | 49% (35–73) · 23/26 |
| **`bev_revisado`** (correção) | **28% (22–33) · 22/26** | **90% (70–96) · 23/26** |
| `hev_nimh_corolla` | 29% (22–35) · 22/26 | 35% (28–62) · 23/26 |

### 3.2 Destino na aposentadoria (faixa 80–100% da vida, % das decisões)

| Perfil | Conservador | Só fade |
|---|---|---|
| `v0_ilustrativo` | 96% reciclagem | 90% reciclagem, 10% potência |
| `bev_revisado` | 96% reciclagem | **60% reciclagem, 40% energia estacionária** |
| `hev_nimh_corolla` | 96% reciclagem | 92% reciclagem |

Na grade de sensibilidade (frete × preço do novo, decisão na aposentadoria), a fração média que vai
para reuso é 5,3% no conservador e **31,3% no só fade** com `bev_revisado` (antes 14,7%).

### 3.3 Quanto vale pagar por 4 p.p. de SOH (break-even do reparo, R$ por pack de 10 kWh, mediana)

| Perfil e cenário | 30% da vida | 60% | 90% |
|---|---:|---:|---:|
| `v0` conservador (antes) | 466 | 0 | 0 |
| `bev_revisado` conservador | 656 | 0 | 0 |
| `v0` só fade (antes) | 1.228 | 272 | 0 |
| `bev_revisado` só fade | 1.759 | 1.777 | **1.090** |

### 3.4 Ganho médio de seguir a recomendação em vez de reciclar (R$/pack de 10 kWh, mesmo instante)

| Perfil e cenário | 30% | 60% | 90% |
|---|---:|---:|---:|
| `bev_revisado` conservador | 590 | 90 | 58 |
| `bev_revisado` só fade | 4.874 | 2.780 | 1.075 |

Mede duas decisões alternativas **no mesmo ponto da vida** (o pack ainda rodaria na 1ª vida entre um
ponto e outro, então comparar instantes diferentes seria enganoso).

### 3.5 Perfil híbrido: remanufatura contra reciclagem

O custo de remanufaturar um pack de 1,3 kWh **não tem fonte pública**, então não foi assumido: o script
calcula o **custo de equilíbrio** — quanto pode custar remanufaturar antes de a reciclagem voltar a
vencer.

| Cenário | Momento | Packs em que a remanufatura é elegível | Custo de equilíbrio (R$/pack, mediana) |
|---|---|---:|---:|
| conservador | 30% | 21/26 | ~920 |
| conservador | 60% | 10/26 | negativo (não compensa) |
| só fade | 30% | 21/26 | ~2.000 |
| só fade | 60% | 10/26 | ~900 |

O valor esperado da remanufatura (mediana, elegíveis, aos 30%) é R$ 1.374 (conservador) e R$ 2.451
(só fade), contra R$ 663 da reciclagem: cerca de **2× e 3,7×**. **Não** é a razão de 5–8× que sai de
comparar preço de venda do remanufaturado (R$ 8–10 mil) com valor de metal contido (R$ 1,2–1,7 mil):
essa razão é **bruta**, antes do custo de remanufatura, do valor do núcleo e da vida restante, e não
deve ser citada como ganho.

## 4. Interpretação

1. **No cenário conservador nada mudou** (28% nos três perfis). Isso indica que ali a janela é
   determinada pela elegibilidade (piso de SOH) e pela vida até a falha, não pelo preço. A conclusão
   "na aposentadoria a reciclagem domina" (96%) se mantém.
2. **No cenário só fade a correção muda muito**: com o sistema estacionário a R$ 3.100/kWh, a 2ª vida em
   energia fica lucrativa até tarde (janela mediana em 90%, 40% das decisões na aposentadoria vão para
   energia estacionária, reparo vale R$ 1.090 mesmo aos 90%).
3. **A previsão feita antes da rodada ("a janela fecha mais tarde") estava certa só em um dos dois
   cenários.** No conservador não se confirmou.
4. **O que decide é o modelo de vida, não o preço.** O laço fechado com packs remontados mostra que a
   vida real fica entre os dois cenários na maioria dos pares. Logo, a janela real está entre 28% e
   90% para o perfil `bev_revisado`, e o protótipo **não consegue estreitar esse intervalo**.
5. O achado qualitativo é robusto: em todos os perfis e cenários, decidir na aposentadoria é pior que
   decidir antes (ganho contra reciclar cai monotonicamente de 30% para 90% da vida).

## 5. Limitações desta revisão

- **O perfil `hev_nimh_corolla` usa dinâmica de envelhecimento de íon-lítio.** Só a economia é de
  NiMH. Serve como sensibilidade econômica, **não** como simulação do pack do Corolla. O piso de SOH de
  85% do destino de origem também vem de tração de íon-lítio.
- Frete e certificação escalam por kWh. Para um pack de 80 kg e 1,3 kWh isso subestima o frete (que
  escala com massa). Não altera a conclusão do perfil, pois o valor de reposição domina, mas o custo de
  equilíbrio da remanufatura fica otimista.
- Os preços de reposição e de sistema estacionário vêm de imprensa (confiança média), não de tabela
  oficial.
- O preço de venda de 2ª vida entra como proporção do novo (`price_factor`), sem fonte para os
  destinos de potência e backup.

## 6. Como reproduzir

```bash
python scripts/rerun_premissas.py     # ~3 min; grava docs/data/revisao_premissas.json
```

## 7. Pendências que dependem de fora

- Cotar com remanufaturadores o custo de remanufatura e o valor pago pelo núcleo.
- Obter dado de degradação de 2ª vida real para escolher entre os cenários de vida.
- Base de conhecimento (KB), Business Case publicado e laudo interativo ainda citam os números da
  versão anterior (janela em 28–49%); precisam ser atualizados para 28–90%.
