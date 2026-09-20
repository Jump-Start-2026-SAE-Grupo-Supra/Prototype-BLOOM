# BLOOM — protótipo F0

**Equipe Supra · Jump Start 2026 · Missão 2 — Economia Circular Inteligente**

Protótipo do motor proposto no dossiê do projeto: **laudo de saúde sem ciclagem completa → gates de
segurança → destino de maior valor por bateria**, com incerteza declarada e a conta de cada decisão à
vista. Roda sobre dados públicos de baterias de íon-lítio (26 packs, 9.613 ciclos) e corresponde à fase
**F0 — gêmeo digital offline** do dossiê: nenhum hardware novo.

- 📓 **Notebook:** [`notebooks/BLOOM_prototipo.ipynb`](notebooks/BLOOM_prototipo.ipynb) — a narrativa completa, executada, com gráficos.
- 🌐 **Site:** [`docs/index.html`](docs/index.html) — laudo interativo por pack, com glossário, o dataset explicado, o método de desenvolvimento, as equações com fonte e a seção do pack do Corolla híbrido. Abra o arquivo no navegador.

## O que o protótipo responde

Os dois primeiros testes de falsificação do dossiê (§5), com dados:

1. **Dá para estimar o estado sem ciclar o pack, com erro útil para decisão?**
   Sim, para packs de primeira vida: o *laudo rápido* (~20 min de bancada, sem histórico) erra
   1,7 p.p. de SOH fora da amostra, com intervalo P10–P90 que cobre 78% dos casos —
   contra 5,3 p.p. só pela idade e 0,6 p.p. com a descarga completa.
   **Não, para packs remontados** sem um ensaio de recomissão: o modelo superestima o SOH
   (erro 5,1 p.p. → 3,0 p.p. com um ensaio de âncora). O laudo passa a dizer "inconclusivo".
2. **O roteamento recomendaria segunda vida em quantidade suficiente, ou a reciclagem domina?**
   Depende do momento e do cenário de vida. Na aposentadoria (~90% da vida) **a reciclagem domina no cenário
   conservador** (5,3% dos packs vão para reuso, na média da grade de sensibilidade) — em linha com Guan et al.
   (2025) e com o caso Relectrify —, mas não no só fade, onde a média é 31,3% e chega a 69% na célula mais
   favorável. Aos 30% da vida, 42–88% dos packs vão para reuso nos dois cenários: o que pesa é o **momento** da
   decisão e o preço do produto novo que o reuso substitui (o frete pesa menos). O produto é o destino de maior
   valor **a cada momento** da vida e o ponto em que ele deixa de valer — quando e para onde são uma decisão só —,
   não a triagem do que já sobrou.

3. **Existe uma janela de realocação mensurável, e onde ela fecha?** (`pipeline.decision_windows`)
   Sim, mas **onde** ela fecha é a maior incerteza do protótipo. Varrendo 14 pontos da vida de cada pack,
   a janela em que algum destino de reuso ainda vence a reciclagem fecha na **mediana em 28% da vida**
   (quartis 22%–33%) no cenário conservador, e em **90%** (70%–96%) no cenário só fade. O processo atual
   decide perto de 100% da vida, então em ambos os cenários há espaço — mas o intervalo defensável é
   **28%–90%**, não um número. Os três packs remontados de 2ª vida nunca abrem janela (os gates G5/G6 os
   bloqueiam por incerteza) e, no conservador, o pack 20 (29 ciclos) também não.
   **O que move o resultado é o modelo de vida, não o preço** ([`PREMISSAS.md`](PREMISSAS.md)): corrigir o
   preço do produto novo em 18/09/2026 não mudou o cenário conservador em um ponto sequer e levou o só fade
   de 49% a 90%. Estreitar esse intervalo exige degradação medida em 2ª vida real — dado que este dataset
   não tem.
   **Cuidado com o que os 28% medem:** no conservador, 92 dos 96 pontos em que o reuso vence são *continuar
   na tração*; só 2 packs recebem recomendação de 2ª vida, e essa janela fecha em **15%** da vida. No só
   fade, quase todo o reuso é 2ª vida (257 de 259 pontos, quase toda em energia estacionária) e a janela de
   2ª vida fecha em **90%**.

4. **Vale pagar por reparo antes de decidir o destino?** (`pipeline.repair_value`)
   Depende do momento e do cenário. Para um pack de 10 kWh no cenário conservador, um ganho de 4 p.p. de SOH
   vale, na **mediana, R$ 656** aos 30% da vida (máx. R$ 1.514; muda o destino em 9 de 26 packs) e **R$ 0**
   aos 60% e aos 90%. No só fade o valor não cai: R$ 1.759 aos 30%, R$ 1.777 aos 60% e ainda R$ 1.090 aos
   90% — e com um ganho grande (15 p.p.) o reparo vale R$ 5.796 aos 90%. Não é uma confirmação independente
   do item 3: usa o mesmo roteador e o mesmo modelo de vida, e quando a reciclagem já é o destino base um
   ganho moderado não muda a decisão. O *uplift* não é calibrado: é varrido, e a saída é o **break-even**,
   não um preço.

Os itens 3 e 4 estão no notebook (seções 10 e 11) e no site. Todos os números acima são do perfil de
premissas `bev_revisado` — ver [Perfis de premissas](#perfis-de-premissas).

## Arquitetura

```
battery_alt_dataset/ (3,2 GB, não versionado)
        │  scripts/build_dataset.py      bloom/extract.py
        ▼
data/processed/cycles.parquet  ── 9.613 ciclos × 51 colunas
        │
        ├─ bloom/health.py    SOH P10/P50/P90 (quantis + calibração conformal), validação por pack
        ├─ bloom/safety.py    gates G1–G6: vetos e restrições com evidência numérica
        ├─ bloom/router.py    VPL por destino sob incerteza, garantia, valor da informação, laudo
        ├─ bloom/config.py    destinos e perfis de premissas econômicas (ver PREMISSAS.md)
        └─ bloom/pipeline.py  orquestração, laço fechado, sensibilidade,
                              janela de decisão e break-even do reparo
                │
                ├─ notebooks/BLOOM_prototipo.ipynb
                ├─ scripts/export_site.py --perfil … → docs/index.html
                └─ scripts/rerun_premissas.py → docs/data/revisao_premissas.json
```

| Módulo do dossiê | Implementação | Base |
|---|---|---|
| Health Intelligence | `health.py` | Wei et al. 2022; Kumar et al. 2023; Liu et al. 2024 |
| Gates de segurança | `safety.py` | Lai et al. 2021 §3.2.3; Li et al. 2024; Geng et al. 2026 §6 |
| Lifecycle Router | `router.py` | Guan et al. 2025 §4.2.2; Kumar et al. 2023 |
| Garantia como produto | `router.py` (prêmio) | Dossiê §3.4 |
| Battery Passport | gate de procedência G6, âncora de recomissão | Guan et al. 2025 §3 |
| Janela de realocação | `pipeline.decision_windows` | achado próprio (F0) |
| Reparo/remanufatura como **ação** | `pipeline.repair_value` (break-even) | Wang et al. 2023 (efeito barril) |

## Como rodar

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows (Linux/macOS: .venv/bin/pip)

# opcional: regerar a tabela de ciclos a partir dos CSVs brutos (~8 min)
.venv/Scripts/python scripts/build_dataset.py --jobs 6

# notebook
.venv/Scripts/python scripts/make_notebook.py
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/BLOOM_prototipo.ipynb

# site (perfil bev_revisado por padrão; ~2 min)
.venv/Scripts/python scripts/export_site.py --perfil bev_revisado
.venv/Scripts/python scripts/export_site.py --html-only      # só reaplica o template ao JSON existente

# reexecuta as análises de decisão sob os três perfis (grava docs/data/revisao_premissas.json)
.venv/Scripts/python scripts/rerun_premissas.py
```

O dataset bruto é o **Randomized & Recommissioned Battery Dataset** (Fricke, Nascimento & Viana, 2023;
NASA PCoE / UCF). Descompacte em `battery_alt_dataset/battery_alt_dataset/`. A tabela processada
`data/processed/cycles.parquet` já está no repositório, então notebook e site rodam sem o bruto.

**Ver o site localmente** (o repositório é privado, então não há GitHub Pages):

```bash
# opção 1: abrir o arquivo direto
start docs/index.html            # Windows

# opção 2: servidor local (evita restrições de file:// em alguns navegadores)
.venv/Scripts/python -m http.server 8000 --directory docs
# depois abra http://localhost:8000
```

A página é um único HTML com os resultados embutidos — não precisa de internet (só as fontes, que caem
para a fonte do sistema se estiver offline).

## Perfis de premissas

A camada econômica não tem um conjunto único de preços: tem **perfis de produto**
(`bloom/config.py::profile_settings`), porque o que o reuso substitui muda de produto para produto. O que
cada um assume, e de onde vem, está em [`PREMISSAS.md`](PREMISSAS.md).

| perfil | pack | produto novo que o reuso substitui | onde aparece |
|---|---|---|---|
| `bev_revisado` | 10 kWh, íon-lítio | R$ 1.250/kWh (reposição de elétrico) · R$ 3.100/kWh (sistema estacionário) | **padrão** do notebook e do site |
| `v0_ilustrativo` | 10 kWh, íon-lítio | R$ 900/kWh · R$ 1.800/kWh | premissas originais; `export_site.py --perfil v0_ilustrativo` |
| `hev_nimh_corolla` | 1,3 kWh, Ni-MH | R$ 13.077/kWh (reposição de híbrido), remanufaturada a 53% do novo | seção do Corolla no site |

Só a camada econômica muda entre perfis: saúde, gates e modelo de estresse são os mesmos, e
`v0_ilustrativo` reproduz exatamente os resultados publicados antes da revisão.

No perfil do Corolla a pergunta deixa de ser "2ª vida ou reciclagem" — não existe sistema estacionário de
1,3 kWh — e passa a ser **remanufaturar para a mesma aplicação, ou reciclar**. Como o custo de remanufatura
não tem fonte pública, o motor não o assume: resolve o **custo de equilíbrio**, que no cenário conservador é
~R$ 920 por pack aos 30% da vida e negativo dos 60% em diante. Está no site, em "O pack do Corolla".

## Limitações

- **Íon-lítio 18650, não Ni-MH** — o insumo imediato da Toyota no Brasil é Ni-MH.
- **Pack de 2 células**, não pack automotivo com BMS.
- **Ensaio acelerado (4–8 C, missões perto de 100 °C)** — vida em destinos reais é extrapolação;
  por isso há dois cenários (*conservador* e *só fade*), e o laço fechado mostra que em 4 dos 6 pares origem → pack remontado a vida real fica entre eles.
- **Falha abrupta sem precursor elétrico** — vida residual sai com intervalo largo.
- **Só três packs remontados** — a calibração da segunda vida é frágil.
- **Sem EIS nem sensor de gás** — gates térmico e de autodescarga são proxies.
- **Economia com premissas ilustrativas** (`bloom/config.py`) — preços, frete e regras brasileiras
  precisam de fonte. Com as premissas atuais o VPL da reciclagem é **negativo** (−R$ 300 por pack de
  10 kWh): o motor está escolhendo o menor prejuízo, não o maior lucro. O valor de material é o
  número que decide o sinal, e é o primeiro a cotar.
- **A posição da janela não está resolvida** — 28% a 90% da vida conforme o modelo de vida, e nenhum preço
  desfaz isso. É a maior incerteza do protótipo, e só cede com degradação medida em 2ª vida real. Ver
  [`PREMISSAS.md`](PREMISSAS.md), que documenta a correção do preço do produto novo e os perfis.
- **Reparo não é calibrado** — o *uplift* de SOH é varrido, não medido. Não há packs reparados neste
  dataset; a saída é break-even, e calibrá-la é trabalho de F2.

As referências e a análise que orientaram o desenho estão em `referencias/` (não versionada).
