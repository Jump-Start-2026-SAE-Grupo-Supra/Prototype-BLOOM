# BLOOM — protótipo F0

**Equipe Supra · Jump Start 2026 · Missão 2 — Economia Circular Inteligente**

Protótipo do motor proposto no dossiê do projeto: **laudo de saúde sem ciclagem completa → gates de
segurança → destino de maior valor por bateria**, com incerteza declarada e a conta de cada decisão à
vista. Roda sobre dados públicos de baterias de íon-lítio (26 packs, 9.613 ciclos) e corresponde à fase
**F0 — gêmeo digital offline** do dossiê: nenhum hardware novo.

- 📓 **Notebook:** [`notebooks/BLOOM_prototipo.ipynb`](notebooks/BLOOM_prototipo.ipynb) — a narrativa completa, executada, com gráficos.
- 🌐 **Site:** [`docs/index.html`](docs/index.html) — laudo interativo por pack, pronto para o GitHub Pages.

## O que o protótipo responde

Os dois primeiros testes de falsificação do dossiê (§5), com dados:

1. **Dá para estimar o estado sem ciclar o pack, com erro útil para decisão?**
   Sim, para packs de primeira vida: o *laudo rápido* (~20 min de bancada, sem histórico) erra
   1,7 p.p. de SOH fora da amostra, com intervalo P10–P90 que cobre 78% dos casos —
   contra 5,3 p.p. só pela idade e 0,6 p.p. com a descarga completa.
   **Não, para packs remontados** sem um ensaio de recomissão: o modelo superestima o SOH
   (erro 5,1 p.p. → 3,0 p.p. com um ensaio de âncora). O laudo passa a dizer "inconclusivo".
2. **O roteamento recomendaria segunda vida em quantidade suficiente, ou a reciclagem domina?**
   Depende do momento. Na aposentadoria (~90% da vida) **a reciclagem domina** em quase toda a grade de
   sensibilidade — em linha com Guan et al. (2025) e com o caso Relectrify. Aos 30% da vida, 31–88% dos packs
   vão para reuso: o que pesa é o **momento** da decisão e o preço do produto novo que o reuso substitui
   (o frete pesa menos). O produto é a decisão no momento certo, não a triagem do que já sobrou.

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
        ├─ bloom/config.py    destinos e premissas econômicas explícitas (ilustrativas)
        └─ bloom/pipeline.py  orquestração, laço fechado com a 2ª vida, sensibilidade
                │
                ├─ notebooks/BLOOM_prototipo.ipynb
                └─ scripts/export_site.py → docs/index.html
```

| Módulo do dossiê | Implementação | Base |
|---|---|---|
| Health Intelligence | `health.py` | Wei et al. 2022; Kumar et al. 2023; Liu et al. 2024 |
| Gates de segurança | `safety.py` | Lai et al. 2021 §3.2.3; Li et al. 2024; Geng et al. 2026 §6 |
| Lifecycle Router | `router.py` | Guan et al. 2025 §4.2.2; Kumar et al. 2023 |
| Garantia como produto | `router.py` (prêmio) | Dossiê §3.4 |
| Battery Passport | gate de procedência G6, âncora de recomissão | Guan et al. 2025 §3 |

## Como rodar

```bash
python -m venv .venv
.venv/Scripts/pip install -r requirements.txt        # Windows (Linux/macOS: .venv/bin/pip)

# opcional: regerar a tabela de ciclos a partir dos CSVs brutos (~8 min)
.venv/Scripts/python scripts/build_dataset.py --jobs 6

# notebook
.venv/Scripts/python scripts/make_notebook.py
.venv/Scripts/jupyter nbconvert --to notebook --execute --inplace notebooks/BLOOM_prototipo.ipynb

# site
.venv/Scripts/python scripts/export_site.py
```

O dataset bruto é o **Randomized & Recommissioned Battery Dataset** (Fricke, Nascimento & Viana, 2023;
NASA PCoE / UCF). Descompacte em `battery_alt_dataset/battery_alt_dataset/`. A tabela processada
`data/processed/cycles.parquet` já está no repositório, então notebook e site rodam sem o bruto.

**GitHub Pages:** em *Settings → Pages*, publique a pasta `/docs` do branch `main`.

## Limitações

- **Íon-lítio 18650, não Ni-MH** — o insumo imediato da Toyota no Brasil é Ni-MH.
- **Pack de 2 células**, não pack automotivo com BMS.
- **Ensaio acelerado (4–8 C, missões perto de 100 °C)** — vida em destinos reais é extrapolação;
  por isso há dois cenários (*conservador* e *só fade*), e o laço fechado mostra que em 4 dos 6 pares origem → pack remontado a vida real fica entre eles.
- **Falha abrupta sem precursor elétrico** — vida residual sai com intervalo largo.
- **Só três packs remontados** — a calibração da segunda vida é frágil.
- **Sem EIS nem sensor de gás** — gates térmico e de autodescarga são proxies.
- **Economia com premissas ilustrativas** (`bloom/config.py`) — preços, frete e regras brasileiras
  precisam de fonte.

As referências e a análise que orientaram o desenho estão em `referencias/` (não versionada).
