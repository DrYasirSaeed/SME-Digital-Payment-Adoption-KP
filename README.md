# Digital Payment Mandates and SME Financial Inclusion in KP

**Dr. Yasir Saeed** — KUST ORIC Research Grant 2025-26  
Kohat University of Science and Technology (KUST)

---

## Overview

This repository contains the full analytical pipeline for the study:

> *"Digital Payment Mandates and SME Financial Inclusion in Khyber Pakhtunkhwa, Pakistan"*

The study examines how awareness of regulatory **safe harbours** influences informal SMEs' adoption of formal QR-based digital payment systems, with a focus on whether **tax anxiety** (audit fear, tax burden belief) mediates this relationship and how effects vary by geographic zone and vendor type.

---

## Research Design

| Element | Detail |
|---|---|
| Setting | Khyber Pakhtunkhwa (KP), Pakistan |
| Unit of analysis | Informal SME / trader |
| Key exposure | C3 — Safe Harbour Awareness |
| Primary outcome | B1b — Formal QR Registration |
| Mediators | D7 (Audit Fear), C4 (Tax Burden Belief) |
| Heterogeneity | Zone (urban/peri-urban/rural), Vendor type |
| Data | Hypothetical survey (generated; see `data/`) |

---

## Repository Structure

```
SME-Digital-Payment-Adoption-KP/
├── README.md
├── LICENSE
├── requirements.txt
├── run_all.py                        # Master pipeline runner
│
├── data/
│   └── generate_hypothetical_data.py # Generates synthetic survey data
│
├── scripts/
│   ├── 01_descriptives.py            # Step 1-2:  Descriptives & cross-tabs
│   ├── 02_probit_primary.py          # Step 3:    Primary probit model
│   ├── 03_factor_analysis.py         # Step 4-5:  Factor analysis
│   ├── 04_ordered_logit.py           # Step 6:    Ordered logit (willingness)
│   ├── 05_heterogeneity.py           # Step 7:    Heterogeneity by zone/vendor
│   ├── 06_mediation.py               # Step 8:    Mediation analysis
│   ├── 07_coordination_wedge.py      # Step 9:    Coordination wedge
│   ├── 08_policy_simulation.py       # Step 10:   Policy scenario projections
│   └── 09_qualitative_coding.py      # Step 11:   AI-assisted qualitative coding
│
└── outputs/                          # Created at runtime (gitignored)
    ├── tables/                       # CSV result tables
    └── figures/                      # PNG charts
```

---

## Pipeline Steps

| Step | Script | Description |
|------|--------|-------------|
| 1–2 | `01_descriptives.py` | Summary statistics; QR registration and safe harbour awareness by zone |
| 3 | `02_probit_primary.py` | Primary probit: C3 → B1b with full controls; average marginal effects (AME) and VIF |
| 4–5 | `03_factor_analysis.py` | PCA/EFA on barrier items; factor loadings heatmap and scree plot |
| 6 | `04_ordered_logit.py` | Ordered logit on willingness-to-adopt (B2); coefficients and AMEs |
| 7 | `05_heterogeneity.py` | Sub-group probit by zone and vendor type; forest plot of AMEs |
| 8 | `06_mediation.py` | Baron–Kenny mediation + bootstrapped indirect effects for D7 and C4 |
| 9 | `07_coordination_wedge.py` | Coordination wedge analysis; probit and ordered logit on peer-norm measures |
| 10 | `08_policy_simulation.py` | Policy simulation: project registration rates under awareness intervention scenarios |
| 11 | `09_qualitative_coding.py` | AI-assisted thematic coding of open-ended responses (F1) using Claude API |

---

## Qualitative Coding Themes

The qualitative step uses a pre-specified 11-category taxonomy:

`TAX_ANXIETY` · `PROCESS_COMPLEXITY` · `AWARENESS_GAP` · `CUSTOMER_DEMAND`  
`INFRASTRUCTURE` · `TRUST_DEFICIT` · `COST_BARRIER` · `PEER_NORMS`  
`POSITIVE_NUDGE` · `SMALL_SCALE` · `OTHER`

When an `ANTHROPIC_API_KEY` environment variable is present, responses are coded via the Claude API. Otherwise the script falls back to rule-based keyword matching.

---

## Installation

```bash
git clone https://github.com/DrYasirSaeed/SME-Digital-Payment-Adoption-KP.git
cd SME-Digital-Payment-Adoption-KP
pip install -r requirements.txt
```

To enable AI-assisted qualitative coding, set your Anthropic API key:

```bash
# Linux / macOS
export ANTHROPIC_API_KEY=sk-...

# Windows (PowerShell)
$env:ANTHROPIC_API_KEY = "sk-..."
```

---

## Running the Pipeline

**Full pipeline (recommended):**

```bash
python run_all.py
```

**Single step:**

```bash
python data/generate_hypothetical_data.py   # generate data first
python scripts/01_descriptives.py
python scripts/06_mediation.py              # any individual step
```

All outputs are written to `outputs/tables/` (CSV) and `outputs/figures/` (PNG).

---

## Key Outputs

| File | Content |
|------|---------|
| `outputs/tables/03a_probit_primary_ame.csv` | Main probit AMEs — effect of safe harbour awareness |
| `outputs/tables/08d_mediation_summary.csv` | Full mediation summary (paths a, b, c, c′, indirect) |
| `outputs/tables/07a_heterogeneity_by_zone.csv` | AMEs by zone |
| `outputs/tables/10a_policy_simulation_overall.csv` | Registration rate projections by scenario |
| `outputs/figures/08_mediation_diagram.png` | Mediation pathway diagram |
| `outputs/figures/10_policy_simulation_scenarios.png` | Policy projection chart |

---

## Dependencies

Core packages (see `requirements.txt` for pinned versions):

- `numpy`, `pandas`, `scipy` — data handling and statistics
- `statsmodels` — probit, ordered logit, OLS
- `scikit-learn` — factor analysis / PCA
- `matplotlib`, `seaborn` — visualisation
- `anthropic` — Claude API for qualitative coding (optional)

---

## Citation

If you use this code or pipeline in your work, please cite:

```
Saeed, Y. (2026). Digital Payment Mandates and SME Financial Inclusion
in Khyber Pakhtunkhwa, Pakistan. KUST ORIC Research Grant 2025-26.
https://github.com/DrYasirSaeed/SME-Digital-Payment-Adoption-KP
```

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgements

Funded by the **KUST Office of Research, Innovation and Commercialisation (ORIC)**, Research Grant 2025-26.
