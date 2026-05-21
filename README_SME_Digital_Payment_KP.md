# Digital Payment Mandates and SME Financial Inclusion in KP

**Evidence from the Cashless Khyber Pakhtunkhwa Initiative**

---

## Overview

This repository contains the replication code, codebook, and data dictionary for the research project:

> **Digital Payment Mandates and SME Financial Inclusion: Evidence from the Cashless Khyber Pakhtunkhwa Initiative**
>
> Dr. Yasir Saeed, Department of Economics, Kohat University of Science and Technology (KUST), Kohat, KP, Pakistan.

The study examines the determinants of QR code adoption among small and informal traders in Kohat district, with a particular focus on measuring awareness of the two-year safe harbour provision in the Khyber Pakhtunkhwa Digital Payments Act 2026 (passed unanimously by the KP Provincial Assembly on April 6, 2026). The provision protects newly registered traders from new direct sales tax liability for two years after QR code registration — a policy instrument designed to neutralise tax anxiety as a barrier to formalisation.

---

## Research Context

| Item | Detail |
|---|---|
| Funding | KUST Office of Research, Innovation and Commercialization (ORIC), Research Grant Award 2025–26 |
| Grant Category | Lecturer |
| Principal Investigator | Dr. Yasir Saeed |
| Affiliation | Department of Economics, KUST, Kohat |
| Contact | yasirsaeed@kust.edu.pk |
| Study Area | Kohat District, Khyber Pakhtunkhwa, Pakistan |
| Sample Size | 300 traders |
| Sample Design | 3×2 purposive stratified (3 market types × 2 vendor categories) |
| Field Period | Project Months 3–5 (post-EIRB approval) |
| Policy Context | KP Digital Payments Act 2026; Cashless KP Programme (March 2025–) |

---

## Research Questions

1. What is the current level of QR code registration and safe harbour awareness among Kohat traders, disaggregated by market type and vendor category?
2. What demand-side and supply-side barriers prevent informal traders from adopting QR-based digital payments, and do these barriers differ by vendor type?
3. Does safe harbour awareness reduce tax anxiety perceptions, and does this reduction translate into higher current registration and stated willingness to adopt?
4. What are the policy implications for KPRA and KPITB on awareness campaign design, registration simplification, and enforcement sequencing?

---

## Theoretical Framework

The study integrates three theoretical strands:

- **Technology Acceptance Model** (Davis, 1989; Venkatesh and Davis, 2000) — perceived usefulness and ease of use as structural adoption predictors
- **Financial Inclusion Literature** (Demirgüç-Kunt et al., 2022) — banking access and prior digital experience as enablers
- **Behavioural Tax Compliance** (Allingham and Sandmo, 1972; Kleven et al., 2011; Slemrod, 2019) — tax anxiety as the specific channel through which the safe harbour provision operates
- **Coordination Failure** (Jack and Suri, 2011) — mutual misperception between merchants and customers as a second binding constraint independent of tax anxiety

---

## Survey Instrument

The structured questionnaire contains **44 items** across six sections:

| Section | Items | Content |
|---|---|---|
| A | A1–A8 | Trader Profile |
| B | B1a, B1b, B2–B6 | Current Digital Payment Status |
| C | C1–C9 | Policy Awareness and Perceptions (includes 2 corrected knowledge-test items) |
| D | D1–D12 | Adoption Barriers — Likert scale 1–5 (includes D11–D12 coordination wedge items) |
| E | E1–E6 | Willingness to Adopt and Policy Preferences |
| F | F1 | Open-Ended Item (AI-assisted qualitative coding) |

**Key design features:**
- **B1b** (formal QR registration) is the primary binary dependent variable, separated from B1a (display) to avoid definitional conflation
- **C3** is a four-option multiple-choice knowledge test (not self-report) measuring safe harbour awareness without information contamination — KEY independent variable
- **C9** is a knowledge test measuring awareness of the technical support and Rapid Dispute Resolution mandate
- **D11–D12** measure the coordination wedge: trader estimates of customer digital payment demand and peer registration expectations
- **F1** is an open-ended narrative item coded using AI-assisted qualitative analysis

---

## Analysis Plan

All analysis is implemented in **Python 3.11** using open-source libraries:

```
statsmodels 0.14    — probit, ordered logit, OLS
scikit-learn 1.4    — factor analysis (varimax rotation)
scipy               — chi-square tests, mediation analysis
pandas / numpy      — data management
```

Ten pre-specified analytical steps:

1. Descriptive statistics and frequency tables
2. Safe harbour awareness cross-tabulations (chi-square)
3. **Primary probit model** — DV: B1b (formal QR registration), Key IV: C3 (safe harbour awareness), market × vendor interaction terms, trader controls
4. Factor analysis on D1–D10 barrier items (varimax, 4 factors)
5. Probit with factor scores (multicollinearity reduction)
6. **Ordered logit** — DV: E1 (willingness to adopt, 1–5) — robustness check
7. Split-sample heterogeneity analysis by market type and vendor category
8. **Mediation analysis** — C3 → tax anxiety (C4, D7) → B1b pathway
9. **Coordination wedge analysis** — D11 and D12 in probit and ordered logit
10. Policy simulation — adoption rates under full awareness and dual-intervention scenarios

---

## Repository Structure

```
SME-Digital-Payment-Adoption-KP/
│
├── README.md                          ← This file
│
├── data/
│   ├── codebook.md                    ← Variable definitions, scales, analytical roles
│   └── data_dictionary.xlsx           ← Full codebook in Excel format
│
├── scripts/
│   ├── 01_descriptives.py             ← Step 1-2: Descriptive stats and cross-tabs
│   ├── 02_probit_primary.py           ← Step 3: Primary probit model
│   ├── 03_factor_analysis.py          ← Step 4-5: Factor analysis and probit with factors
│   ├── 04_ordered_logit.py            ← Step 6: Ordered logit robustness
│   ├── 05_heterogeneity.py            ← Step 7: Split-sample analysis
│   ├── 06_mediation.py                ← Step 8: Mediation analysis
│   ├── 07_coordination_wedge.py       ← Step 9: D11-D12 coordination wedge
│   └── 08_policy_simulation.py        ← Step 10: Policy scenarios
│
├── outputs/
│   ├── tables/                        ← LaTeX and CSV result tables
│   └── figures/                       ← Charts and visualisations
│
└── instrument/
    └── questionnaire_v3.md            ← Survey instrument (44 items, plain text)
```

> **Note:** Raw survey data is not uploaded to this repository in compliance with respondent privacy obligations and the study's ethics protocol. The cleaned analysis dataset (no names, phone numbers, shop names, or addresses) will be made available upon journal acceptance. All replication code will be deposited here upon journal submission.

---

## Data Privacy Statement

This study involves human subjects. All respondents provided verbal informed consent. No personally identifying information was recorded on survey forms. Business revenue was collected in categorical bands. Contact numbers from follow-up consent are stored separately and never linked to the questionnaire data. The analysis dataset contains no names, phone numbers, shop names, or addresses.

---

## Related Repositories

| Repository | Description |
|---|---|
| [Stockmarket-Sentiment-Analysis](https://github.com/DrYasirSaeed/Stockmarket-Sentiment-Analysis) | FinBERT-based NLP pipeline for KSE-100 macroeconomic news sentiment |
| [Shariah-Complaince-Research](https://github.com/DrYasirSaeed/Shariah-Complaince-Research) | Shariah screening compliance and firm profitability, PSX-listed firms |

---

## Citation

If you use the code or instrument from this repository, please cite:

```
Saeed, Y. (2026). Digital Payment Mandates and SME Financial Inclusion:
Evidence from the Cashless Khyber Pakhtunkhwa Initiative.
Working Paper. Department of Economics, KUST, Kohat.
GitHub: https://github.com/DrYasirSaeed/SME-Digital-Payment-Adoption-KP
```

---

## Status

| Component | Status |
|---|---|
| Survey instrument (v3, 44 items) | ✅ Complete |
| Codebook and data dictionary | ✅ Complete |
| Analysis scripts (tested on hypothetical data) | ✅ Complete |
| ORIC grant application | ✅ Submitted |
| EIRB presentation | 🔄 Scheduled June 4, 2026 |
| Field data collection | ⏳ Pending grant award |
| Journal submission | ⏳ Pending field data |

---

*Department of Economics, Kohat University of Science and Technology (KUST)*
*KUST ORIC Research Grant 2025–26*
