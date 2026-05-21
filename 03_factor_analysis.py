"""
08_policy_simulation.py
Step 10: Policy simulation — adoption rates under full safe harbour awareness
         and dual-intervention (awareness + process simplification) scenarios

Uses the fitted primary probit AME of C3 to project:
  Scenario 1 — Baseline (observed awareness rate)
  Scenario 2 — Full awareness campaign (C3=1 for all traders)
  Scenario 3 — Dual intervention: full awareness + D3/D4 process complexity
                reduced from median to 2 (simplified registration)
  Scenario 4 — Triple intervention: Scenario 3 + D11 customer demand estimate
                increased by 2 (customer-side demand campaign)

Also produces zone-specific and vendor-specific projections.

Digital Payment Mandates and SME Financial Inclusion in KP
Dr. Yasir Saeed, KUST ORIC Research Grant 2025-26

Outputs
-------
outputs/tables/10a_policy_simulation_overall.csv
outputs/tables/10b_policy_simulation_by_zone.csv
outputs/figures/10_policy_simulation_scenarios.png
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

df = pd.read_csv("data/hypothetical_survey_data.csv").dropna(subset=["B1b_registered_binary"])

df["zone1"] = (df["A1_zone"] == 1).astype(int)
df["zone2"] = (df["A1_zone"] == 2).astype(int)
df["zone1_food"] = df["zone1"] * df["vendor_food"]
df["zone2_food"] = df["zone2"] * df["vendor_food"]
df["customer_demand_share"] = df["D11_customer_demand_est"] / 10.0
df["tax_anxiety"] = df["D7_audit_fear"]
df["payment_trust"] = df["C5_payment_trust"]

REGRESSORS = [
    "C3_safe_harbour_aware",
    "zone1", "zone2",
    "vendor_food", "zone1_food", "zone2_food",
    "A3_years_operating", "A4_daily_revenue",
    "has_bank_account", "has_mobile_wallet", "has_smartphone",
    "tax_anxiety", "payment_trust",
    "customer_demand_share", "D12_peer_will_register",
]

X = sm.add_constant(df[REGRESSORS])
y = df["B1b_registered_binary"].astype(int)
model = Probit(y, X).fit(disp=False, cov_type="HC1")

print("Step 10 — Policy Simulation\n")
print(f"Primary probit fitted. N={int(model.nobs)}, Pseudo-R2={model.prsquared:.4f}\n")


def predict_scenario(df_scenario, model, regressors):
    """Return predicted registration probabilities for each observation."""
    X_s = sm.add_constant(df_scenario[regressors], has_constant="add")
    return model.predict(X_s)


# ============================================================
# Define scenarios
# ============================================================

def scenario_baseline(df):
    return df.copy()

def scenario_full_awareness(df):
    s = df.copy()
    s["C3_safe_harbour_aware"] = 1
    # Full awareness also reduces tax anxiety (via mediation) — apply partial adjustment
    # Treat as conservative: only change C3 directly
    return s

def scenario_dual(df):
    s = df.copy()
    s["C3_safe_harbour_aware"] = 1
    # Process complexity reduction: D3 (know how) and D4 (process complex) → floor at 2
    # In our model these are captured through factor scores but we approximate
    # by boosting customer_demand_share modestly (+0.15 = 1.5 more customers per 10)
    s["tax_anxiety"] = s["tax_anxiety"].clip(upper=3)  # anxiety partially reduced
    return s

def scenario_triple(df):
    s = scenario_dual(df)
    # Customer-side demand push: D11 estimate increases by 2 units (20 pp share)
    s["customer_demand_share"] = (s["D11_customer_demand_est"] + 2).clip(upper=10) / 10.0
    # Peer norm update: traders expect more peers to register (D12 shifts down by 1)
    s["D12_peer_will_register"] = (s["D12_peer_will_register"] - 1).clip(lower=1)
    return s


scenario_funcs = [
    ("S1: Baseline", scenario_baseline),
    ("S2: Full Awareness\n(C3=1 for all)", scenario_full_awareness),
    ("S3: Dual Intervention\n(Awareness + Reduced Tax Anxiety)", scenario_dual),
    ("S4: Triple Intervention\n(S3 + Customer Demand Push)", scenario_triple),
]


# ============================================================
# Overall projections
# ============================================================

overall_rows = []
baseline_rate = None

for label, func in scenario_funcs:
    df_s = func(df)
    predicted = predict_scenario(df_s, model, REGRESSORS)
    proj_rate = predicted.mean() * 100
    if baseline_rate is None:
        baseline_rate = proj_rate
    additional_traders = int((proj_rate - baseline_rate) / 100 * 300)
    overall_rows.append({
        "Scenario": label.replace("\n", " "),
        "Projected_Registration_Rate_Pct": round(proj_rate, 1),
        "Absolute_Increase_Over_Baseline_Pct": round(proj_rate - baseline_rate, 1),
        "Additional_Traders_Registered_of_300": additional_traders,
    })
    print(f"{label.replace(chr(10), ' '): <55} → {proj_rate:.1f}%")

overall_df = pd.DataFrame(overall_rows)
overall_df.to_csv("outputs/tables/10a_policy_simulation_overall.csv", index=False)


# ============================================================
# Zone-specific projections
# ============================================================

ZONE_LABELS = {1: "Zone 1 (Central)", 2: "Zone 2 (Secondary)", 3: "Zone 3 (Neighbourhood)"}
zone_rows = []

for zone in [1, 2, 3]:
    zdf = df[df["A1_zone"] == zone].copy()
    baseline_z = predict_scenario(scenario_baseline(zdf), model, REGRESSORS).mean() * 100
    for label, func in scenario_funcs:
        df_s = func(zdf)
        proj_rate = predict_scenario(df_s, model, REGRESSORS).mean() * 100
        zone_rows.append({
            "Zone": ZONE_LABELS[zone],
            "Scenario": label.replace("\n", " "),
            "Projected_Rate_Pct": round(proj_rate, 1),
            "Increase_Over_Zone_Baseline_Pct": round(proj_rate - baseline_z, 1),
        })

zone_df = pd.DataFrame(zone_rows)
zone_df.to_csv("outputs/tables/10b_policy_simulation_by_zone.csv", index=False)
print("\nZone-level simulation:")
print(zone_df.pivot(index="Zone", columns="Scenario", values="Projected_Rate_Pct").to_string())


# ============================================================
# FIGURE 10  Policy scenarios chart
# ============================================================

scenario_labels = [r["Scenario"].split("(")[0].strip() for r in overall_rows]
scenario_rates = [r["Projected_Registration_Rate_Pct"] for r in overall_rows]
bar_colors = ["#9CAAB6", "#2D6A9F", "#E87722", "#3BAA75"]

fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(range(len(scenario_labels)), scenario_rates,
              color=bar_colors, width=0.55, edgecolor="white")

for bar, rate, row in zip(bars, scenario_rates, overall_rows):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
            f"{rate:.1f}%\n(+{row['Absolute_Increase_Over_Baseline_Pct']:.1f}pp)",
            ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(range(len(scenario_labels)))
ax.set_xticklabels(scenario_labels, fontsize=10)
ax.set_ylabel("Projected Formal QR Registration Rate (%)", fontsize=11)
ax.set_title("Policy Simulation: Projected QR Registration Rates Under Alternative Interventions\n"
             "Kohat District Survey — Hypothetical Data (N=300)", fontsize=11)
ax.set_ylim(0, 80)
ax.spines[["top", "right"]].set_visible(False)
ax.axhline(scenario_rates[0], color="#9CAAB6", linestyle="--", linewidth=1.2, label="Baseline")
ax.legend(fontsize=9, loc="upper left")

fig.text(0.5, -0.02,
         "S2: Full awareness campaign. S3: Awareness + tax anxiety reduction. S4: S3 + customer demand push.",
         ha="center", fontsize=8.5, style="italic")

plt.tight_layout()
plt.savefig("outputs/figures/10_policy_simulation_scenarios.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nFigure 10 saved.\nStep 10 complete.\n")
print("="*60)
print("ALL 10 ANALYTICAL STEPS COMPLETE")
print("Tables saved to: outputs/tables/")
print("Figures saved to: outputs/figures/")
print("="*60)
