"""
02_probit_primary.py
Step 3: Primary probit — effect of safe harbour awareness on QR registration

Dependent variable  : B1b_registered_binary  (1 = formally registered)
Key independent var : C3_safe_harbour_aware   (1 = passed safe harbour quiz)
Controls            : zone dummies (ref=Zone 3), vendor_food,
                      has_bank_account, D11_customer_demand_est

Outputs
-------
outputs/tables/03a_probit_primary_ame.csv    — full model average marginal effects
outputs/tables/03b_vif_check.csv             — variance inflation factors
outputs/tables/03c_probit_restricted_ame.csv — restricted model (zone + vendor only)
outputs/figures/03_ame_forest_plot.png       — forest plot of primary AMEs

Digital Payment Mandates and SME Financial Inclusion in KP
Dr. Yasir Saeed, KUST ORIC Research Grant 2025-26
"""

import os
import warnings

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

warnings.filterwarnings("ignore")

DATA_PATH = "data/hypothetical_survey_data.csv"
TABLES    = "outputs/tables"
FIGURES   = "outputs/figures"
os.makedirs(TABLES,  exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
df = pd.read_csv(DATA_PATH)

# Zone dummies (Zone 3 = reference category)
df["zone1"] = (df["A1_zone"] == 1).astype(int)
df["zone2"] = (df["A1_zone"] == 2).astype(int)

OUTCOME  = "B1b_registered_binary"
FULL_CONTROLS = [
    "C3_safe_harbour_aware",
    "zone1", "zone2",
    "vendor_food",
    "has_bank_account",
    "D11_customer_demand_est",
]
RESTRICTED_CONTROLS = [
    "C3_safe_harbour_aware",
    "zone1", "zone2",
    "vendor_food",
]

print(f"\nN = {len(df)}")
print(f"Registration rate: {df[OUTCOME].mean():.1%}")
print(f"C3 awareness rate: {df['C3_safe_harbour_aware'].mean():.1%}")

# ---------------------------------------------------------------------------
# 2. Full probit model + average marginal effects
# ---------------------------------------------------------------------------
X_full = sm.add_constant(df[FULL_CONTROLS])
y      = df[OUTCOME]

probit_full   = sm.Probit(y, X_full).fit(disp=False)
margeff_full  = probit_full.get_margeff()
ame_df        = margeff_full.summary_frame().reset_index()
ame_df.columns = ["variable", "AME", "SE", "z", "p_value", "CI_lower", "CI_upper"]
ame_df["sig"] = ame_df["p_value"].apply(
    lambda p: "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
)
ame_df = ame_df[ame_df["variable"] != "const"]

ame_df.to_csv(os.path.join(TABLES, "03a_probit_primary_ame.csv"), index=False)
print("\nFull model AMEs saved.")
print(ame_df[["variable", "AME", "SE", "p_value", "sig"]].to_string(index=False))

# ---------------------------------------------------------------------------
# 3. VIF check on full model regressors
# ---------------------------------------------------------------------------
X_vif = df[FULL_CONTROLS].copy()
vif_data = pd.DataFrame({
    "Variable": FULL_CONTROLS,
    "VIF": [
        variance_inflation_factor(X_vif.values, i)
        for i in range(X_vif.shape[1])
    ],
})
vif_data["VIF"] = vif_data["VIF"].round(3)
vif_data.to_csv(os.path.join(TABLES, "03b_vif_check.csv"), index=False)
print("\nVIF check:")
print(vif_data.to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Restricted probit (zone + vendor only, no additional controls)
# ---------------------------------------------------------------------------
X_restricted    = sm.add_constant(df[RESTRICTED_CONTROLS])
probit_restr    = sm.Probit(y, X_restricted).fit(disp=False)
margeff_restr   = probit_restr.get_margeff()
ame_restr_df    = margeff_restr.summary_frame().reset_index()
ame_restr_df.columns = [
    "variable", "AME_restricted", "SE_restricted", "z_restricted",
    "p_restricted", "CI_lower_restricted", "CI_upper_restricted",
]
ame_restr_df = ame_restr_df[ame_restr_df["variable"] != "const"]
ame_restr_df.to_csv(os.path.join(TABLES, "03c_probit_restricted_ame.csv"), index=False)
print("\nRestricted model AMEs saved.")

# ---------------------------------------------------------------------------
# 5. Forest plot of full model AMEs
# ---------------------------------------------------------------------------
LABEL_MAP = {
    "C3_safe_harbour_aware": "Safe harbour\nawareness (C3)",
    "zone1":                 "Zone 1 (Central)",
    "zone2":                 "Zone 2 (Peripheral)",
    "vendor_food":           "Food/grocery\nvendor",
    "has_bank_account":      "Has bank account",
    "D11_customer_demand_est": "Customer digital\ndemand (D11)",
}

plot_df = ame_df.copy()
plot_df["label"] = plot_df["variable"].map(LABEL_MAP).fillna(plot_df["variable"])
plot_df = plot_df.sort_values("AME", ascending=True).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(7, 4.5))

colors = ["#c0392b" if p < 0.10 else "#7f8c8d" for p in plot_df["p_value"]]
ax.barh(plot_df["label"], plot_df["AME"], xerr=1.96 * plot_df["SE"],
        color=colors, capsize=4, height=0.55, error_kw={"linewidth": 1.2})
ax.axvline(0, color="black", linewidth=0.8, linestyle="--")

ax.set_xlabel("Average Marginal Effect on P(QR Registered)", fontsize=10)
ax.set_title("Primary Probit: Average Marginal Effects\n(red = p < 0.10)", fontsize=11)
ax.xaxis.set_major_formatter(mticker.PercentFormatter(xmax=1, decimals=1))
ax.tick_params(axis="y", labelsize=9)

for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

plt.tight_layout()
fig.savefig(os.path.join(FIGURES, "03_ame_forest_plot.png"), dpi=150, bbox_inches="tight")
plt.close()
print("\nForest plot saved.")

print(f"\nStep 3 complete. Pseudo-R² = {probit_full.prsquared:.3f}  "
      f"Log-lik = {probit_full.llf:.1f}")
