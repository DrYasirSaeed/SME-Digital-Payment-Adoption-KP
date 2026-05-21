"""
04_ordered_logit.py
Step 6: Ordered logit — robustness check on willingness to adopt

Dependent variable  : E1_willingness (1=not willing, 5=very willing)
Key independent var : C3_safe_harbour_aware
Controls            : same as primary probit

Digital Payment Mandates and SME Financial Inclusion in KP
Dr. Yasir Saeed, KUST ORIC Research Grant 2025-26

Outputs
-------
outputs/tables/06a_ordered_logit_coefficients.csv
outputs/tables/06b_ordered_logit_ame.csv
outputs/figures/06_willingness_distribution_by_awareness.png
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.miscmodels.ordinal_model import OrderedModel
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

df = pd.read_csv("data/hypothetical_survey_data.csv").dropna(subset=["E1_willingness"])

# Zone dummies (Zone 3 reference)
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

y_ordered = df["E1_willingness"].astype(int)
X = df[REGRESSORS]

# Ordered logit
model_ol = OrderedModel(y_ordered, X, distr="logit")
result_ol = model_ol.fit(method="bfgs", disp=False)

print("Step 6 — Ordered Logit Results\n")
print(result_ol.summary())

# Save coefficients
coef_df = pd.DataFrame({
    "Variable": result_ol.model.exog_names,
    "Coefficient": result_ol.params.round(4),
    "SE": result_ol.bse.round(4),
    "z": result_ol.tvalues.round(3),
    "p_value": result_ol.pvalues.round(4),
})

def stars(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""

coef_df["sig"] = coef_df["p_value"].apply(stars)
coef_df.to_csv("outputs/tables/06a_ordered_logit_coefficients.csv", index=False)

# ============================================================
# Approximate marginal effects at means (probability of E1=5)
# ============================================================

# Extract predicted probabilities for each outcome
pred_probs = result_ol.predict()
pred_df = pd.DataFrame(pred_probs, columns=[f"P(E1={j})" for j in range(1, 6)])

# Average predicted probability of being "Very willing" (E1=5)
# comparing aware vs not-aware, holding others at mean
df_mean = pd.DataFrame([X.mean().values], columns=REGRESSORS)

df_aware = df_mean.copy()
df_aware["C3_safe_harbour_aware"] = 1
df_not_aware = df_mean.copy()
df_not_aware["C3_safe_harbour_aware"] = 0

p_aware = result_ol.predict(df_aware)
p_not_aware = result_ol.predict(df_not_aware)

ame_e5_aware = p_aware.iloc[0, 4]
ame_e5_not = p_not_aware.iloc[0, 4]

print(f"\nP(E1=5 | C3=1, others at mean): {ame_e5_aware:.4f}")
print(f"P(E1=5 | C3=0, others at mean): {ame_e5_not:.4f}")
print(f"Marginal difference at E1=5:    {ame_e5_aware - ame_e5_not:.4f}")

ame_simple_df = pd.DataFrame({
    "Outcome": [f"P(E1={j})" for j in range(1, 6)],
    "P_C3_aware": p_aware.iloc[0].values.round(4),
    "P_C3_not_aware": p_not_aware.iloc[0].values.round(4),
    "Difference": (p_aware.iloc[0].values - p_not_aware.iloc[0].values).round(4),
})
ame_simple_df.to_csv("outputs/tables/06b_ordered_logit_ame.csv", index=False)


# ============================================================
# FIGURE 6  Willingness distribution by C3 awareness
# ============================================================

fig, axes = plt.subplots(1, 2, figsize=(11, 5), sharey=True)

for ax, aware_val, label, color in zip(
    axes, [0, 1],
    ["Safe Harbour NOT Aware (C3=0)", "Safe Harbour AWARE (C3=1)"],
    ["#9CAAB6", "#2D6A9F"]
):
    sub = df[df["C3_safe_harbour_aware"] == aware_val]["E1_willingness"]
    vc = sub.value_counts().sort_index()
    bars = ax.bar(vc.index, vc.values / len(sub) * 100, color=color, width=0.6, edgecolor="white")
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                f"{h:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xlabel("Willingness to Register (E1)\n1=Not willing  →  5=Very willing", fontsize=10)
    ax.set_ylabel("Percentage of Respondents (%)", fontsize=10)
    ax.set_title(label, fontsize=10, fontweight="bold", color=color)
    ax.set_xticks([1, 2, 3, 4, 5])
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_ylim(0, 55)

fig.suptitle("Willingness to Register for QR Code by Safe Harbour Awareness\nKohat District Survey — Hypothetical Data", fontsize=11)
plt.tight_layout()
plt.savefig("outputs/figures/06_willingness_distribution_by_awareness.png", dpi=150)
plt.close()
print("Figure 6 saved.\nStep 6 complete.\n")
