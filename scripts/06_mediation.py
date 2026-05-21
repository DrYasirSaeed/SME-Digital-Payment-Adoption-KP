"""
06_mediation.py
Step 8: Mediation analysis — C3 safe harbour awareness → tax anxiety (C4, D7) → B1b registration

Tests whether the C3 effect on formal registration operates through the tax anxiety channel.
Uses the product-of-coefficients method (Baron and Kenny pathway verification +
bootstrapped indirect effects via scipy).

Pathway:
  a: C3 → tax_anxiety (OLS, since mediator is ordinal Likert)
  b: tax_anxiety → B1b | C3 (probit, treating mediator as continuous)
  c: C3 → B1b (total probit effect)
  c': C3 → B1b | tax_anxiety (direct effect probit)
  indirect = a × b (product of coefficients)

Bootstrap 95% CI on indirect effect confirms mediation.

Digital Payment Mandates and SME Financial Inclusion in KP
Dr. Yasir Saeed, KUST ORIC Research Grant 2025-26

Outputs
-------
outputs/tables/08a_mediation_path_a.csv
outputs/tables/08b_mediation_path_c.csv
outputs/tables/08c_mediation_path_c_prime.csv
outputs/tables/08d_mediation_summary.csv
outputs/figures/08_mediation_diagram.png
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
import os

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

np.random.seed(42)
N_BOOT = 1000

df = pd.read_csv("data/hypothetical_survey_data.csv").dropna(subset=["B1b_registered_binary"])
df["customer_demand_share"] = df["D11_customer_demand_est"] / 10.0
df["zone1"] = (df["A1_zone"] == 1).astype(int)
df["zone2"] = (df["A1_zone"] == 2).astype(int)

# Two mediators tested: D7 (audit fear barrier) and C4 (tax burden belief)
# Primary mediator: D7_audit_fear (structural barrier item, pre-specified)
# Secondary mediator: C4_tax_burden_belief (perception item)

CONTROLS = ["zone1", "zone2", "vendor_food",
            "A3_years_operating", "A4_daily_revenue",
            "has_bank_account", "has_mobile_wallet", "has_smartphone",
            "customer_demand_share", "D12_peer_will_register"]


def mediation_analysis(df, mediator_col, mediator_label):
    print(f"\n{'='*60}")
    print(f"Mediation via {mediator_col} ({mediator_label})")
    print(f"{'='*60}")

    sub = df.dropna(subset=[mediator_col, "B1b_registered_binary", "C3_safe_harbour_aware"] + CONTROLS)

    # PATH a: C3 → mediator (OLS)
    Xa = sm.add_constant(sub[["C3_safe_harbour_aware"] + CONTROLS])
    Ma = sm.OLS(sub[mediator_col].astype(float), Xa).fit(cov_type="HC1")
    a_coef = Ma.params["C3_safe_harbour_aware"]
    a_se = Ma.bse["C3_safe_harbour_aware"]
    a_p = Ma.pvalues["C3_safe_harbour_aware"]
    print(f"\nPath a (C3 → {mediator_col}): coef={a_coef:.4f}, SE={a_se:.4f}, p={a_p:.4f}")

    path_a_df = pd.DataFrame({
        "Variable": Ma.model.exog_names,
        "Coefficient": Ma.params.round(4),
        "SE": Ma.bse.round(4),
        "t": Ma.tvalues.round(3),
        "p_value": Ma.pvalues.round(4),
    })
    path_a_df.to_csv(f"outputs/tables/08a_mediation_path_a_{mediator_col}.csv", index=False)

    # PATH c: C3 → B1b (total effect probit, no mediator)
    Xc = sm.add_constant(sub[["C3_safe_harbour_aware"] + CONTROLS])
    Mc = Probit(sub["B1b_registered_binary"].astype(int), Xc).fit(disp=False, cov_type="HC1")
    ame_c = Mc.get_margeff(at="mean")
    c_ame = ame_c.margeff[CONTROLS.__len__() > -1 and
                          list(Mc.model.exog_names[1:]).index("C3_safe_harbour_aware")]
    # safer extraction
    reg_names = list(Mc.model.exog_names[1:])  # drop 'const'
    c3_idx = reg_names.index("C3_safe_harbour_aware")
    c_ame = ame_c.margeff[c3_idx]
    c_se = ame_c.margeff_se[c3_idx]
    c_p = ame_c.pvalues[c3_idx]
    print(f"Path c (C3 → B1b, total): AME={c_ame:.4f}, SE={c_se:.4f}, p={c_p:.4f}")

    # PATH c' (direct): C3 → B1b with mediator included
    Xcp = sm.add_constant(sub[["C3_safe_harbour_aware", mediator_col] + CONTROLS])
    Mcp = Probit(sub["B1b_registered_binary"].astype(int), Xcp).fit(disp=False, cov_type="HC1")
    ame_cp = Mcp.get_margeff(at="mean")
    reg_names_cp = list(Mcp.model.exog_names[1:])
    c3_idx_cp = reg_names_cp.index("C3_safe_harbour_aware")
    med_idx_cp = reg_names_cp.index(mediator_col)
    cp_ame = ame_cp.margeff[c3_idx_cp]
    cp_se = ame_cp.margeff_se[c3_idx_cp]
    cp_p = ame_cp.pvalues[c3_idx_cp]
    b_ame = ame_cp.margeff[med_idx_cp]
    b_se = ame_cp.margeff_se[med_idx_cp]
    b_p = ame_cp.pvalues[med_idx_cp]
    print(f"Path c' (C3 → B1b, direct): AME={cp_ame:.4f}, SE={cp_se:.4f}, p={cp_p:.4f}")
    print(f"Path b ({mediator_col} → B1b | C3): AME={b_ame:.4f}, SE={b_se:.4f}, p={b_p:.4f}")

    # Indirect effect (product of coefficients, delta method SE)
    indirect = a_coef * b_ame
    indirect_se_delta = np.sqrt((b_ame ** 2) * (a_se ** 2) + (a_coef ** 2) * (b_se ** 2))
    print(f"\nIndirect effect (a × b): {indirect:.4f} (delta SE: {indirect_se_delta:.4f})")

    # Bootstrap CI for indirect effect
    boot_indirect = []
    for _ in range(N_BOOT):
        idx = np.random.choice(len(sub), len(sub), replace=True)
        boot_sub = sub.iloc[idx].reset_index(drop=True)
        try:
            Xa_b = sm.add_constant(boot_sub[["C3_safe_harbour_aware"] + CONTROLS])
            Ma_b = sm.OLS(boot_sub[mediator_col].astype(float), Xa_b).fit()
            a_b = Ma_b.params["C3_safe_harbour_aware"]

            Xcp_b = sm.add_constant(boot_sub[["C3_safe_harbour_aware", mediator_col] + CONTROLS])
            Mcp_b = Probit(boot_sub["B1b_registered_binary"].astype(int), Xcp_b).fit(disp=False)
            ame_b = Mcp_b.get_margeff(at="mean")
            b_b = ame_b.margeff[list(Mcp_b.model.exog_names[1:]).index(mediator_col)]
            boot_indirect.append(a_b * b_b)
        except Exception:
            continue

    boot_indirect = np.array(boot_indirect)
    ci_lower = np.percentile(boot_indirect, 2.5)
    ci_upper = np.percentile(boot_indirect, 97.5)
    mediation_significant = (ci_lower > 0) or (ci_upper < 0)

    print(f"Bootstrap 95% CI for indirect effect: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Mediation supported (CI excludes 0): {mediation_significant}")

    proportion_mediated = indirect / c_ame if abs(c_ame) > 0.001 else np.nan
    print(f"Proportion of total effect mediated: {proportion_mediated:.3f}")

    summary = {
        "Mediator": mediator_label,
        "Path_a_coef": round(a_coef, 4),
        "Path_a_p": round(a_p, 4),
        "Path_b_AME": round(b_ame, 4),
        "Path_b_p": round(b_p, 4),
        "Path_c_total_AME": round(c_ame, 4),
        "Path_c_p": round(c_p, 4),
        "Path_c_prime_direct_AME": round(cp_ame, 4),
        "Path_c_prime_p": round(cp_p, 4),
        "Indirect_effect": round(indirect, 4),
        "Bootstrap_CI_lower": round(ci_lower, 4),
        "Bootstrap_CI_upper": round(ci_upper, 4),
        "Mediation_significant": mediation_significant,
        "Proportion_mediated": round(proportion_mediated, 3) if not np.isnan(proportion_mediated) else np.nan,
    }
    return summary


print("Step 8 — Mediation Analysis\n")

summary_d7 = mediation_analysis(df, "D7_audit_fear", "Tax/Audit Fear Barrier (D7)")
summary_c4 = mediation_analysis(df, "C4_tax_burden_belief", "Tax Burden Belief (C4)")

mediation_summary_df = pd.DataFrame([summary_d7, summary_c4])
mediation_summary_df.to_csv("outputs/tables/08d_mediation_summary.csv", index=False)
print("\nMediation summary saved.")


# ============================================================
# FIGURE 8  Mediation path diagram
# ============================================================

fig, ax = plt.subplots(figsize=(10, 5))
ax.set_xlim(0, 10)
ax.set_ylim(0, 5)
ax.axis("off")

# Boxes
boxes = {
    "C3": (1.5, 2.5),
    "MEDIAT": (5.0, 4.0),
    "B1b": (8.5, 2.5),
}
box_labels = {
    "C3": "C3\nSafe Harbour\nAwareness",
    "MEDIAT": "Tax Anxiety\n(D7 Audit Fear)",
    "B1b": "B1b\nFormal QR\nRegistration",
}
for key, (x, y) in boxes.items():
    ax.add_patch(mpatches.FancyBboxPatch((x - 1.0, y - 0.55), 2.0, 1.1,
                 boxstyle="round,pad=0.1",
                 facecolor="#EBF3FB" if key != "MEDIAT" else "#FEF3E2",
                 edgecolor="#2D6A9F" if key != "MEDIAT" else "#E87722",
                 linewidth=2))
    ax.text(x, y, box_labels[key], ha="center", va="center",
            fontsize=10, fontweight="bold",
            color="#2D6A9F" if key != "MEDIAT" else "#E87722")

# Arrows
arrow_kw = dict(arrowstyle="-|>", color="#555555", lw=1.5,
                connectionstyle="arc3,rad=0.0")

# C3 → Mediator (path a)
ax.annotate("", xy=(4.0, 3.8), xytext=(2.5, 3.1),
            arrowprops=dict(**arrow_kw))
a_text = f"a = {summary_d7['Path_a_coef']:.3f}\np = {summary_d7['Path_a_p']:.3f}"
ax.text(3.2, 3.7, a_text, ha="center", va="center", fontsize=9, color="#E87722")

# Mediator → B1b (path b)
ax.annotate("", xy=(7.5, 3.1), xytext=(6.0, 3.8),
            arrowprops=dict(**arrow_kw))
b_text = f"b = {summary_d7['Path_b_AME']:.3f}\np = {summary_d7['Path_b_p']:.3f}"
ax.text(6.9, 3.7, b_text, ha="center", va="center", fontsize=9, color="#E87722")

# C3 → B1b direct (path c')
ax.annotate("", xy=(7.5, 2.5), xytext=(2.5, 2.5),
            arrowprops=dict(arrowstyle="-|>", color="#2D6A9F", lw=2.0,
                            connectionstyle="arc3,rad=0.0"))
c_text = f"c' (direct) = {summary_d7['Path_c_prime_direct_AME']:.3f}\nc (total) = {summary_d7['Path_c_total_AME']:.3f}"
ax.text(5.0, 2.1, c_text, ha="center", va="center", fontsize=9, color="#2D6A9F")

# Indirect effect label
indirect_text = (
    f"Indirect (a×b) = {summary_d7['Indirect_effect']:.4f}\n"
    f"95% Boot CI [{summary_d7['Bootstrap_CI_lower']:.4f}, {summary_d7['Bootstrap_CI_upper']:.4f}]"
)
ax.text(5.0, 0.6, indirect_text, ha="center", va="center", fontsize=9.5,
        style="italic",
        bbox=dict(boxstyle="round", facecolor="#F0FFF4", edgecolor="#3BAA75", alpha=0.9))

ax.set_title("Mediation Analysis: Safe Harbour Awareness → Tax Anxiety → QR Registration\n"
             "Kohat District Survey — Hypothetical Data", fontsize=11, y=0.98)
plt.tight_layout()
plt.savefig("outputs/figures/08_mediation_diagram.png", dpi=150)
plt.close()
print("Figure 8 saved.\nStep 8 complete.\n")
