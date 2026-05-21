"""
03_factor_analysis.py
Steps 4-5: Exploratory Factor Analysis on Section D barrier items
           + probit regression using factor scores

Step 4 -- EFA (PCA extraction, varimax rotation) on D1-D10 barrier items
          Identify latent barrier dimensions underlying non-adoption
Step 5 -- Probit regression: B1b_registered_binary on factor scores + C3 + controls
          Tests whether barrier dimensions predict formal QR registration

Digital Payment Mandates and SME Financial Inclusion in KP
Dr. Yasir Saeed, KUST ORIC Research Grant 2025-26

Outputs
-------
outputs/tables/04a_factor_loadings.csv
outputs/tables/04b_factor_variance.csv
outputs/tables/05a_probit_factor_ame.csv
outputs/figures/04_factor_loadings_heatmap.png
outputs/figures/05_scree_plot.png
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import Probit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import os

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

df = pd.read_csv("data/hypothetical_survey_data.csv").dropna(subset=["B1b_registered_binary"])

print("Step 4-5 -- Factor Analysis and Factor-Score Probit\n")

# ============================================================
# Step 4 -- EFA via PCA with varimax rotation (manual)
# ============================================================

BARRIER_ITEMS = [
    "D1_no_smartphone",
    "D2_internet_unreliable",
    "D3_dont_know_how",
    "D4_process_complex",
    "D5_fees_high",
    "D6_customers_prefer_cash",
    "D7_audit_fear",
    "D8_distrust_safety",
    "D9_amounts_too_small",
    "D10_no_peer_pressure",
]

ITEM_LABELS = {
    "D1_no_smartphone": "D1 No Smartphone",
    "D2_internet_unreliable": "D2 Unreliable Internet",
    "D3_dont_know_how": "D3 Lack of Know-how",
    "D4_process_complex": "D4 Complex Process",
    "D5_fees_high": "D5 High Fees",
    "D6_customers_prefer_cash": "D6 Cash Preference",
    "D7_audit_fear": "D7 Audit/Tax Fear",
    "D8_distrust_safety": "D8 Safety Distrust",
    "D9_amounts_too_small": "D9 Small Amounts",
    "D10_no_peer_pressure": "D10 No Peer Pressure",
]

sub = df[BARRIER_ITEMS].dropna()
idx = sub.index

scaler = StandardScaler()
X_scaled = scaler.fit_transform(sub)

# Determine number of factors via eigenvalues (Kaiser criterion: eigenvalue > 1)
pca_full = PCA()
pca_full.fit(X_scaled)
eigenvalues = pca_full.explained_variance_
n_factors = int((eigenvalues > 1).sum())
n_factors = max(n_factors, 2)  # at least 2
print(f"Eigenvalues > 1: {n_factors} factors retained")
print(f"Eigenvalues: {eigenvalues.round(3)}")

# Fit PCA with n_factors
pca = PCA(n_components=n_factors)
factor_scores_raw = pca.fit_transform(X_scaled)
loadings = pca.components_.T  # shape: (n_items, n_factors)

# Varimax rotation (raw scores × rotation matrix approximation)
def varimax(loadings_matrix, max_iter=1000, tol=1e-6):
    """Varimax rotation via Jacobi sweep algorithm."""
    p, k = loadings_matrix.shape
    rotation = np.eye(k)
    for _ in range(max_iter):
        old_rotation = rotation.copy()
        for i in range(k - 1):
            for j in range(i + 1, k):
                L = loadings_matrix @ rotation
                u = L[:, i] ** 2 - L[:, j] ** 2
                v = 2 * L[:, i] * L[:, j]
                A = np.sum(u)
                B = np.sum(v)
                C = np.sum(u ** 2 - v ** 2)
                D = 2 * np.sum(u * v)
                num = D - 2 * A * B / p
                den = C - (A ** 2 - B ** 2) / p
                theta = np.arctan2(num, den) / 4
                rot2 = np.eye(k)
                rot2[i, i] = np.cos(theta)
                rot2[j, j] = np.cos(theta)
                rot2[i, j] = -np.sin(theta)
                rot2[j, i] = np.sin(theta)
                rotation = rotation @ rot2
        if np.max(np.abs(rotation - old_rotation)) < tol:
            break
    return loadings_matrix @ rotation, rotation

rotated_loadings, rot_matrix = varimax(loadings)
rotated_scores = factor_scores_raw @ rot_matrix

factor_names = [f"Factor{i+1}" for i in range(n_factors)]

# Save factor loadings
loadings_df = pd.DataFrame(
    np.round(rotated_loadings, 4),
    index=[ITEM_LABELS[c] for c in BARRIER_ITEMS],
    columns=factor_names,
)
loadings_df.index.name = "Item"
loadings_df.to_csv("outputs/tables/04a_factor_loadings.csv")
print("\nFactor loadings (varimax):")
print(loadings_df.to_string())

# Save variance explained
var_explained = pca.explained_variance_ratio_ * 100
cum_var = np.cumsum(var_explained)
var_df = pd.DataFrame({
    "Factor": factor_names,
    "Eigenvalue": eigenvalues[:n_factors].round(4),
    "Variance_Explained_Pct": np.round(var_explained, 2),
    "Cumulative_Variance_Pct": np.round(cum_var, 2),
})
var_df.to_csv("outputs/tables/04b_factor_variance.csv", index=False)
print("\nVariance explained:")
print(var_df.to_string(index=False))

# Attach factor scores to main dataframe
scores_df = pd.DataFrame(rotated_scores, index=idx, columns=factor_names)
df = df.join(scores_df, how="left")

# ============================================================
# FIGURE 4 -- Factor loadings heatmap
# ============================================================

fig, ax = plt.subplots(figsize=(max(6, n_factors * 1.8), 7))
heatmap_data = loadings_df.values.astype(float)
im = ax.imshow(heatmap_data, cmap="RdYlBu_r", vmin=-1, vmax=1, aspect="auto")
plt.colorbar(im, ax=ax, label="Loading")
ax.set_xticks(range(n_factors))
ax.set_xticklabels(factor_names, fontsize=11)
ax.set_yticks(range(len(BARRIER_ITEMS)))
ax.set_yticklabels([ITEM_LABELS[c] for c in BARRIER_ITEMS], fontsize=10)
for i in range(len(BARRIER_ITEMS)):
    for j in range(n_factors):
        val = heatmap_data[i, j]
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=9, color="black" if abs(val) < 0.6 else "white")
ax.set_title("Rotated Factor Loadings (Varimax)\nSection D Barrier Items",
             fontsize=12)
plt.tight_layout()
plt.savefig("outputs/figures/04_factor_loadings_heatmap.png", dpi=150)
plt.close()
print("\nFigure 4 (factor loadings heatmap) saved.")

# ============================================================
# FIGURE 5 -- Scree plot
# ============================================================

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(range(1, len(eigenvalues) + 1), eigenvalues,
        marker="o", color="#2D6A9F", linewidth=2, markersize=7)
ax.axhline(1.0, color="#E87722", linestyle="--", linewidth=1.5, label="Kaiser criterion (eigenvalue=1)")
ax.fill_between(range(1, n_factors + 1), eigenvalues[:n_factors], alpha=0.15, color="#2D6A9F",
                label=f"Retained ({n_factors} factors)")
ax.set_xlabel("Factor Number", fontsize=11)
ax.set_ylabel("Eigenvalue", fontsize=11)
ax.set_title("Scree Plot — Section D Barrier Items\nKohat District Survey", fontsize=11)
ax.legend(fontsize=9)
ax.spines[["top", "right"]].set_visible(False)
plt.tight_layout()
plt.savefig("outputs/figures/05_scree_plot.png", dpi=150)
plt.close()
print("Figure 5 (scree plot) saved.")

# ============================================================
# Step 5 -- Probit with factor scores
# ============================================================

df["zone1"] = (df["A1_zone"] == 1).astype(int)
df["zone2"] = (df["A1_zone"] == 2).astype(int)

FACTOR_CONTROLS = ["C3_safe_harbour_aware", "zone1", "zone2",
                   "vendor_food", "has_bank_account"] + factor_names

sub5 = df.dropna(subset=FACTOR_CONTROLS + ["B1b_registered_binary"])
X5 = sm.add_constant(sub5[FACTOR_CONTROLS])
y5 = sub5["B1b_registered_binary"].astype(int)
model5 = Probit(y5, X5).fit(disp=False, cov_type="HC1")
ame5 = model5.get_margeff(at="mean")

ame_df = pd.DataFrame({
    "Variable": list(model5.model.exog_names[1:]),
    "AME": ame5.margeff.round(4),
    "SE": ame5.margeff_se.round(4),
    "z": ame5.tvalues.round(3),
    "p_value": ame5.pvalues.round(4),
})

def stars(p):
    if p < 0.01: return "***"
    elif p < 0.05: return "**"
    elif p < 0.10: return "*"
    return ""

ame_df["sig"] = ame_df["p_value"].apply(stars)
ame_df.to_csv("outputs/tables/05a_probit_factor_ame.csv", index=False)

print("\nStep 5 -- Factor-Score Probit AMEs:")
print(ame_df.to_string(index=False))
print(f"\nN={int(model5.nobs)}, Pseudo-R2={model5.prsquared:.4f}")

print("\nStep 4-5 complete.\n")
