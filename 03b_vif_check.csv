"""
generate_hypothetical_data.py
Digital Payment Mandates and SME Financial Inclusion in KP
Dr. Yasir Saeed, KUST ORIC Research Grant 2025-26

Generates a hypothetical dataset of N=300 traders that mirrors the
44-item survey instrument (Annexure B, v3). Run this once to produce
data/hypothetical_survey_data.csv. Replace with the real cleaned
fieldwork dataset before running any analytical scripts.

Variable names follow the instrument exactly so that all downstream
scripts can be used without modification after real data is imported.
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(seed=42)
N = 300


def generate_data() -> pd.DataFrame:
    rows = []

    for i in range(N):
        row = {}
        row["respondent_id"] = f"KHT-{i+1:04d}"

        # ---- SECTION A: Trader Profile ----
        row["A1_zone"] = RNG.choice([1, 2, 3], p=[1/3, 1/3, 1/3])

        row["A2_business_type"] = RNG.choice(
            [1, 2, 3, 4, 5, 6], p=[0.30, 0.20, 0.15, 0.10, 0.15, 0.10]
        )

        row["A3_years_operating"] = RNG.choice(
            [1, 2, 3, 4], p=[0.15, 0.30, 0.30, 0.25]
        )

        row["A4_daily_revenue"] = RNG.choice(
            [1, 2, 3, 4], p=[0.25, 0.35, 0.25, 0.15]
        )

        row["A5_employees"] = RNG.choice(
            [1, 2, 3, 4, 5], p=[0.40, 0.30, 0.15, 0.10, 0.05]
        )

        row["A6_bank_account"] = RNG.choice([1, 2, 3], p=[0.40, 0.15, 0.45])

        row["A7_mobile_wallet"] = RNG.choice([1, 2, 3], p=[0.30, 0.20, 0.50])

        row["A8_smartphone"] = RNG.choice([1, 2, 3], p=[0.60, 0.35, 0.05])

        # ---- SECTION B: Digital Payment Status ----
        # QR display probability increases with zone 1 vs 3 and bank access
        zone_boost = {1: 0.15, 2: 0.05, 3: -0.10}[row["A1_zone"]]
        bank_boost = 0.10 if row["A6_bank_account"] == 1 else 0
        p_display = min(max(0.25 + zone_boost + bank_boost, 0.05), 0.95)
        row["B1a_qr_display"] = int(RNG.random() < p_display)

        if row["B1a_qr_display"] == 1:
            # Formal registration conditional on display
            p_reg = 0.55 + zone_boost * 0.5
            reg_choice = RNG.choice([1, 2, 3], p=[p_reg, 1 - p_reg - 0.05, 0.05])
            row["B1b_qr_registered"] = reg_choice  # 1=yes registered (DV)
            row["B2_provider"] = RNG.choice([1, 2, 3, 4, 5], p=[0.20, 0.35, 0.30, 0.10, 0.05])
            row["B3_daily_customers"] = RNG.choice([1, 2, 3, 4], p=[0.20, 0.40, 0.25, 0.15])
            row["B4_no_qr_situation"] = np.nan
        else:
            row["B1b_qr_registered"] = np.nan
            row["B2_provider"] = np.nan
            row["B3_daily_customers"] = np.nan
            row["B4_no_qr_situation"] = RNG.choice(
                [1, 2, 3, 4], p=[0.05, 0.25, 0.50, 0.20]
            )

        row["B5_received_digital"] = RNG.choice([1, 2, 3], p=[0.15, 0.30, 0.55])

        row["B6_inspected"] = RNG.choice(
            [1, 2], p=[0.20 + (zone_boost * 0.5), 0.80 - (zone_boost * 0.5)]
        )

        # ---- SECTION C: Policy Awareness ----
        row["C1_cashless_kp_awareness"] = RNG.choice(
            [1, 2, 3],
            p=[0.10 + zone_boost * 0.3, 0.25, 0.65 - zone_boost * 0.3]
        )

        row["C2_mandate_awareness"] = RNG.choice(
            [1, 2, 3],
            p=[0.20 + zone_boost * 0.3, 0.55, 0.25 - zone_boost * 0.3]
        )

        # C3 KEY IV: safe harbour knowledge test (binary: 1=correct)
        # Awareness probability higher in Zone 1, with bank account,
        # with mobile wallet use, and with prior government contact
        p_aware = (
            0.18
            + (0.12 if row["A1_zone"] == 1 else -0.05 if row["A1_zone"] == 3 else 0)
            + (0.08 if row["A6_bank_account"] == 1 else 0)
            + (0.06 if row["A7_mobile_wallet"] == 1 else 0)
            + (0.10 if row["B6_inspected"] == 1 else 0)
        )
        p_aware = min(max(p_aware, 0.05), 0.85)
        row["C3_safe_harbour_aware"] = int(RNG.random() < p_aware)

        # C4: tax burden belief (1=strongly agree harm, 5=strongly disagree)
        # Aware traders have lower tax anxiety
        base_c4 = 2.1 + (0.8 * row["C3_safe_harbour_aware"])
        row["C4_tax_burden_belief"] = int(
            np.clip(RNG.normal(base_c4, 1.0), 1, 5)
        )

        row["C5_payment_trust"] = int(np.clip(RNG.normal(2.8, 1.1), 1, 5))

        base_c6 = 2.0 + (0.7 * row["C3_safe_harbour_aware"])
        row["C6_record_fear"] = int(np.clip(RNG.normal(base_c6, 1.0), 1, 5))

        row["C7_training_received"] = RNG.choice(
            [1, 2, 3, 4],
            p=[0.05 + zone_boost * 0.1, 0.10, 0.15, 0.70 - zone_boost * 0.1]
        )

        row["C8_would_register_if_no_tax"] = RNG.choice(
            [1, 2, 3, 4, 5],
            p=[0.35, 0.30, 0.15, 0.12, 0.08]
        )

        # C9: technical support mandate awareness (binary: 1=correct)
        p_c9 = 0.15 + (0.10 if row["C3_safe_harbour_aware"] == 1 else 0)
        row["C9_dispute_resolution_aware"] = int(RNG.random() < p_c9)

        # ---- SECTION D: Barriers (Likert 1-5) ----
        barrier_means = {
            "D1_no_smartphone":         2.5 - (0.8 if row["A8_smartphone"] == 1 else 0),
            "D2_internet_unreliable":   3.0 - (0.4 * zone_boost * 3),
            "D3_dont_know_how":         3.5 - (0.5 if row["C7_training_received"] < 4 else 0),
            "D4_process_complex":       3.4,
            "D5_fees_high":             3.2,
            "D6_customers_prefer_cash": 3.6,
            "D7_audit_fear":            3.8 - (0.9 * row["C3_safe_harbour_aware"]),
            "D8_distrust_safety":       3.1,
            "D9_amounts_too_small":     2.8,
            "D10_no_peer_pressure":     3.3,
        }
        for var, mu in barrier_means.items():
            row[var] = int(np.clip(RNG.normal(mu, 1.0), 1, 5))

        # D11: estimated customer demand out of 10
        base_d11 = 3.0 + (1.0 if row["C3_safe_harbour_aware"] == 1 else 0)
        row["D11_customer_demand_est"] = int(np.clip(RNG.normal(base_d11, 2.0), 0, 10))

        row["D12_peer_will_register"] = RNG.choice(
            [1, 2, 3, 4, 5], p=[0.05, 0.15, 0.25, 0.30, 0.25]
        )

        # ---- SECTION E: Willingness to Adopt ----
        # E1 ordered DV (1=not willing, 5=very willing)
        base_e1 = (
            2.2
            + (0.8 * row["C3_safe_harbour_aware"])
            + (0.4 if row["A6_bank_account"] == 1 else 0)
            + (0.3 if row["A7_mobile_wallet"] == 1 else 0)
            - (0.4 * (row["D7_audit_fear"] / 5))
            + (0.4 * (row["D11_customer_demand_est"] / 10))
        )
        row["E1_willingness"] = int(np.clip(RNG.normal(base_e1, 1.1), 1, 5))

        # E2: up to two policy preferences (store as two columns)
        prefs = RNG.choice(range(1, 8), size=2, replace=False)
        row["E2_pref_1"] = int(prefs[0])
        row["E2_pref_2"] = int(prefs[1])

        row["E3_paymir_awareness"] = RNG.choice([1, 2, 3], p=[0.08, 0.22, 0.70])

        row["E4_cash_incentive_effect"] = RNG.choice(
            [1, 2, 3, 4, 5], p=[0.30, 0.35, 0.18, 0.10, 0.07]
        )

        row["E5_trusted_info_source"] = RNG.choice(
            [1, 2, 3, 4, 5, 6, 7],
            p=[0.15, 0.15, 0.10, 0.15, 0.15, 0.15, 0.15]
        )

        row["E6_followup_consent"] = RNG.choice([1, 2], p=[0.55, 0.45])

        # ---- SECTION F: Open-ended (placeholder text) ----
        open_ended_pool = [
            "Darr hai ke FBR aa jaye ga",
            "Mujhe pata nahi tha ke kaise register karein",
            "Mere grahak cash hi dete hain",
            "Internet sahi nahi hota yahan",
            "Registration ka tareeqa bohat mushkil lagta hai",
            "Fees zyada hain wallets ki",
            "Jab safe harbour ka pata chala to main ne register kar liya",
            "Hamari dukan chhoti hai digital ki zaroorat nahi",
            "Sab log cash mein karte hain toh main kyun badulon?",
            "Paisa account mein aata hai toh KPRA dekh leta hai",
        ]
        row["F1_open_ended"] = RNG.choice(open_ended_pool)

        rows.append(row)

    df = pd.DataFrame(rows)

    # ---- Derived binary DV for regression ----
    # B1b == 1 means formally registered. For traders without QR (B1b NaN), set to 0.
    df["B1b_registered_binary"] = df["B1b_qr_registered"].apply(
        lambda x: 1 if x == 1 else 0
    )

    # Vendor category (food=1 based on A2=1; non-food=0)
    df["vendor_food"] = (df["A2_business_type"] == 1).astype(int)

    # Banking access binary
    df["has_bank_account"] = (df["A6_bank_account"] == 1).astype(int)

    # Mobile wallet binary
    df["has_mobile_wallet"] = (df["A7_mobile_wallet"] == 1).astype(int)

    # Smartphone binary
    df["has_smartphone"] = (df["A8_smartphone"] == 1).astype(int)

    return df


if __name__ == "__main__":
    import os
    os.makedirs("data", exist_ok=True)
    df = generate_data()
    df.to_csv("data/hypothetical_survey_data.csv", index=False)
    print(f"Dataset generated: {len(df)} rows, {len(df.columns)} columns")
    print(df.head(3).T)
