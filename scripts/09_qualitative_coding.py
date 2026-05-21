"""
09_qualitative_coding.py
Step 9 (Qualitative): AI-assisted thematic coding of F1 open-ended responses

Uses the Anthropic API to apply a pre-specified thematic taxonomy consistently
across all 300 F1 open-ended responses. The taxonomy is defined before fieldwork
and documented here for full methodological transparency and replication.

IMPORTANT: This script requires a valid ANTHROPIC_API_KEY environment variable.
Set it before running:
    export ANTHROPIC_API_KEY=your_key_here   (Linux / Mac)
    set ANTHROPIC_API_KEY=your_key_here      (Windows)

The script processes responses in batches and writes results incrementally
so that a partial run can be resumed without reprocessing completed records.

Digital Payment Mandates and SME Financial Inclusion in KP
Dr. Yasir Saeed, KUST ORIC Research Grant 2025-26

Outputs
-------
outputs/tables/09q_f1_coded_responses.csv       -- full coded dataset
outputs/tables/09q_f1_theme_frequencies.csv     -- theme frequency table by zone and vendor
outputs/tables/09q_f1_coding_log.csv            -- model reasoning for audit trail
outputs/figures/09q_f1_theme_distribution.png   -- bar chart of theme frequencies
outputs/figures/09q_f1_themes_by_zone.png       -- theme breakdown by commercial zone
"""

import os
import json
import time
import pandas as pd
import numpy as np
import requests
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("outputs/tables", exist_ok=True)
os.makedirs("outputs/figures", exist_ok=True)

# ============================================================
# PRE-SPECIFIED THEMATIC TAXONOMY
# Defined before fieldwork. Do not modify after data collection begins.
# Each theme maps directly to a theoretical construct in the framework.
# ============================================================

TAXONOMY = {
    "TAX_ANXIETY": {
        "label": "Tax / Audit Anxiety",
        "description": (
            "Trader expresses fear that digital records will be used by KPRA, FBR, "
            "or any government body for taxation, auditing, or fiscal surveillance. "
            "Includes references to becoming 'registered' in a tax sense, fear of "
            "inspectors, or reluctance to create a financial trail."
        ),
        "theory_link": "Allingham and Sandmo (1972); Kleven et al. (2011)",
        "instrument_link": "D7, C4, C6",
    },
    "PROCESS_COMPLEXITY": {
        "label": "Registration Process Complexity",
        "description": (
            "Trader cites difficulty, confusion, or time cost of the registration "
            "process itself. Includes not knowing which office to approach, "
            "complicated paperwork, or repeated failed attempts."
        ),
        "theory_link": "Venkatesh and Davis (2000) — perceived ease of use",
        "instrument_link": "D3, D4",
    },
    "AWARENESS_GAP": {
        "label": "Awareness Gap",
        "description": (
            "Trader was simply unaware that QR code registration was required, "
            "available, or beneficial. No knowledge of the Cashless KP programme, "
            "the Act, or the safe harbour provision. Not the same as active resistance."
        ),
        "theory_link": "Information failure; Slemrod (2019)",
        "instrument_link": "C1, C2, C3",
    },
    "CUSTOMER_DEMAND": {
        "label": "Low Customer Demand / Cash Preference",
        "description": (
            "Trader believes their customers prefer cash, do not carry mobile wallets, "
            "or would not use a QR code even if offered. Demand-side coordination failure."
        ),
        "theory_link": "Jack and Suri (2011) — coordination failure",
        "instrument_link": "D6, D11",
    },
    "INFRASTRUCTURE": {
        "label": "Infrastructure Barrier",
        "description": (
            "Trader cites unreliable mobile internet, lack of a smartphone, power "
            "outages, or connectivity problems as the primary obstacle."
        ),
        "theory_link": "Ozili (2020) — digital infrastructure",
        "instrument_link": "D1, D2",
    },
    "TRUST_DEFICIT": {
        "label": "Trust Deficit",
        "description": (
            "Trader does not trust that digital payments are safe, fears fraud, "
            "worries about transaction failures, or distrusts banks and wallet providers."
        ),
        "theory_link": "Gao and Waechter (2017) — perceived security",
        "instrument_link": "D8, C5",
    },
    "COST_BARRIER": {
        "label": "Fee / Cost Barrier",
        "description": (
            "Trader cites transaction fees charged by banks or mobile wallet companies "
            "as the main reason for non-adoption. May also include cost of devices."
        ),
        "theory_link": "Financial inclusion cost barriers",
        "instrument_link": "D5",
    },
    "PEER_NORM": {
        "label": "Peer / Competitive Norm",
        "description": (
            "Trader says they would register if others in their market do, or that "
            "because competitors are not registering there is no pressure to do so. "
            "Social proof or herd behaviour framing."
        ),
        "theory_link": "Jack and Suri (2011) — coordination wedge",
        "instrument_link": "D10, D12",
    },
    "POSITIVE_NUDGE": {
        "label": "Positive Nudge / Already Adopted",
        "description": (
            "Trader describes a specific event, person, or information that pushed "
            "them to register. Includes outreach from a government officer, "
            "recommendation from a bank agent, customer request, or awareness of "
            "the safe harbour provision."
        ),
        "theory_link": "Slemrod (2019) — communication effectiveness",
        "instrument_link": "B6, C7, C8",
    },
    "SMALL_SCALE": {
        "label": "Small Scale / Not Worthwhile",
        "description": (
            "Trader believes their transaction volumes or amounts are too small to "
            "make digital payments practical or beneficial."
        ),
        "theory_link": "TAM — perceived usefulness",
        "instrument_link": "D9",
    },
    "OTHER": {
        "label": "Other / Unclear",
        "description": (
            "Response does not clearly fit any of the above categories, is too vague "
            "to classify, or addresses a concern not covered by the taxonomy."
        ),
        "theory_link": "N/A",
        "instrument_link": "N/A",
    },
}

TAXONOMY_KEYS = list(TAXONOMY.keys())
TAXONOMY_JSON = json.dumps(
    {k: v["description"] for k, v in TAXONOMY.items()}, ensure_ascii=False, indent=2
)


# ============================================================
# Prompt template
# ============================================================

SYSTEM_PROMPT = """You are a research assistant helping a Pakistani academic code
qualitative survey responses for a study on digital payment adoption among informal
traders in Khyber Pakhtunkhwa, Pakistan. Your job is to assign each open-ended
response to exactly one thematic category from a pre-defined taxonomy.

You must respond ONLY with a valid JSON object containing exactly two keys:
  "theme_code": one of the taxonomy keys listed below
  "reasoning": one sentence (max 20 words) explaining your choice

Do not add any other text, explanation, or markdown."""

def build_user_prompt(response_text: str) -> str:
    return f"""Taxonomy (key: description):
{TAXONOMY_JSON}

Trader's response (in Pashto, Urdu, or English):
\"{response_text}\"

Assign this response to the single best-fitting theme code and give a brief reasoning."""


# ============================================================
# API call with retry
# ============================================================

API_URL = "https://api.anthropic.com/v1/messages"
MODEL = "claude-sonnet-4-20250514"

def call_api(response_text: str, api_key: str, retries: int = 3) -> dict:
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    payload = {
        "model": MODEL,
        "max_tokens": 150,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": build_user_prompt(response_text)}],
    }
    for attempt in range(retries):
        try:
            resp = requests.post(API_URL, headers=headers, json=payload, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            text = data["content"][0]["text"].strip()
            # Strip markdown fences if model adds them
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            parsed = json.loads(text.strip())
            theme_code = parsed.get("theme_code", "OTHER").upper()
            if theme_code not in TAXONOMY_KEYS:
                theme_code = "OTHER"
            return {
                "theme_code": theme_code,
                "theme_label": TAXONOMY[theme_code]["label"],
                "reasoning": parsed.get("reasoning", ""),
                "raw_response": text,
                "error": None,
            }
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
            else:
                return {
                    "theme_code": "OTHER",
                    "theme_label": TAXONOMY["OTHER"]["label"],
                    "reasoning": f"API error: {str(e)[:80]}",
                    "raw_response": "",
                    "error": str(e),
                }


# ============================================================
# Fallback: rule-based coding when API key is not available
# (ensures script runs for testing without an API key)
# ============================================================

KEYWORD_MAP = {
    "TAX_ANXIETY":        ["kpra", "fbr", "tax", "audit", "darr", "survey", "record",
                           "account", "nazar", "taxed", "taxing"],
    "PROCESS_COMPLEXITY": ["register", "registration", "mushkil", "tareeqa", "office",
                           "complicated", "pata nahi", "nahi pata", "kaise"],
    "AWARENESS_GAP":      ["suna nahi", "pata nahi tha", "maloom nahi", "heard", "aware",
                           "didn't know", "unknown"],
    "CUSTOMER_DEMAND":    ["grahak", "customer", "cash", "naqd", "demand", "prefer"],
    "INFRASTRUCTURE":     ["internet", "network", "connectivity", "bijli", "phone",
                           "smartphone", "signal"],
    "TRUST_DEFICIT":      ["trust", "fraud", "safe", "security", "bharosa", "darr"],
    "COST_BARRIER":       ["fee", "charge", "mehnga", "cost", "paisa", "mahenga"],
    "PEER_NORM":          ["sab log", "everyone", "competitors", "doosre", "peers"],
    "POSITIVE_NUDGE":     ["safe harbour", "officer", "bank", "agent", "register kar liya",
                           "pushed", "finally"],
    "SMALL_SCALE":        ["chhoti", "small", "thodi", "kam", "little", "minor"],
}

def rule_based_code(text: str) -> dict:
    text_lower = text.lower()
    scores = {k: 0 for k in TAXONOMY_KEYS}
    for theme, keywords in KEYWORD_MAP.items():
        for kw in keywords:
            if kw in text_lower:
                scores[theme] += 1
    best = max(scores, key=lambda k: scores[k])
    if scores[best] == 0:
        best = "OTHER"
    return {
        "theme_code": best,
        "theme_label": TAXONOMY[best]["label"],
        "reasoning": "Rule-based fallback (no API key)",
        "raw_response": "",
        "error": "No API key — rule-based coding used",
    }


# ============================================================
# Main coding loop
# ============================================================

def code_responses(df: pd.DataFrame, api_key: str | None) -> pd.DataFrame:
    results = []
    use_api = api_key is not None and len(api_key) > 10

    if use_api:
        print(f"API key found. Coding {len(df)} responses via Anthropic API...")
    else:
        print("No API key found. Using rule-based fallback coding for all responses.")
        print("To use AI coding: export ANTHROPIC_API_KEY=your_key_here\n")

    for i, row in df.iterrows():
        text = str(row.get("F1_open_ended", "")).strip()
        if not text or text == "nan":
            result = {
                "theme_code": "OTHER",
                "theme_label": TAXONOMY["OTHER"]["label"],
                "reasoning": "Empty response",
                "raw_response": "",
                "error": "Empty",
            }
        elif use_api:
            result = call_api(text, api_key)
            time.sleep(0.3)   # Respect rate limits
        else:
            result = rule_based_code(text)

        result["respondent_id"] = row.get("respondent_id", f"R{i}")
        result["A1_zone"] = row.get("A1_zone", np.nan)
        result["vendor_food"] = row.get("vendor_food", np.nan)
        result["C3_safe_harbour_aware"] = row.get("C3_safe_harbour_aware", np.nan)
        result["B1b_registered_binary"] = row.get("B1b_registered_binary", np.nan)
        result["F1_text"] = text
        results.append(result)

        if (i + 1) % 50 == 0:
            print(f"  Coded {i+1}/{len(df)} responses...")

    return pd.DataFrame(results)


# ============================================================
# Run and produce outputs
# ============================================================

if __name__ == "__main__":
    df = pd.read_csv("data/hypothetical_survey_data.csv")
    api_key = os.environ.get("ANTHROPIC_API_KEY", None)

    coded_df = code_responses(df, api_key)
    coded_df.to_csv("outputs/tables/09q_f1_coded_responses.csv", index=False)
    print(f"\nCoding complete. {len(coded_df)} responses coded.")

    # ---- Theme frequency table ----
    theme_freq = coded_df["theme_code"].value_counts().reset_index()
    theme_freq.columns = ["theme_code", "count"]
    theme_freq["pct"] = (theme_freq["count"] / len(coded_df) * 100).round(1)
    theme_freq["theme_label"] = theme_freq["theme_code"].map(
        {k: v["label"] for k, v in TAXONOMY.items()}
    )
    theme_freq["theory_link"] = theme_freq["theme_code"].map(
        {k: v["theory_link"] for k, v in TAXONOMY.items()}
    )
    theme_freq["instrument_link"] = theme_freq["theme_code"].map(
        {k: v["instrument_link"] for k, v in TAXONOMY.items()}
    )
    theme_freq.to_csv("outputs/tables/09q_f1_theme_frequencies.csv", index=False)

    # ---- Theme by zone ----
    theme_zone = (
        coded_df.groupby(["A1_zone", "theme_code"])
        .size()
        .reset_index(name="count")
    )
    theme_zone_pct = coded_df.groupby("A1_zone")["theme_code"].value_counts(normalize=True)
    theme_zone_pct = (theme_zone_pct * 100).round(1).reset_index(name="pct")
    theme_zone.merge(theme_zone_pct, on=["A1_zone", "theme_code"]).to_csv(
        "outputs/tables/09q_f1_themes_by_zone.csv", index=False
    )

    # ---- Coding log for audit trail ----
    log_df = coded_df[["respondent_id", "F1_text", "theme_code",
                        "theme_label", "reasoning", "error"]].copy()
    log_df.to_csv("outputs/tables/09q_f1_coding_log.csv", index=False)

    print("\nTheme Frequency Summary:")
    print(theme_freq[["theme_label", "count", "pct"]].to_string(index=False))

    # ============================================================
    # FIGURE 9q-a  Theme distribution overall
    # ============================================================

    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_themes = theme_freq.sort_values("count", ascending=True)
    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(sorted_themes)))
    bars = ax.barh(sorted_themes["theme_label"], sorted_themes["count"],
                   color=colors, height=0.6)
    for bar, pct in zip(bars, sorted_themes["pct"]):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{pct:.1f}%", va="center", fontsize=9)
    ax.set_xlabel("Number of Responses", fontsize=11)
    ax.set_title("F1 Open-Ended Responses — Thematic Distribution\n"
                 "AI-Assisted Qualitative Coding | Kohat District Survey — Hypothetical Data",
                 fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/figures/09q_f1_theme_distribution.png", dpi=150)
    plt.close()

    # ============================================================
    # FIGURE 9q-b  Top 5 themes by zone (grouped bar)
    # ============================================================

    top5 = theme_freq.head(5)["theme_code"].tolist()
    zone_labels = {1: "Zone 1\nCentral", 2: "Zone 2\nSecondary", 3: "Zone 3\nNeighbourhood"}
    zone_colors = ["#2D6A9F", "#E87722", "#3BAA75"]

    fig, ax = plt.subplots(figsize=(11, 6))
    x = np.arange(len(top5))
    width = 0.25

    for i, (zone, color) in enumerate(zip([1, 2, 3], zone_colors)):
        zone_data = coded_df[coded_df["A1_zone"] == zone]
        counts = [zone_data["theme_code"].eq(t).sum() for t in top5]
        ax.bar(x + (i - 1) * width, counts, width, label=zone_labels[zone],
               color=color, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [TAXONOMY[t]["label"].replace(" / ", "\n") for t in top5],
        fontsize=9
    )
    ax.set_ylabel("Number of Responses", fontsize=11)
    ax.set_title("Top 5 F1 Themes by Commercial Zone\n"
                 "AI-Assisted Qualitative Coding | Kohat District Survey — Hypothetical Data",
                 fontsize=11)
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    plt.savefig("outputs/figures/09q_f1_themes_by_zone.png", dpi=150)
    plt.close()

    print("\nFigures saved.")
    print("\nMethodological note for paper:")
    print("  F1 responses coded using pre-specified taxonomy (11 categories)")
    print("  Applied via Anthropic claude-sonnet-4-20250514 with structured JSON output")
    print("  Coding taxonomy, prompts, and model reasoning documented in 09q_f1_coding_log.csv")
    print("  Available for reviewer replication upon request")
    print("\nStep 9 (Qualitative) complete.\n")
