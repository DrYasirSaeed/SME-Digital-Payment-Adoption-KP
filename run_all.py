"""
run_all.py
Master runner — executes all 10 analytical steps in sequence.

Digital Payment Mandates and SME Financial Inclusion in KP
Dr. Yasir Saeed, KUST ORIC Research Grant 2025-26

Usage (from the project root directory):
    python run_all.py

To use the real cleaned fieldwork dataset instead of hypothetical data:
    Replace data/hypothetical_survey_data.csv with the cleaned field data
    (same column names, same coding conventions) and run python run_all.py again.
"""

import subprocess
import sys
import os
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))

STEPS = [
    ("Data Generation",      "data/generate_hypothetical_data.py"),
    ("Step 1-2: Descriptives and Cross-tabs",    "scripts/01_descriptives.py"),
    ("Step 3: Primary Probit",                   "scripts/02_probit_primary.py"),
    ("Step 4-5: Factor Analysis",                "scripts/03_factor_analysis.py"),
    ("Step 6: Ordered Logit",                    "scripts/04_ordered_logit.py"),
    ("Step 7: Heterogeneity Analysis",           "scripts/05_heterogeneity.py"),
    ("Step 8: Mediation Analysis",               "scripts/06_mediation.py"),
    ("Step 9: Coordination Wedge",               "scripts/07_coordination_wedge.py"),
    ("Step 10: Policy Simulation",               "scripts/08_policy_simulation.py"),
    ("Step 11: Qualitative Coding (F1)",         "scripts/09_qualitative_coding.py"),
]


def run_step(name, script):
    print(f"\n{'='*65}")
    print(f"  {name}")
    print(f"  Script: {script}")
    print(f"{'='*65}")
    t0 = time.time()
    result = subprocess.run(
        [sys.executable, script],
        capture_output=False,
        text=True,
    )
    elapsed = time.time() - t0
    if result.returncode != 0:
        print(f"\n  ERROR in {name} (exit code {result.returncode})")
        return False
    print(f"\n  Completed in {elapsed:.1f}s")
    return True


if __name__ == "__main__":
    print("\nDigital Payment Mandates and SME Financial Inclusion in KP")
    print("Dr. Yasir Saeed — KUST ORIC Research Grant 2025-26")
    print("Running full analytical pipeline...\n")

    all_ok = True
    for name, script in STEPS:
        ok = run_step(name, script)
        if not ok:
            print(f"\nPipeline halted at: {name}")
            sys.exit(1)

    print("\n" + "="*65)
    print("  PIPELINE COMPLETE — all outputs written to:")
    print("  outputs/tables/   — CSV result tables")
    print("  outputs/figures/  — PNG charts")
    print("="*65 + "\n")
