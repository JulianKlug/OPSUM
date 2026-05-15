"""
Compute the dataset-richness statistics cited in the END paper revision
(paragraph added under "Study population" in response to the reviewer
asking for examples of the number of datapoints extracted per variable).

All counts exclude imputed values:
  - Lab counts come from `logs/descriptive_stats.csv`, which counts raw
    measurements before any imputation.
  - Time-series counts (vitals, NIHSS, imaging) come from
    `preprocessed_features.csv` filtered to `source == 'EHR'` for EHR
    signals and `source == 'stroke_registry'` for static registry-derived
    variables. The three imputed sources
    (`*_locf_imputed`, `*_pop_imputed`, `*_pop_imputed_locf_imputed`)
    are excluded.

Usage:
    python end_paper_dataset_richness.py
"""

from pathlib import Path

import pandas as pd

DATA_DIR = Path(
    "/mnt/data1/klug/datasets/opsum/short_term_outcomes/with_imaging/"
    "gsu_Extraction_20220815_prepro_30012026_154047"
)
FEATURES_CSV = DATA_DIR / "preprocessed_features_30012026_154047.csv"
LOGS_DIR = DATA_DIR / "logs_30012026_154047"
DESCRIPTIVE_STATS_CSV = LOGS_DIR / "descriptive_stats.csv"
MEDIAN_OBS_CSV = LOGS_DIR / "median_observations_per_case_admission_id.csv"

VITALS_VARIABLES = [
    "median_heart_rate",
    "median_systolic_blood_pressure",
    "median_diastolic_blood_pressure",
    "median_mean_blood_pressure",
    "median_oxygen_saturation",
    "median_respiratory_rate",
]
NIHSS_VARIABLE = "median_NIHSS"

LAB_EXAMPLES = [
    "sodium",
    "potassium",
    "hematocrite",
    "hemoglobine",
    "creatinine",
    "leucocytes",
    "glucose",
    "INR",
    "proteine C-reactive",
]

CTP_QUANTITATIVE_VARIABLES = [
    "tmax_gt_4",
    "tmax_gt_6",
    "tmax_gt_8",
    "tmax_gt_10",
    "cbf_lt_20",
    "cbf_lt_30",
    "cbf_lt_34",
    "cbf_lt_38",
    "cbv_lt_34",
    "cbv_lt_38",
    "cbv_lt_42",
]
CTP_QUALITATIVE_VARIABLES = [
    "hypoperfusion_with_mismatch",
    "hypoperfusion_without_mismatch",
]
CTA_VARIABLES = [
    "vascular_occlusion",
    "vascular_stenosis_over_50p",
]


def load_non_imputed_feature_counts() -> pd.Series:
    """Return per-variable counts from the long-format features table,
    restricted to non-imputed sources (`EHR` and `stroke_registry`).
    """
    keep_sources = {"EHR", "stroke_registry"}
    counts = {}
    chunksize = 1_000_000
    for chunk in pd.read_csv(
        FEATURES_CSV,
        usecols=["sample_label", "source"],
        chunksize=chunksize,
    ):
        chunk = chunk[chunk["source"].isin(keep_sources)]
        for label, n in chunk["sample_label"].value_counts().items():
            counts[label] = counts.get(label, 0) + int(n)
    return pd.Series(counts).sort_values(ascending=False)


def main() -> None:
    print("=" * 78)
    print("END paper -- dataset-richness statistics (paragraph under Study population)")
    print("=" * 78)

    # --- Cohort headline figures ---------------------------------------------
    # 2657 admissions x 72 h = 191,304 h ~ "190,000 hours of monitoring".
    # The 2.29 Mio total datapoint figure cited in the abstract is the upstream
    # raw extraction total; we reproduce the post-pipeline lower bound here
    # for cross-check.
    n_admissions = 2657
    print(f"\nAdmissions: {n_admissions}")
    print(f"Hours of monitoring (admissions * 72h): {n_admissions * 72:,}")

    print("\n--- Per-variable non-imputed counts from preprocessed_features.csv ---")
    feature_counts = load_non_imputed_feature_counts()
    print(f"Total non-imputed rows (EHR + stroke_registry): {int(feature_counts.sum()):,}")
    print(f"Unique non-imputed variables: {feature_counts.shape[0]}")

    # --- Vital signs (hourly summary counts) ---------------------------------
    print("\nVital signs -- hourly windows with >=1 raw measurement:")
    for var in VITALS_VARIABLES:
        print(f"  {var:>35}: {feature_counts.get(var, 0):>8,}")
    print(f"  {NIHSS_VARIABLE:>35}: {feature_counts.get(NIHSS_VARIABLE, 0):>8,}")

    # --- Lab examples (raw measurement counts) -------------------------------
    print("\nLabs -- raw measurement counts (from logs/descriptive_stats.csv):")
    descriptive = pd.read_csv(DESCRIPTIVE_STATS_CSV)
    descriptive_idx = descriptive.set_index("dosage_label")["count"].astype(int)
    for lab in LAB_EXAMPLES:
        print(f"  {lab:>25}: {descriptive_idx.get(lab, 0):>8,}")

    print("\nLabs -- median observations per admission (logs/median_observations_per_case_admission_id.csv):")
    median_obs = pd.read_csv(MEDIAN_OBS_CSV).set_index("dosage_label")[
        "median_observations_per_case_admission_id"
    ]
    for lab in LAB_EXAMPLES:
        if lab in median_obs.index:
            print(f"  {lab:>25}: {median_obs[lab]:>4}")

    # --- Imaging -------------------------------------------------------------
    print("\nImaging -- non-imputed counts per variable:")
    print("  CTA-derived (source=stroke_registry):")
    cta_total = 0
    for var in CTA_VARIABLES:
        n = int(feature_counts.get(var, 0))
        cta_total += n
        pct = 100 * n / n_admissions
        print(f"    {var:>32}: {n:>6,}  ({pct:.0f}% of admissions)")

    print("  Qualitative perfusion CT (source=stroke_registry):")
    ctp_qual_total = 0
    for var in CTP_QUALITATIVE_VARIABLES:
        n = int(feature_counts.get(var, 0))
        ctp_qual_total += n
        pct = 100 * n / n_admissions
        print(f"    {var:>32}: {n:>6,}  ({pct:.0f}% of admissions)")

    print("  Quantitative perfusion CT (source=EHR, thresholded volumes):")
    ctp_quant_total = 0
    ctp_quant_counts = []
    for var in CTP_QUANTITATIVE_VARIABLES:
        n = int(feature_counts.get(var, 0))
        ctp_quant_total += n
        ctp_quant_counts.append(n)
        pct = 100 * n / n_admissions
        print(f"    {var:>32}: {n:>6,}  ({pct:.0f}% of admissions)")

    n_imaging_variables = (
        len(CTA_VARIABLES) + len(CTP_QUALITATIVE_VARIABLES) + len(CTP_QUANTITATIVE_VARIABLES)
    )
    imaging_total = cta_total + ctp_qual_total + ctp_quant_total

    print("\nImaging summary:")
    print(f"  Total imaging-derived variables: {n_imaging_variables}")
    print(f"  Total non-imputed imaging datapoints: {imaging_total:,}")
    print(
        f"  Quantitative perfusion CT thresholds range: "
        f"{min(ctp_quant_counts):,}-{max(ctp_quant_counts):,}"
    )


if __name__ == "__main__":
    main()
