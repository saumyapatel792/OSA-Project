"""
mesa_loader.py  ·  OSA-CDSS
============================
Handles loading and preprocessing the MESA Sleep Study dataset
(Multi-Ethnic Study of Atherosclerosis — BioLINCC / NHLBI).

Download instructions
---------------------
1. Register at: https://biolincc.nhlbi.nih.gov/studies/mesa/
2. Request access to the MESA Sleep Study ancillary dataset.
3. After approval, download the CSV (usually named mesa-sleep-dataset-*.csv
   or similar). Place the file in the same folder as this script.
4. Set MESA_CSV_PATH below to the exact filename you downloaded.

Column mapping
--------------
MESA column          →  Our model feature
────────────────────────────────────────────
age5c                →  Age
bmi5c                →  BMI
neckgirth5c          →  Neck   (neck circumference, cm)
slpapnea5            →  Snoring proxy (0-4 snoring/breathing scale)
gender1              →  Gender (1=Male, 2=Female)
avgO2nrem5 / avgO2   →  SpO2   (average overnight oxygen saturation)
diabetes5c           →  Diabetes (0/1)
ahi_c0h3a            →  AHI    (events/hour, AASM 3% desaturation rule)

AHI → Risk mapping  (AASM 2023 guidelines)
-------------------------------------------
AHI  < 5   →  Low       (Normal)
AHI  5–14  →  Moderate  (Mild OSA)
AHI 15–29  →  High      (Moderate OSA)
AHI >= 30  →  Severe    (Severe OSA)
"""

import os
import pandas as pd
import numpy as np

# ── CONFIGURATION ─────────────────────────────────────────────────────────────
# Set this to the path of your downloaded MESA CSV file.
# Accepted filenames (any of these will be detected automatically):
MESA_CSV_CANDIDATES = [
    "mesa-sleep-dataset-0.5.0.csv",
    "mesa_sleep.csv",
    "mesa-sleep.csv",
    "MESASleep.csv",
    "mesa_dataset.csv",
    "mesa.csv",
]

# ── AHI → RISK LABEL ─────────────────────────────────────────────────────────
def ahi_to_risk(ahi: float) -> str:
    """Convert AHI value to OSA risk category using AASM 2023 guidelines."""
    if ahi < 5:
        return "Low"
    elif ahi < 15:
        return "Moderate"
    elif ahi < 30:
        return "High"
    else:
        return "Severe"


def ahi_to_binary(ahi: float) -> int:
    """Binary label for model training: 0 = No/Mild OSA, 1 = Moderate/Severe OSA."""
    return 1 if ahi >= 15 else 0


# ── MESA COLUMN DETECTION ─────────────────────────────────────────────────────
# MESA distributes data with slightly different column names across versions.
# This map lists all known aliases for each feature we need.
COLUMN_ALIASES = {
    "Age":      ["age5c", "age5", "age", "Age"],
    "BMI":      ["bmi5c", "bmi5", "bmi", "BMI"],
    "Neck":     ["neckgirth5c", "neckgirth5", "neck5c", "neck5", "neck", "Neck"],
    "Snoring":  ["slpapnea5", "snoring5", "snoring", "slpapnea", "Snoring"],
    "Gender":   ["gender1", "gender", "sex", "Gender", "Sex"],
    "SpO2":     ["avgO2nrem5", "avgO2nrem", "avgo2nrem5", "avgo2nrem",
                 "avgO25", "avgo25", "spo2", "SpO2"],
    "Diabetes": ["diabetes5c", "diabetes5", "diabetes", "Diabetes"],
    "AHI":      ["ahi_c0h3a", "ahi_c0h4a", "ahi5", "ahi", "AHI",
                 "ahiu5c", "ahi_a0h3a"],
}


def _find_column(df_cols: list, feature: str) -> str | None:
    """Return the first matching column alias found in the DataFrame."""
    aliases = COLUMN_ALIASES.get(feature, [feature])
    df_lower = {c.lower(): c for c in df_cols}
    for alias in aliases:
        if alias in df_cols:
            return alias
        if alias.lower() in df_lower:
            return df_lower[alias.lower()]
    return None


def _find_mesa_file() -> str | None:
    """Search current directory and parent for a MESA CSV file."""
    search_dirs = [".", "data", "dataset", "datasets", "mesa"]
    for d in search_dirs:
        for name in MESA_CSV_CANDIDATES:
            path = os.path.join(d, name)
            if os.path.isfile(path):
                return path
        # Also try any CSV in that dir with 'mesa' in the name
        if os.path.isdir(d):
            for f in os.listdir(d):
                if f.lower().endswith(".csv") and "mesa" in f.lower():
                    return os.path.join(d, f)
    return None


# ── MAIN LOADER ───────────────────────────────────────────────────────────────
def load_mesa(csv_path: str = None) -> tuple[pd.DataFrame, dict]:
    """
    Load and preprocess the MESA Sleep Study dataset.

    Parameters
    ----------
    csv_path : str, optional
        Explicit path to the MESA CSV. If None, auto-detects.

    Returns
    -------
    df : pd.DataFrame
        Clean DataFrame with columns:
        Age, BMI, Neck, Snoring, Gender (int 0/1), SpO2,
        Diabetes (int 0/1), AHI, Risk (str), OSA (int 0/1)

    info : dict
        Metadata about the load: n_rows, n_dropped, columns_mapped,
        ahi_distribution, source
    """
    # ── 1. Find the file ─────────────────────────────────────────────────────
    path = csv_path or _find_mesa_file()
    if path is None or not os.path.isfile(path):
        raise FileNotFoundError(
            "MESA CSV not found.\n\n"
            "Please download the dataset from:\n"
            "  https://biolincc.nhlbi.nih.gov/studies/mesa/\n\n"
            "Then place the CSV in the project folder and name it:\n"
            "  mesa-sleep-dataset-0.5.0.csv  (or any name containing 'mesa')\n\n"
            "See mesa_loader.py for full instructions."
        )

    # ── 2. Load raw CSV ──────────────────────────────────────────────────────
    raw = pd.read_csv(path, low_memory=False)
    n_raw = len(raw)
    cols = raw.columns.tolist()

    # ── 3. Map columns ───────────────────────────────────────────────────────
    col_map = {}
    missing_features = []
    for feature in ["Age", "BMI", "Neck", "Snoring", "Gender", "SpO2", "Diabetes", "AHI"]:
        found = _find_column(cols, feature)
        if found:
            col_map[found] = feature
        else:
            missing_features.append(feature)

    if missing_features:
        raise ValueError(
            f"Could not find columns for: {missing_features}\n"
            f"Available columns in your CSV: {cols[:30]}\n\n"
            "Please open mesa_loader.py and add the correct column names "
            "to COLUMN_ALIASES for the missing features."
        )

    df = raw[list(col_map.keys())].rename(columns=col_map).copy()

    # ── 4. Type coercion ─────────────────────────────────────────────────────
    numeric_cols = ["Age", "BMI", "Neck", "Snoring", "SpO2", "Diabetes", "AHI"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # ── 5. Gender encoding ───────────────────────────────────────────────────
    # MESA encodes gender as 1=Male, 2=Female; convert to 1=Male, 0=Female
    if df["Gender"].max() > 1:
        df["Gender"] = (df["Gender"] == 1).astype(int)   # 1=Male, 0=Female
    else:
        # Already 0/1 — keep as-is (0=Female, 1=Male)
        df["Gender"] = df["Gender"].astype(int)

    # ── 6. Snoring normalisation ─────────────────────────────────────────────
    # MESA uses a 0–4 sleep-disordered breathing scale.
    # Rescale to 1–5 to match our slider input.
    sn_max = df["Snoring"].max()
    if sn_max <= 4:
        df["Snoring"] = (df["Snoring"] + 1).clip(1, 5)   # 0–4 → 1–5
    df["Snoring"] = df["Snoring"].round().astype("Int64")

    # ── 7. Diabetes encoding ─────────────────────────────────────────────────
    # Ensure binary 0/1 (some versions use 0/1/2; treat >=1 as diabetic)
    df["Diabetes"] = (df["Diabetes"] >= 1).astype(int)

    # ── 8. Drop rows missing critical features ───────────────────────────────
    critical = ["Age", "BMI", "AHI", "SpO2"]
    n_before = len(df)
    df.dropna(subset=critical, inplace=True)
    n_dropped_na = n_before - len(df)

    # ── 9. Impute Neck and Snoring if partially missing ──────────────────────
    df["Neck"] = df["Neck"].fillna(df["Neck"].median())
    df["Snoring"] = df["Snoring"].fillna(df["Snoring"].mode()[0])

    # ── 10. Clip physiological outliers ──────────────────────────────────────
    n_before = len(df)
    df = df[
        (df["Age"].between(18, 90)) &
        (df["BMI"].between(15, 70)) &
        (df["SpO2"].between(60, 100)) &
        (df["AHI"].between(0, 200)) &
        (df["Neck"].between(20, 70))
    ]
    n_dropped_outliers = n_before - len(df)

    # ── 11. Add risk labels ──────────────────────────────────────────────────
    df["Risk"] = df["AHI"].apply(ahi_to_risk)
    df["OSA"]  = df["AHI"].apply(ahi_to_binary)

    # ── 12. Final column order ───────────────────────────────────────────────
    df = df[["Age", "BMI", "Neck", "Snoring", "Gender", "SpO2",
             "Diabetes", "AHI", "Risk", "OSA"]].reset_index(drop=True)

    # ── 13. Build info dict ──────────────────────────────────────────────────
    risk_dist = df["Risk"].value_counts().to_dict()
    info = {
        "source":          path,
        "n_raw":           n_raw,
        "n_loaded":        len(df),
        "n_dropped_na":    n_dropped_na,
        "n_dropped_outliers": n_dropped_outliers,
        "columns_mapped":  col_map,
        "risk_distribution": risk_dist,
        "ahi_mean":        round(df["AHI"].mean(), 2),
        "ahi_median":      round(df["AHI"].median(), 2),
        "spo2_mean":       round(df["SpO2"].mean(), 2),
        "bmi_mean":        round(df["BMI"].mean(), 2),
        "class_balance":   {
            "OSA (≥15 AHI)":    int((df["OSA"] == 1).sum()),
            "No OSA (<15 AHI)": int((df["OSA"] == 0).sum()),
        },
    }

    return df, info


# ── TRAINING SPLIT HELPER ─────────────────────────────────────────────────────
def get_training_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Extract feature matrix X and binary target y from a loaded MESA DataFrame.
    Matches exactly the 7 features expected by the VotingClassifier in app.py.

    Returns
    -------
    X : pd.DataFrame  (Age, BMI, Neck, Snoring, Gender, SpO2, Diabetes)
    y : pd.Series     (0 = No/Mild OSA, 1 = Moderate/Severe OSA)
    """
    feature_cols = ["Age", "BMI", "Neck", "Snoring", "Gender", "SpO2", "Diabetes"]
    X = df[feature_cols].copy()
    y = df["OSA"].copy()
    return X, y


# ── FALLBACK: ENHANCED SYNTHETIC DATA ─────────────────────────────────────────
def generate_realistic_synthetic(n: int = 2000, seed: int = 42) -> pd.DataFrame:
    """
    Generate synthetic training data with realistic clinical correlations.
    Used as fallback when the MESA CSV is not yet available.

    This is substantially better than the original flat random generation
    because it respects known clinical relationships:
      - BMI and neck circumference are positively correlated
      - Male gender increases OSA risk
      - Older age increases OSA risk
      - Higher BMI suppresses SpO2
      - AHI is derived from a weighted clinical model (not random)
    """
    rng = np.random.default_rng(seed)

    # Demographics
    age    = rng.integers(20, 80, n).astype(float)
    gender = rng.integers(0, 2, n)   # 0=Female, 1=Male

    # BMI: normally distributed, slightly right-skewed
    bmi = np.clip(rng.normal(28, 6, n), 16, 55)

    # Neck: correlated with BMI + gender (men have larger necks)
    neck = np.clip(
        bmi * 0.7 + gender * 3.5 + rng.normal(0, 3, n) + 15,
        28, 62
    )

    # Snoring: correlated with BMI, neck, gender
    snoring_score = (bmi / 45) * 2 + (neck / 55) * 1.5 + gender * 0.8 + rng.normal(0, 0.5, n)
    snoring = np.clip(np.round(snoring_score).astype(int), 1, 5)

    # SpO2: inversely related to BMI and AHI risk
    spo2 = np.clip(
        99 - (bmi - 18) * 0.3 - rng.normal(0, 2, n),
        72, 99
    )

    # Diabetes: ~25% prevalence, higher in older/heavier
    diabetes_prob = 0.05 + (age / 200) + (np.clip(bmi - 25, 0, 20) / 100)
    diabetes = (rng.random(n) < diabetes_prob).astype(int)

    # AHI: derived from weighted clinical risk factors (literature-based weights)
    # Key predictors: BMI, neck, male gender, age, SpO2 deficit
    ahi_latent = (
        (bmi   - 25) * 1.2 +          # BMI above normal
        (neck  - 35) * 1.8 +          # Neck circumference
        gender        * 8.0 +          # Male sex
        (age   - 40) * 0.25 +         # Aging
        (100 - spo2)  * 1.5 +          # SpO2 deficit
        snoring        * 2.5 +          # Snoring intensity
        diabetes       * 4.0 +          # Metabolic comorbidity
        rng.normal(0, 6, n)            # Residual variance
    )
    ahi = np.clip(ahi_latent, 0, 100)

    df = pd.DataFrame({
        "Age":      age.astype(int),
        "BMI":      bmi.round(1),
        "Neck":     neck.round(1),
        "Snoring":  snoring,
        "Gender":   gender,
        "SpO2":     spo2.round(1),
        "Diabetes": diabetes,
        "AHI":      ahi.round(1),
        "Risk":     [ahi_to_risk(a) for a in ahi],
        "OSA":      [ahi_to_binary(a) for a in ahi],
    })
    return df


# ── DATASET STATUS CHECK ─────────────────────────────────────────────────────
def check_mesa_available() -> tuple[bool, str]:
    """
    Returns (is_available: bool, message: str).
    Used by app.py to decide whether to load real or synthetic data.
    """
    path = _find_mesa_file()
    if path:
        return True, path
    return False, (
        "MESA dataset not found. Using enhanced synthetic data for training.\n"
        "Download MESA from: https://biolincc.nhlbi.nih.gov/studies/mesa/\n"
        "Place the CSV in the project folder to enable real-data training."
    )
