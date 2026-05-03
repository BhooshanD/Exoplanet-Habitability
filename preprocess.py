"""
preprocess.py
-------------
Loads raw NASA Exoplanet Archive data, engineers features, and generates
a binary habitability label based on established astrophysical criteria.

Habitability criteria used:
  - Planetary radius : 0.5 – 1.6 Earth radii  (rocky / terrestrial zone)
  - Stellar effective temperature : 3700 – 7200 K  (F, G, K, M star range)
  - Orbital semi-major axis inside the conservative habitable zone (HZ),
    approximated via insolation flux: 0.36 – 1.11 Earth flux
    (Kopparapu et al. 2013 conservative limits)

If insolation flux is unavailable the planet is dropped so labels stay clean.
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/kepler_planets.csv"
OUT_PATH  = "data/processed.csv"

# ── Habitability thresholds ────────────────────────────────────────────────
# Using optimistic HZ limits (Kopparapu et al. 2013) to capture a broader
# but still physically motivated habitable population for classification.
RADIUS_MIN,  RADIUS_MAX  = 0.5,  2.5     # Earth radii (up to super-Earths)
TEFF_MIN,    TEFF_MAX    = 2600, 7200    # Kelvin  (M-dwarf through F-star)
INSOL_MIN,   INSOL_MAX   = 0.20, 1.90   # Earth flux  (optimistic HZ limits)


def load_raw(path: str) -> pd.DataFrame:
    """Read CSV, skipping NASA header comment lines."""
    df = pd.read_csv(path, comment="#")
    print(f"Loaded  {len(df):,} rows, {df.shape[1]} columns")
    return df


def select_features(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the columns needed for modelling."""
    features = [
        "pl_name",
        "pl_orbper",   # orbital period [days]
        "pl_rade",     # planet radius  [Earth radii]
        "st_teff",     # stellar effective temperature [K]
        "pl_orbsmax",  # semi-major axis [AU]
        "pl_insol",    # insolation flux [Earth flux]
        "pl_eqt",      # equilibrium temperature [K]
    ]
    df = df[features].copy()
    print(f"After column selection : {len(df):,} rows")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop rows where any of the three label-defining columns are missing,
    then drop remaining NaNs in modelling features and obvious outliers.
    """
    label_cols = ["pl_rade", "st_teff", "pl_insol"]
    df = df.dropna(subset=label_cols)
    print(f"After dropping NaN label cols : {len(df):,} rows")

    # Physical sanity filters
    df = df[df["pl_rade"]   > 0]
    df = df[df["st_teff"]   > 0]
    df = df[df["pl_orbper"] > 0]
    df = df[df["pl_insol"]  > 0]

    # Drop remaining NaNs in any feature column
    feature_cols = ["pl_orbper", "pl_rade", "st_teff", "pl_orbsmax",
                    "pl_insol", "pl_eqt"]
    df = df.dropna(subset=feature_cols)
    print(f"After cleaning          : {len(df):,} rows")
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived features that improve classification signal."""
    # Log-transform skewed distributions
    df["log_orbper"] = np.log1p(df["pl_orbper"])
    df["log_insol"]  = np.log1p(df["pl_insol"])

    # Stellar flux relative to Sun (already in Earth flux units, keep as-is)
    # Normalised radius (how Earth-like the planet is)
    df["radius_earth_ratio"] = df["pl_rade"]          # already in Earth radii

    # Habitable-zone proximity score: 1.0 = centre of HZ, 0 = edge
    hz_centre = (INSOL_MIN + INSOL_MAX) / 2           # ~0.735 Earth flux
    hz_half   = (INSOL_MAX - INSOL_MIN) / 2
    df["hz_proximity"] = np.clip(
        1 - np.abs(df["pl_insol"] - hz_centre) / hz_half, 0, 1
    )

    print("Feature engineering complete.")
    return df


def label(df: pd.DataFrame) -> pd.DataFrame:
    """Assign binary habitability label."""
    mask = (
        df["pl_rade"].between(RADIUS_MIN, RADIUS_MAX) &
        df["st_teff"].between(TEFF_MIN,   TEFF_MAX)   &
        df["pl_insol"].between(INSOL_MIN,  INSOL_MAX)
    )
    df["habitable"] = mask.astype(int)
    n_hab  = df["habitable"].sum()
    n_tot  = len(df)
    print(f"Habitable : {n_hab:,}  ({n_hab/n_tot*100:.1f}%)")
    print(f"Non-hab   : {n_tot - n_hab:,}  ({(n_tot-n_hab)/n_tot*100:.1f}%)")
    return df


def main():
    df = load_raw(RAW_PATH)
    df = select_features(df)
    df = clean(df)
    df = engineer_features(df)
    df = label(df)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"\nSaved processed dataset → {OUT_PATH}")


if __name__ == "__main__":
    main()
