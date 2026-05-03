"""
feature_importance.py
---------------------
Analyses and visualises which astrophysical features most strongly
influence the habitability prediction, using:
  1. Random Forest feature importances (Gini impurity reduction)
  2. Logistic Regression coefficient magnitudes
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import joblib
import os

PROCESSED_PATH = "data/processed.csv"
OUT_DIR        = "outputs"

FEATURE_COLS = [
    "log_orbper",
    "pl_rade",
    "st_teff",
    "pl_orbsmax",
    "log_insol",
    "pl_eqt",
    "hz_proximity",
]

FEATURE_LABELS = {
    "log_orbper":       "Orbital Period (log)",
    "pl_rade":          "Planetary Radius",
    "st_teff":          "Stellar Temperature",
    "pl_orbsmax":       "Semi-Major Axis",
    "log_insol":        "Insolation Flux (log)",
    "pl_eqt":           "Equilibrium Temperature",
    "hz_proximity":     "HZ Proximity Score",
}


def rf_importance_plot(rf_model, ax):
    importances = rf_model.feature_importances_
    std         = np.std(
        [tree.feature_importances_ for tree in rf_model.estimators_], axis=0
    )
    indices = np.argsort(importances)[::-1]

    labels = [FEATURE_LABELS[FEATURE_COLS[i]] for i in indices]
    vals   = importances[indices]
    errs   = std[indices]

    colors = plt.cm.viridis(np.linspace(0.2, 0.85, len(labels)))
    bars   = ax.barh(labels[::-1], vals[::-1], xerr=errs[::-1],
                     color=colors, edgecolor="white", linewidth=0.6,
                     capsize=4, error_kw={"elinewidth": 1.2, "ecolor": "#555"})

    ax.set_xlabel("Mean Decrease in Impurity", fontsize=11)
    ax.set_title("Random Forest — Feature Importance", fontsize=13, fontweight="bold")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    print("\nRandom Forest Feature Importances:")
    for i in indices:
        print(f"  {FEATURE_LABELS[FEATURE_COLS[i]]:<35} {importances[i]:.4f}")


def lr_coefficient_plot(lr_model, scaler, ax):
    coefs  = lr_model.coef_[0]
    labels = [FEATURE_LABELS[c] for c in FEATURE_COLS]
    indices = np.argsort(np.abs(coefs))

    vals   = coefs[indices]
    lbls   = [labels[i] for i in indices]
    colors = ["#d62728" if v < 0 else "#2ca02c" for v in vals]

    ax.barh(lbls, vals, color=colors, edgecolor="white", linewidth=0.6)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (standardised)", fontsize=11)
    ax.set_title("Logistic Regression — Feature Coefficients",
                 fontsize=13, fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.4)

    # Legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#2ca02c", label="Increases habitability probability"),
        Patch(facecolor="#d62728", label="Decreases habitability probability"),
    ]
    ax.legend(handles=legend_elements, fontsize=9, loc="lower right")

    print("\nLogistic Regression Coefficients (standardised features):")
    for i in np.argsort(np.abs(coefs))[::-1]:
        print(f"  {labels[i]:<35} {coefs[i]:+.4f}")


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    rf_model = joblib.load(f"{OUT_DIR}/random_forest.pkl")
    lr_model = joblib.load(f"{OUT_DIR}/logistic_regression.pkl")
    scaler   = joblib.load(f"{OUT_DIR}/scaler.pkl")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Exoplanet Habitability — Feature Importance Analysis",
                 fontsize=15, fontweight="bold", y=1.01)

    rf_importance_plot(rf_model, axes[0])
    lr_coefficient_plot(lr_model, scaler, axes[1])

    plt.tight_layout()
    out_path = f"{OUT_DIR}/feature_importance.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"\nPlot saved → {out_path}")


if __name__ == "__main__":
    main()
