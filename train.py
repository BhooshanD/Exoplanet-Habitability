"""
train.py
--------
Trains Logistic Regression and Random Forest classifiers to predict
exoplanet habitability. Reports accuracy and F1-score for both models
and saves results to outputs/.
"""

import pandas as pd
import numpy as np
import os
import json

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing   import StandardScaler
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier
from sklearn.metrics         import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import joblib

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
TARGET_COL = "habitable"

RANDOM_STATE = 42


def load_data():
    df = pd.read_csv(PROCESSED_PATH)
    X  = df[FEATURE_COLS]
    y  = df[TARGET_COL]
    print(f"Dataset : {len(df):,} samples  |  "
          f"Habitable: {y.sum():,}  Non-hab: {(1-y).sum():,}")
    return X, y


def build_models():
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def evaluate(name, model, X_train, X_test, y_train, y_test, scaler=None):
    if scaler:
        X_tr = scaler.transform(X_train)
        X_te = scaler.transform(X_test)
    else:
        X_tr, X_te = X_train, X_test

    model.fit(X_tr, y_train)
    y_pred = model.predict(X_te)

    acc = accuracy_score(y_test, y_pred)
    f1  = f1_score(y_test, y_pred, average="binary")

    print(f"\n{'='*50}")
    print(f"  {name}")
    print(f"{'='*50}")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  F1-Score : {f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=["Non-Habitable", "Habitable"]))
    return acc, f1, y_pred


def cross_validate(name, model, X, y, scaler=None):
    X_use = scaler.transform(X) if scaler else X
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_use, y, cv=cv, scoring="f1", n_jobs=-1)
    print(f"  5-Fold CV F1 ({name}): {scores.mean():.4f} ± {scores.std():.4f}")
    return scores.mean(), scores.std()


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train : {len(X_train):,}  |  Test : {len(X_test):,}")

    scaler = StandardScaler()
    scaler.fit(X_train)

    models  = build_models()
    results = {}

    for name, model in models.items():
        # Logistic Regression needs scaling; RF does not strictly require it
        use_scaler = (name == "Logistic Regression")
        acc, f1, _ = evaluate(
            name, model,
            X_train, X_test, y_train, y_test,
            scaler=scaler if use_scaler else None,
        )
        cv_mean, cv_std = cross_validate(
            name, model, X_train, y_train,
            scaler=scaler if use_scaler else None,
        )
        results[name] = {
            "accuracy": round(acc, 4),
            "f1_score": round(f1,  4),
            "cv_f1_mean": round(cv_mean, 4),
            "cv_f1_std":  round(cv_std,  4),
        }
        # Save model artefact
        slug = name.lower().replace(" ", "_")
        joblib.dump(model,  f"{OUT_DIR}/{slug}.pkl")
        print(f"  Saved model → {OUT_DIR}/{slug}.pkl")

    joblib.dump(scaler, f"{OUT_DIR}/scaler.pkl")

    # Persist results summary
    with open(f"{OUT_DIR}/results.json", "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n\nResults summary saved → {OUT_DIR}/results.json")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
