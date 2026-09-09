

from __future__ import annotations

import io
import json
import os
import re
import sys
import time
import types
import warnings
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# 
# Constants
# 

APP_DIR = os.path.dirname(os.path.abspath('Medicine_Details.csv'))
DEFAULT_CSV = os.path.join(APP_DIR, "Medicine_Details.csv")
MODEL_PATH = os.path.join(APP_DIR, "model.pkl")

RANDOM_STATE = 42
TRAIN_SIZE = 0.70
VAL_SIZE = 0.15
TEST_SIZE = 0.15
CV_FOLDS = 3
MAX_ONEHOT_CATEGORIES = 50

TARGET = "final_price"

RAW_COLUMNS = [
    "disease_name",
    "med_name",
    "final_price",
    "prescription_required",
    "drug_varient",
    "drug_manufacturer",
    "drug_manufacturer_origin",
    "generic_name",
]

NUMERIC_FEATURES = [
    "rx_required",
    "pack_size",
    "strength_value",
    "molecule_count",
    "quantity_value",
]

CATEGORICAL_FEATURES = [
    "disease",
    "dosage_form",
    "manufacturer",
    "origin",
    "strength_unit",
    "molecule",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

FORM_PATTERNS = [
    ("injection", "Injection"),
    ("infusion", "Injection"),
    ("vaccine", "Injection"),
    ("syringe", "Injection"),
    ("prefilled", "Injection"),
    ("vial", "Injection"),
    ("ampoule", "Injection"),
    ("oral solution", "Oral Solution"),
    ("suspension", "Suspension"),
    ("syrup", "Syrup"),
    ("tablet", "Tablet"),
    ("caplet", "Tablet"),
    ("capsule", "Capsule"),
    ("rotacap", "Capsule"),
    ("softgel", "Capsule"),
    ("soft gel", "Capsule"),
    ("inhaler", "Inhaler"),
    ("respule", "Inhaler"),
    ("nebul", "Inhaler"),
    ("spray", "Spray"),
    ("drops", "Drops"),
    ("drop", "Drops"),
    ("face wash", "Face Wash"),
    ("body wash", "Soap"),
    ("cleanser", "Face Wash"),
    ("shampoo", "Shampoo"),
    ("soap", "Soap"),
    ("wash", "Face Wash"),
    ("ointment", "Ointment"),
    ("cream", "Cream"),
    ("gel", "Gel"),
    ("lotion", "Lotion"),
    ("balm", "Balm"),
    ("liniment", "Liniment"),
    ("mousse", "Mousse"),
    ("foam", "Foam"),
    ("paint", "Paint"),
    ("sachet", "Sachet"),
    ("granule", "Granules"),
    ("powder", "Powder"),
    ("suppository", "Suppository"),
    ("pessary", "Suppository"),
    ("patch", "Patch"),
    ("test strip", "Test Strip"),
    ("strip", "Test Strip"),
    ("kit", "Kit"),
    ("oil", "Oil"),
    ("solution", "Solution"),
]

NUMBER_RE = re.compile(r"(\d+(?:\.\d+)?)")
PACK_FROM_NAME_RE = re.compile(r"(\d+)\s*['\u2019]?s\b", re.IGNORECASE)
QUANTITY_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(?:ml|milliliter|litre|liter|gm|grams?|g|l)\b",
    re.IGNORECASE,
)
GENERIC_STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|ug|gm|g|ml|%|iu|units?|billion|million|ppm|au|cu)(?![a-z])",
    re.IGNORECASE,
)
MEDNAME_STRENGTH_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(mg|mcg|gm|%|ml|iu)(?![a-z])", re.IGNORECASE
)
UNIT_NORMALIZATION = {"ug": "mcg", "g": "gm", "unit": "u", "units": "u"}


# 
# Cleaning and parsing
# 


def load_raw_csv(source):
    """Read only the modelling-relevant raw columns as strings."""
    return pd.read_csv(source, dtype=str, usecols=RAW_COLUMNS)


def parse_price(text):
    """Extract the first number from a price string like 'Rs 50.00' or 'Error'."""
    if not isinstance(text, str):
        return np.nan
    match = NUMBER_RE.search(text)
    return float(match.group(1)) if match else np.nan


def extract_dosage_form(varient, med_name):
    """Derive the dosage form from drug_varient, falling back to med_name."""
    varient = varient if isinstance(varient, str) else ""
    med_name = med_name if isinstance(med_name, str) else ""
    match = re.search(r"\d+\s+([A-Za-z]+)\(s\)", varient)
    if match and match.group(1) in ("Tablet", "Capsule"):
        return match.group(1)
    text = (varient + " " + med_name).lower()
    for keyword, label in FORM_PATTERNS:
        if keyword in text:
            return label
    return "Other"


def extract_pack_size(varient, med_name):
    """Parse unit count per pack from the variant string or the medicine name."""
    if isinstance(varient, str):
        match = re.match(r"^\*?\s*(\d+)", varient.strip())
        if match:
            return float(match.group(1))
    if isinstance(med_name, str):
        match = PACK_FROM_NAME_RE.search(med_name)
        if match:
            return float(match.group(1))
    return np.nan


def extract_quantity(med_name):
    """Parse liquid/topical volume or weight from the medicine name."""
    if not isinstance(med_name, str):
        return np.nan
    match = QUANTITY_RE.search(med_name)
    return float(match.group(1)) if match else np.nan


def parse_strength_and_molecule(generic_name, med_name):
    """Extract strength, strength unit, molecule name, and molecule count.

    Tries the generic name first, then falls back to the brand name. When the
    generic name lists several salts separated by '+', only the first one is
    kept as the main molecule and the total count is recorded.
    """
    strength_value = np.nan
    strength_unit = "unknown"
    molecule = "unknown"
    molecule_count = np.nan

    if isinstance(generic_name, str) and generic_name.strip():
        cleaned = generic_name.replace("Generic Name", "").strip()
        match = GENERIC_STRENGTH_RE.search(cleaned)
        if match:
            strength_value = float(match.group(1))
            strength_unit = UNIT_NORMALIZATION.get(
                match.group(2).lower(), match.group(2).lower()
            )
        head = cleaned[: match.start()] if match else cleaned
        head = head.split("+")[0].strip(" +-/,")
        if head:
            molecule = head.lower()
        molecule_count = float(cleaned.count("+") + 1)
    elif isinstance(med_name, str):
        match = MEDNAME_STRENGTH_RE.search(med_name)
        if match:
            strength_value = float(match.group(1))
            strength_unit = UNIT_NORMALIZATION.get(
                match.group(2).lower(), match.group(2).lower()
            )

    return strength_value, strength_unit, molecule, molecule_count


def clean_data(raw):
    """Run cleaning steps and return a tidy table plus a short cleaning log.

    The returned frame contains the modelling features plus the original
    display columns needed for the catalog (med_name, generic_name,
    prescription_required, drug_varient).
    """
    log = []
    df = raw.copy()
    rows_in = len(df)

    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    log.append(f"Duplicate rows removed: {removed} ({removed / max(before, 1) * 100:.1f}%)")

    df[TARGET] = df["final_price"].map(parse_price)
    bad_target = int(df[TARGET].isna().sum())
    df = df.dropna(subset=[TARGET]).copy()
    log.append(f"Rows dropped with missing or unparsable price ('Error'): {bad_target}")

    # Disease: strip trailing '(N)' artefacts like 'Fever (2)'.
    df["disease"] = (
        df["disease_name"].str.replace(r"\s*\(\d+\)\s*$", "", regex=True).str.strip()
    )

    # Prescription flag.
    df["rx_required"] = (
        df["prescription_required"].fillna("").str.strip().eq("Rx required").astype(int)
    )

    # Manufacturer: strip leading 'Mkt: ...' markers.
    df["manufacturer"] = (
        df["drug_manufacturer"]
        .str.replace(r"^\*\s*Mkt:\s*", "", regex=True)
        .str.strip()
    )
    df.loc[df["manufacturer"].isna() | df["manufacturer"].eq(""), "manufacturer"] = "Unknown"

    # Origin: normalise to India / Imported / Unknown.
    origin_raw = (
        df["drug_manufacturer_origin"]
        .str.replace(r"^\*\s*Country of Origin:\s*", "", regex=True)
        .str.strip()
    )
    df["origin"] = np.where(
        origin_raw.isna() | origin_raw.eq("NA") | origin_raw.eq(""),
        "Unknown",
        np.where(origin_raw.eq("India"), "India", "Imported"),
    )

    df["dosage_form"] = [
        extract_dosage_form(v, m)
        for v, m in zip(df["drug_varient"], df["med_name"])
    ]
    df["pack_size"] = [
        extract_pack_size(v, m)
        for v, m in zip(df["drug_varient"], df["med_name"])
    ]
    df["quantity_value"] = df["med_name"].map(extract_quantity)
    parsed = [
        parse_strength_and_molecule(g, m)
        for g, m in zip(df["generic_name"], df["med_name"])
    ]
    df["strength_value"] = [p[0] for p in parsed]
    df["strength_unit"] = [p[1] for p in parsed]
    df["molecule"] = [p[2] for p in parsed]
    df["molecule_count"] = [p[3] for p in parsed]

    log.append(
        "Missing numerics (pack_size, strength_value, molecule_count, "
        "quantity_value): kept as NaN and median-imputed inside the pipeline."
    )
    log.append(
        'Missing categoricals filled with an explicit "unknown" level.'
    )
    log.append(f"Rows in: {rows_in}, rows out: {len(df)}")

    # Keep display columns that the catalog needs alongside the features.
    display_cols = ["med_name", "generic_name", "prescription_required", "drug_varient"]
    return df[FEATURES + [TARGET] + display_cols].reset_index(drop=True), log


# 
# Derived catalog fields: recommendation and usage text
# 


USAGE_BY_FORM = {
    "Tablet": "Take by mouth with a glass of water. Swallow whole unless the label says otherwise. Follow the dose and timing prescribed by your doctor.",
    "Capsule": "Swallow the capsule whole with water. Do not crush or open it unless instructed by your doctor or pharmacist.",
    "Syrup": "Shake well before use. Measure the dose with the provided spoon or cup. Take by mouth as directed.",
    "Oral Solution": "Measure the recommended dose carefully. Take by mouth, with or without food as advised.",
    "Suspension": "Shake the bottle well before each use. Measure the dose accurately and take by mouth.",
    "Injection": "Given by a doctor or nurse into a vein, muscle, or under the skin. Do not attempt to inject yourself unless trained.",
    "Inhaler": "Shake the inhaler, breathe out fully, place the mouthpiece in your mouth, and breathe in deeply while pressing the canister. Hold your breath for a few seconds.",
    "Spray": "Shake if required and spray into the nose or mouth as directed, usually once or twice into each nostril.",
    "Drops": "Instil the prescribed number of drops into the affected eye or ear. Do not touch the tip to the eye or ear.",
    "Ointment": "Apply a thin layer to the affected skin area. Wash hands before and after use unless the hands are the treated area.",
    "Cream": "Apply a thin layer to clean, dry skin on the affected area. Wash hands after use.",
    "Gel": "Apply a thin layer to the affected area and allow it to dry if directed. Wash hands after use.",
    "Lotion": "Shake well and apply to the affected skin using clean hands or a cotton pad.",
    "Balm": "Apply a small amount to the affected area and rub in gently.",
    "Face Wash": "Wet the face, apply a small amount, lather gently, and rinse thoroughly. Use as a daily cleanser.",
    "Soap": "Use as a regular wash on the affected area or as directed for the condition.",
    "Shampoo": "Apply to wet hair, massage into the scalp, leave for the recommended time, and rinse thoroughly.",
    "Powder": "Apply to the affected dry area as directed, usually after bathing or as needed.",
    "Suppository": "Insert into the rectum or vagina as directed. Wash hands before and after use.",
    "Patch": "Apply to clean, dry, hairless skin as directed. Do not cut unless instructed.",
    "Oil": "Apply a small amount to the affected area or use as directed for the condition.",
    "Solution": "Apply or use as directed, depending on the affected area.",
    "Kit": "Use the components as instructed in the package leaflet. Follow the order and method described.",
    "Sachet": "Mix the contents with water or take as directed on the packet. Do not use partial sachets unless advised.",
    "Granules": "Mix with water or food as directed and take by mouth.",
    "Other": "Follow the instructions on the pack or the advice of your doctor and pharmacist.",
}

RECOMMENDATION_BY_DOSAGE_FORM = {
    "Tablet": "Oral solid dosage, convenient for travel and regular dosing. Suitable when a precise, repeatable dose is needed.",
    "Capsule": "Easier to swallow than tablets for some patients and can mask taste. Suitable for medicines that are unstable in liquid form.",
    "Syrup": "Useful for children and people who have difficulty swallowing solids. Dose can be adjusted more flexibly.",
    "Oral Solution": "Fast-acting oral form with flexible dosing. Suitable when exact small doses are needed.",
    "Suspension": "Suitable for medicines that do not dissolve well. Often used in children or for topical/oral use.",
    "Injection": "Rapid effect and reliable absorption. Used when oral administration is not possible or when fast action is needed.",
    "Inhaler": "Delivers medicine directly to the lungs. Preferred for asthma and other respiratory conditions.",
    "Spray": "Quick and convenient for nasal or local use. Useful for allergies and localised symptoms.",
    "Drops": "Targeted application for eyes or ears.Useful when only a small local dose is needed.",
    "Ointment": "Longer contact with the skin than creams. Suitable for thicker or drier skin areas.",
    "Cream": "Absorbs more easily than ointments. Suitable for moist or inflamed skin areas.",
    "Gel": "Cooling and non-greasy feel. Suitable for localised skin applications and pain relief.",
    "Lotion": "Easy to spread over large areas. Suitable for sunburn, rashes, or dry skin over a wide area.",
    "Balm": "Soothes and protects the skin or lips. Suitable for minor irritation and dry areas.",
    "Face Wash": "For daily cleansing of the face. Suitable for acne-prone or sensitive skin as directed.",
    "Soap": "General cleansing and antiseptic use where appropriate.",
    "Shampoo": "For scalp and hair conditions such as dandruff or irritation.",
    "Powder": "Helps keep moist areas dry. Suitable for skin folds and minor irritation.",
    "Suppository": "Useful when oral intake is not possible or for local rectal or vaginal treatment.",
    "Patch": "Provides steady release over time and avoids the stomach. Suitable for long-term pain relief or hormone delivery.",
    "Oil": "Moisturising or medicated application for skin and scalp.",
    "Solution": "Versatile form for local or general use depending on the product.",
    "Kit": "Combines several items for a complete treatment plan. Follow the kit instructions.",
    "Sachet": "Single-dose convenience and portable. Suitable for powders and dry mixes.",
    "Granules": "Easy to mix and take by mouth. Suitable for children and people with swallowing difficulty.",
    "Other": "Check the specific product information for the intended use and benefits.",
}


def build_catalog_columns(df):
    """Add recommendation and usage text to a cleaned catalog frame."""
    df = df.copy()

    df["usage"] = df["dosage_form"].map(USAGE_BY_FORM).fillna(
        "Follow the instructions on the pack or the advice of your doctor and pharmacist."
    )
    df["recommendation"] = df["dosage_form"].map(RECOMMENDATION_BY_DOSAGE_FORM).fillna(
        "Check the specific product information for the intended use and benefits."
    )

    # Strengthen the recommendation with a short disease-specific note when
    # the disease field is meaningful.
    def disease_note(row):
        disease = row.get("disease", "")
        rx = row.get("rx_required", 0)
        notes = []
        if isinstance(disease, str) and disease.strip() and disease.strip().lower() != "unknown":
            notes.append(f"Used for: {disease.strip()}.")
        if int(row.get("rx_required", 0)):
            notes.append("Prescription-only medicine. Use only under medical supervision.")
        else:
            notes.append("Available without prescription. Read the label and use as directed.")
        if isinstance(row.get("origin", ""), str) and row["origin"].strip().lower() == "india":
            notes.append("Manufactured in India.")
        elif isinstance(row.get("origin", ""), str) and row["origin"].strip().lower() == "imported":
            notes.append("Imported product.")
        return " ".join(notes)

    df["note"] = df.apply(disease_note, axis=1)

    return df


# 
# Preprocessing and model pipeline (kept from the original pipeline)
# 


class OutlierCapper(BaseEstimator, TransformerMixin):
    """Clip numeric columns to IQR fences learned at fit time."""

    def fit(self, X, y=None):
        array = np.asarray(X, dtype=float)
        q1 = np.nanquantile(array, 0.25, axis=0)
        q3 = np.nanquantile(array, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - 1.5 * iqr
        self.upper_ = q3 + 1.5 * iqr
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            raise ValueError("input_features is required for OutlierCapper")
        return np.asarray(input_features, dtype=object)


# Pickle identity fix for the Streamlit runtime.
if __name__ == "__main__":
    _script_namespace = globals()
    _registered = sys.modules.get("medicine")
    if (
        _registered is None
        or getattr(_registered, "OutlierCapper", None)
        is not _script_namespace.get("OutlierCapper")
    ):
        _mirror = types.ModuleType("medicine")
        _mirror.__dict__.update(_script_namespace)
        sys.modules["medicine"] = _mirror
    OutlierCapper.__module__ = "medicine"


def build_preprocessor():
    """Build the ColumnTransformer for numeric and categorical features."""
    numeric_branch = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("capper", OutlierCapper()),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_branch = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    max_categories=MAX_ONEHOT_CATEGORIES,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [
            ("num", numeric_branch, NUMERIC_FEATURES),
            ("cat", categorical_branch, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def candidate_models():
    """Return the candidate regressors keyed by display name."""
    return {
        "Linear Regression": LinearRegression(),
        "Ridge": Ridge(random_state=RANDOM_STATE),
        "Lasso": Lasso(max_iter=20000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestRegressor(
            n_estimators=150, min_samples_leaf=2, n_jobs=-1, random_state=RANDOM_STATE
        ),
        "HistGradientBoosting": HistGradientBoostingRegressor(
            random_state=RANDOM_STATE
        ),
    }


def build_pipeline(model_name, model=None):
    """Wrap preprocessing and a regressor into one Pipeline."""
    if model is None:
        model = candidate_models()[model_name]
    return Pipeline([("preprocess", build_preprocessor()), ("model", model)])


def regression_metrics(y_true, y_pred):
    """Compute evaluation metrics."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": root_mean_squared_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
    }


def split_data(df):
    """Define X/y and split into train/val/test."""
    X = df[FEATURES]
    y = df[TARGET]
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    val_fraction = VAL_SIZE / (VAL_SIZE + TRAIN_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_fraction, random_state=RANDOM_STATE
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def cap_target(y_train, y_val, y_test):
    """Apply IQR capping to the target using train-only bounds."""
    q1, q3 = np.percentile(y_train, [25, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return (
        y_train.clip(lower, upper),
        y_val.clip(lower, upper),
        y_test.clip(lower, upper),
        float(lower),
        float(upper),
    )


def train_candidates(df, progress=None):
    """Fit every candidate model on the train split and evaluate on validation."""
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    y_train_c, y_val_c, y_test_c, cap_low, cap_high = cap_target(
        y_train, y_val, y_test
    )

    names = list(candidate_models())
    rows = []
    for index, name in enumerate(names):
        pipeline = build_pipeline(name)
        started = time.perf_counter()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            pipeline.fit(X_train, y_train_c)
        elapsed = time.perf_counter() - started
        y_pred = pipeline.predict(X_val)
        scores = regression_metrics(y_val_c, y_pred)
        rows.append({"Model": name, **scores, "Fit seconds": round(elapsed, 2)})
        if progress is not None:
            progress(index + 1, len(names), name)

    results = pd.DataFrame(rows).sort_values("R2", ascending=False).reset_index(
        drop=True
    )
    best = results.sort_values(["R2", "RMSE"], ascending=[False, True]).iloc[0]
    split_info = {
        "train_rows": int(len(X_train)),
        "val_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
        "target_cap_low": cap_low,
        "target_cap_high": cap_high,
    }
    return results, str(best["Model"]), split_info


PARAM_GRIDS = {
    "Ridge": {"model__alpha": [0.1, 1.0, 10.0, 100.0]},
    "Lasso": {"model__alpha": [0.001, 0.01, 0.1, 1.0]},
    "Random Forest": {"model__n_estimators": [150, 300], "model__max_depth": [None, 20]},
    "HistGradientBoosting": {"model__learning_rate": [0.05, 0.1], "model__max_iter": [200, 400], "model__max_leaf_nodes": [31, 63]},
    "Linear Regression": {},
}


def cv_grid_size(grid):
    """Count total parameter combinations in a grid dict."""
    total = 1
    for values in grid.values():
        total *= len(values)
    return total


def tune_and_finalize(df, best_name, model_path=MODEL_PATH, progress=None):
    """Tune the best model with GridSearchCV, retrain on train+val, and save."""
    def note(text):
        if progress is not None:
            progress(text)

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    y_train_c, y_val_c, y_test_c, _, _ = cap_target(y_train, y_val, y_test)
    grid = PARAM_GRIDS.get(best_name, {})

    note(f"GridSearchCV over {best_name} ({cv_grid_size(grid)} combos x {CV_FOLDS} folds)")
    if grid:
        search = GridSearchCV(
            build_pipeline(best_name),
            grid,
            cv=CV_FOLDS,
            scoring="r2",
            n_jobs=1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            search.fit(X_train, y_train_c)
        tuned_pipeline = search.best_estimator_
        best_params = search.best_params_
        cv_score = float(search.best_score_)
    else:
        tuned_pipeline = build_pipeline(best_name)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            tuned_pipeline.fit(X_train, y_train_c)
        best_params = {}
        cv_score = float(r2_score(y_val_c, tuned_pipeline.predict(X_val)))

    note("Retraining tuned model on train + validation")
    final_pipeline = clone(tuned_pipeline)
    X_final = pd.concat([X_train, X_val])
    y_final = pd.concat([y_train_c, y_val_c])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        final_pipeline.fit(X_final, y_final)

    note("Final evaluation on the untouched test split")
    final_metrics = regression_metrics(y_test_c, final_pipeline.predict(X_test))

    note(f"Saving pipeline to {model_path}")
    joblib.dump(final_pipeline, model_path)

    feature_names = final_pipeline.named_steps["preprocess"].get_feature_names_out()
    report = {
        "best_model": best_name,
        "best_params": best_params,
        "grid_search_cv_r2": cv_score,
        "final_test_metrics": final_metrics,
        "split_info": {
            "train_rows": int(len(X_train)),
            "val_rows": int(len(X_val)),
            "test_rows": int(len(X_test)),
            "final_fit_rows": int(len(X_final)),
        },
        "n_model_inputs": int(len(feature_names)),
        "model_path": model_path,
        "pipeline": final_pipeline,
    }
    return report


# 
# Self-test
# 


def selftest(csv_path=None):
    """Run loading, cleaning, training, tuning, saving, and reload check."""
    path = csv_path or DEFAULT_CSV
    logging_basic_config_done = False

    def log_info(msg):
        nonlocal logging_basic_config_done
        if not logging_basic_config_done:
            import logging
            logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
            logging_basic_config_done = True
        import logging
        logging.info(msg)

    log_info(f"Loading {path}")
    raw = load_raw_csv(path)
    df, log = clean_data(raw)
    catalog = build_catalog_columns(df)
    for line in log:
        log_info(f"[clean] {line}")

    def model_progress(done, total, name):
        log_info(f"[train] {done}/{total} fitted: {name}")

    results, best_name, split_info = train_candidates(df, progress=model_progress)
    log_info(f"[compare] best model: {best_name}")

    report = tune_and_finalize(df, best_name, progress=lambda text: log_info(f"[tune] {text}"))
    printable = {k: v for k, v in report.items() if k != "pipeline"}
    log_info(f"[final] report: {json.dumps(printable, indent=2, default=str)}")

    reloaded = joblib.load(MODEL_PATH)
    sample = df[FEATURES].head(5)
    preds = reloaded.predict(sample)
    log_info(f"[reload] predictions: {np.round(preds, 2).tolist()}")
    return 0


# 
# Streamlit UI - catalog first, price prediction second
# 


@st.cache_data(show_spinner="Loading and cleaning medicines.csv ...")
def cached_load_and_clean(path, mtime, size):
    """Cache cleaning results per file version."""
    raw = load_raw_csv(path)
    return clean_data(raw)


def load_and_clean_any(source_path=None, uploaded=None):
    """Load and clean from a disk path or an uploaded file object."""
    if uploaded is not None:
        raw = load_raw_csv(io.BytesIO(uploaded.getvalue()))
        return clean_data(raw)
    stat = os.stat(source_path)
    return cached_load_and_clean(source_path, stat.st_mtime, stat.st_size)


def styled_container(title, content_rows):
    """Render a small two-column definition list for a drug detail card."""
    st.markdown(f"**{title}**")
    for label, value in content_rows:
        if value in (None, "", float("nan"), "nan", "unknown", "Unknown"):
            display = "Not specified"
        elif isinstance(value, float):
            if pd.isna(value):
                display = "Not specified"
            else:
                display = f"{value:,.2f}"
        else:
            display = str(value)
        st.markdown(f"| {label} | {display} |")
    st.markdown("")


def render_catalog_tab(df, catalog):
    """Searchable medicine catalog with a detail page for each drug."""
    if catalog is None or df is None:
        st.info("Load medicines.csv from the sidebar to browse the catalog.")
        return

    st.subheader("Medicine Catalog")

    search = st.text_input(
        "Search by medicine name, generic name, disease, or manufacturer",
        placeholder="e.g. paracetamol, fever, Cipla",
    )

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        dosage_filter = st.selectbox(
            "Dosage form",
            ["All"] + sorted(catalog["dosage_form"].unique()),
        )
    with col2:
        origin_filter = st.selectbox(
            "Origin",
            ["All", "India", "Imported", "Unknown"],
        )
    with col3:
        rx_filter = st.selectbox(
            "Prescription",
            ["All", "Rx required", "No prescription"],
        )
    with col4:
        sort_by = st.selectbox(
            "Sort by",
            ["Price: low to high", "Price: high to low", "Medicine name", "Disease"],
        )

    filtered = catalog.copy()

    if search:
        needle = search.lower()
        mask = (
            catalog["med_name"].fillna("").str.lower().str.contains(needle, na=False)
            | catalog["generic_name"].fillna("").str.lower().str.contains(needle, na=False)
            | catalog["disease"].fillna("").str.lower().str.contains(needle, na=False)
            | catalog["manufacturer"].fillna("").str.lower().str.contains(needle, na=False)
        )
        filtered = catalog[mask]

    if dosage_filter != "All":
        filtered = filtered[filtered["dosage_form"] == dosage_filter]
    if origin_filter != "All":
        filtered = filtered[filtered["origin"] == origin_filter]
    if rx_filter == "Rx required":
        filtered = filtered[filtered["rx_required"] == 1]
    elif rx_filter == "No prescription":
        filtered = filtered[filtered["rx_required"] == 0]

    if sort_by == "Price: low to high":
        filtered = filtered.sort_values(TARGET, ascending=True)
    elif sort_by == "Price: high to low":
        filtered = filtered.sort_values(TARGET, ascending=False)
    elif sort_by == "Medicine name":
        filtered = filtered.sort_values("med_name")
    elif sort_by == "Disease":
        filtered = filtered.sort_values("disease")

    total = len(filtered)
    st.caption(f"Showing {total:,} medicine{'s' if total != 1 else ''}")

    if total == 0:
        st.info("No medicines match the current filters. Try different search terms or reset the filters.")
        return

    # Summary table: a compact browsable list.
    summary = filtered[["med_name", "generic_name", "disease", "dosage_form", "manufacturer", "origin", "final_price"]].copy()
    summary["final_price"] = summary["final_price"].map(lambda v: f"Rs {v:,.2f}" if pd.notna(v) else "N/A")
    summary.columns = ["Medicine", "Generic name", "Disease", "Dosage form", "Manufacturer", "Origin", "Price"]
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Select a medicine for full details")

    med_name = st.selectbox(
        "Medicine",
        sorted(filtered["med_name"].dropna().unique()),
        help="Choose a medicine to see its full details, recommendation, and usage.",
    )

    if med_name:
        rows = filtered[filtered["med_name"] == med_name]
        if len(rows) == 0:
            st.info("Medicine not found in the filtered set.")
            return
        row = rows.iloc[0]

        st.markdown("## " + str(med_name))

        with st.expander("Medicine details", expanded=True):
            styled_container("Product information", [
                ("Generic name", row.get("generic_name")),
                ("Brand / medicine name", row.get("med_name")),
                ("Disease / condition", row.get("disease")),
                ("Dosage form", row.get("dosage_form")),
                ("Manufacturer", row.get("manufacturer")),
                ("Country of origin", row.get("origin")),
                ("Pack size (units)", row.get("pack_size")),
                ("Strength", f"{row.get('strength_value')} {row.get('strength_unit')}" if pd.notna(row.get("strength_value")) else "Not specified"),
                ("Molecule", row.get("molecule")),
                ("Molecule count", int(row.get("molecule_count")) if pd.notna(row.get("molecule_count")) else "Not specified"),
                ("Quantity (ml/gm)", row.get("quantity_value")),
                ("Prescription required", "Yes" if int(row.get("rx_required", 0)) else "No"),
                ("Drug variant", row.get("drug_varient")),
                ("Final price", f"Rs {row.get(TARGET):,.2f}"),
            ])

        st.markdown("## Recommendation")
        st.markdown(row.get("recommendation", "No recommendation available."))
        st.markdown(row.get("note", ""))

        st.markdown("## How to use")
        st.markdown(row.get("usage", "No usage information available."))

        st.divider()
        st.markdown("## Similar medicines")
        similar = filtered[
            (filtered["dosage_form"] == row["dosage_form"])
            & (filtered["med_name"] != med_name)
        ].head(8)
        if len(similar) > 0:
            similar_out = similar[["med_name", "generic_name", "disease", "final_price"]].copy()
            similar_out["final_price"] = similar_out["final_price"].map(lambda v: f"Rs {v:,.2f}" if pd.notna(v) else "N/A")
            similar_out.columns = ["Medicine", "Generic name", "Disease", "Price"]
            st.dataframe(similar_out, use_container_width=True, hide_index=True)
        else:
            st.caption("No similar medicines in the current filtered set.")


def render_predict_tab():
    """Predict a price from feature inputs using the saved pipeline."""
    st.subheader("Price Prediction")
    pipeline = st.session_state.get("final_pipeline")
    if pipeline is None and os.path.exists(MODEL_PATH):
        pipeline = joblib.load(MODEL_PATH)
    if pipeline is None:
        st.info("Train and finalize a model first, or use the catalog to browse prices.")
        return

    df = st.session_state.get("clean_df")

    with st.form("predict_form"):
        c1, c2 = st.columns(2)
        with c1:
            disease = st.selectbox(
                "Disease",
                sorted(df["disease"].unique()) if df is not None else ["unknown"],
            )
            dosage_form = st.selectbox(
                "Dosage form",
                sorted(df["dosage_form"].unique()) if df is not None else ["Tablet"],
            )
            manufacturer = st.text_input("Manufacturer", value="Unknown")
            origin = st.selectbox("Origin", ["India", "Imported", "Unknown"])
            rx_required = st.checkbox("Prescription required (Rx)")
        with c2:
            molecule = st.text_input("Molecule (generic name)", value="unknown")
            strength_unit = st.selectbox(
                "Strength unit",
                sorted(df["strength_unit"].unique()) if df is not None else ["mg"],
            )
            strength_value = st.number_input("Strength value", min_value=0.0, value=500.0)
            molecule_count = st.number_input("Molecule count", min_value=1, max_value=8, value=1, step=1)
            pack_size = st.number_input("Pack size", min_value=0.0, value=10.0)
            quantity_value = st.number_input("Quantity (ml/gm)", min_value=0.0, value=0.0)
        submitted = st.form_submit_button("Predict price")

    if submitted:
        row = pd.DataFrame([{
            "rx_required": int(rx_required),
            "pack_size": float(pack_size),
            "strength_value": float(strength_value),
            "molecule_count": float(molecule_count),
            "quantity_value": float(quantity_value),
            "disease": disease,
            "dosage_form": dosage_form,
            "manufacturer": manufacturer,
            "origin": origin,
            "strength_unit": strength_unit,
            "molecule": molecule,
        }])[FEATURES]
        prediction = float(pipeline.predict(row)[0])
        st.markdown(f"**Predicted final price: Rs {prediction:,.2f}**")


def render_train_tab(df):
    """Train, compare, tune, and save the price-prediction model."""
    st.subheader("Model Training (price prediction)")
    st.caption("This tab builds and saves a price-prediction model. The main view is the catalog tab above.")

    if st.button("Train and compare models", type="primary"):
        bar = st.progress(0.0, text="Fitting models ...")

        def progress(done, total, name):
            bar.progress(done / total, text=f"Fitted {done}/{total}: {name}")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            results, best_name, split_info = train_candidates(df, progress=progress)
        bar.empty()
        st.session_state.update({
            "results_df": results,
            "best_name": best_name,
            "split_info": split_info,
            "report": None,
        })

    results = st.session_state.get("results_df")
    if results is None:
        return st.info("Press the button above to train models.")

    st.dataframe(
        results.style.format({
            "MAE": "{:.2f}",
            "MSE": "{:.2f}",
            "RMSE": "{:.2f}",
            "R2": "{:.4f}",
        }),
        use_container_width=True,
    )
    best_name = st.session_state["best_name"]
    st.success(f"Best model: {best_name} (R2 = {results.loc[results['Model'] == best_name, 'R2'].iloc[0]:.4f})")

    if st.button("Tune and finalize best model", type="primary"):
        with st.status("Running tuning pipeline ...", expanded=True) as status:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                report = tune_and_finalize(df, best_name, progress=lambda text: status.write(text))
            status.update(label="Tuning complete", state="complete")
        st.session_state.update({"report": report, "final_pipeline": report["pipeline"]})

    report = st.session_state.get("report")
    if report is not None:
        st.write("**Best hyperparameters**")
        st.json(report["best_params"] if report["best_params"] else {"note": "No tunable hyperparameters"})
        metrics = report["final_test_metrics"]
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("MAE", f"Rs {metrics['MAE']:.2f}")
        c2.metric("MSE", f"{metrics['MSE']:,.2f}")
        c3.metric("RMSE", f"Rs {metrics['RMSE']:.2f}")
        c4.metric("R2", f"{metrics['R2']:.4f}")
        st.success(f"Pipeline saved to {report['model_path']}")


def render_data_tab(df, log):
    """Show cleaning log and a small data summary."""
    st.subheader("Cleaning summary")
    for line in log:
        st.write("- " + line)
    if df is not None:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total rows", f"{len(df):,}")
        c2.metric("Features", len(FEATURES))
        c3.metric("Median price", f"Rs {df[TARGET].median():.2f}")
        c4.metric("Max price", f"Rs {df[TARGET].max():,.2f}")


def main():
    """Run the Streamlit application."""
    st.set_page_config(page_title="Medicine Catalog", layout="wide")

    st.markdown(
        """
        <style>
        .main-header { font-size: 2.2rem; font-weight: 800; color: #0e1117; text-align: center; margin-bottom: 0.4rem; }
        .sub-header { font-size: 1.1rem; color: #555; text-align: center; margin-bottom: 1.5rem; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown('<h1 class="main-header">Medicine Catalog and Price Book</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Searchable medicine catalog with drug details, recommendations, and usage instructions.</p>', unsafe_allow_html=True)

    st.sidebar.header("Data source")
    csv_path = st.sidebar.text_input("CSV path", value=DEFAULT_CSV)
    uploaded = st.sidebar.file_uploader("Or upload medicines.csv", type=["csv"])

    if st.sidebar.button("Load and clean data", type="primary"):
        try:
            df, log = load_and_clean_any(csv_path, uploaded)
            catalog = build_catalog_columns(df)
            st.session_state.update({
                "clean_df": df,
                "clean_log": log,
                "catalog": catalog,
                "results_df": None,
                "best_name": None,
                "report": None,
                "final_pipeline": None,
            })
        except Exception as exc:
            st.sidebar.error(f"Load failed: {exc}")

    if "clean_df" not in st.session_state and uploaded is None and os.path.exists(csv_path):
        try:
            df, log = load_and_clean_any(csv_path)
            catalog = build_catalog_columns(df)
            st.session_state.update({
                "clean_df": df,
                "clean_log": log,
                "catalog": catalog,
            })
        except Exception as exc:
            st.session_state["load_error"] = str(exc)

    if "load_error" in st.session_state:
        st.error(f"Auto-load failed: {st.session_state['load_error']}")

    df = st.session_state.get("clean_df")
    log = st.session_state.get("clean_log", [])
    catalog = st.session_state.get("catalog")

    tab_catalog, tab_data, tab_train, tab_predict = st.tabs(
        ["Catalog", "Data summary", "Model training", "Price prediction"]
    )

    with tab_catalog:
        render_catalog_tab(df, catalog)

    with tab_data:
        render_data_tab(df, log)

    with tab_train:
        render_train_tab(df) if df is not None else st.info("Load data first.")

    with tab_predict:
        render_predict_tab()


if "--selftest" in sys.argv:
    argument_path = sys.argv[sys.argv.index("--selftest") + 1] if len(sys.argv) > sys.argv.index("--selftest") + 1 else None
    raise SystemExit(selftest(argument_path))

if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
        if get_script_run_ctx() is not None:
            main()
        else:
            print(__doc__)
            print("Start the UI:  streamlit run app.py")
            print("Self-test:     python app.py --selftest")
    except ImportError:
        print(__doc__)
        print("Start the UI:  streamlit run app.py")
        print("Self-test:     python app.py --selftest")
