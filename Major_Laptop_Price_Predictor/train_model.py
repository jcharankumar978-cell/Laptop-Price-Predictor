"""Train the Laptop Price Predictor model.

Dataset: CampusX laptop-price-predictor-regression-project/laptop_data.csv
Features used by the report: brand, RAM (GB), processor family, storage (GB).
Target: Price from the source dataset.
"""
from pathlib import Path
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

DATA_URL = "https://raw.githubusercontent.com/campusx-official/laptop-price-predictor-regression-project/main/laptop_data.csv"
MODEL_PATH = Path(__file__).with_name("laptop_price_model.joblib")

def parse_ram(value):
    match = re.search(r"(\d+(?:\.\d+)?)", str(value))
    return float(match.group(1)) if match else np.nan

def parse_storage(value):
    """Convert mixed SSD/HDD/Flash storage strings to total GB."""
    text = str(value).upper().replace(",", "")
    total = 0.0
    matches = re.findall(r"(\d+(?:\.\d+)?)\s*(TB|GB)", text)
    for amount, unit in matches:
        total += float(amount) * (1024 if unit == "TB" else 1)
    return total if total > 0 else np.nan

def processor_family(value):
    cpu = str(value).lower()
    if "core i9" in cpu: return "Intel Core i9"
    if "core i7" in cpu: return "Intel Core i7"
    if "core i5" in cpu: return "Intel Core i5"
    if "core i3" in cpu: return "Intel Core i3"
    if "core m" in cpu: return "Intel Core M"
    if "celeron" in cpu: return "Intel Celeron"
    if "pentium" in cpu: return "Intel Pentium"
    if "atom" in cpu: return "Intel Atom"
    if "amd" in cpu: return "AMD"
    if "intel" in cpu: return "Intel Other"
    return "Other"

def prepare_features(raw):
    required = {"Company", "Ram", "Cpu", "Memory", "Price"}
    missing = required - set(raw.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    data = pd.DataFrame({
        "Brand": raw["Company"].astype("string").fillna("Unknown").str.strip(),
        "RAM_GB": raw["Ram"].map(parse_ram),
        "Processor_Family": raw["Cpu"].map(processor_family),
        "Storage_GB": raw["Memory"].map(parse_storage),
        "Price": pd.to_numeric(raw["Price"], errors="coerce")
    })
    data = data.replace([np.inf, -np.inf], np.nan).dropna(subset=["Price"])
    return data

def train():
    print("Downloading laptop dataset...")
    raw = pd.read_csv(DATA_URL)
    data = prepare_features(raw)
    X = data[["Brand", "RAM_GB", "Processor_Family", "Storage_GB"]]
    y = data["Price"]

    categorical = ["Brand", "Processor_Family"]
    numeric = ["RAM_GB", "Storage_GB"]
    preprocessor = ColumnTransformer([
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical),
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median"))
        ]), numeric)
    ])
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", LinearRegression())
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42
    )
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    metrics = {
        "MAE": float(mean_absolute_error(y_test, predictions)),
        "RMSE": float(np.sqrt(mean_squared_error(y_test, predictions))),
        "R2": float(r2_score(y_test, predictions)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "dataset_rows_after_cleaning": int(len(data)),
        "features": list(X.columns),
        "target": "Price",
        "split": "80% train / 20% test",
        "random_state": 42
    }
    bundle = {
        "model": pipeline,
        "feature_columns": list(X.columns),
        "brand_options": sorted(X["Brand"].dropna().unique().tolist()),
        "processor_options": sorted(X["Processor_Family"].dropna().unique().tolist()),
        "metrics": metrics
    }
    joblib.dump(bundle, MODEL_PATH)
    print("\nHeld-out test results")
    for key, value in metrics.items():
        print(f"{key}: {value}")
    print(f"\nSaved model bundle: {MODEL_PATH}")
    return bundle

if __name__ == "__main__":
    train()
