"""
src/utils/preprocessing.py
Preprocesamiento reutilizable para Titanic (simple y reproducible).
"""

from typing import List, Tuple
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import joblib

SEED = 42

NUM_FEATURES = ["Age", "Fare", "SibSp", "Parch"]
CAT_FEATURES = ["Pclass", "Sex", "Embarked"]


def build_preprocessor(
    num_features: List[str] = NUM_FEATURES, cat_features: List[str] = CAT_FEATURES
) -> ColumnTransformer:
    num_pipeline = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]
    )
    cat_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        [("num", num_pipeline, num_features), ("cat", cat_pipeline, cat_features)]
    )
    return preprocessor


def prepare_Xy(
    df: pd.DataFrame, target_col: str = "Survived"
) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()

    # Quick feature engineering: extract title from Name si existe
    if "Name" in df.columns:

        def extract_title(name):
            try:
                return name.split(",")[1].split(".")[0].strip()
            except Exception:
                return "Unknown"

        df["Title"] = df["Name"].apply(extract_title)
        common_titles = ["Mr", "Mrs", "Miss", "Master"]
        df["Title"] = df["Title"].apply(lambda t: t if t in common_titles else "Other")
    else:
        # Inferencia sin Name → asignar un valor por defecto
        df["Title"] = "Mr"  # o "Other" según prefieras

    # Feature extra: FamilySize
    df["FamilySize"] = df.get("SibSp", 0).fillna(0) + df.get("Parch", 0).fillna(0)

    # Columnas categóricas
    cat_features = CAT_FEATURES + ["Title"]

    # Columnas finales a usar en el pipeline
    features = NUM_FEATURES + cat_features + ["FamilySize"]

    # Verificar que todas existan
    for c in features:
        if c not in df.columns:
            # Crear columna con valor por defecto si falta
            df[c] = 0 if c in NUM_FEATURES + ["FamilySize"] else "Other"

    X = df[features]
    y = df[target_col] if target_col in df.columns else None
    return X, y


def save_pipeline(pipe, path: str):
    joblib.dump(pipe, path)
