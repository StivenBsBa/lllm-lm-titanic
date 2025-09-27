"""
Predicción con modelo entrenado Titanic.
"""

import joblib
from pathlib import Path
import pandas as pd
from src.utils.preprocessing import prepare_Xy

# --- Rutas ---
ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "artifacts" / "logistic_regression_pipeline.joblib"


def load_model():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {MODEL_PATH}. Ejecuta train.py primero."
        )
    return joblib.load(MODEL_PATH)


def make_prediction(input_data: dict):
    """
    input_data: diccionario con las features esperadas
    """
    model = load_model()

    # Convertir input a DataFrame
    df = pd.DataFrame([input_data])

    # Preprocesar columnas (usar prepare_Xy solo para asegurar formato correcto)
    X, _ = prepare_Xy(df, target_col=None)

    # Predicciones
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0][1]

    return {
        "prediction": int(pred),
        "probability": float(proba * 100),  # convertir a porcentaje
    }
