"""
Entrenamiento del modelo Titanic con preprocesamiento.
"""

import pandas as pd
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib

from src.utils.preprocessing import build_preprocessor, prepare_Xy, save_pipeline, SEED

# --- Rutas ---
ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "data" / "processed" / "train_processed.csv"
ARTIFACTS_PATH = ROOT / "artifacts"
ARTIFACTS_PATH.mkdir(exist_ok=True)
MODEL_PATH = ARTIFACTS_PATH / "logistic_regression_pipeline.joblib"


def train():
    print("[INFO] Cargando dataset procesado...")
    df = pd.read_csv(DATA_PATH)

    # Dividir en train y test
    X, y = prepare_Xy(df, target_col="Survived")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )

    # Pipeline = preprocesador + modelo
    preprocessor = build_preprocessor()
    model = LogisticRegression(max_iter=1000, random_state=SEED)
    pipe = Pipeline([("preprocessor", preprocessor), ("model", model)])

    print("[INFO] Entrenando modelo...")
    pipe.fit(X_train, y_train)

    # Evaluación simple
    y_pred = pipe.predict(X_test)
    print("[INFO] Métricas en validación:")
    print(classification_report(y_test, y_pred))

    # Guardar pipeline completo
    save_pipeline(pipe, MODEL_PATH)
    print(f"[INFO] Modelo guardado en {MODEL_PATH}")


if __name__ == "__main__":
    train()
