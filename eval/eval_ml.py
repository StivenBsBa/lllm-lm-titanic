import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import pandas as pd
import joblib
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.utils.preprocessing import prepare_Xy

# Rutas
ARTIFACTS_PATH = PROJECT_ROOT / "artifacts"
DATA_PATH = PROJECT_ROOT / "data/processed"

# Cargar modelo entrenado
model_path = ARTIFACTS_PATH / "logistic_regression_pipeline.joblib"
model = joblib.load(model_path)

# Cargar datos de validación/test
df = pd.read_csv(DATA_PATH / "train_processed.csv")
X, y = prepare_Xy(df)

# Predecir
y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:, 1]

# Métricas
acc = accuracy_score(y, y_pred)
f1 = f1_score(y, y_pred)
cm = confusion_matrix(y, y_pred)

print("--- Evaluación Modelo ML ---")
print(f"Accuracy: {acc:.4f}")
print(f"F1 Score: {f1:.4f}")
print("Matriz de Confusión:")
print(cm)
print("\nReporte Completo:")
print(classification_report(y, y_pred))
