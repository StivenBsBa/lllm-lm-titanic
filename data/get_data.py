import pandas as pd
from pathlib import Path
import argparse

# Ruta base del proyecto: carpeta actual (donde está tu script)
ROOT = Path(__file__).resolve().parents[1]  # <-- nivel superior de este script

# Carpetas de datos
RAW_DIR = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
RAW_DIR.mkdir(parents=True, exist_ok=True)
PROC_DIR.mkdir(parents=True, exist_ok=True)

# URL pública del Titanic
TITANIC_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)


def download_titanic() -> pd.DataFrame:
    """Descarga Titanic desde GitHub y lo guarda en data/raw."""
    raw_path = RAW_DIR / "train.csv"
    print(f"[INFO] Descargando Titanic desde {TITANIC_URL}")
    df = pd.read_csv(TITANIC_URL)
    df.to_csv(raw_path, index=False)
    print(f"[INFO] Guardado en {raw_path} (shape={df.shape})")
    return df


def basic_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Imputaciones
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode().iloc[0])

    # Feature extra: tamaño de familia
    if {"SibSp", "Parch"}.issubset(df.columns):
        df["FamilySize"] = df["SibSp"].fillna(0) + df["Parch"].fillna(0)

    # Categóricas a string
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str)

    return df


def main(dataset: str = "titanic"):
    if dataset != "titanic":
        raise ValueError("Por ahora solo soporta dataset='titanic'")

    df = download_titanic()
    df_proc = basic_preprocessing(df)

    proc_path = PROC_DIR / "train_processed.csv"
    df_proc.to_csv(proc_path, index=False)
    print(f"[INFO] Guardado dataset procesado en {proc_path} (shape={df_proc.shape})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="titanic", help="Dataset a descargar (default=titanic)"
    )
    args = parser.parse_args()
    main(args.dataset)
