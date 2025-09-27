import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from api.main import app  # Ahora Python encontrará el paquete api

# Cliente de prueba FastAPI
client = TestClient(app)


@pytest.fixture
def example_passenger():
    """Datos de ejemplo de un pasajero para la prueba."""
    return {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 7.25,
        "Embarked": "S",
    }


def test_predict_endpoint(example_passenger):
    response = client.post("/ml/predict", json=example_passenger)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert "probability" in data
    assert isinstance(data["prediction"], int)
    assert data["prediction"] in [0, 1]


def test_predict_invalid_data():
    """Prueba que se manejen datos inválidos correctamente."""
    invalid_passenger = {"Pclass": 3}  # Datos incompletos
    response = client.post("/ml/predict", json=invalid_passenger)

    # Debe devolver 422 Unprocessable Entity
    assert response.status_code == 422
