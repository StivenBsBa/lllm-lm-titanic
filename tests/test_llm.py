from fastapi.testclient import TestClient
import sys
from pathlib import Path


project_root = Path(__file__).resolve().parents[1]
sys.path.append(str(project_root))

from api.main import app

client = TestClient(app)


def test_ask_llm_success():
    """Prueba una pregunta exitosa al sistema RAG."""
    query_data = {"question": "Cual es la precision del modelo?"}
    response = client.post("/llm/ask", json=query_data)
    assert response.status_code == 200
    json_response = response.json()
    assert "answer" in json_response
    assert (
        "81.8%" in json_response["answer"]
        or "precisión" in json_response["answer"].lower()
    )
