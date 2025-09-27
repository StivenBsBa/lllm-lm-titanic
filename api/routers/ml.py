from fastapi import APIRouter
from api.schemas.passenger import Passenger
from src.ml.predict import make_prediction

router = APIRouter()


@router.post("/predict")
def ml_survival(passenger: Passenger):
    """
    Endpoint de predicción con modelo ML.
    """
    input_data = passenger.model_dump()

    # Calcular FamilySize si no viene
    if input_data.get("FamilySize") is None:
        input_data["FamilySize"] = input_data.get("SibSp", 0) + input_data.get(
            "Parch", 0
        )

    result = make_prediction(input_data)
    return result
