from pydantic import BaseModel, Field


class Passenger(BaseModel):
    Pclass: int = Field(..., example=3)
    Sex: str = Field(..., example="male")
    Age: float = Field(..., example=22.0)
    SibSp: int = Field(..., example=1)
    Parch: int = Field(..., example=0)
    Fare: float = Field(..., example=7.25)
    Embarked: str = Field(..., example="S")
    FamilySize: int | None = Field(
        None,
        description="Número de familiares a bordo. Si no se pasa, se calcula como SibSp + Parch.",
        example=1,
    )
