from pydantic import BaseModel


class LLMQuery(BaseModel):
    question: str

    class Config:
        json_schema_extra = {"example": {"question": "¿Qué haces?"}}
