from fastapi import FastAPI
from api.routers import ml, llm

# Inicializar la aplicación
app = FastAPI(
    title="API de Predicción Titanic + LLM",
    description="API con endpoints de ML y LLM",
)

# Registrar routers
app.include_router(ml.router, prefix="/ml", tags=["ML"])
app.include_router(llm.router, prefix="/llm", tags=["LLM"])
