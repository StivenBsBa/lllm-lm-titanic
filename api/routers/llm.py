from fastapi import APIRouter, HTTPException
from api.schemas.llm_query import LLMQuery
from src.llm.rag import create_rag_chain, create_vector_store

router = APIRouter()

# Inicialización de RAG al arrancar el router
create_vector_store()
rag_chain = create_rag_chain()


@router.post("/ask")
def ask_llm(query: LLMQuery):
    """
    Recibe una pregunta y devuelve la respuesta generada por el sistema RAG.
    """
    try:
        answer = rag_chain.invoke(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
