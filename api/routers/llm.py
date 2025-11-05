from fastapi import APIRouter, HTTPException
from api.schemas.llm_query import LLMQuery
from src.llm.rag import create_rag_chain

router = APIRouter()

# Inicialización de RAG al arrancar el router
# create_rag_chain() ya maneja la creación del vector store si no existe
rag_chain = None
rag_error = None

try:
    rag_chain = create_rag_chain()
    print("✅ RAG chain inicializado correctamente")
except Exception as e:
    import traceback

    error_msg = str(e)
    error_trace = traceback.format_exc()
    print(f"❌ Error al inicializar RAG chain: {error_msg}")
    print(f"Traceback completo:\n{error_trace}")
    rag_error = error_msg
    rag_chain = None


@router.post("/ask")
def ask_llm(query: LLMQuery):
    """
    Recibe una pregunta y devuelve la respuesta generada por el sistema RAG.
    """
    global rag_chain, rag_error

    # Si no está inicializado, intentar inicializar de nuevo
    if rag_chain is None:
        try:
            rag_chain = create_rag_chain()
            rag_error = None
            print("✅ RAG chain inicializado correctamente (reintento)")
        except Exception as e:
            import traceback

            error_msg = str(e)
            error_trace = traceback.format_exc()
            print(f"❌ Error al inicializar RAG chain (reintento): {error_msg}")
            print(f"Traceback completo:\n{error_trace}")
            rag_error = error_msg
            raise HTTPException(
                status_code=503,
                detail=f"El sistema RAG no está disponible. Error: {error_msg}. Verifica la configuración (GOOGLE_API_KEY, knowledge_base, etc.)",
            )

    try:
        answer = rag_chain.invoke(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
