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


@router.get("/status")
def rag_status():
    """
    Endpoint de diagnóstico para verificar el estado del sistema RAG.
    """
    import os
    from pathlib import Path
    
    status = {
        "rag_chain_initialized": rag_chain is not None,
        "error": rag_error,
        "checks": {}
    }
    
    # Verificar GOOGLE_API_KEY
    api_key = os.getenv("GOOGLE_API_KEY")
    status["checks"]["google_api_key"] = {
        "configured": api_key is not None and api_key != "",
        "length": len(api_key) if api_key else 0
    }
    
    # Verificar knowledge_base
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
    knowledge_files = list(KNOWLEDGE_BASE_PATH.glob("**/*.txt")) if KNOWLEDGE_BASE_PATH.exists() else []
    status["checks"]["knowledge_base"] = {
        "exists": KNOWLEDGE_BASE_PATH.exists(),
        "files_count": len(knowledge_files),
        "files": [str(f.relative_to(PROJECT_ROOT)) for f in knowledge_files]
    }
    
    # Verificar vector_store
    VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"
    index_faiss = VECTOR_STORE_PATH / "index.faiss"
    index_pkl = VECTOR_STORE_PATH / "index.pkl"
    status["checks"]["vector_store"] = {
        "directory_exists": VECTOR_STORE_PATH.exists(),
        "index_faiss_exists": index_faiss.exists(),
        "index_pkl_exists": index_pkl.exists(),
        "complete": index_faiss.exists() and index_pkl.exists()
    }
    
    return status


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
                detail=f"El sistema RAG no está disponible. Error: {error_msg}. Verifica la configuración (GOOGLE_API_KEY, knowledge_base, etc.)"
            )
    
    try:
        answer = rag_chain.invoke(query.question)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
