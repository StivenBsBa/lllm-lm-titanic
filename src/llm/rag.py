from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from pathlib import Path

# --- Configuración de Rutas ---
PROJECT_ROOT = Path(__file__).resolve().parents[2]
KNOWLEDGE_BASE_PATH = PROJECT_ROOT / "knowledge_base"
VECTOR_STORE_PATH = PROJECT_ROOT / "vector_store"


def create_vector_store():
    """
    Crea y guarda una base de datos vectorial si no existe.
    """
    if VECTOR_STORE_PATH.exists():
        print("La base de datos vectorial ya existe. Omitiendo creación.")
        return

    print("Creando la base de datos vectorial...")
    # Cargar los documentos de la base de conocimiento
    loader = DirectoryLoader(KNOWLEDGE_BASE_PATH, glob="**/*.txt")
    documents = loader.load()

    # Dividir los documentos en trozos más pequeños (chunks)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    # Crear los embeddings (convierte texto a vectores)
    # Usamos un modelo de Hugging Face que se ejecuta localmente en ollama
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # Crear la base de datos vectorial (FAISS) y guardarla localmente en ollama
    vector_store = FAISS.from_documents(chunks, embeddings)
    vector_store.save_local(str(VECTOR_STORE_PATH))
    print(f"Base de datos vectorial creada y guardada en: {VECTOR_STORE_PATH}")


def create_rag_chain():
    """
    Carga la base de datos vectorial y crea una cadena RAG para responder preguntas.
    """
    # Cargar los embeddings y la base de datos vectorial
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    vector_store = FAISS.load_local(
        str(VECTOR_STORE_PATH), embeddings, allow_dangerous_deserialization=True
    )

    # Crear un "retriever" para buscar en la base de datos vectorial
    retriever = vector_store.as_retriever()

    # Definir el LLM (usando Ollama con el modelo llama3)
    llm = Ollama(model="llama3")

    template = """
    Responde la pregunta basándote únicamente en el siguiente contexto:
    {context}

    Pregunta: {question}d
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Crear la cadena RAG
    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    return rag_chain
