# Reporte del Componente LLM: Asistente RAG del Proyecto Titanic

Este documento detalla el diseño, la implementación y la evaluación del componente de Large Language Model (LLM) para el proyecto Titanic.

## 1. Descripción General del Componente

- **Objetivo:** Crear un asistente de preguntas y respuestas (Q&A) que responda preguntas sobre el proyecto de ML del Titanic de manera precisa y basada en hechos, evitando alucinaciones.
- **Enfoque:** **Retrieval-Augmented Generation (RAG)** para garantizar que las respuestas estén "ancladas" a la base de conocimiento específica del proyecto.
- **Stack Tecnológico:**
  - **Orquestación:** LangChain
  - **Modelo de Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (local)
  - **Base de Datos Vectorial:** FAISS (local)
  - **Modelo LLM:** `llama3` a través de Ollama (local)

---

## 2. Diseño del Pipeline RAG

1. **Base de Conocimiento:** Carpeta `/knowledge_base/` con archivos `.txt` que incluyen resúmenes del proyecto, métricas y hallazgos de EDA.
2. **Indexación (Indexing):**
   - Dividir documentos en chunks de 500 caracteres con solapamiento de 50.
   - Vectorización usando embeddings.
   - Guardado en índice FAISS local (`/vector_store`).
3. **Recuperación (Retrieval):**
   - Pregunta → vector → búsqueda en FAISS.
   - Recupera los chunks más relevantes.
4. **Generación (Generation):**
   - Los chunks recuperados se inyectan en un prompt junto con la pregunta.
   - Prompt utilizado:

```text
Responde la pregunta basándote únicamente en el siguiente contexto:
{context}

Pregunta: {question}
```
