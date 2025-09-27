# llm_eval.py
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.llm.rag import create_rag_chain

# Crear cadena RAG
rag_chain = create_rag_chain()

# Preguntas de prueba (mini human eval o set Q&A)
questions = [
    "¿Que haces?",
    "¿Qué precisión tiene el modelo?",
    "¿Quién fue el capitán del Titanic?",
]

print("--- Evaluación LLM / RAG ---")
for q in questions:
    answer = rag_chain.invoke(q)
    print(f"P: {q}")
    print(f"R: {answer}\n")
