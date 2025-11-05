# Proyecto End-to-End: ML + LLM (Predicción Titanic)

Este repositorio contiene un proyecto completo de Machine Learning y LLM, desde el análisis de datos hasta el despliegue de una API con pruebas y documentación. El objetivo es predecir la supervivencia de los pasajeros del Titanic y responder preguntas sobre el proyecto utilizando un sistema RAG.

---

## 🚀 Features

- **Modelo de ML Supervisado:** Un pipeline de clasificación (Regresión Logística) para predecir la supervivencia.
- **Asistente LLM (RAG):** Un sistema de eneration que responde preguntas basándose en documentos del proyecto para evitar alucinaciones.
- **API Robusta:** Un servicio web construido con FastAPI que expone dos endpoints: `/predict` para el modelo de ML y `/llm/ask` para el asistente RAG.
- **Pruebas Automatizadas:** Pruebas de integración para la API usando `pytest`.
- **Documentación Completa:** Incluye un `MODEL_CARD.md` para el modelo de ML y un `LLM_REPORT.md` para el componente de IA Generativa.
- **Reproducibilidad:** El proyecto está diseñado para ser completamente reproducible, con scripts para descarga de datos, entrenamiento y un entorno definido.

---

## 🛠️ Tech Stack

- **Lenguaje:** Python 3.10+
- **Análisis y Modelado:** Pandas, Scikit-learn, NumPy
- **API:** FastAPI, Uvicorn
- **Componente LLM:** LangChain, Google Gemini API, FAISS, Sentence-Transformers
- **Pruebas:** Pytest

---

## 📂 Estructura del Proyecto

```
├── api/
│   ├── routers/
│   │   ├── llm.py
│   │   └── ml.py
│   ├── schemas/
│   │   ├── llm_query.py
│   │   └── passenger.py
│   └── main.py
├── artifacts/
│   └── logistic_regression_pipeline.joblib
├── data/
│   └── get_data.py
├── eval/
│   ├── eval_ml.py
│   └── llm_eval.py
├── knowledge_base/
│   └── project_summary.txt
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_baseline.ipynb
├── src/
│   ├── llm/
│   │   └── rag.py
│   ├── ml/
│   │   ├── predict.py
│   │   └── train.py
│   └── utils/
│       └── preprocessing.py
├── tests/
│   ├── test_llm.py
│   └── test_ml.py
├── vector_store/
│   ├── index.faiss
│   └── index.pkl
├── LLM_REPORT.md
├── MODEL_CARD.md
├── README.md
└── requirements.txt
```

---

## 📋 ¿Qué se hizo?

1. **Procesamiento de Datos:** Limpieza y transformación de los datos originales del Titanic para crear conjuntos de entrenamiento y prueba.
2. **Entrenamiento de Modelos ML:** Se entrenó un modelo de regresión logística y se guardó el pipeline en `artifacts/`.
3. **Implementación de LLM y RAG:** Se desarrolló un sistema de recuperación aumentada por generación (RAG) usando LangChain y FAISS, almacenando los índices en `vector_store/`.
4. **API REST:** Se creó una API con FastAPI para exponer endpoints de predicción y consulta LLM.
5. **Evaluación:** Scripts y notebooks para analizar el desempeño de los modelos y el sistema LLM.
6. **Pruebas:** Pruebas automatizadas para asegurar la robustez del sistema.

## 🏗️ ¿Cómo se trabaja?

### 0. Configuración de API Key de Google Gemini

Antes de usar el sistema LLM, necesitas configurar tu API key de Google Gemini:

1. Obtén tu API key desde [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Crea un archivo `.env` en la raíz del proyecto con:
   ```
   GOOGLE_API_KEY=tu_api_key_aqui
   ```

### 1. Instalación de dependencias

```bash
pip install -r requirements.txt
```

### 2. Procesamiento de datos

```bash
python data/get_data.py
```

### 3. Entrenamiento ML

```bash
python3 -m src.ml.train
```

- Pipeline guardado en `/artifacts/logistic_regression_pipeline.joblib`

### 4. Inferencia ML

```bash
python3 -m src.ml.predict
```

### 5. Evaluación ML

```bash
python eval/eval_ml.py
```

### 6. Crear Vector Store RAG

```bash
python src/llm/rag.py
```

### 7. Evaluación LLM / RAG

```bash
python eval/llm_eval.py
```

### 8. Correr API FastAPI

```bash
uvicorn api.main:app --reload
```

Endpoints disponibles:

- `POST /predict` → Recibe features del pasajero y devuelve predicción + probabilidad.
- `POST /llm/ask` → Recibe pregunta y devuelve respuesta + metadatos (citaciones, latencia, tokens).

### 9. Pruebas

```bash
pytest tests/
```

### 10. Notebooks

Revisar `notebooks/` para EDA y experimentos de modelado.

## 💡 Recomendaciones

- Revisa los notebooks para entender el flujo de datos y experimentos realizados.
- Consulta la documentación en los archivos `MODEL_CARD.md` y `LLM_REPORT.md` para detalles técnicos de los modelos.
