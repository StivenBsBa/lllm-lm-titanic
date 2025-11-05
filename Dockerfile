# --- Etapa base: Python estable y liviano ---
FROM python:3.11-slim-bookworm AS base


# --- Etapa 1: Dependencias del sistema ---
FROM base AS system-deps

# Instalar solo dependencias mínimas necesarias (sentence-transformers usa wheels pre-compilados)
# build-essential solo se necesita si hay que compilar desde fuente, pero no es necesario
# Para CPU sin GPU, no necesitamos dependencias adicionales del sistema
RUN apt-get update && apt-get clean && rm -rf /var/lib/apt/lists/*


# --- Etapa 2: Dependencias Python ---
FROM system-deps AS python-deps

# Usar requirements-prod.txt para producción (sin dependencias de testing ni librerías innecesarias)
COPY requirements-prod.txt requirements.txt

# Actualizar pip e instalar solo dependencias esenciales para producción
# Usar --no-cache-dir para reducir tamaño de imagen
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Pre-descargar el modelo de embeddings específico que usamos (evita descarga en runtime)
# Esto reduce el tamaño de la descarga y acelera el inicio
# El modelo se cachea en ~/.cache/huggingface/ dentro del contenedor
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"

# --- Etapa 3: Imagen final ---
FROM python-deps AS final

# Crear directorio de trabajo
WORKDIR /app

# Copiar el código fuente necesario para ejecutar la aplicación
COPY api/ ./api/
COPY src/ ./src/
COPY artifacts/ ./artifacts/
COPY knowledge_base/ ./knowledge_base/

# Crear directorio para vector_store si no existe (se creará en runtime si es necesario)
RUN mkdir -p vector_store

EXPOSE 8000

# Cambiar el comando para usar la ruta correcta del módulo
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]