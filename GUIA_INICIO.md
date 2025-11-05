# 🚀 Guía Completa: De la Construcción al Uso del Sistema

Esta guía te lleva paso a paso desde construir la imagen Docker hasta usar todos los endpoints de la API.

---

## 📋 Prerrequisitos

1. **Docker y Docker Compose** instalados
2. **API Key de Google Gemini** (obtener en [Google AI Studio](https://makersuite.google.com/app/apikey))

---

## 🔧 PASO 1: Configuración Inicial

### 1.1 Crear archivo `.env`

Crea un archivo `.env` en la raíz del proyecto con tu API key:

```bash
# En la raíz del proyecto
echo "GOOGLE_API_KEY=tu_api_key_aqui" > .env
```

**Importante:** Reemplaza `tu_api_key_aqui` con tu API key real de Google Gemini.

### 1.2 Verificar archivos necesarios

Asegúrate de que existan estos archivos/directorios:

- ✅ `artifacts/logistic_regression_pipeline.joblib` (modelo ML entrenado)
- ✅ `knowledge_base/project_summary.txt` (o al menos un archivo `.txt`)
- ✅ `requirements-prod.txt` (ya existe)

---

## 🐳 PASO 2: Construir la Imagen Docker

### 2.1 Construir la imagen

```bash
docker compose build
```

**Tiempo estimado:** 3-10 minutos (primera vez)

**¿Qué hace?**
- Descarga la imagen base de Python
- Instala todas las dependencias de `requirements-prod.txt`
- Pre-descarga el modelo de embeddings (`sentence-transformers/all-MiniLM-L6-v2`)
- Prepara el contenedor

### 2.2 Verificar que la construcción fue exitosa

Deberías ver al final:
```
✔ ml-llm-project-backend  Built
```

---

## 🚀 PASO 3: Iniciar el Contenedor

### 3.1 Levantar el servicio

```bash
docker compose up
```

O si quieres ver los logs en tiempo real:

```bash
docker compose up -d  # En segundo plano
docker compose logs -f  # Ver logs
```

### 3.2 Verificar que el contenedor está corriendo

```bash
docker compose ps
```

Deberías ver:
```
NAME          IMAGE                      STATUS
llm-titanic    ml-llm-project-backend     Up
```

### 3.3 Revisar los logs

Busca estos mensajes importantes:

✅ **Éxito:**
```
INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
Creando la base de datos vectorial...  (solo la primera vez)
Base de datos vectorial creada y guardada en: /app/vector_store
```

❌ **Errores comunes:**
- `GOOGLE_API_KEY no está configurada` → Revisa tu archivo `.env`
- `No se encontraron documentos en knowledge_base` → Asegúrate de tener archivos `.txt` en `knowledge_base/`

---

## ✅ PASO 4: Verificar que la API Funciona

### 4.1 Verificar endpoint de salud (opcional)

Si agregaste un endpoint de salud, puedes probar:

```bash
curl http://localhost:5101/docs
```

Esto abrirá la documentación interactiva de FastAPI en tu navegador.

### 4.2 Verificar que el servidor está escuchando

```bash
curl http://localhost:5101/docs
```

O abre en tu navegador: `http://localhost:5101/docs`

---

## 📡 PASO 5: Llamar a los Endpoints

### 5.1 Endpoint de ML: Predicción de Supervivencia

**Endpoint:** `POST /ml/predict`

**URL completa:** `http://localhost:5101/ml/predict`

**Ejemplo con curl:**

```bash
curl -X POST "http://localhost:5101/ml/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
  }'
```

**Ejemplo con Python:**

```python
import requests

url = "http://localhost:5101/ml/predict"
data = {
    "Pclass": 3,
    "Sex": "male",
    "Age": 22.0,
    "SibSp": 1,
    "Parch": 0,
    "Fare": 7.25,
    "Embarked": "S"
}

response = requests.post(url, json=data)
print(response.json())
```

**Respuesta esperada:**

```json
{
  "prediction": 0,
  "probability": 15.23
}
```

**Campos requeridos:**
- `Pclass`: Clase del pasajero (1, 2, o 3)
- `Sex`: "male" o "female"
- `Age`: Edad del pasajero
- `SibSp`: Número de hermanos/esposos a bordo
- `Parch`: Número de padres/hijos a bordo
- `Fare`: Precio del boleto
- `Embarked`: Puerto de embarque ("C", "Q", o "S")
- `FamilySize`: (Opcional) Se calcula automáticamente como SibSp + Parch

---

### 5.2 Endpoint de LLM: Preguntas al Sistema RAG

**Endpoint:** `POST /llm/ask`

**URL completa:** `http://localhost:5101/llm/ask`

**Ejemplo con curl:**

```bash
curl -X POST "http://localhost:5101/llm/ask" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "¿Qué precisión tiene el modelo?"
  }'
```

**Ejemplo con Python:**

```python
import requests

url = "http://localhost:5101/llm/ask"
data = {
    "question": "¿Qué precisión tiene el modelo?"
}

response = requests.post(url, json=data)
print(response.json())
```

**Respuesta esperada:**

```json
{
  "answer": "El modelo tiene una precisión del 81.8%..."
}
```

**Nota:** La primera llamada puede tardar más porque el sistema crea el vector store si no existe.

---

## 🌐 PASO 6: Usar la Documentación Interactiva

FastAPI genera automáticamente documentación interactiva:

1. Abre en tu navegador: `http://localhost:5101/docs`
2. Verás la interfaz Swagger UI con todos los endpoints
3. Puedes probar los endpoints directamente desde el navegador

---

## 🔍 Verificación y Troubleshooting

### Ver logs del contenedor

```bash
docker compose logs -f
```

### Detener el contenedor

```bash
docker compose down
```

### Reiniciar el contenedor

```bash
docker compose restart
```

### Reconstruir desde cero (si hay problemas)

```bash
docker compose down
docker compose build --no-cache
docker compose up
```

---

## 📊 Resumen de Endpoints

| Endpoint | Método | Descripción | URL |
|----------|--------|-------------|-----|
| `/ml/predict` | POST | Predicción de supervivencia | `http://localhost:5101/ml/predict` |
| `/llm/ask` | POST | Preguntas al sistema RAG | `http://localhost:5101/llm/ask` |
| `/docs` | GET | Documentación interactiva | `http://localhost:5101/docs` |

---

## ⚠️ Errores Comunes y Soluciones

### 1. "GOOGLE_API_KEY no está configurada"
**Solución:** Verifica que el archivo `.env` existe y tiene la API key correcta.

### 2. "ModuleNotFoundError"
**Solución:** Reconstruye la imagen: `docker compose build --no-cache`

### 3. "No se encontraron documentos en knowledge_base"
**Solución:** Asegúrate de tener al menos un archivo `.txt` en `knowledge_base/`

### 4. "Error: could not open /app/vector_store/index.faiss"
**Solución:** Esto ya está resuelto automáticamente. El sistema creará el vector store si no existe.

### 5. Puerto 5101 ya en uso
**Solución:** Cambia el puerto en `docker-compose.yml` (línea 6)

---

## 🎯 Checklist Final

- [ ] Archivo `.env` creado con `GOOGLE_API_KEY`
- [ ] Modelo ML existe en `artifacts/logistic_regression_pipeline.joblib`
- [ ] Archivos en `knowledge_base/` existen
- [ ] Imagen Docker construida exitosamente
- [ ] Contenedor corriendo (`docker compose ps`)
- [ ] API accesible en `http://localhost:5101/docs`
- [ ] Endpoint `/ml/predict` funciona
- [ ] Endpoint `/llm/ask` funciona

---

## 📝 Ejemplo Completo de Uso

```python
import requests

BASE_URL = "http://localhost:5101"

# 1. Predicción ML
ml_response = requests.post(
    f"{BASE_URL}/ml/predict",
    json={
        "Pclass": 1,
        "Sex": "female",
        "Age": 25.0,
        "SibSp": 1,
        "Parch": 0,
        "Fare": 50.0,
        "Embarked": "S"
    }
)
print("Predicción ML:", ml_response.json())

# 2. Pregunta al LLM
llm_response = requests.post(
    f"{BASE_URL}/llm/ask",
    json={"question": "¿Qué características tiene el modelo?"}
)
print("Respuesta LLM:", llm_response.json())
```

---

¡Listo! Tu sistema debería estar funcionando completamente. 🎉

