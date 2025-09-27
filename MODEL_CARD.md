# Model Card: Titanic Survival Prediction

Esta ficha de modelo proporciona información esencial sobre el modelo de predicción de supervivencia del Titanic.

## 1. Detalles del Modelo

- **Descripción:** Modelo de clasificación binaria entrenado para predecir si un pasajero del RMS Titanic habría sobrevivido al naufragio.
- **Tipo de Modelo:** Regresión Logística implementada con la librería Scikit-learn en Python.
- **Desarrollado por:** Brayan Barajas
- **Fecha:** 26 de septiembre de 2025

---

## 2. Datos de Entrenamiento

- **Dataset:** "Titanic - Machine Learning from Disaster" (Kaggle)
- **Fuente:** [https://www.kaggle.com/c/titanic](https://www.kaggle.com/c/titanic)
- **Preprocesamiento:**
  - Valores nulos:
    - `Age`: Imputados con la **mediana**.
    - `Embarked`: Imputados con la **moda**.
  - Codificación:
    - `Sex`, `Embarked`, `Pclass`, `Title`: One-Hot Encoding.
  - Escalado:
    - Características numéricas (`Age`, `Fare`, `SibSp`, `Parch`) con **StandardScaler**.
- **Variables descartadas:** `PassengerId`, `Name`, `Ticket`, `Cabin`.

---

## 3. Datos de Evaluación

- **División de Datos:** Entrenamiento 80%, validación 20% (estratificado)
- **Métricas de Rendimiento:**
  - **Accuracy:** 0.808
  - **F1-Score (Clase 1 - Sobrevive):** 0.737
  - **Reporte Completo:**
    ```
    Precision  Recall  F1-Score  Support
    0 0.82 0.87 0.85 549
    1 0.78 0.70 0.74 342
    ```

---

## 4. Consideraciones Éticas y Limitaciones

- **Sesgos en los Datos:**
  - Género: "Mujeres y niños primero" → mayor probabilidad de supervivencia para mujeres.
  - Socioeconómico: Pasajeros de 1ra clase (`Pclass`=1) favorecidos.
- **Limitaciones del Modelo:**
  - No considera variables ausentes: ubicación exacta, condición física, decisiones individuales.
  - Rendimiento limitado por la calidad y características del dataset histórico.
- **Riesgos Potenciales:**
  - Sobreinterpretación de predicciones. Una predicción de "No Sobrevive" es estadística, no histórica.

---

## 5. Reproducibilidad

- Modelo entrenado usando `python src/ml/train.py`
- Pipeline guardado en `/artifacts/logistic_regression_pipeline.joblib`
- Evaluación reproducible con `python eval/eval_ml.py`
- Semilla fija (`SEED=42`) para garantizar consistencia
