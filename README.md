# Proyecto de Analítica Predictiva: Estimación de Costos en Servicios de Salud

### Curso de Ciencia de Datos - Facultad de Ingeniería Industrial

## Integrantes

* JUAN DAVID DIAZ CEPEDA
* NATALIA ALEJANDRA MENDIETA ORTEGA
* GUSTAVO ADOLFO RAMIREZ TAFUR
* NATALIA SOLER JIMENEZ

---

## Descripción General

Este proyecto desarrolla una plataforma de analítica predictiva orientada a la estimación de gastos médicos anuales de una cartera de 1,338 asegurados. El sistema integra procesos de análisis exploratorio de datos (EDA), entrenamiento de modelos de machine learning y simulación predictiva mediante interfaces interactivas desarrolladas en Streamlit.

El enfoque implementado permite:

* Analizar patrones estadísticos y distribuciones de variables médicas y demográficas.
* Evaluar el comportamiento financiero de los costos asociados a pacientes.
* Comparar múltiples algoritmos de regresión supervisada.
* Generar predicciones en tiempo real para nuevos perfiles clínicos.
* Interpretar la importancia relativa de cada variable dentro del modelo seleccionado.

---

# Vista General del Sistema

## Dashboard de Visualización y Análisis Exploratorio

La primera aplicación corresponde al módulo de análisis exploratorio de datos (`app_visualizacion.py`), diseñado para examinar distribuciones, correlaciones y patrones generales de comportamiento en la base de asegurados.

<img width="1839" height="858" alt="image" src="https://github.com/user-attachments/assets/d64afacb-1b1d-4ed1-9d33-a86148205b37" />


Características principales:

* Distribución de gastos médicos.
* Comparación entre fumadores y no fumadores.
* Relación entre IMC y costos médicos.
* Segmentación por región y género.
* Métricas descriptivas generales.
* Histogramas, gráficos de dispersión y análisis categórico.

---

## Plataforma Predictiva y Simulación de Costos

La segunda aplicación corresponde al módulo de inferencia y simulación predictiva (`app_ml.py`). Este componente permite ingresar información de nuevos pacientes y obtener estimaciones automáticas de gastos médicos utilizando distintos modelos entrenados.

<img width="1741" height="885" alt="image" src="https://github.com/user-attachments/assets/1d5f0921-7196-404c-867c-a3157a415595" />


Funciones principales:

* Predicción de costos médicos en tiempo real.
* Comparación paralela entre modelos.
* Visualización de métricas de rendimiento.
* Interpretabilidad mediante importancia de variables.
* Evaluación del impacto de hábitos y condiciones físicas sobre el costo estimado.

---

# Estructura del Repositorio

```text
Proyecto/
│
├── data/
│   └── gastos_medicos.csv
│
├── models/
│   ├── random_forest.pkl
│   ├── decision_tree.pkl
│   ├── linear_regression.pkl
│   └── metricas_modelos.csv
│
├── images/
│   ├── dashboard_eda.png
│   └── dashboard_ml.png
│
├── entrenamiento.py
├── app_visualizacion.py
├── app_ml.py
├── requirements.txt
└── README.md
```

Descripción de componentes:

| Archivo / Directorio      | Descripción                                                                  |
| ------------------------- | ---------------------------------------------------------------------------- |
| `data/gastos_medicos.csv` | Dataset histórico utilizado para entrenamiento y validación.                 |
| `models/`                 | Almacena modelos serializados y métricas generadas durante el entrenamiento. |
| `images/`                 | Carpeta destinada a capturas y recursos visuales del proyecto.               |
| `entrenamiento.py`        | Pipeline principal de preprocesamiento, entrenamiento y evaluación.          |
| `app_visualizacion.py`    | Dashboard interactivo de análisis exploratorio de datos.                     |
| `app_ml.py`               | Plataforma de simulación predictiva basada en machine learning.              |
| `requirements.txt`        | Dependencias necesarias para la ejecución del entorno.                       |

---

# Arquitectura de Modelado

El núcleo analítico del proyecto se implementó en `entrenamiento.py`, donde se construyó un pipeline unificado de preparación de datos y entrenamiento supervisado.

## Preprocesamiento de Datos

Se aplicaron estrategias diferenciadas según el tipo de variable:

### Variables Numéricas

Las variables cuantitativas fueron normalizadas mediante estandarización (`StandardScaler`) para reducir diferencias de escala y mejorar la estabilidad matemática de los modelos.

### Variables Categóricas

Las variables cualitativas fueron transformadas usando codificación One-Hot Encoding, eliminando la primera categoría para evitar multicolinealidad.

---

# Modelos Evaluados

## 1. Regresión Lineal (`LinearRegression`)

Modelo base utilizado para establecer una referencia estadística inicial y analizar relaciones lineales simples entre variables.

## 2. Árbol de Regresión (`DecisionTreeRegressor`)

Modelo no lineal configurado con profundidad máxima de 5 niveles para reducir sobreajuste y mejorar interpretabilidad.

## 3. Random Forest (`RandomForestRegressor`)

Modelo de ensamble basado en múltiples árboles de decisión independientes, implementado con 100 estimadores para estabilizar la varianza y capturar interacciones complejas entre variables.

---

# Resultados y Evaluación

## Comparación de Rendimiento

Los resultados muestran que el modelo `RandomForestRegressor` obtiene consistentemente:

* Mayor coeficiente de determinación ($R^2$).
* Menor error cuadrático medio (RMSE).
* Mejor capacidad de generalización sobre datos no observados.

La superioridad de los modelos basados en árboles se explica por la naturaleza no lineal de los datos médicos y financieros analizados.

---

## Interacciones Relevantes Detectadas

Uno de los patrones más importantes encontrados corresponde a la interacción entre:

* Índice de Masa Corporal (IMC)
* Hábito de tabaquismo

Un paciente fumador con obesidad presenta incrementos exponenciales en sus gastos médicos, comportamiento que no puede modelarse adecuadamente mediante regresión lineal clásica.

Los árboles de decisión y Random Forest logran identificar estas interacciones complejas mediante segmentaciones jerárquicas del espacio de datos.

---

# Control de Sobreajuste

Dado que los modelos de ensamble tienden a memorizar fácilmente el conjunto de entrenamiento, se implementaron mecanismos de regularización:

* Restricción de profundidad máxima.
* Control del tamaño mínimo de muestras.
* Separación entrenamiento/prueba.
* Validación mediante métricas comparativas.

La diferencia controlada entre métricas de entrenamiento y prueba evidencia estabilidad y capacidad de generalización.

---

# Modelo Seleccionado para Producción

El modelo seleccionado para despliegue fue `RandomForestRegressor`, debido a:

* Mayor precisión predictiva.
* Mejor manejo de relaciones no lineales.
* Mayor robustez frente a ruido y variabilidad.
* Mejor desempeño en escenarios financieros complejos.

Desde la perspectiva de ingeniería financiera aplicada a salud, subestimar costos médicos representa un riesgo económico significativo. Por ello, el uso de modelos de ensamble se considera técnicamente justificado.

---

# Tecnologías Utilizadas

* Python 3.x
* Pandas
* NumPy
* Scikit-Learn
* Streamlit
* Matplotlib
* Seaborn
* Joblib

---

# Guía de Ejecución

## 1. Instalación de Dependencias

```bash
pip install -r requirements.txt
```

---

## 2. Entrenamiento de Modelos

```bash
python entrenamiento.py
```

Este proceso:

* Limpia y transforma los datos.
* Entrena los modelos.
* Calcula métricas.
* Guarda modelos serializados en `/models`.

---

## 3. Ejecución del Dashboard EDA

```bash
streamlit run app_visualizacion.py
```

Módulo orientado al análisis exploratorio y visualización estadística.

---

## 4. Ejecución del Módulo Predictivo

```bash
streamlit run app_ml.py
```

