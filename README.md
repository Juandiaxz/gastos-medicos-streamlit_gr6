# Proyecto de Analítica Predictiva: Estimación de Costos en Servicios de Salud
**Curso de Ciencia de Datos - Facultad de Ingeniería Industrial**

Este proyecto desarrolla un sistema de evaluación y predicción para estimar los gastos médicos anuales de una cartera de 1,338 asegurados. A través de un enfoque híbrido, el sistema combina un pipeline de preparación y entrenamiento de modelos de regresión con interfaces interactivas para el análisis exploratorio y la consulta de predicciones en tiempo real.

---

## Estructura del Repositorio

* `data/gastos_medicos.csv`: Datos históricos con variables demográficas, antropométricas y hábitos de los asegurados.
* `models/`: Directorio destinado al almacenamiento de los modelos entrenados en formato serializado y sus respectivos registros de rendimiento.
* `entrenamiento.py`: Script de backend encargado del preprocesamiento de variables y el ajuste de los algoritmos.
* `app_visualizacion.py`: Interfaz orientada al análisis exploratorio de datos, distribución de frecuencias y comportamiento de variables macro.
* `app_ml.py`: Módulo de inferencia que permite ingresar nuevos perfiles de pacientes, comparar modelos en paralelo y evaluar la importancia de los atributos.
* `requirements.txt`: Declaración de librerías y dependencias del entorno de ejecución.

---

## Resumen del Entorno de Desarrollo y Modelado

El núcleo técnico se concentra en `entrenamiento.py`. Aquí se unificó el tratamiento de los datos mediante transformadores de columnas: las variables cuantitativas se estandarizaron para corregir diferencias de escala, mientras que las cualitativas se transformaron mediante codificación binaria (One-Hot Encoding) omitiendo la primera categoría para prevenir problemas de multicolinealidad.

Se configuraron y compararon tres arquitecturas distintas con el fin de evaluar su capacidad de generalización:

1. Regresión Lineal (`LinearRegression`): Utilizado como modelo base para identificar tendencias aditivas simples.
2. Árbol de Regresión (`DecisionTreeRegressor`): Restringido a una profundidad máxima de 5 niveles para permitir segmentaciones no lineales básicas sin caer en sobreajuste.
3. Random Forest (`RandomForestRegressor`): Ensamble de 100 estimadores independientes, implementado con el fin de estabilizar la varianza global y capturar interacciones complejas.

---

## Análisis de Resultados y Criterio de Selección

### Rendimiento Relativo
Los resultados en el conjunto de test muestran de manera consistente que el algoritmo de Random Forest obtiene el coeficiente de determinación (R²) más alto y el menor error cuadrático medio (RMSE). 

La superioridad de los modelos basados en árboles frente a la regresión lineal radica en la naturaleza de los datos. La regresión lineal asume un impacto constante por variable; sin embargo, en este escenario existe un comportamiento marcadamente no lineal y multiplicativo. El caso más evidente ocurre con la interacción entre el tabaquismo y el índice de masa corporal (IMC): un paciente fumador con un IMC estándar no experimenta el mismo incremento exponencial en sus costos que un paciente fumador que además presenta un diagnóstico de obesidad (IMC mayor a 30). Los árboles de decisión logran identificar y aislar estas interacciones mediante sus divisiones nodales.

### Control de Sobreajuste
Es importante notar que los modelos de ensamble tienden a memorizar con facilidad el conjunto de entrenamiento (alcanzando valores de R² cercanos al 98%). Para mitigar este riesgo en el bosque aleatorio, se aplicaron restricciones de profundidad y tamaño de muestras. La diferencia controlada entre las métricas de train y test confirma que el modelo es estable frente a información nueva.

### Propuesta de Despliegue
Para el entorno de producción se seleccionó el modelo de Random Forest. En la gestión del riesgo financiero dentro de la ingeniería de servicios de salud, subestimar el costo de un siniestro debido a las limitaciones de un modelo lineal plano representa un peligro económico crítico, justificando por completo el uso de un modelo de ensamble.

---

## Guía de Ejecución

Para poner en marcha los componentes del proyecto, ejecute los siguientes comandos en la terminal desde el directorio raíz:

### 1. Instalación de librerías
Asegure las dependencias necesarias en su entorno local:
```bash
pip install -r requirements.txt