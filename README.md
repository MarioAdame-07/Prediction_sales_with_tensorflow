# Prediction_sales_with_tensorflow
Desarrollo e implementacion de un un modelo de redes neuronales recurrentes (LSTM) en tensorflow/keras para predecir la demanda de ventas de un dataset minorista con poco mas de 1 millon de registros

---

## 🛠️ Tecnologías Utilizadas
* **Python**: Lenguaje principal de desarrollo.
* **TensorFlow / Keras**: Construcción, entrenamiento y evaluación del modelo de aprendizaje profundo (LSTM).
* **Pandas & NumPy**: Carga, manipulación de datos, agregación diaria y estructuración de ventanas temporales.
* **Scikit-Learn**: Escalado de datos (`MinMaxScaler`) y cálculo de métricas de error (MAE, RMSE).
* **Matplotlib & Seaborn**: Visualización de la serie temporal y comparación gráfica de valores reales vs. predichos.

---


# CODIGO
Para el desarrollo de este proyecto se opto por implementar un código en Python seccionado en 6 fases para mejorar su comprensión y facilidad para modificar de acuerdo a los requerimientos de cada usuario  

El objetivo principal es capturar la estacionalidad anual y el comportamiento semanal de la demanda para anticipar los niveles de ventas futuros y optimizar la gestión de inventario.

## ⚙️ Estructura del Código (Pipeline en 6 Fases)
Para el desarrollo de este proyecto se optó por implementar un código en Python seccionado en **6 fases** para mejorar su comprensión, mantenibilidad y facilitar su adaptación a diferentes conjuntos de datos:

1. **Fase 1: Carga y Exploración de Datos**: Conversión de tipos de datos de fecha, agregación diaria de ventas globales e inspección visual de la serie temporal.
2. **Fase 2: Preprocesamiento y Escalado**: Normalización de los datos de ventas en el rango $[0, 1]$ mediante `MinMaxScaler` y separación secuencial en conjuntos de Entrenamiento (80%) y Prueba (20%).
3. **Fase 3: Creación de Ventanas Temporales (*Windowing*)**: Transformación de la serie temporal a un problema de aprendizaje supervisado utilizando ventanas de historial de 30 días para predecir el día siguiente.
4. **Fase 4: Arquitectura del Modelo**: Diseño de una red neuronal secuencial con capas `LSTM` apiladas, capas `Dropout` (0.2) para prevenir sobreajuste y compilación con el optimizador `Adam`.
5. **Fase 5: Entrenamiento y Evaluación**: Entrenamiento de la red durante 20 épocas, desescalado de valores a su magnitud real e inspección de métricas de desempeño (**MAE** y **RMSE**).
6. **Fase 6: Visualización de Resultados y Exportación**: Generación de gráficos comparativos (*Ventas Reales vs. Predicción LSTM*) y guardado del modelo entrenado (`modelo_demanda_lstm.h5`).

---

## 📊 Resultados e Impacto
* **Captura de Tendencias y Estacionalidad**: El modelo logró adaptarse eficazmente a las fluctuaciones anuales y a las variaciones semanales ("picos de sierra") sin presentar desfase temporal.
* **Evaluación en Datos No Vistos**: La predicción sobre el 20% de prueba demostró una alta precisión en la estimación de la curva de demanda.

---

## 📁 Estructura del Repositorio
* `retail_sales.py: Script principal estructurado en las 6 fases de desarrollo.
* `retail_sales.csv`: Dataset original de ventas minoristas.
* `modelo_demanda_lstm.h5`: Modelo entrenado exportado.
* `ventas_totales_diarias.png`: Gráfica exploratoria de la serie temporal (2019-2023).
* `prediccion_lstm_resultados.png`: Gráfica comparativa de la predicción del modelo frente a los datos reales.
