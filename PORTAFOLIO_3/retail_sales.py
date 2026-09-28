
############################# FASE 1 ####################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. IMPORTAMOS DATASET
df = pd.read_csv(r'C:\Users\madam\PORTAFOLIO_3\retail_sales.csv')
# 2. PASAMOS A FORMATO FECHA
df['date'] = pd.to_datetime(df['date'], format = '%d/%m/%Y')
# 3. AGRUPAMOS POR FECHAS PARA OBTENER EL TOTAL DE VENTAS DIARIAS
df_daily = df.groupby('date')['sales'].sum().reset_index()
# 4. ORDENAMOS CRONOLOGICAMENTE
df_daily = df_daily.sort_values('date').reset_index(drop=True)

# 5. VISUALIZAR LAS PRIMERAS FILAS
print("--- Primeras 5 filas del total diario ---")
print(df_daily.head())

# 6. MOSTRAMOS FRAFICO DE LA SERIE TEMPORAL DE VENTAS DIARIAS
plt.figure(figsize=(12,5))
plt.plot(df_daily['date'], df_daily['sales'], color = '#007acc', linewidth =1)
plt.title('VENTAS TOTALES DIARIAS (2019-2023)')
plt.xlabel('FECHA')
plt.ylabel('VENTAS TOTALES')
plt.grid(True, linestyle = '--', alpha = 0.5)
plt.show()

############################# FASE 2 ####################################

from sklearn.preprocessing import MinMaxScaler

# 1. Extraer los valores de la columna 'sales' como un arreglo numpy 2D
sales_data = df_daily[['sales']].values

# 2. Normalizar las ventas en el rango [0, 1]
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_sales = scaler.fit_transform(sales_data)

# 3. Definir la proporción para el conjunto de entrenamiento (80%)
train_size = int(len(scaled_sales) * 0.80)
test_size = len(scaled_sales) - train_size

# 4. Dividir de forma secuencial
train_data = scaled_sales[0:train_size]
test_data = scaled_sales[train_size:len(scaled_sales)]

print(f"Total de registros diarios: {len(scaled_sales)}")
print(f"Registros de entrenamiento (Train): {len(train_data)}")
print(f"Registros de prueba (Test): {len(test_data)}")

############################# FASE 3 ####################################

# Definir la ventana de tiempo (30 días de historial)
time_step = 30

def create_dataset(dataset, time_step=1):
    X, y = [], []
    for i in range(len(dataset) - time_step):
        X.append(dataset[i:(i + time_step), 0])
        y.append(dataset[i + time_step, 0])
    return np.array(X), np.array(y)

# Crear conjuntos X e y para Train y Test
X_train, y_train = create_dataset(train_data, time_step)
X_test, y_test = create_dataset(test_data, time_step)

# Redimensionar la entrada a 3D para TensorFlow: (muestras, pasos_temporales, características)
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

print(f"Forma de X_train: {X_train.shape}")
print(f"Forma de X_test: {X_test.shape}")

############################# FASE 4 ####################################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

# Construir el modelo secuencial
model = Sequential([
    # Primera capa LSTM con Dropout
    LSTM(64, return_sequences=True, input_shape=(time_step, 1)),
    Dropout(0.2),
    
    # Segunda capa LSTM
    LSTM(32, return_sequences=False),
    Dropout(0.2),
    
    # Capa densa de salida (1 valor de predicción)
    Dense(1)
])

# Compilar el modelo usando Adam y Error Cuadrático Medio
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

############################# FASE 5 ####################################

from sklearn.metrics import mean_absolute_error, mean_squared_error

# 1. Entrenar la red
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=20,
    batch_size=32,
    verbose=1
)

# 2. Generar predicciones
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)

# 3. Deshacer el escalado para volver a las ventas reales
train_predict_real = scaler.inverse_transform(train_predict)
y_train_real = scaler.inverse_transform(y_train.reshape(-1, 1))

test_predict_real = scaler.inverse_transform(test_predict)
y_test_real = scaler.inverse_transform(y_test.reshape(-1, 1))

# 4. Calcular métricas de rendimiento en el conjunto de prueba
mae = mean_absolute_error(y_test_real, test_predict_real)
rmse = np.sqrt(mean_squared_error(y_test_real, test_predict_real))

print("\n--- Métricas de Desempeño (Conjunto de Prueba) ---")
print(f"Error Absoluto Medio (MAE): {mae:.2f} unidades")
print(f"Raíz del Error Cuadrático Medio (RMSE): {rmse:.2f} unidades")
############################# FASE 6 ####################################

# Graficar comparativa en el conjunto de prueba
plt.figure(figsize=(14, 6))
plt.plot(y_test_real, label='Ventas Reales (Test)', color='#2b5c8f', alpha=0.8)
plt.plot(test_predict_real, label='Predicción LSTM', color='#e74c3c', linestyle='--')
plt.title('Predicción de Demanda de Ventas Diarias con TensorFlow (LSTM)')
plt.xlabel('Días del periodo de prueba')
plt.ylabel('Ventas Totales')
plt.legend()
plt.grid(True, linestyle=':', alpha=0.6)
plt.show()

# Guardar el modelo entrenado
model.save('modelo_demanda_lstm.h5')
print("¡Modelo guardado exitosamente como 'modelo_demanda_lstm.h5'!")