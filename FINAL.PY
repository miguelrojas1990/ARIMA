#!pip install pandas matplotlib numpy statsmodels pmdarima scikit-learn
#!pip install pandas matplotlib numpy statsmodels pmdarima scikit-learn

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from sklearn.metrics import mean_squared_error
from math import sqrt

# Datos sinteticos de poblacion y felicidad
datos_ciudades_felicidad = pd.DataFrame({
    'Ciudad': ['Tokyo', 'Delhi', 'Shanghai', 'Sao Paulo', 'Mumbai', 'Beijing', 'Estambul', 'Karachi', 'Dhaka', 'Ciudad de Mexico'],
    'Poblacion': [37797531, 29399141, 26317104, 21192735, 20876485, 19416096, 15029231, 14910352, 14799100, 12619078],
    'Anio': [2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024],
    'Felicidad': [110, 113, 115, 27, 20, 42, 46, 136, 133, 43]})

# Poner los datos del 2015 al 2020, en donde sí tenemos datos
for i in range(2015, 2020):
    datos_ciudades_felicidad = pd.concat([datos_ciudades_felicidad, pd.DataFrame({
        'Ciudad': ['Tokyo', 'Delhi', 'Shanghai', 'Sao Paulo', 'Mumbai', 'Beijing', 'Estambul', 'Karachi', 'Dhaka', 'Ciudad de Mexico'],
        'Poblacion': [37797531, 29399141, 26317104, 21192735, 20876485, 19416096, 15029231, 14910352, 14799100, 12619078],
        'Anio': [i] * 10,
        'Felicidad': [np.random.randint(20, 150) for _ in range(10)]})], ignore_index=True)
    
#Predecimos algunos datos de los años siguientes
for i in range(2020, 2026):
    datos_ciudades_felicidad = pd.concat([datos_ciudades_felicidad, pd.DataFrame({
        'Ciudad': ['Tokyo', 'Delhi', 'Shanghai', 'Sao Paulo', 'Mumbai', 'Beijing', 'Estambul', 'Karachi', 'Dhaka', 'Ciudad de Mexico'],
        'Poblacion': [37797531, 29399141, 26317104, 21192735, 20876485, 19416096, 15029231, 14910352, 14799100, 12619078],
        'Anio': [i] * 10,
        'Felicidad': [np.random.randint(20, 150) for _ in range(10)] })], ignore_index=True)

# juntamos todos los datos para tener un promedio de las tendencias
tendencia_promedio = datos_ciudades_felicidad.groupby('Anio')['Felicidad'].mean()

# Gráfica del promedio
plt.figure(figsize=(12, 6))
plt.plot(tendencia_promedio.index, tendencia_promedio.values, label='Tendencia Promedio', linestyle='dashed', marker='x')

# Transformamos los datos a una serie temporal
serie_temporal = datos_ciudades_felicidad.groupby('Anio')['Felicidad'].mean()
serie_temporal

# El test de Dickey-Fuller
result = adfuller(serie_temporal)
print('Estadística de prueba de Dickey-Fuller:', result[0])
print('Valor p:', result[1])

# Diferenciación y nuevo test de Dickey-Fuller
serie_temporal_diff = serie_temporal.diff().dropna()
result_diff = adfuller(serie_temporal_diff)
print('Estadística de prueba de Dickey-Fuller después de diferenciación:', result_diff[0])
print('Valor p después de diferenciación:', result_diff[1])

# Serie temporal y diferenciada
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
plt.plot(serie_temporal, label='Felicidad Promedio')
plt.title('Serie Temporal de Felicidad Promedio')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(serie_temporal_diff, label='Diferenciada')
plt.title('Serie Temporal Diferenciada de Felicidad Promedio')
plt.legend()

plt.tight_layout()
plt.show()

# Funciones de Autocorrelación ACF 
plt.figure(figsize=(12, 6))

# ACF
plt.subplot(2, 1, 1)
plot_acf(serie_temporal_diff.values, ax=plt.gca())  # Utilizar .values para evitar la escala
plt.title('Función de Autocorrelación (ACF) de la Serie Temporal Diferenciada')
plt.tight_layout()
plt.show()

# Propuestas de modelos ARIMA
modelo_auto_arima = auto_arima(serie_temporal, seasonal=False, suppress_warnings=True)
orden_auto_arima = modelo_auto_arima.order
print('Propuesta de orden de modelo auto_arima:', orden_auto_arima)

#Primera respuesta AUTOREGRESION, 
#si es 1 significa que está relacionada con la

#Diferenciación: 
#cantidad de veces que se ha restado la observación actual de la observación anterior

#MA (Media movil):
#el error residual de la observación actual 
#está relacionado con el error residual de la observación anterior

# Gráfica del promedio
plt.figure(figsize=(12, 6))
plt.plot(serie_temporal.index, serie_temporal.values, label='Felicidad Promedio')
plt.plot(tendencia_promedio.index, tendencia_promedio.values, label='Tendencia Promedio', linestyle='dashed', marker='x')
plt.title('Serie Temporal de Felicidad Promedio y Tendencia')
plt.xlabel('Año')
plt.ylabel('Felicidad Promedio')
plt.legend()
plt.grid(True)
plt.show()

# Entrenar modelo ARIMA
modelo_arima = ARIMA(serie_temporal.loc['2020':], order=orden_auto_arima)
modelo_arima_entrenado = modelo_arima.fit()

# Generar predicciones
fechas_prediccion = pd.date_range(start='2020-01-01', end='2025-01-01', freq='A')
predicciones = modelo_arima_entrenado.get_forecast(steps=len(fechas_prediccion)).predicted_mean
predicciones.index = fechas_prediccion

# Imprimir predicciones
print("Predicciones ARIMA:")
print(predicciones)

# Obtener las predicciones y sus fechas
predicciones_df = pd.DataFrame({'Fecha': predicciones.index, 'Prediccion': predicciones.values})
print(predicciones_df)