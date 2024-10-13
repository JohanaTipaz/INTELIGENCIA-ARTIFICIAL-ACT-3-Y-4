import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt  # Importar la biblioteca para visualización

def crear_dataset_simulado(n_sample=1000):
    """
    Crea un dataset simulado para un sistema de transporte.

    Args:
        n_sample (int): Número de muestras a generar. Por defecto es 1000.

    Returns:
        DataFrame: DataFrame que contiene el dataset simulado.
    """
    np.random.seed(42)  # Para reproducibilidad

    # Generamos los datos simulados
    data = {
        'fecha_hora': pd.date_range(start='2024-01-01', periods=n_sample, freq='H'),
        'estacion_origen': np.random.choice(['Estacion_A', 'Estacion_B', 'Estacion_C'], n_sample),
        'estacion_destino': np.random.choice(['Estacion_A', 'Estacion_B', 'Estacion_C'], n_sample),
        'pasajeros': np.random.poisson(20, n_sample),  # Distribución de pasajeros simulada
        'tiempo_viaje_minutos': np.random.normal(loc=15, scale=5, size=n_sample).clip(min=5),
        'tipo_transporte': np.random.choice(['bus', 'metro', 'tren'], n_sample),
        'clima': np.random.choice(['soleado', 'lluvioso', 'nublado'], n_sample),
        'densidad_trafico': np.random.uniform(0, 1, n_sample)  # Densidad de tráfico simulada
    }

    df_simulado = pd.DataFrame(data)  
    return df_simulado

# Crear el dataset simulado
df_transporte = crear_dataset_simulado()  

# Mostramos los primeros 5 registros del dataset
print(df_transporte.head())

# Separar características y la variable objetivo
X = df_transporte.drop(columns=['pasajeros', 'fecha_hora'])
y = df_transporte['pasajeros']

# Dividir el conjunto de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocesamiento de las variables categóricas
categorical_cols = ['estacion_origen', 'estacion_destino', 'tipo_transporte', 'clima']
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), categorical_cols)
    ], remainder='passthrough')

# Crear el pipeline con el preprocesador y el modelo
model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Entrenar el modelo
model.fit(X_train, y_train)

# Realizar predicciones
y_pred = model.predict(X_test)

# Evaluar el modelo
mse = mean_squared_error(y_test, y_pred)
print(f'Error cuadrático medio (MSE): {mse}')

# Imprimir algunas predicciones y valores reales
for i in range(10):
    print(f'Valor real: {y_test.values[i]}, Predicción: {y_pred[i]}')

# Visualizar resultados
plt.scatter(y_test, y_pred, alpha=0.5)
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--')  # Línea de referencia
plt.xlabel('Valores Reales')
plt.ylabel('Predicciones')
plt.title('Valores Reales vs Predicciones')
plt.show()
