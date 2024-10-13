import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Configuración de la semilla para la reproducibilidad
np.random.seed(42)

# Definición de parámetros
NUM_RECORDS = 1000
horarios = [f"{h:02}:00" for h in range(24)]  # Generar horas como cadenas (00:00 a 23:00)
rutas = ['Ruta A', 'Ruta B', 'Ruta C', 'Ruta D']
dias = ['Lunes', 'Martes', 'Miércoles', 'Jueves', 'Viernes', 'Sábado', 'Domingo']

# Creación del dataset
data = {
    'Fecha': np.random.choice(pd.date_range(start='2023-01-01', end='2023-01-31'), NUM_RECORDS),
    'Hora': np.random.choice(horarios, NUM_RECORDS),
    'Ruta': np.random.choice(rutas, NUM_RECORDS),
    'Dia': np.random.choice(dias, NUM_RECORDS),
    'Pasajeros': np.random.randint(0, 100, NUM_RECORDS),
    'Congestion': np.random.choice(['Bajo', 'Moderado', 'Alto'], NUM_RECORDS)
}

dataset = pd.DataFrame(data)

# Guardar el dataset en un archivo CSV
dataset.to_csv('datos_transporte_masivo.csv', index=False)
print("Dataset creado y guardado como datos_transporte_masivo.csv")

# Desarrollo de un Modelo de Aprendizaje No Supervisado
dataset = pd.read_csv('datos_transporte_masivo.csv')

# Preprocesamiento: seleccionar las variables numéricas
X = dataset[['Pasajeros']]

# Escalado de los datos
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Definición y entrenamiento del modelo K-means
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)

# Agregar etiquetas de cluster al dataset
dataset['Cluster'] = kmeans.labels_

# Visualización de los clusters
plt.figure(figsize=(10, 6))
plt.scatter(dataset['Fecha'], dataset['Pasajeros'], c=dataset['Cluster'], cmap='viridis')
plt.title('Clusters de Pasajeros en Transporte Masivo')
plt.xlabel('Fecha')
plt.ylabel('Número de Pasajeros')
plt.xticks(rotation=45)
plt.grid()
plt.show()