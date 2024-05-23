# Carga de las librerías necesarias
from tkinter.ttk import Notebook
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Análisis de Datos del Dataset de OLX Cars

Este Notebook realiza un análisis exploratorio y modelado predictivo del dataset de autos usados listados en OLX. 
El objetivo es entender las características de los autos y predecir sus precios de forma efectiva.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

##Cargamos el data set 

# Cargando el dataset
data_path = 'C:\Users\Sergi\OneDrive\Documentos'
data = pd.read_csv(data_path)
data.head()

# Mostrar los nombres de las columnas
print(data.columns)

##Como proceso de data wrarling revisamos los DATOS FALTANTES y buscamos columnas con valores faltantes 

# Información general del dataset
data.info()

## Limpieza de Datos

El proceso de limpieza incluye la eliminación de duplicados, el manejo de valores faltantes y la corrección de tipos de datos erróneos.

# Eliminación de duplicados
data.drop_duplicates(inplace=True)

# Identificar columnas numericas
numeric_cols = data.select_dtypes(include=np.number).columns

# Llenar valores faltantes solo en columnas numericas
data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

# Conversión de tipos de datos erróneos
data['Price'] = pd.to_numeric(data['Price'], errors='coerce')
data.info()

## Análisis Exploratorio de Datos (EDA)

Realizamos algunas visualizaciones para entender la distribución de los precios y la relación entre el año del coche y su precio.

# Visualización de la distribución de precios
plt.figure(figsize=(10, 6))
sns.histplot(data['Price'], kde=True)
plt.title('Distribución de Precios')
plt.show()

# Boxplot de precios para detectar valores atípicos
plt.figure(figsize=(10, 6))
sns.boxplot(x=data['Price'])
plt.title('Boxplot de Precios')
plt.show()

# Relación entre el año del coche y el precio
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Year', y='Price', data=data)
plt.title('Precio vs Año del Coche')
plt.show()


## Modelado Predictivo

Preparamos los datos para el modelo, entrenamos un modelo de regresión lineal y evaluamos su rendimiento.


X = data[['Year', "KM's driven"]]
y = data['Price']

# Dividir los datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo de regresión lineal
model = LinearRegression()
model.fit(X_train, y_train)

# Predicciones y evaluacion del modelo
predictions = model.predict(X_test)
mse = mean_squared_error(y_test, predictions)
print(f"El MSE del modelo es: {mse}")