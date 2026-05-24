# agregamos librerias necesarias
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# librerías de Machine Learning
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# metricas
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# guardador de modelo
import joblib

# 1. CARGA DE DATOS
iris = load_iris()

# se crea el DataFrame
df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

# agregamos la columna objetivo
df["species"] = iris.target

# convertimos números a nombres
df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

print("""
----------------------------
Primeras filas del dataset:
----------------------------""")
print(df.head())


# 2. IINFORMACION GENERAL
print("""
----------------------------
Información del dataset:
----------------------------""")
print(df.info())

print("""
----------------------------
Valores nulos:
----------------------------""")
print(df.isnull().sum())

print("""
----------------------------
Estadísticas:
----------------------------""")
print(df.describe())


# 3. EDA (graficos y visualizaciones)

# hist = histograma 
df.hist(figsize=(10, 8))
plt.suptitle("Distribución de Variables")
plt.show()

# pairplot (seaborn)
sns.pairplot(df, hue="species") #"hue" nos hara categorizar las especies clasificadas por colores
plt.show()

# heatmap de correlación
plt.figure(figsize=(8,6))

sns.heatmap(
    df.drop("species", axis=1).corr(),
    annot=True,
    cmap="Blues"
)

plt.title("Matriz de Correlación")
plt.show()

# 4. PREPARACIÓN DE DATOS

# vaiable predictoras
X = df.drop("species", axis=1)

# variable objetivo
y = df["species"]

# dividimos dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("-----------------------------")
print("Datos de entrenamiento:", X_train.shape)
print("-----------------------------")
print("Datos de prueba:", X_test.shape)
print("-----------------------------")

# 5. TRAINING MODEL (entrenamiento del modelo)

# creamos modelo Random Forest
model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

# entrenar modelo, es decir lo adecuamos a los datos y configuracion anteriormente dada (100,42)
model.fit(X_train, y_train)
print("\nModelo entrenado correctamente (RandomForestClassifier)")

# 6. PREDICCIONES

y_pred = model.predict(X_test)

# 7. METRICAS 

accuracy = accuracy_score(y_test, y_pred)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted" # "weighted" calcula la precisión de cada clase y luego hace un promedio ponderado según el número de muestras de cada clase.
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted"
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted"
)

print("""
----------------------------
Métricas del Modelo:
----------------------------""")

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")  
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")

print("""
----------------------------
Reporte de Clasificación:
----------------------------""")
print(classification_report(y_test, y_pred))

# 8. CONFUSION MATRIX (matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,5))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Purples",
)

plt.title("Matriz de confusión")
plt.xlabel("Predicción")
plt.ylabel("Valor Real")
plt.show()

# 9. GUARDAR MODELO (joblib) 

#Ya está guardado pero lo dejamos comentado por si quiere probar la linea de guardado
#para volver otra vez a guardar el archivo .pkl

#joblib.dump(model, "iris_model.pkl")

#print("""
#----------------------------
#Modelo guardado como: iris_model.pkl
#----------------------------""")