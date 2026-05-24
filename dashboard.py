# IRIS SPECIES CLASSIFICATION DASHBOARD

# importar librerias

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.datasets import load_iris
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# config de la pagina

st.set_page_config(
    page_title="Iris Classification",
    layout="wide"
)

st.title("Iris Species Classification")
st.markdown("---")
st.markdown("""##### Students:
- Cortés Cure Iván José
            
- Mercado Rachath Esteban
            
##### Group:
- Minería de Datos | 18038""")


# Cargamos dataset

iris = load_iris()

df = pd.DataFrame(
    iris.data,
    columns=iris.feature_names
)

df["species"] = iris.target

df["species"] = df["species"].map({
    0: "setosa",
    1: "versicolor",
    2: "virginica"
})

# Preparamos los DATOS 

X = df.drop("species", axis=1)
y = df["species"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Cargamos el modelo 

model = joblib.load("iris_model.pkl")

# Metricas de predicción

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(
    y_test,
    y_pred,
    average="weighted"
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

# Herramienta ajustable de entrada de datos (sidebar)

st.sidebar.header("Medidas de la flor")

sepal_length = st.sidebar.slider(
    "Sepal Length (Longitud del Sépalo)",
    4.0,
    8.0,
    5.4
)

sepal_width = st.sidebar.slider(
    "Sepal Width (Ancho del Sépalo)",
    2.0,
    4.5,
    3.4
)

petal_length = st.sidebar.slider(
    "Petal Length (Longitud del Pétalo)",
    1.0,
    7.0,
    1.3
)

petal_width = st.sidebar.slider(
    "Petal Width (Ancho del Pétalo)",
    0.1,
    2.5,
    0.2
)


# Nueva muestra

new_sample = pd.DataFrame({
    "sepal length (cm)": [sepal_length],
    "sepal width (cm)": [sepal_width],
    "petal length (cm)": [petal_length],
    "petal width (cm)": [petal_width]
})

st.markdown("---")
# Predicción de la muestra

prediction = model.predict(new_sample)

st.subheader("Predicted Species (Especie Predicha)")

st.success(f"La flor pertenece a el grupo de: {prediction[0].upper()}")

st.markdown("---")
# Metricas visuales del modelo

st.subheader("Model Metrics (Métricas del modelo)")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Accuracy", f"{accuracy:.2f}")
col2.metric("Precision", f"{precision:.2f}")
col3.metric("Recall", f"{recall:.2f}")
col4.metric("F1 Score", f"{f1:.2f}")

st.markdown("---")
# Histogramas

st.subheader("Histogram")

fig_hist = px.histogram(
    df,
    x="petal length (cm)",
    color="species",
    marginal="box",
    nbins=30
)

st.plotly_chart(fig_hist, use_container_width=True)

st.markdown("---")
# Matriz de dispersión

st.subheader("Scatter Matrix (Matriz de dispersión)")

fig_scatter = px.scatter_matrix(
    df,
    dimensions=[
        "sepal length (cm)",
        "sepal width (cm)",
        "petal length (cm)",
        "petal width (cm)"
    ],
    color="species"
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("---")
# Heatmap

st.subheader("Correlation Heatmap (Heatmap de correlación)")

fig, ax = plt.subplots(figsize=(8,6))

sns.heatmap(
    df.drop("species", axis=1).corr(),
    annot=True,
    cmap="Blues",
    ax=ax
)

st.pyplot(fig)

st.markdown("---")
# Grafico de dispersión 3D

st.subheader("3D Scatter Plot (Grafico de dispersión 3D)")

# Dataset original
fig_3d = px.scatter_3d(
    df,
    x="sepal length (cm)",
    y="petal length (cm)",
    z="petal width (cm)",
    color="species",
    opacity=0.7
)

# Agregar nueva muestra
fig_3d.add_scatter3d(
    x=[sepal_length],
    y=[petal_length],
    z=[petal_width],
    mode='markers',
    marker=dict(
        size=10,
        color='purple'
    ),
    name='nueva flor'
)

st.plotly_chart(fig_3d, use_container_width=True)

st.markdown("---")
# Tabla de datos

st.subheader("Iris Dataset")

st.dataframe(df)

st.markdown("---")
# Informacion adiciolal de la pagina 

st.markdown("Developed for Data Mining Final Project")