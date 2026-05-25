#  Iris Species Classification Dashboard

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange?logo=scikitlearn&logoColor=white)
![Status](https://img.shields.io/badge/Status-Completed-brightgreen)

An interactive Machine Learning dashboard that classifies Iris flower species in real time using a **Random Forest** model and a **Streamlit** interface.

>  Course: Data Mining — **18038** | Universidad de la Costa (CUC)

---

##  Students

| Name | GitHub |
|------|--------|
| Cortés Cure Iván José | [@ijcure](https://github.com/ijcure) |
| Mercado Rachath Esteban | [@EstebanDMR](https://github.com/EstebanDMR) |

---

##  Objectives

- Understand and analyze the Iris dataset
- Train a classification model using Random Forest
- Evaluate model performance with standard metrics
- Build an interactive dashboard for real-time prediction
- Visualize dataset behavior and model insights

---

##  Project Structure

```
📁 Iris-Classification-Dashboard/
├── dashboard.py          # Streamlit dashboard (main app)
├── iris_training.py      # Model training script
├── iris_model.pkl        # Trained Random Forest model
├── iris.csv              # Iris dataset
├── requirements.txt      # Python dependencies
└── important.txt         # Quick setup guide
```

---

##  Dataset

The **Iris dataset** contains 150 samples across 3 species:

| Species | Samples |
|---------|---------|
| Iris Setosa | 50 |
| Iris Versicolor | 50 |
| Iris Virginica | 50 |

**Features used for classification:**
- Sepal Length (cm)
- Sepal Width (cm)
- Petal Length (cm)
- Petal Width (cm)

---

##  Machine Learning Model

**Algorithm:** Random Forest Classifier

**Metrics evaluated:**

| Metric | Description |
|--------|-------------|
| Accuracy | Overall correct predictions |
| Precision | Correctness of positive predictions |
| Recall | Coverage of actual positives |
| F1 Score | Harmonic mean of precision and recall |

---

##  Dashboard Features

The Streamlit dashboard includes:

-  Interactive sliders for flower measurements
-  Real-time species prediction
-  Model performance metrics
-  Histograms & Scatter Matrix
-  Correlation Heatmap
-  3D Scatter Plot
-  Feature Importance chart
-  Confusion Matrix

---

##  Technologies Used

| Category | Tools |
|----------|-------|
| Language | Python 3.10+ |
| ML | Scikit-learn |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib, Seaborn |
| Data | Pandas, NumPy |

---

##  How to Run

### 1. Clone the repository
```bash
git clone https://github.com/ijcure/Iris-Classification-Dashboard.git
cd Iris-Classification-Dashboard
```

### 2. Create and activate a virtual environment *(recommended)*
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Train the model
```bash
python iris_training.py
```
This will generate the `iris_model.pkl` file.

### 5. Launch the dashboard
```bash
python -m streamlit run dashboard.py
```

Your browser will open the dashboard automatically at `http://localhost:8501`.

---

## License

This project was developed for academic purposes at **Universidad de la Costa (CUC)**.
