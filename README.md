# CricMetrics Pro: AI-Powered Multi-Task Cricket Performance Analytics, Talent Tier Classification & Fair Market Valuation
### Case Study: Player Performance Analysis | Academic & Enterprise Machine Learning System

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](https://share.streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.6-orange.svg)](https://scikit-learn.org/)
[![Dataset: Real IPL 2008-2024](https://img.shields.io/badge/Dataset-260k%20IPL%20Deliveries-green.svg)](https://cricsheet.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Assigned Problem Statement:**  
> *"A sports organization wants to investigate measurable factors associated with player performance (With Proper Justification)."*  
> **Student-Formulated Project Title:**  
> **CricMetrics Pro: AI-Powered Multi-Task Cricket Performance Analytics, Talent Tier Classification & Fair Market Valuation**  
> **Sponsoring Stakeholder:**  
> Professional T20 Cricket Franchise Board of Directors, Director of Cricket Operations, and Scouting / Auction Committee.

---

## 🌟 Executive Summary & Deliverables Mapping

CricMetrics Pro is an end-to-end Machine Learning decision support platform built on **260,920 real deliveries across 1,095 IPL matches (2008–2024)**, covering **619 qualified professional cricketers**. The project strictly fulfills all 8 assigned academic and industrial deliverables:

| Deliverable | Academic Objective | System Implementation |
| :--- | :--- | :--- |
| **1. Problem Definition** | Formulate problem & justification | Defined multi-task continuous rating estimation, 3-tier classification, and unsupervised archetype discovery to eliminate scouting heuristics and auction overbidding. |
| **2. Dataset & Docs** | Identify & justify real dataset | 17 seasons of ball-by-ball IPL match telemetry (`data/processed/cricket_players_clean.csv`), complete data dictionary, and data quality observations. |
| **3. Exploratory Analysis** | EDA & factor insights | Pearson correlation of measurable metrics ($r = 0.84$ for clutch index), strike rate vs average quadrant matrix, and multi-skill role radars. |
| **4. Preprocessing** | Cleaning, scaling & encoding | Zero-division role imputation, outlier thresholds ($\ge 3$ matches), One-Hot categorical encoding, and StandardScaler normalization. |
| **5. Model Development** | Train & compare ML algorithms | **3 Regressors** (Linear, Polynomial, Random Forest) + **4 Classifiers** (KNN, Random Forest, Logistic Regression, Decision Tree) + **K-Means ($k=5$)** + **2D PCA**. |
| **6. Model Evaluation** | Rigorous evaluation & error study | MAE, RMSE, $R^2$, 5-Fold Stratified CV, Macro F1, interactive confusion matrices, Gini feature importance, and documented real-world limitations. |
| **7. Streamlit Application** | Functional interactive web app | Live interactive dashboard with 1-click iconic player presets (Kohli, Bumrah, Russell, Narine, Pandya), live predictions, and PCA map. |
| **8. Final Documentation** | Technical report & defense evidence | Complete `FINAL_REPORT.md`, Jupyter Notebook, user guides, and viva defense reference cards. |

---

## 📊 Machine Learning Model Leaderboards

### 1. Supervised Regression (Overall Rating & Fair Auction Valuation)
*Target: `overall_performance_rating` (Continuous score 50.0 to 95.0)*

| Model Algorithm | Test $R^2$ Score | 5-Fold CV $R^2$ | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Evaluation Verdict |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Polynomial Regression (Degree 2)** | **0.9941** | **0.9956** | **0.5147** | **0.6648** | 🏆 **Top Accuracy** (Captures non-linear metric synergy) |
| **Random Forest Regressor** | **0.9780** | **0.9522** | **0.9412** | **1.2837** | 🌲 **Most Robust** (Ensemble of 150 bagged trees) |
| **Multiple Linear Regression** | 0.9490 | 0.8797 | 1.4820 | 1.9534 | Baseline Linear Model |

### 2. Supervised Classification (Talent Tier Prediction)
*Target: `performance_tier` (3 Classes: Elite / Marquee, Core / Star, Developing / Squad)*

| Classifier Algorithm | Test Accuracy | 5-Fold Stratified CV | Macro Precision | Macro Recall | Macro F1-Score | Evaluation Verdict |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **K-Nearest Neighbors (KNN, k=7)** | **95.16%** | 90.30% | **0.9521** | **0.9504** | **0.9501** | 🏆 **Top Test Accuracy** |
| **Random Forest Classifier** | **94.35%** | **94.75%** | **0.9468** | **0.9452** | **0.9459** | 🛡️ **Highest CV Generalization** |
| **Multinomial Logistic Regression** | 87.90% | 92.53% | 0.8872 | 0.8841 | 0.8853 | Probabilistic Baseline |
| **Decision Tree Classifier (CART)** | 87.90% | 89.90% | 0.8835 | 0.8802 | 0.8812 | White-Box Decision Logic |

### 3. Unsupervised Tactical Archetypes (K-Means, $k=5$)
*Algorithm: K-Means Clustering on 10 standardized performance traits ($Silhouette = 0.2676$)*
1. **0: Top-Order Anchor & Accumulator** (Virat Kohli, Shikhar Dhawan, KL Rahul)
2. **1: Powerplay & Middle-Overs Pace Specialist** (Jasprit Bumrah, Lasith Malinga, Bhuvneshwar Kumar)
3. **2: Death-Overs Finisher & Power Hitter** (Andre Russell, Heinrich Klaasen, Kieron Pollard)
4. **3: Defensive Middle-Overs Economy Spinner** (Sunil Narine, Rashid Khan, Yuzvendra Chahal)
5. **4: Elite Dual-Threat All-Rounder** (Hardik Pandya, Ravindra Jadeja, Shane Watson)

---

## 🚀 Quickstart Guide

### 1. Installation & Setup
```bash
# Clone / navigate to project repository
cd /Users/laxmanpatel/Desktop/ML

# Activate virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Train and Serialize Models
```bash
python3 src/train_cricket_models.py
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to access the 6-part dashboard:
1. 🏛️ **Deliverable 1: Problem Definition** (Formal definition, business justification, KPI cards)
2. 📋 **Deliverable 2: Dataset & Preprocessing** (Interactive database, data dictionary, quality steps)
3. 📊 **Deliverable 3: Exploratory Analysis (EDA)** (Correlations, role radars, quadrant matrices)
4. ⚡ **Deliverable 4: AI Valuation Engine** (1-click presets, live regression & classification predictions)
5. 🧩 **Deliverable 5: Tactical Archetypes & PCA** (2D latent space scatter plot with cluster inspection)
6. 🏆 **Deliverable 6: Benchmarks & Evaluation** (Side-by-side leaderboards, confusion matrices, limitations)

### 4. Run the Jupyter Notebook
```bash
jupyter notebook notebooks/cricket_player_performance.ipynb
```

---

## 📁 Repository Structure

```
├── app.py                             # Full 6-Deliverable Streamlit Web Application
├── requirements.txt                   # Production package dependencies
├── README.md                          # Executive project documentation & quickstart
├── FINAL_REPORT.md                    # Comprehensive academic viva report
├── data/
│   ├── raw/                           # Raw match event logs (deliveries & matches)
│   └── processed/
│       └── cricket_players_clean.csv  # 619 cleaned player profiles across 17 seasons
├── models/
│   ├── scaler.joblib                  # Fitted StandardScaler
│   ├── linear_regression.joblib       # Multiple Linear Regression model
│   ├── polynomial_regression.joblib   # Degree 2 Polynomial Regression model
│   ├── random_forest_regressor.joblib # Random Forest Regressor model
│   ├── knn_classifier.joblib          # K-Nearest Neighbors Classifier model
│   ├── random_forest_classifier.joblib# Random Forest Classifier model
│   ├── logistic_regression.joblib     # Multinomial Logistic Regression model
│   ├── decision_tree_classifier.joblib# Decision Tree Classifier model
│   ├── kmeans_model.joblib            # K-Means Clustering model (k=5)
│   ├── pca_model.joblib               # 2D Principal Component Analysis model
│   ├── feature_metadata.json          # Schema & feature mapping definitions
│   └── metrics_summary.json           # Serialized test & CV metrics
├── notebooks/
│   └── cricket_player_performance.ipynb # Executed end-to-end Jupyter Notebook
├── docs/                              # PDF and Markdown guides
└── src/
    ├── preprocess_cricket_data.py     # Data extraction and aggregation pipeline
    ├── train_cricket_models.py        # Model training and evaluation script
    └── create_cricket_notebook.py     # Clean notebook generator script
```

---

## 📜 License & Acknowledgments
* **License:** MIT Open Source License.
* **Data Acknowledgments:** Historical match delivery data courtesy of [Cricsheet](https://cricsheet.org/).
