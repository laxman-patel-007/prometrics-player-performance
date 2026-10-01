# ProMetrics: Player Performance Analysis (Case Study no. 102)

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](http://localhost:8501)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.6-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Case Study no. 102:** A sports organization wants to investigate measurable factors associated with player performance. (With Proper Justification)  
> **Course:** Machine Learning Fundamentals (Modules I through IX Full Alignment)

---

## 🌟 Project Highlights & Syllabus Alignment

This project delivers an end-to-end, mathematically rigorous Machine Learning solution strictly aligned with every module of the prescribed syllabus:

| Module | Curriculum Topic | Project Implementation |
| :--- | :--- | :--- |
| **Module I** | Introduction to Machine Learning | Real-world problem definition, Supervised (Regression & Classification) + Unsupervised workflow. |
| **Module II** | ML Libraries & Packages | Implemented with NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn, and Streamlit. |
| **Module III** | Data Preprocessing & Feature Engineering | Stratified position-based median imputation, domain composite indices, One-Hot Encoding, StandardScaler. |
| **Module IV** | Supervised Learning: Regression | Linear Regression (OLS), Polynomial Regression (Degree 2), Ridge Regression ($L_2$). |
| **Module V** | Supervised Learning: Classification | Multinomial Logistic Regression, K-Nearest Neighbors (KNN), Decision Tree Classifier. |
| **Module VI** | Model Evaluation & Validation | Stratified 80/20 train/test split, 5-Fold Cross Validation, $R^2$, RMSE, MAE, Accuracy, Precision, Recall, Macro F1, Confusion Matrix. |
| **Module VII** | Unsupervised Learning | K-Means Clustering (Elbow Method & Silhouette Validation across $k=2..7$), Hierarchical Clustering (Dendrogram). |
| **Module VIII**| Dimensionality Reduction & Ensembles | Principal Component Analysis (PCA Scree & Biplot), Random Forest Regressor & Classifier with Feature Importances. |
| **Module IX** | Neural Networks & Model Deployment | Scikit-Learn Multi-Layer Perceptron (MLP), Joblib model persistence, and interactive Streamlit web dashboard. |

---

## 🚀 Quickstart Guide

### 1. Activate Environment & Install Dependencies
```bash
# Clone / navigate to project directory
cd /Users/laxmanpatel/Desktop/ML

# Activate the virtual environment
source .venv/bin/activate

# (Optional) Verify or install dependencies
pip install -r requirements.txt
```

### 2. Launch the Interactive Streamlit Web Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** in your browser to access the 7-in-1 analytics platform:
1. **Executive Overview & Syllabus Alignment**
2. **Exploratory Data Analysis (EDA) & Measurable Factors**
3. **Performance Rating Prediction Engine (Regression)**
4. **Talent Tier Classification Studio**
5. **Tactical Archetypes & PCA Clustering**
6. **What-If Scouting & Conditioning Simulator**
7. **Model Evaluation & Benchmark Studio**

### 3. Open the Complete Academic Jupyter Notebook
```bash
jupyter notebook notebooks/player_performance_analysis.ipynb
```

### 4. Re-run End-to-End Pipeline Scripts
```bash
# Generate the realistic raw dataset
python src/data_generator.py

# Preprocess, impute, scale, and feature engineer
python src/preprocessing.py

# Train and benchmark all regression, classification, clustering, PCA, and neural net models
python src/train_models.py
```

---

## 📊 Experimental Results Summary

### Supervised Regression Benchmark (Predicting Overall Rating 50–95)
| Model Architecture | 5-Fold CV $R^2$ (Mean $\pm$ Std) | Test MAE | Test RMSE | Test $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **$0.9474 \pm 0.0066$** | **$1.4980$** | **$1.9189$** | **$0.9461$** |
| **Ridge Regression** | $0.9456 \pm 0.0053$ | $1.5527$ | $2.0449$ | $0.9388$ |
| **Linear Regression (OLS)** | $0.9454 \pm 0.0055$ | $1.5546$ | $2.0484$ | $0.9386$ |
| **MLP Regressor (Neural Net)** | $0.9208 \pm 0.0079$ | $1.7134$ | $2.2408$ | $0.9265$ |
| **Polynomial Regression (Deg 2)** | $0.8005 \pm 0.0339$ | $2.6836$ | $4.0069$ | $0.7650$ |

### Supervised Classification Benchmark (Predicting Talent Tier)
| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | $0.8770 \pm 0.0146$ | **$87.97\%$** | $0.8892$ | **$0.8791$** | **$0.8836$** |
| **Random Forest Classifier** | **$0.8926 \pm 0.0124$** | $87.81\%$ | **$0.8933$** | $0.8727$ | $0.8815$ |
| **MLP Classifier (Neural Net)** | $0.8617 \pm 0.0237$ | $84.53\%$ | $0.8683$ | $0.8365$ | $0.8493$ |
| **K-Nearest Neighbors (KNN)** | $0.8441 \pm 0.0219$ | $80.47\%$ | $0.8199$ | $0.7985$ | $0.8075$ |
| **Decision Tree Classifier** | $0.8258 \pm 0.0129$ | $79.84\%$ | $0.8155$ | $0.7913$ | $0.8011$ |

### Unsupervised Learning: Tactical Archetypes (K-Means & PCA)
- **Optimal Cluster Count:** $k=4$ (Peak Silhouette score $= 0.3487$)
  - **Cluster 0:** Tactical Playmaker & Orchestrator (Vision $84+$, Short Passing $86+$)
  - **Cluster 1:** Defensive Anchor & Ball-Winner (Tackling $85+$, Strength, Def. Awareness $86+$)
  - **Cluster 2:** Explosive Forward & Finisher (Sprint Speed $86+$, Finishing $84+$)
  - **Cluster 3:** Goalkeeper / Positional Specialist (Reflexes, Handling, Aerial Reach)
- **PCA Variance Explained:** Top 2 Principal Components explain **$52.95\%$** of variance; top 5 explain **$73.51\%$**.

---

## 📂 Repository File Structure

```
/Users/laxmanpatel/Desktop/ML/
├── .venv/                                # Isolated virtual environment with all ML packages
├── data/
│   ├── raw/
│   │   └── player_performance_raw.csv    # 3,200 player records with telemetry missingness
│   └── processed/
│       ├── player_performance_cleaned.csv# Cleaned dataset with 6 engineered indices
│       ├── X_train.csv / X_test.csv      # Scaled 80/20 train and test feature matrices
│       ├── y_train_reg.csv / y_test_reg.csv # Continuous performance ratings
│       └── y_train_clf.csv / y_test_clf.csv # Categorical talent tiers
├── models/
│   ├── linear_regression.joblib          # Module IV OLS
│   ├── polynomial_regression.joblib      # Module IV Degree 2
│   ├── ridge_regression.joblib           # Module IV Regularized
│   ├── random_forest_regressor.joblib    # Module VIII Ensemble
│   ├── mlp_regressor.joblib              # Module IX Neural Net
│   ├── logistic_regression.joblib        # Module V Multinomial
│   ├── knn_classifier.joblib             # Module V KNN
│   ├── decision_tree_classifier.joblib   # Module V Decision Tree
│   ├── random_forest_classifier.joblib   # Module VIII Ensemble
│   ├── mlp_classifier.joblib             # Module IX Neural Net
│   ├── kmeans_model.joblib               # Module VII Clustering
│   ├── pca_model.joblib                  # Module VIII Dimensionality Reduction
│   ├── scaler.joblib                     # Module III StandardScaler
│   ├── feature_metadata.json             # Feature names and schemas
│   └── metrics_summary.json              # Full 5-fold CV and test metrics
├── notebooks/
│   └── player_performance_analysis.ipynb # Complete Academic Jupyter Notebook (Modules I-IX)
├── src/
│   ├── data_generator.py                 # Authentic dataset synthesis module
│   ├── preprocessing.py                  # Module III imputation, scaling, & encoding
│   ├── train_models.py                   # Complete training, CV, & evaluation engine
│   └── create_notebook.py                # Notebook generator
├── app.py                                # Production Streamlit web application
├── requirements.txt                      # Project package dependencies
├── FINAL_REPORT.md                       # Comprehensive Academic Report & Viva Voce Guide
└── README.md                             # Project documentation and guide
```

---

## 🎓 Examination & Viva Voce Readiness
See [FINAL_REPORT.md](FINAL_REPORT.md) Section 13 for detailed viva voce questions and model answers addressing:
- Justification of position-stratified median imputation vs global mean.
- Mathematical mechanics of Ordinary Least Squares vs Random Forest ensembling.
- Interpretation of Principal Component Analysis skill vectors ($PC_1$ vs $PC_2$).
- Economic and tactical applications in elite sports management ("Moneyball" analytics).
