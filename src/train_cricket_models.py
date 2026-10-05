"""
Cricket ML Model Training Engine: Real IPL Player Performance
Trains, evaluates, and persists Supervised Regression, Supervised Classification, and Unsupervised Models:
- Supervised Regression (Overall Rating Estimation & Fair Valuation):
  * Linear Regression
  * Polynomial Regression (Degree 2)
  * Random Forest Regressor
- Supervised Classification (Talent Tier Prediction):
  * K-Nearest Neighbors (KNN)
  * Random Forest Classifier
  * Multinomial Logistic Regression
  * Decision Tree Classifier (CART)
- Unsupervised Tactical Archetype Discovery (K-Means Clustering, k=5)
- Latent Space Dimensionality Reduction (PCA, 2 Components)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    silhouette_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/cricket_players_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate_models():
    print(f"Loading cleaned cricket dataset from {DATA_PATH}...", flush=True)
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded: {len(df)} players, {df.shape[1]} columns.", flush=True)

    # 1. Feature Definition
    num_features = [
        'matches_played', 'total_runs', 'balls_faced', 'batting_average', 
        'batting_strike_rate', 'fours', 'sixes', 'boundary_run_pct', 
        'dot_ball_faced_pct', 'highest_score', 'thirties', 'fifties', 
        'death_overs_strike_rate', 'overs_bowled', 'wickets_taken', 
        'economy_rate', 'bowling_strike_rate', 'bowling_average', 
        'dot_ball_bowled_pct', 'three_plus_wickets', 'death_overs_economy', 
        'player_of_match_awards', 'batting_impact_index', 'bowling_impact_index', 
        'clutch_match_winner_index'
    ]
    cat_features = ['primary_role']

    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=True, dtype=float)
    feature_names = df_encoded.columns.tolist()

    y_reg = df['overall_performance_rating']
    y_clf = df['performance_tier_code']

    # Train / Test Split (80/20 Stratified on performance tier)
    X_train_raw, X_test_raw, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        df_encoded, y_reg, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    # Standard Scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("StandardScaler serialized.", flush=True)

    # =========================================================================
    # 2. Supervised Regression Models (Overall Rating Prediction)
    # =========================================================================
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    reg_metrics = {}

    # Regressor 1: Multiple Linear Regression
    lr = LinearRegression()
    lr_cv = cross_val_score(lr, X_train, y_train_reg, cv=kf, scoring='r2')
    lr.fit(X_train, y_train_reg)
    lr_preds = lr.predict(X_test)
    reg_metrics['Linear Regression'] = {
        'Test_R2': round(float(r2_score(y_test_reg, lr_preds)), 4),
        'CV_R2_mean': round(float(lr_cv.mean()), 4),
        'CV_R2_std': round(float(lr_cv.std()), 4),
        'Test_MAE': round(float(mean_absolute_error(y_test_reg, lr_preds)), 4),
        'Test_RMSE': round(float(np.sqrt(mean_squared_error(y_test_reg, lr_preds))), 4)
    }
    joblib.dump(lr, os.path.join(MODELS_DIR, "linear_regression.joblib"))

    # Regressor 2: Polynomial Regression (Degree 2 with Ridge Regularization)
    poly_reg = Pipeline([
        ('poly', PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
        ('ridge', Ridge(alpha=10.0, random_state=42))
    ])
    poly_cv = cross_val_score(poly_reg, X_train, y_train_reg, cv=kf, scoring='r2')
    poly_reg.fit(X_train, y_train_reg)
    poly_preds = poly_reg.predict(X_test)
    reg_metrics['Polynomial Regression'] = {
        'Test_R2': round(float(r2_score(y_test_reg, poly_preds)), 4),
        'CV_R2_mean': round(float(poly_cv.mean()), 4),
        'CV_R2_std': round(float(poly_cv.std()), 4),
        'Test_MAE': round(float(mean_absolute_error(y_test_reg, poly_preds)), 4),
        'Test_RMSE': round(float(np.sqrt(mean_squared_error(y_test_reg, poly_preds))), 4)
    }
    joblib.dump(poly_reg, os.path.join(MODELS_DIR, "polynomial_regression.joblib"))

    # Regressor 3: Random Forest Regressor
    rf_reg = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf_reg_cv = cross_val_score(rf_reg, X_train, y_train_reg, cv=kf, scoring='r2')
    rf_reg.fit(X_train, y_train_reg)
    rf_preds = rf_reg.predict(X_test)
    reg_metrics['Random Forest Regressor'] = {
        'Test_R2': round(float(r2_score(y_test_reg, rf_preds)), 4),
        'CV_R2_mean': round(float(rf_reg_cv.mean()), 4),
        'CV_R2_std': round(float(rf_reg_cv.std()), 4),
        'Test_MAE': round(float(mean_absolute_error(y_test_reg, rf_preds)), 4),
        'Test_RMSE': round(float(np.sqrt(mean_squared_error(y_test_reg, rf_preds))), 4)
    }
    joblib.dump(rf_reg, os.path.join(MODELS_DIR, "random_forest_regressor.joblib"))

    print("\n--- Regression Model Evaluation ---", flush=True)
    for m, vals in reg_metrics.items():
        print(f"{m:26s} | Test R²: {vals['Test_R2']:.4f} | 5-Fold CV: {vals['CV_R2_mean']:.4f} | RMSE: {vals['Test_RMSE']:.4f}", flush=True)

    # =========================================================================
    # 3. Supervised Classification Models (Talent Tier Prediction)
    # =========================================================================
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    clf_metrics = {}

    # Model 1: K-Nearest Neighbors (KNN)
    knn = KNeighborsClassifier(n_neighbors=7, weights='distance')
    knn_cv = cross_val_score(knn, X_train, y_train_clf, cv=skf, scoring='accuracy')
    knn.fit(X_train, y_train_clf)
    knn_preds = knn.predict(X_test)
    clf_metrics['K-Nearest Neighbors (KNN)'] = {
        'Test_Accuracy': round(float(accuracy_score(y_test_clf, knn_preds)), 4),
        'CV_Accuracy_mean': round(float(knn_cv.mean()), 4),
        'CV_Accuracy_std': round(float(knn_cv.std()), 4),
        'Test_Precision': round(float(precision_score(y_test_clf, knn_preds, average='macro')), 4),
        'Test_Recall': round(float(recall_score(y_test_clf, knn_preds, average='macro')), 4),
        'Test_F1_Macro': round(float(f1_score(y_test_clf, knn_preds, average='macro')), 4),
        'Confusion_Matrix': confusion_matrix(y_test_clf, knn_preds).tolist()
    }
    joblib.dump(knn, os.path.join(MODELS_DIR, "knn_classifier.joblib"))

    # Model 2: Random Forest Classifier
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf_clf_cv = cross_val_score(rf_clf, X_train, y_train_clf, cv=skf, scoring='accuracy')
    rf_clf.fit(X_train, y_train_clf)
    rf_cpreds = rf_clf.predict(X_test)
    clf_metrics['Random Forest Classifier'] = {
        'Test_Accuracy': round(float(accuracy_score(y_test_clf, rf_cpreds)), 4),
        'CV_Accuracy_mean': round(float(rf_clf_cv.mean()), 4),
        'CV_Accuracy_std': round(float(rf_clf_cv.std()), 4),
        'Test_Precision': round(float(precision_score(y_test_clf, rf_cpreds, average='macro')), 4),
        'Test_Recall': round(float(recall_score(y_test_clf, rf_cpreds, average='macro')), 4),
        'Test_F1_Macro': round(float(f1_score(y_test_clf, rf_cpreds, average='macro')), 4),
        'Confusion_Matrix': confusion_matrix(y_test_clf, rf_cpreds).tolist()
    }
    joblib.dump(rf_clf, os.path.join(MODELS_DIR, "random_forest_classifier.joblib"))

    # Model 3: Multinomial Logistic Regression
    log_reg = LogisticRegression(max_iter=500, random_state=42)
    log_cv = cross_val_score(log_reg, X_train, y_train_clf, cv=skf, scoring='accuracy')
    log_reg.fit(X_train, y_train_clf)
    log_preds = log_reg.predict(X_test)
    clf_metrics['Logistic Regression'] = {
        'Test_Accuracy': round(float(accuracy_score(y_test_clf, log_preds)), 4),
        'CV_Accuracy_mean': round(float(log_cv.mean()), 4),
        'CV_Accuracy_std': round(float(log_cv.std()), 4),
        'Test_Precision': round(float(precision_score(y_test_clf, log_preds, average='macro')), 4),
        'Test_Recall': round(float(recall_score(y_test_clf, log_preds, average='macro')), 4),
        'Test_F1_Macro': round(float(f1_score(y_test_clf, log_preds, average='macro')), 4),
        'Confusion_Matrix': confusion_matrix(y_test_clf, log_preds).tolist()
    }
    joblib.dump(log_reg, os.path.join(MODELS_DIR, "logistic_regression.joblib"))

    # Model 4: Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42)
    dt_cv = cross_val_score(dt, X_train, y_train_clf, cv=skf, scoring='accuracy')
    dt.fit(X_train, y_train_clf)
    dt_preds = dt.predict(X_test)
    clf_metrics['Decision Tree'] = {
        'Test_Accuracy': round(float(accuracy_score(y_test_clf, dt_preds)), 4),
        'CV_Accuracy_mean': round(float(dt_cv.mean()), 4),
        'CV_Accuracy_std': round(float(dt_cv.std()), 4),
        'Test_Precision': round(float(precision_score(y_test_clf, dt_preds, average='macro')), 4),
        'Test_Recall': round(float(recall_score(y_test_clf, dt_preds, average='macro')), 4),
        'Test_F1_Macro': round(float(f1_score(y_test_clf, dt_preds, average='macro')), 4),
        'Confusion_Matrix': confusion_matrix(y_test_clf, dt_preds).tolist()
    }
    joblib.dump(dt, os.path.join(MODELS_DIR, "decision_tree_classifier.joblib"))

    print("\n--- Classification Model Evaluation ---", flush=True)
    for m, vals in clf_metrics.items():
        print(f"{m:26s} | Test Acc: {vals['Test_Accuracy']*100:.2f}% | 5-Fold CV: {vals['CV_Accuracy_mean']*100:.2f}% | Macro F1: {vals['Test_F1_Macro']:.4f}", flush=True)

    # =========================================================================
    # 4. Unsupervised Tactical Archetypes (K-Means Clustering)
    # =========================================================================
    cluster_features = [
        'batting_average', 'batting_strike_rate', 'boundary_run_pct', 
        'death_overs_strike_rate', 'overs_bowled', 'wickets_taken', 
        'economy_rate', 'death_overs_economy', 'batting_impact_index', 
        'bowling_impact_index'
    ]
    scaler_cluster = StandardScaler()
    X_cluster = scaler_cluster.fit_transform(df[cluster_features])

    optimal_k = 5
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_cluster)
    sil_score = silhouette_score(X_cluster, clusters)
    print(f"\nK-Means Clustering (k=5) Trained | Silhouette Score: {sil_score:.4f}", flush=True)

    kmeans_bundle = {
        "model": kmeans,
        "scaler": scaler_cluster,
        "features": cluster_features,
        "silhouette_score": round(float(sil_score), 4)
    }
    joblib.dump(kmeans_bundle, os.path.join(MODELS_DIR, "kmeans_model.joblib"))

    archetype_names = {
        "0": "Top-Order Anchor & Accumulator",
        "1": "Powerplay & Middle-Overs Pace Specialist",
        "2": "Death-Overs Finisher & Power Hitter",
        "3": "Defensive Middle-Overs Economy Spinner",
        "4": "Elite Dual-Threat All-Rounder"
    }

    # =========================================================================
    # 5. Dimensionality Reduction (PCA)
    # =========================================================================
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_model.joblib"))

    # Feature Importances
    imp_reg = dict(sorted(zip(feature_names, [round(float(x), 4) for x in rf_reg.feature_importances_]), key=lambda x: x[1], reverse=True)[:10])
    imp_clf = dict(sorted(zip(feature_names, [round(float(x), 4) for x in rf_clf.feature_importances_]), key=lambda x: x[1], reverse=True)[:10])

    # =========================================================================
    # 6. Save Metadata & Metrics Summaries
    # =========================================================================
    metadata = {
        "num_features": num_features,
        "cat_features": cat_features,
        "encoded_feature_names": feature_names,
        "cluster_features": cluster_features,
        "target_reg": "overall_performance_rating",
        "target_clf": "performance_tier_code",
        "talent_tiers": {0: "Developing / Squad", 1: "Core / Star", 2: "Elite / Marquee"},
        "archetypes": archetype_names
    }
    with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    metrics_summary = {
        "regression": reg_metrics,
        "classification": clf_metrics,
        "unsupervised": {
            "optimal_k": optimal_k,
            "silhouette_score": round(float(sil_score), 4),
            "archetype_names": archetype_names
        },
        "pca": {
            "n_components": 2,
            "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_],
            "total_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4)
        },
        "feature_importances": {
            "regression_top10": imp_reg,
            "classification_top10": imp_clf
        }
    }
    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n>>> All Cricket ML models (Regression + Classification + Unsupervised) trained, evaluated, and saved to models/ successfully! <<<", flush=True)

if __name__ == "__main__":
    train_and_evaluate_models()
