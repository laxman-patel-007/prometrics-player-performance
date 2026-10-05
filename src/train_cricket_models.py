"""
Cricket ML Model Training Engine: Real IPL Player Performance
Trains, evaluates, and persists Supervised Classification and Unsupervised Models:
- Supervised Talent Tier Classification (Elite / Marquee, Core / Star, Developing / Squad)
  * K-Nearest Neighbors (KNN)
  * Random Forest Classifier
  * Logistic Regression
  * Decision Tree (CART)
- Unsupervised Tactical Archetype Discovery (K-Means Clustering, k=5)
- Latent Space Dimensionality Reduction (PCA, 2 Components)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
    silhouette_score
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data/processed/cricket_players_clean.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

def train_and_evaluate_models():
    print(f"Loading cleaned cricket dataset from {DATA_PATH}...")
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset Loaded: {len(df)} players, {df.shape[1]} columns.")

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

    y_clf = df['performance_tier_code']

    # Train / Test Split (80/20 Stratified on performance tier)
    X_train_raw, X_test_raw, y_train_clf, y_test_clf = train_test_split(
        df_encoded, y_clf, test_size=0.20, random_state=42, stratify=y_clf
    )

    # Standard Scaling
    scaler = StandardScaler()
    X_train = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names)
    X_test = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names)

    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.joblib"))
    print("StandardScaler serialized.")

    # 2. Supervised Classification Models (Talent Tier Prediction)
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

    print("\n--- Classification Model Evaluation ---")
    for m, vals in clf_metrics.items():
        print(f"{m:30s} | Test Acc: {vals['Test_Accuracy']*100:.2f}% | 5-Fold CV: {vals['CV_Accuracy_mean']*100:.2f}% | Macro F1: {vals['Test_F1_Macro']:.4f}")

    # 3. Unsupervised Tactical Archetypes (K-Means Clustering)
    cluster_features = [
        'batting_average', 'batting_strike_rate', 'boundary_run_pct', 
        'death_overs_strike_rate', 'overs_bowled', 'wickets_taken', 
        'economy_rate', 'bowling_strike_rate', 'dot_ball_bowled_pct', 
        'death_overs_economy', 'clutch_match_winner_index'
    ]
    X_cluster = X_train[cluster_features]

    inertias = []
    silhouettes = []
    k_range = list(range(2, 8))
    for k in k_range:
        km_test = KMeans(n_clusters=k, random_state=42, n_init=10)
        k_labels = km_test.fit_predict(X_cluster)
        inertias.append(float(km_test.inertia_))
        silhouettes.append(float(silhouette_score(X_cluster, k_labels)))

    optimal_k = 5
    final_kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    final_kmeans.fit(X_cluster)
    joblib.dump({"model": final_kmeans, "features": cluster_features}, os.path.join(MODELS_DIR, "kmeans_model.joblib"))

    archetype_names = {
        "0": "Tactical Anchor & Top-Order Accumulator",
        "1": "High-Impact Pace Spearhead & Death Bowler",
        "2": "Explosive Death-Over Finisher & Boundary Hitter",
        "3": "Mystery / Control Spin Maestro",
        "4": "Elite Dual-Threat All-Rounder"
    }

    # 4. Dimensionality Reduction (PCA)
    pca = PCA(n_components=2, random_state=42)
    pca.fit(X_train)
    joblib.dump(pca, os.path.join(MODELS_DIR, "pca_model.joblib"))

    # Feature Importances from Random Forest Classifier
    imp_clf = dict(sorted(zip(feature_names, [round(float(x), 4) for x in rf_clf.feature_importances_]), key=lambda x: x[1], reverse=True)[:10])

    # 5. Save Metadata & Metrics Summaries
    metadata = {
        "num_features": num_features,
        "cat_features": cat_features,
        "encoded_feature_names": feature_names,
        "cluster_features": cluster_features,
        "target_clf": "performance_tier_code",
        "talent_tiers": {0: "Developing / Squad", 1: "Core / Star", 2: "Elite / Marquee"}
    }
    with open(os.path.join(MODELS_DIR, "feature_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    metrics_summary = {
        "classification": clf_metrics,
        "unsupervised": {
            "optimal_k": optimal_k,
            "k_range": k_range,
            "inertias": inertias,
            "silhouette_scores": silhouettes,
            "archetype_names": archetype_names
        },
        "pca": {
            "explained_variance_ratio": [round(float(x), 4) for x in pca.explained_variance_ratio_],
            "total_variance_explained": round(float(pca.explained_variance_ratio_.sum()), 4)
        },
        "feature_importances": {
            "classification_top10": imp_clf
        }
    }
    with open(os.path.join(MODELS_DIR, "metrics_summary.json"), "w") as f:
        json.dump(metrics_summary, f, indent=2)

    print("\n>>> All Cricket Classification & Unsupervised ML models trained and saved to models/ successfully! <<<")

if __name__ == "__main__":
    train_and_evaluate_models()
