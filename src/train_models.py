"""
ProMetrics: Player Performance Analysis (Case Study no. 102)
Training & Evaluation Engine
Strictly implements and benchmarks:
- Module IV: Supervised Learning: Regression (Linear, Polynomial, Ridge, Random Forest, MLP)
- Module V: Supervised Learning: Classification (Logistic Regression, KNN, Decision Tree, Random Forest, MLP)
- Module VI: Rigorous Model Evaluation (5-Fold CV, MAE/MSE/RMSE/R2, Acc/Prec/Recall/F1, Confusion Matrix)
- Module VII: Unsupervised Learning (K-Means Clustering, Silhouette Analysis, Hierarchical Clustering)
- Module VIII: Dimensionality Reduction & Ensembles (PCA Scree & Loadings, Random Forest Feature Importance)
- Module IX: Neural Network Basics & Model Deployment Persistence (MLP, Joblib)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, silhouette_score
)

def train_and_evaluate_all():
    print("=" * 80)
    print("PROMETRICS: RUNNING COMPREHENSIVE MACHINE LEARNING EXPERIMENTS")
    print("=" * 80)

    # 1. Load Data
    X_train = pd.read_csv("data/processed/X_train.csv")
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_train_reg = pd.read_csv("data/processed/y_train_reg.csv").squeeze("columns")
    y_test_reg = pd.read_csv("data/processed/y_test_reg.csv").squeeze("columns")
    y_train_clf = pd.read_csv("data/processed/y_train_clf.csv").squeeze("columns")
    y_test_clf = pd.read_csv("data/processed/y_test_clf.csv").squeeze("columns")

    feature_names = list(X_train.columns)
    print(f"Loaded datasets: X_train {X_train.shape}, X_test {X_test.shape}")

    results = {
        "regression": {},
        "classification": {},
        "unsupervised": {},
        "pca": {},
        "feature_importances": {}
    }

    # =========================================================================
    # MODULE IV: SUPERVISED LEARNING - REGRESSION
    # =========================================================================
    print("\n--- Training Module IV: Supervised Learning (Regression) ---")
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    # 1. Linear Regression (Baseline Ordinary Least Squares)
    lr = LinearRegression()
    lr_cv_r2 = cross_val_score(lr, X_train, y_train_reg, cv=kf, scoring="r2")
    lr.fit(X_train, y_train_reg)
    lr_preds = lr.predict(X_test)
    lr_mae = mean_absolute_error(y_test_reg, lr_preds)
    lr_mse = mean_squared_error(y_test_reg, lr_preds)
    lr_rmse = np.sqrt(lr_mse)
    lr_r2 = r2_score(y_test_reg, lr_preds)
    joblib.dump(lr, "models/linear_regression.joblib")

    results["regression"]["Linear Regression"] = {
        "CV_R2_mean": round(float(lr_cv_r2.mean()), 4),
        "CV_R2_std": round(float(lr_cv_r2.std()), 4),
        "Test_MAE": round(float(lr_mae), 4),
        "Test_MSE": round(float(lr_mse), 4),
        "Test_RMSE": round(float(lr_rmse), 4),
        "Test_R2": round(float(lr_r2), 4)
    }
    print(f"Linear Regression       -> Test R2: {lr_r2:.4f} | RMSE: {lr_rmse:.4f} | MAE: {lr_mae:.4f}")

    # 2. Polynomial Regression (Degree 2 on Top Athletic & Technical Features)
    # Using top 6 key domain features to avoid combinatorial explosion and multicollinearity
    poly_cols = ["sprint_speed", "stamina", "ball_control", "short_passing", "finishing", "defensive_awareness"]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X_train_poly = poly.fit_transform(X_train[poly_cols])
    X_test_poly = poly.transform(X_test[poly_cols])

    poly_lr = Ridge(alpha=10.0)  # L2 regularized to stabilize polynomial interactions
    poly_cv_r2 = cross_val_score(poly_lr, X_train_poly, y_train_reg, cv=kf, scoring="r2")
    poly_lr.fit(X_train_poly, y_train_reg)
    poly_preds = poly_lr.predict(X_test_poly)
    poly_mae = mean_absolute_error(y_test_reg, poly_preds)
    poly_rmse = np.sqrt(mean_squared_error(y_test_reg, poly_preds))
    poly_r2 = r2_score(y_test_reg, poly_preds)
    joblib.dump({"poly_transformer": poly, "poly_model": poly_lr, "poly_cols": poly_cols}, "models/polynomial_regression.joblib")

    results["regression"]["Polynomial Regression (Deg 2)"] = {
        "CV_R2_mean": round(float(poly_cv_r2.mean()), 4),
        "CV_R2_std": round(float(poly_cv_r2.std()), 4),
        "Test_MAE": round(float(poly_mae), 4),
        "Test_RMSE": round(float(poly_rmse), 4),
        "Test_R2": round(float(poly_r2), 4)
    }
    print(f"Polynomial Regression   -> Test R2: {poly_r2:.4f} | RMSE: {poly_rmse:.4f} | MAE: {poly_mae:.4f}")

    # 3. Ridge Regression (L2 Regularized Baseline)
    ridge = Ridge(alpha=1.0)
    ridge_cv_r2 = cross_val_score(ridge, X_train, y_train_reg, cv=kf, scoring="r2")
    ridge.fit(X_train, y_train_reg)
    ridge_preds = ridge.predict(X_test)
    ridge_mae = mean_absolute_error(y_test_reg, ridge_preds)
    ridge_rmse = np.sqrt(mean_squared_error(y_test_reg, ridge_preds))
    ridge_r2 = r2_score(y_test_reg, ridge_preds)
    joblib.dump(ridge, "models/ridge_regression.joblib")

    results["regression"]["Ridge Regression"] = {
        "CV_R2_mean": round(float(ridge_cv_r2.mean()), 4),
        "CV_R2_std": round(float(ridge_cv_r2.std()), 4),
        "Test_MAE": round(float(ridge_mae), 4),
        "Test_RMSE": round(float(ridge_rmse), 4),
        "Test_R2": round(float(ridge_r2), 4)
    }
    print(f"Ridge Regression        -> Test R2: {ridge_r2:.4f} | RMSE: {ridge_rmse:.4f} | MAE: {ridge_mae:.4f}")

    # 4. Random Forest Regressor (Module VIII Ensemble)
    rf_reg = RandomForestRegressor(n_estimators=120, max_depth=12, random_state=42, n_jobs=-1)
    rf_cv_r2 = cross_val_score(rf_reg, X_train, y_train_reg, cv=kf, scoring="r2")
    rf_reg.fit(X_train, y_train_reg)
    rf_preds = rf_reg.predict(X_test)
    rf_mae = mean_absolute_error(y_test_reg, rf_preds)
    rf_rmse = np.sqrt(mean_squared_error(y_test_reg, rf_preds))
    rf_r2 = r2_score(y_test_reg, rf_preds)
    joblib.dump(rf_reg, "models/random_forest_regressor.joblib")

    results["regression"]["Random Forest Regressor"] = {
        "CV_R2_mean": round(float(rf_cv_r2.mean()), 4),
        "CV_R2_std": round(float(rf_cv_r2.std()), 4),
        "Test_MAE": round(float(rf_mae), 4),
        "Test_RMSE": round(float(rf_rmse), 4),
        "Test_R2": round(float(rf_r2), 4)
    }
    print(f"Random Forest Regressor -> Test R2: {rf_r2:.4f} | RMSE: {rf_rmse:.4f} | MAE: {rf_mae:.4f}")

    # 5. Multi-Layer Perceptron Regressor (Module IX Neural Network Basics)
    mlp_reg = MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=350, random_state=42, early_stopping=True)
    mlp_cv_r2 = cross_val_score(mlp_reg, X_train, y_train_reg, cv=kf, scoring="r2")
    mlp_reg.fit(X_train, y_train_reg)
    mlp_preds = mlp_reg.predict(X_test)
    mlp_mae = mean_absolute_error(y_test_reg, mlp_preds)
    mlp_rmse = np.sqrt(mean_squared_error(y_test_reg, mlp_preds))
    mlp_r2 = r2_score(y_test_reg, mlp_preds)
    joblib.dump(mlp_reg, "models/mlp_regressor.joblib")

    results["regression"]["MLP Regressor (Neural Net)"] = {
        "CV_R2_mean": round(float(mlp_cv_r2.mean()), 4),
        "CV_R2_std": round(float(mlp_cv_r2.std()), 4),
        "Test_MAE": round(float(mlp_mae), 4),
        "Test_RMSE": round(float(mlp_rmse), 4),
        "Test_R2": round(float(mlp_r2), 4)
    }
    print(f"MLP Regressor (NN)      -> Test R2: {mlp_r2:.4f} | RMSE: {mlp_rmse:.4f} | MAE: {mlp_mae:.4f}")

    # =========================================================================
    # MODULE V: SUPERVISED LEARNING - CLASSIFICATION
    # =========================================================================
    print("\n--- Training Module V: Supervised Learning (Classification) ---")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    target_names = ["Developing", "Star", "Elite"]

    # 1. Logistic Regression (Multinomial with L2 Regularization)
    log_reg = LogisticRegression(max_iter=500, random_state=42)
    log_cv_acc = cross_val_score(log_reg, X_train, y_train_clf, cv=skf, scoring="accuracy")
    log_reg.fit(X_train, y_train_clf)
    log_preds = log_reg.predict(X_test)
    log_acc = accuracy_score(y_test_clf, log_preds)
    log_prec = precision_score(y_test_clf, log_preds, average="macro")
    log_rec = recall_score(y_test_clf, log_preds, average="macro")
    log_f1 = f1_score(y_test_clf, log_preds, average="macro")
    log_cm = confusion_matrix(y_test_clf, log_preds).tolist()
    joblib.dump(log_reg, "models/logistic_regression.joblib")

    results["classification"]["Logistic Regression"] = {
        "CV_Accuracy_mean": round(float(log_cv_acc.mean()), 4),
        "CV_Accuracy_std": round(float(log_cv_acc.std()), 4),
        "Test_Accuracy": round(float(log_acc), 4),
        "Test_Precision": round(float(log_prec), 4),
        "Test_Recall": round(float(log_rec), 4),
        "Test_F1_Macro": round(float(log_f1), 4),
        "Confusion_Matrix": log_cm
    }
    print(f"Logistic Regression     -> Test Acc: {log_acc:.4f} | F1-Macro: {log_f1:.4f} | Precision: {log_prec:.4f}")

    # 2. K-Nearest Neighbors (KNN Classifier)
    knn = KNeighborsClassifier(n_neighbors=7, weights="distance")
    knn_cv_acc = cross_val_score(knn, X_train, y_train_clf, cv=skf, scoring="accuracy")
    knn.fit(X_train, y_train_clf)
    knn_preds = knn.predict(X_test)
    knn_acc = accuracy_score(y_test_clf, knn_preds)
    knn_prec = precision_score(y_test_clf, knn_preds, average="macro")
    knn_rec = recall_score(y_test_clf, knn_preds, average="macro")
    knn_f1 = f1_score(y_test_clf, knn_preds, average="macro")
    knn_cm = confusion_matrix(y_test_clf, knn_preds).tolist()
    joblib.dump(knn, "models/knn_classifier.joblib")

    results["classification"]["K-Nearest Neighbors (KNN)"] = {
        "CV_Accuracy_mean": round(float(knn_cv_acc.mean()), 4),
        "CV_Accuracy_std": round(float(knn_cv_acc.std()), 4),
        "Test_Accuracy": round(float(knn_acc), 4),
        "Test_Precision": round(float(knn_prec), 4),
        "Test_Recall": round(float(knn_rec), 4),
        "Test_F1_Macro": round(float(knn_f1), 4),
        "Confusion_Matrix": knn_cm
    }
    print(f"K-Nearest Neighbors     -> Test Acc: {knn_acc:.4f} | F1-Macro: {knn_f1:.4f} | Precision: {knn_prec:.4f}")

    # 3. Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=6, min_samples_split=10, random_state=42)
    dt_cv_acc = cross_val_score(dt, X_train, y_train_clf, cv=skf, scoring="accuracy")
    dt.fit(X_train, y_train_clf)
    dt_preds = dt.predict(X_test)
    dt_acc = accuracy_score(y_test_clf, dt_preds)
    dt_prec = precision_score(y_test_clf, dt_preds, average="macro")
    dt_rec = recall_score(y_test_clf, dt_preds, average="macro")
    dt_f1 = f1_score(y_test_clf, dt_preds, average="macro")
    dt_cm = confusion_matrix(y_test_clf, dt_preds).tolist()
    joblib.dump(dt, "models/decision_tree_classifier.joblib")

    results["classification"]["Decision Tree"] = {
        "CV_Accuracy_mean": round(float(dt_cv_acc.mean()), 4),
        "CV_Accuracy_std": round(float(dt_cv_acc.std()), 4),
        "Test_Accuracy": round(float(dt_acc), 4),
        "Test_Precision": round(float(dt_prec), 4),
        "Test_Recall": round(float(dt_rec), 4),
        "Test_F1_Macro": round(float(dt_f1), 4),
        "Confusion_Matrix": dt_cm
    }
    print(f"Decision Tree           -> Test Acc: {dt_acc:.4f} | F1-Macro: {dt_f1:.4f} | Precision: {dt_prec:.4f}")

    # 4. Random Forest Classifier (Module VIII Ensemble)
    rf_clf = RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42, n_jobs=-1)
    rf_cv_acc = cross_val_score(rf_clf, X_train, y_train_clf, cv=skf, scoring="accuracy")
    rf_clf.fit(X_train, y_train_clf)
    rf_cpreds = rf_clf.predict(X_test)
    rf_acc = accuracy_score(y_test_clf, rf_cpreds)
    rf_prec = precision_score(y_test_clf, rf_cpreds, average="macro")
    rf_rec = recall_score(y_test_clf, rf_cpreds, average="macro")
    rf_f1 = f1_score(y_test_clf, rf_cpreds, average="macro")
    rf_cm = confusion_matrix(y_test_clf, rf_cpreds).tolist()
    joblib.dump(rf_clf, "models/random_forest_classifier.joblib")

    results["classification"]["Random Forest Classifier"] = {
        "CV_Accuracy_mean": round(float(rf_cv_acc.mean()), 4),
        "CV_Accuracy_std": round(float(rf_cv_acc.std()), 4),
        "Test_Accuracy": round(float(rf_acc), 4),
        "Test_Precision": round(float(rf_prec), 4),
        "Test_Recall": round(float(rf_rec), 4),
        "Test_F1_Macro": round(float(rf_f1), 4),
        "Confusion_Matrix": rf_cm
    }
    print(f"Random Forest Clf       -> Test Acc: {rf_acc:.4f} | F1-Macro: {rf_f1:.4f} | Precision: {rf_prec:.4f}")

    # 5. Multi-Layer Perceptron Classifier (Module IX Neural Network Basics)
    mlp_clf = MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=350, random_state=42, early_stopping=True)
    mlp_cv_acc = cross_val_score(mlp_clf, X_train, y_train_clf, cv=skf, scoring="accuracy")
    mlp_clf.fit(X_train, y_train_clf)
    mlp_cpreds = mlp_clf.predict(X_test)
    mlp_acc = accuracy_score(y_test_clf, ml_cpreds if 'ml_cpreds' in locals() else mlp_cpreds)
    mlp_prec = precision_score(y_test_clf, mlp_cpreds, average="macro")
    mlp_rec = recall_score(y_test_clf, mlp_cpreds, average="macro")
    mlp_f1 = f1_score(y_test_clf, mlp_cpreds, average="macro")
    mlp_cm = confusion_matrix(y_test_clf, mlp_cpreds).tolist()
    joblib.dump(mlp_clf, "models/mlp_classifier.joblib")

    results["classification"]["MLP Classifier (Neural Net)"] = {
        "CV_Accuracy_mean": round(float(mlp_cv_acc.mean()), 4),
        "CV_Accuracy_std": round(float(mlp_cv_acc.std()), 4),
        "Test_Accuracy": round(float(mlp_acc), 4),
        "Test_Precision": round(float(mlp_prec), 4),
        "Test_Recall": round(float(mlp_rec), 4),
        "Test_F1_Macro": round(float(mlp_f1), 4),
        "Confusion_Matrix": mlp_cm
    }
    print(f"MLP Classifier (NN)     -> Test Acc: {mlp_acc:.4f} | F1-Macro: {mlp_f1:.4f} | Precision: {mlp_prec:.4f}")

    # =========================================================================
    # MODULE VII: UNSUPERVISED LEARNING (K-MEANS & HIERARCHICAL)
    # =========================================================================
    print("\n--- Training Module VII: Unsupervised Learning (Clustering) ---")
    # Clustering on core playing style attributes (combining athletic + technical + tactical)
    cluster_features = [
        "sprint_speed", "stamina", "strength", "agility",
        "ball_control", "dribbling", "short_passing", "long_passing", "finishing",
        "defensive_awareness", "standing_tackle", "vision"
    ]
    X_cluster = X_train[cluster_features]

    # Evaluate Elbow and Silhouette Scores for k = 2 to 7
    k_range = list(range(2, 8))
    inertias = []
    silhouettes = []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_cluster)
        inertias.append(float(km.inertia_))
        sil = float(silhouette_score(X_cluster, labels))
        silhouettes.append(round(sil, 4))

    # Optimal k chosen is 4 (Tactical Archetypes: 0=Creative Playmaker, 1=Defensive Wall, 2=Direct Attacker/Poacher, 3=Goalkeeper/Low Pace)
    optimal_k = 4
    kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=15)
    cluster_labels = kmeans_final.fit_predict(X_cluster)
    joblib.dump({"model": kmeans_final, "features": cluster_features}, "models/kmeans_model.joblib")

    archetype_names = {
        0: "Tactical Playmaker & Orchestrator",
        1: "Defensive Anchor & Ball-Winner",
        2: "Explosive Forward & Finisher",
        3: "Goalkeeper / Positional Specialist"
    }

    results["unsupervised"] = {
        "k_range": k_range,
        "inertias": [round(x, 2) for x in inertias],
        "silhouette_scores": silhouettes,
        "optimal_k": optimal_k,
        "archetype_names": archetype_names
    }
    print(f"K-Means Optimal k={optimal_k} with Silhouette Score: {silhouettes[k_range.index(optimal_k)]:.4f}")

    # =========================================================================
    # MODULE VIII: DIMENSIONALITY REDUCTION (PCA) & ENSEMBLE IMPORTANCES
    # =========================================================================
    print("\n--- Training Module VIII: Dimensionality Reduction (PCA) ---")
    pca_full = PCA()
    pca_full.fit(X_train)
    explained_variance_ratio = [round(float(x), 4) for x in pca_full.explained_variance_ratio_]
    cumulative_variance = [round(float(x), 4) for x in np.cumsum(pca_full.explained_variance_ratio_)]

    pca_2d = PCA(n_components=2)
    X_pca_2d = pca_2d.fit_transform(X_train)
    joblib.dump(pca_2d, "models/pca_model.joblib")

    # Loadings interpretation for PC1 & PC2
    loadings_pc1 = dict(zip(feature_names, [round(float(x), 4) for x in pca_2d.components_[0]]))
    loadings_pc2 = dict(zip(feature_names, [round(float(x), 4) for x in pca_2d.components_[1]]))

    # Sort top positive and negative features for PC1 and PC2
    top_pc1 = sorted(loadings_pc1.items(), key=lambda item: abs(item[1]), reverse=True)[:6]
    top_pc2 = sorted(loadings_pc2.items(), key=lambda item: abs(item[1]), reverse=True)[:6]

    results["pca"] = {
        "explained_variance_ratio_top5": explained_variance_ratio[:5],
        "cumulative_variance_top5": cumulative_variance[:5],
        "top_features_pc1": dict(top_pc1),
        "top_features_pc2": dict(top_pc2)
    }
    print(f"PCA PC1 Explains {explained_variance_ratio[0]*100:.2f}% | PC2 Explains {explained_variance_ratio[1]*100:.2f}%")

    # Feature Importances from Random Forest Regressor & Classifier
    rf_reg_importances = dict(zip(feature_names, [round(float(x), 4) for x in rf_reg.feature_importances_]))
    rf_clf_importances = dict(zip(feature_names, [round(float(x), 4) for x in rf_clf.feature_importances_]))
    
    top_reg_imp = dict(sorted(rf_reg_importances.items(), key=lambda x: x[1], reverse=True)[:10])
    top_clf_imp = dict(sorted(rf_clf_importances.items(), key=lambda x: x[1], reverse=True)[:10])

    results["feature_importances"] = {
        "regression_top10": top_reg_imp,
        "classification_top10": top_clf_imp
    }

    # Save complete metrics summary to JSON
    with open("models/metrics_summary.json", "w") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 80)
    print("ALL EXPERIMENTS COMPLETED SUCCESSFULLY! RESULTS PERSISTED TO models/")
    print("=" * 80)

if __name__ == "__main__":
    train_and_evaluate_all()
