"""
ProMetrics: Player Performance Analysis (Case Study no. 102)
Data Preprocessing & Feature Engineering Pipeline:
- Handling missing data (domain-aware stratified median imputation)
- Categorical encoding (One-Hot Encoding for nominal variables)
- Feature engineering (Domain composite indices & non-linear age interaction)
- Feature scaling (StandardScaler)
- Train/Test Split (Stratified 80/20)
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def preprocess_player_data(raw_csv_path: str = "data/raw/player_performance_raw.csv"):
    print(f"Loading raw dataset from {raw_csv_path}...")
    df = pd.read_csv(raw_csv_path)
    
    # 1. Inspect Missing Data
    missing_counts = df.isnull().sum()
    missing_cols = missing_counts[missing_counts > 0]
    print("Detected missing values:")
    for col, count in missing_cols.items():
        print(f"  - {col}: {count} missing values ({count/len(df)*100:.2f}%)")

    # 2. Domain-Aware Median Imputation (stratified by position)
    # Rationale: Midfield stamina & passing differ fundamentally from Goalkeepers
    impute_cols = ["short_passing", "stamina", "discipline_score", "distance_km_per_90"]
    for col in impute_cols:
        if col in df.columns:
            df[col] = df.groupby("primary_position")[col].transform(lambda x: x.fillna(x.median()))
            # Fallback overall median just in case
            df[col] = df[col].fillna(df[col].median())

    print("Missing values after stratified median imputation:", df.isnull().sum().sum())

    # 3. Domain Feature Engineering
    # A. Athletic Power Index
    df["athletic_power_index"] = (
        0.30 * df["sprint_speed"] +
        0.25 * df["acceleration"] +
        0.25 * df["stamina"] +
        0.20 * df["strength"]
    ).round(2)

    # B. Technical Mastery Index
    df["technical_mastery_index"] = (
        0.30 * df["ball_control"] +
        0.25 * df["dribbling"] +
        0.25 * df["short_passing"] +
        0.20 * df["long_passing"]
    ).round(2)

    # C. Defensive Solidity Index
    df["defensive_solidity_index"] = (
        0.40 * df["defensive_awareness"] +
        0.35 * df["standing_tackle"] +
        0.25 * df["sliding_tackle"]
    ).round(2)

    # D. Attacking Threat Index
    df["attacking_threat_index"] = (
        0.45 * df["finishing"] +
        0.30 * df["shot_power"] +
        0.25 * (df["goals_per_90"] * 60.0).clip(0, 100)
    ).round(2)

    # E. Work Ethic & Stamina Efficiency
    df["stamina_efficiency"] = (
        (df["distance_km_per_90"] / (df["stamina"] + 1e-4)) * 100.0
    ).round(2)

    # F. Non-linear Age Peak Interaction (Age - 27)^2
    df["age_peak_delta_sq"] = ((df["age"] - 27) ** 2).astype(float)

    # 4. Save Cleaned & Engineered Dataset
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    cleaned_path = "data/processed/player_performance_cleaned.csv"
    df.to_csv(cleaned_path, index=False)
    print(f"Saved cleaned dataset with engineered features to {cleaned_path}")

    # 5. Define Feature Sets for Supervised Learning
    # Core measurable numerical features
    num_features = [
        "age", "height_cm", "weight_kg",
        "sprint_speed", "acceleration", "stamina", "strength", "agility", "jumping",
        "ball_control", "dribbling", "short_passing", "long_passing", "crossing", "finishing", "shot_power",
        "defensive_awareness", "standing_tackle", "sliding_tackle",
        "vision", "composure", "aggression", "discipline_score",
        "minutes_played", "goals_per_90", "assists_per_90", "pass_accuracy_pct", "tackle_success_pct", "distance_km_per_90",
        "athletic_power_index", "technical_mastery_index", "defensive_solidity_index", "attacking_threat_index",
        "stamina_efficiency", "age_peak_delta_sq"
    ]

    # Categorical features for One-Hot Encoding
    cat_features = ["primary_position", "preferred_foot", "work_rate_attack", "work_rate_defense"]
    
    # Perform One-Hot Encoding
    df_encoded = pd.get_dummies(df[num_features + cat_features], columns=cat_features, drop_first=True, dtype=float)
    feature_names = list(df_encoded.columns)

    # Target variables
    y_reg = df["overall_performance_rating"]
    y_clf = df["performance_tier_code"]

    # 6. Stratified Train / Test Split
    # Stratified by performance_tier_code to maintain balanced class proportions
    X_train_raw, X_test_raw, y_train_reg, y_test_reg, y_train_clf, y_test_clf = train_test_split(
        df_encoded, y_reg, y_clf,
        test_size=0.20,
        random_state=42,
        stratify=y_clf
    )

    print(f"Training split: {X_train_raw.shape[0]} samples | Test split: {X_test_raw.shape[0]} samples")

    # 7. Feature Scaling
    # Standardize numerical features using StandardScaler (fit on train, transform on test)
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_raw), columns=feature_names, index=X_train_raw.index)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test_raw), columns=feature_names, index=X_test_raw.index)

    # Save Scaler & Metadata
    joblib.dump(scaler, "models/scaler.joblib")
    
    metadata = {
        "num_features": num_features,
        "cat_features": cat_features,
        "encoded_feature_names": feature_names,
        "target_reg": "overall_performance_rating",
        "target_clf": "performance_tier_code",
        "tier_mapping": {0: "Developing", 1: "Star", 2: "Elite"}
    }
    with open("models/feature_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Save train / test splits
    X_train_scaled.to_csv("data/processed/X_train.csv", index=False)
    X_test_scaled.to_csv("data/processed/X_test.csv", index=False)
    y_train_reg.to_csv("data/processed/y_train_reg.csv", index=False)
    y_test_reg.to_csv("data/processed/y_test_reg.csv", index=False)
    y_train_clf.to_csv("data/processed/y_train_clf.csv", index=False)
    y_test_clf.to_csv("data/processed/y_test_clf.csv", index=False)

    print("Preprocessing completed successfully. Processed files written to data/processed/")
    return df, X_train_scaled, X_test_scaled, y_train_reg, y_test_reg, y_train_clf, y_test_clf

if __name__ == "__main__":
    preprocess_player_data()
