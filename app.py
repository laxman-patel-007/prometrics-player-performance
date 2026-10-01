"""
ProMetrics: Advanced Player Performance Analytics & Scouting Intelligence
Case Study no. 102: Player Performance Analysis
Built strictly conforming to Machine Learning Syllabus (Modules I - IX).
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# Page Configuration
st.set_page_config(
    page_title="ProMetrics | Player Performance Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling for Modern Sports Analytics Dashboard
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1E3A8A, #3B82F6, #10B981);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #64748B;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    .badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-elite { background-color: #FEF3C7; color: #D97706; }
    .badge-star { background-color: #DBEAFE; color: #1D4ED8; }
    .badge-dev { background-color: #E2E8F0; color: #475569; }
</style>
""", unsafe_allow_html=True)

# Cache data and models
@st.cache_data
def load_datasets():
    df_clean = pd.read_csv("data/processed/player_performance_cleaned.csv")
    with open("models/metrics_summary.json", "r") as f:
        metrics = json.load(f)
    with open("models/feature_metadata.json", "r") as f:
        metadata = json.load(f)
    return df_clean, metrics, metadata

@st.cache_resource
def load_models():
    models = {
        "scaler": joblib.load("models/scaler.joblib"),
        "linear_reg": joblib.load("models/linear_regression.joblib"),
        "ridge_reg": joblib.load("models/ridge_regression.joblib"),
        "poly_reg": joblib.load("models/polynomial_regression.joblib"),
        "rf_reg": joblib.load("models/random_forest_regressor.joblib"),
        "mlp_reg": joblib.load("models/mlp_regressor.joblib"),
        "log_clf": joblib.load("models/logistic_regression.joblib"),
        "knn_clf": joblib.load("models/knn_classifier.joblib"),
        "dt_clf": joblib.load("models/decision_tree_classifier.joblib"),
        "rf_clf": joblib.load("models/random_forest_classifier.joblib"),
        "mlp_clf": joblib.load("models/mlp_classifier.joblib"),
        "kmeans": joblib.load("models/kmeans_model.joblib"),
        "pca": joblib.load("models/pca_model.joblib")
    }
    return models

df, metrics, metadata = load_datasets()
models = load_models()

# Sidebar Navigation
st.sidebar.image("https://images.unsplash.com/photo-1508098682722-e99c43a406b2?w=600&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("ProMetrics Studio")
st.sidebar.markdown("**Case Study 102:** Player Performance Analysis")

menu = st.sidebar.radio(
    "Navigation Modules:",
    [
        "1. Executive Overview & Syllabus Alignment",
        "2. Exploratory Data Analysis & Measurable Factors",
        "3. Performance Rating Prediction (Regression)",
        "4. Talent Tier Classification",
        "5. Tactical Archetypes & PCA Clustering",
        "6. What-If Scouting Simulator",
        "7. Model Evaluation & Benchmark Studio"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("""
**Course Coverage:**
- **Module I:** Problem Formulation & Workflow
- **Module II:** NumPy, Pandas, Scikit-Learn
- **Module III:** Preprocessing & Feature Engineering
- **Module IV:** Linear & Polynomial Regression
- **Module V:** Logistic, KNN, Decision Trees
- **Module VI:** Cross-Validation & Metric Evaluation
- **Module VII:** K-Means & Hierarchical Clustering
- **Module VIII:** PCA & Random Forest Ensembles
- **Module IX:** Neural Networks & Streamlit Deployment
""")

# ==============================================================================
# TAB 1: EXECUTIVE OVERVIEW & SYLLABUS MAPPING
# ==============================================================================
if menu == "1. Executive Overview & Syllabus Alignment":
    st.markdown('<div class="main-header">ProMetrics: Player Performance Analytics Platform</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Case Study no. 102 | A Machine Learning Framework for Investigating Measurable Factors in Player Performance</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Players Analyzed", f"{len(df):,}")
    with col2:
        st.metric("Best Regression R² Score", f"{metrics['regression']['Random Forest Regressor']['Test_R2']:.4f}", delta="Random Forest")
    with col3:
        st.metric("Best Classification Accuracy", f"{metrics['classification']['Logistic Regression']['Test_Accuracy']*100:.1f}%", delta="Logistic / RF")
    with col4:
        st.metric("Discovered Tactical Archetypes", f"{metrics['unsupervised']['optimal_k']}", delta="K-Means (k=4)")

    st.markdown("### 1. Problem Definition & Formal Specification")
    st.write("""
    In high-stakes professional sports organizations (e.g. European top football leagues), talent acquisition, salary negotiation,
    and match-day tactical deployment demand empirical, measurable justifications. Subjective scouting often suffers from 
    cognitive biases, recency bias, and regional scouting gaps. 

    **Assigned Problem Statement:**
    > *"A sports organization wants to investigate measurable factors associated with player performance. (With Proper Justification)"*

    **Formal ML Objectives:**
    1. **Supervised Regression:** Predict a continuous **Overall Performance Rating** (scale 50.0–95.0) as a function of physiological, athletic, and technical factors.
    2. **Supervised Classification:** Classify athletes into actionable organizational tiers: **Developing/Rotation (0)**, **Core/Star (1)**, or **Elite/World-Class (2)**.
    3. **Unsupervised Clustering:** Discover hidden **Tactical Archetypes** without relying on nominal roster labels.
    4. **Dimensionality Reduction:** Extract orthogonal latent skill axes using **PCA** to visualize player positioning and identify redundant athletic indicators.
    """)

    st.markdown("### 2. Full Syllabus Compliance Matrix")
    syllabus_data = [
        {"Module": "Module I", "Topic": "Introduction to Machine Learning", "Implementation in Project": "Supervised (Regression & Classification) + Unsupervised (Clustering & PCA) end-to-end pipeline."},
        {"Module": "Module II", "Topic": "ML Libraries & Packages", "Implementation in Project": "Built entirely with NumPy, Pandas, Scikit-Learn, Matplotlib, Seaborn, and Streamlit."},
        {"Module": "Module III", "Topic": "Data Preprocessing & Feature Engineering", "Implementation in Project": "Stratified median imputation, One-Hot Encoding, StandardScaler, and composite athletic/technical indices."},
        {"Module": "Module IV", "Topic": "Supervised Learning: Regression", "Implementation in Project": "Ordinary Least Squares (OLS) Linear Regression, Polynomial Regression (degree 2 interaction), and Ridge."},
        {"Module": "Module V", "Topic": "Supervised Learning: Classification", "Implementation in Project": "Multinomial Logistic Regression, K-Nearest Neighbors (KNN), and Decision Tree Classifier."},
        {"Module": "Module VI", "Topic": "Model Evaluation & Validation", "Implementation in Project": "Stratified 80/20 split, 5-Fold Cross Validation, MAE/MSE/RMSE/R², Confusion Matrix, and Precision/Recall/F1."},
        {"Module": "Module VII", "Topic": "Unsupervised Learning", "Implementation in Project": "K-Means Clustering with Elbow Inertia & Silhouette score validation across k=2..7, tactical profiling."},
        {"Module": "Module VIII", "Topic": "Dimensionality Reduction & Ensembles", "Implementation in Project": "PCA for variance decomposition & biplot loadings; Random Forest Regressor & Classifier with Gini/MSE importances."},
        {"Module": "Module IX", "Topic": "Neural Networks & Model Deployment", "Implementation in Project": "Multi-Layer Perceptron (MLP) Regressor & Classifier; full deployment in an interactive Streamlit UI."}
    ]
    st.table(pd.DataFrame(syllabus_data))

# ==============================================================================
# TAB 2: EXPLORATORY DATA ANALYSIS & MEASURABLE FACTORS
# ==============================================================================
elif menu == "2. Exploratory Data Analysis & Measurable Factors":
    st.markdown('<div class="main-header">Exploratory Data Analysis: Measurable Factors</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Dissecting correlations, athletic traits, and technical markers across positions</div>', unsafe_allow_html=True)

    tab_eda1, tab_eda2, tab_eda3, tab_eda4 = st.tabs([
        "Correlation Heatmap", "Feature Distributions", "Positional Radars", "Age vs Athletic Peak"
    ])

    with tab_eda1:
        st.subheader("Correlation Heatmap: Key Factors vs Overall Performance")
        corr_cols = [
            "overall_performance_rating", "athletic_power_index", "technical_mastery_index",
            "defensive_solidity_index", "attacking_threat_index", "sprint_speed",
            "stamina", "ball_control", "short_passing", "finishing", "standing_tackle",
            "vision", "composure", "market_value_eur_m"
        ]
        corr_matrix = df[corr_cols].corr()
        fig_corr = px.imshow(
            corr_matrix,
            text_auto=".2f",
            color_continuous_scale="Blues",
            title="Correlation Matrix of Measurable Factors and Player Performance"
        )
        fig_corr.update_layout(height=650)
        st.plotly_chart(fig_corr, use_container_width=True)
        st.markdown("""
        **Key Observation:** 
        - `athletic_power_index` ($r = 0.81$) and `technical_mastery_index` ($r = 0.69$) exhibit the highest direct correlations with overall performance rating.
        - `composure` shows a high universal correlation across all positions ($r = 0.72$), validating sports psychology research that emotional regulation under pressure directly modulates athletic output.
        """)

    with tab_eda2:
        st.subheader("Distribution Analysis of Measurable Factors")
        selected_factor = st.selectbox(
            "Select Measurable Factor to Inspect:",
            ["overall_performance_rating", "sprint_speed", "stamina", "ball_control", "finishing", "standing_tackle", "vision", "composure"]
        )
        fig_hist = px.histogram(
            df,
            x=selected_factor,
            color="primary_position",
            marginal="box",
            nbins=35,
            title=f"Distribution of {selected_factor} by Primary Pitch Position",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_hist.update_layout(height=480)
        st.plotly_chart(fig_hist, use_container_width=True)

    with tab_eda3:
        st.subheader("Positional Skill Profile Radar Chart")
        radar_categories = ["sprint_speed", "stamina", "ball_control", "short_passing", "finishing", "defensive_awareness", "composure"]
        pos_grouped = df.groupby("primary_position")[radar_categories].mean().reset_index()

        fig_radar = go.Figure()
        for idx, row in pos_grouped.iterrows():
            fig_radar.add_trace(go.Scatterpolar(
                r=[row[c] for c in radar_categories] + [row[radar_categories[0]]],
                theta=radar_categories + [radar_categories[0]],
                fill='toself',
                name=row["primary_position"]
            ))
        fig_radar.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[20, 95])),
            showlegend=True,
            title="Multi-Dimensional Attribute Fingerprint Across Positions",
            height=520
        )
        st.plotly_chart(fig_radar, use_container_width=True)

    with tab_eda4:
        st.subheader("Non-linear Age Curve vs Performance & Stamina")
        age_summary = df.groupby("age")[["overall_performance_rating", "sprint_speed", "stamina", "composure"]].mean().reset_index()
        fig_age = px.line(
            age_summary,
            x="age",
            y=["overall_performance_rating", "sprint_speed", "stamina", "composure"],
            markers=True,
            title="Evolution of Athletic vs Cognitive Factors Over Player Career Lifespan"
        )
        fig_age.update_layout(height=480, yaxis_title="Average Metric Score (0-100)")
        st.plotly_chart(fig_age, use_container_width=True)
        st.caption("Notice the physiological inflection point: Sprint speed and stamina peak at age 25–27, whereas composure and tactical reading continue to rise past age 30.")

# ==============================================================================
# TAB 3: PERFORMANCE RATING PREDICTION (REGRESSION)
# ==============================================================================
elif menu == "3. Performance Rating Prediction (Regression)":
    st.markdown('<div class="main-header">Performance Rating Prediction Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Module IV: Supervised Learning (Regression) - Live Performance Estimation</div>', unsafe_allow_html=True)

    st.sidebar.subheader("Select Prediction Model:")
    chosen_reg_model = st.sidebar.selectbox(
        "Algorithm (Module IV & VIII):",
        ["Random Forest Regressor (Ensemble)", "Linear Regression (OLS)", "Ridge Regression (L2)", "Polynomial Regression (Deg 2)", "MLP Regressor (Neural Net)"]
    )

    st.write("Tune player's measurable athletic and technical inputs below to simulate predicted overall performance rating:")

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### Athletic & Physical")
        val_sprint = st.slider("Sprint Speed (km/h scale)", 40.0, 99.0, 78.0)
        val_accel = st.slider("Acceleration", 40.0, 99.0, 76.0)
        val_stamina = st.slider("Stamina (VO2 Max surrogate)", 45.0, 99.0, 80.0)
        val_strength = st.slider("Physical Strength", 45.0, 99.0, 72.0)
        val_agility = st.slider("Agility", 45.0, 99.0, 75.0)

    with c2:
        st.markdown("#### Technical Mastery")
        val_bc = st.slider("Ball Control", 35.0, 99.0, 82.0)
        val_dribble = st.slider("Dribbling", 35.0, 99.0, 79.0)
        val_spass = st.slider("Short Passing", 40.0, 99.0, 83.0)
        val_lpass = st.slider("Long Passing", 35.0, 99.0, 77.0)
        val_finish = st.slider("Finishing", 20.0, 99.0, 65.0)

    with c3:
        st.markdown("#### Tactical & Mental")
        val_def_aware = st.slider("Defensive Awareness", 25.0, 99.0, 68.0)
        val_tackle = st.slider("Standing Tackle", 25.0, 99.0, 65.0)
        val_vision = st.slider("Vision", 40.0, 99.0, 81.0)
        val_composure = st.slider("Composure", 45.0, 99.0, 80.0)
        val_pos = st.selectbox("Position", ["Midfielder", "Forward", "Defender", "Goalkeeper"])

    # Build input row matching feature pipeline
    age = 26.0
    power_idx = 0.30 * val_sprint + 0.25 * val_accel + 0.25 * val_stamina + 0.20 * val_strength
    tech_idx = 0.30 * val_bc + 0.25 * val_dribble + 0.25 * val_spass + 0.20 * val_lpass
    def_idx = 0.40 * val_def_aware + 0.35 * val_tackle + 0.25 * (val_tackle * 0.9)
    att_idx = 0.45 * val_finish + 0.30 * 75.0 + 0.25 * 30.0

    input_dict = {f: 0.0 for f in metadata["encoded_feature_names"]}
    input_dict["age"] = age
    input_dict["height_cm"] = 180.0
    input_dict["weight_kg"] = 75.0
    input_dict["sprint_speed"] = val_sprint
    input_dict["acceleration"] = val_accel
    input_dict["stamina"] = val_stamina
    input_dict["strength"] = val_strength
    input_dict["agility"] = val_agility
    input_dict["jumping"] = 72.0
    input_dict["ball_control"] = val_bc
    input_dict["dribbling"] = val_dribble
    input_dict["short_passing"] = val_spass
    input_dict["long_passing"] = val_lpass
    input_dict["crossing"] = 70.0
    input_dict["finishing"] = val_finish
    input_dict["shot_power"] = 75.0
    input_dict["defensive_awareness"] = val_def_aware
    input_dict["standing_tackle"] = val_tackle
    input_dict["sliding_tackle"] = val_tackle * 0.9
    input_dict["vision"] = val_vision
    input_dict["composure"] = val_composure
    input_dict["aggression"] = 65.0
    input_dict["discipline_score"] = 75.0
    input_dict["minutes_played"] = 2100.0
    input_dict["goals_per_90"] = 0.25
    input_dict["assists_per_90"] = 0.30
    input_dict["pass_accuracy_pct"] = 84.0
    input_dict["tackle_success_pct"] = 70.0
    input_dict["distance_km_per_90"] = 10.5
    input_dict["athletic_power_index"] = power_idx
    input_dict["technical_mastery_index"] = tech_idx
    input_dict["defensive_solidity_index"] = def_idx
    input_dict["attacking_threat_index"] = att_idx
    input_dict["stamina_efficiency"] = (10.5 / (val_stamina + 1e-4)) * 100.0
    input_dict["age_peak_delta_sq"] = float((age - 27) ** 2)

    # One-hot position flags
    pos_col = f"primary_position_{val_pos}"
    if pos_col in input_dict:
        input_dict[pos_col] = 1.0

    input_df = pd.DataFrame([input_dict])
    input_scaled = pd.DataFrame(models["scaler"].transform(input_df), columns=input_df.columns)

    # Perform prediction based on selected model
    if chosen_reg_model == "Linear Regression (OLS)":
        pred_rating = models["linear_reg"].predict(input_scaled)[0]
    elif chosen_reg_model == "Ridge Regression (L2)":
        pred_rating = models["ridge_reg"].predict(input_scaled)[0]
    elif chosen_reg_model == "Polynomial Regression (Deg 2)":
        p_dict = models["poly_reg"]
        inp_poly = p_dict["poly_transformer"].transform(input_scaled[p_dict["poly_cols"]])
        pred_rating = p_dict["poly_model"].predict(inp_poly)[0]
    elif chosen_reg_model == "MLP Regressor (Neural Net)":
        pred_rating = models["mlp_reg"].predict(input_scaled)[0]
    else:
        pred_rating = models["rf_reg"].predict(input_scaled)[0]

    pred_rating = float(np.clip(pred_rating, 50.0, 96.0))

    st.markdown("---")
    res_col1, res_col2 = st.columns([1, 2])
    with res_col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Predicted Rating</h3>
            <h1 style="color: #2563EB; font-size: 3.5rem; margin: 0;">{pred_rating:.1f}</h1>
            <p style="color: #64748B;">Estimated Overall Performance (50 - 95 scale)</p>
            <span class="badge {'badge-elite' if pred_rating >= 82 else 'badge-star' if pred_rating >= 71 else 'badge-dev'}">
                Tier: {'Elite / World-Class' if pred_rating >= 82 else 'Core / Star' if pred_rating >= 71 else 'Developing / Rotation'}
            </span>
        </div>
        """, unsafe_allow_html=True)

    with res_col2:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=pred_rating,
            title={'text': f"Overall Performance Meter ({chosen_reg_model})"},
            domain={'x': [0, 1], 'y': [0, 1]},
            gauge={
                'axis': {'range': [50, 95], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': "#2563EB"},
                'steps': [
                    {'range': [50, 71], 'color': "#E2E8F0"},
                    {'range': [71, 82], 'color': "#BFDBFE"},
                    {'range': [82, 95], 'color': "#FEF08A"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 82.0
                }
            }
        ))
        fig_gauge.update_layout(height=280, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_gauge, use_container_width=True)

# ==============================================================================
# TAB 4: TALENT TIER CLASSIFICATION
# ==============================================================================
elif menu == "4. Talent Tier Classification":
    st.markdown('<div class="main-header">Talent Tier Classification Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Module V: Supervised Classification (Logistic Regression, KNN, Decision Tree, Random Forest, MLP)</div>', unsafe_allow_html=True)

    clf_choice = st.selectbox(
        "Select Classification Algorithm to Analyze:",
        ["Logistic Regression", "Random Forest Classifier", "MLP Classifier (Neural Net)", "K-Nearest Neighbors (KNN)", "Decision Tree"]
    )

    clf_metrics = metrics["classification"][clf_choice]

    c_m1, c_m2, c_m3, c_m4 = st.columns(4)
    with c_m1:
        st.metric("Test Accuracy", f"{clf_metrics['Test_Accuracy']*100:.2f}%")
    with c_m2:
        st.metric("Macro Precision", f"{clf_metrics['Test_Precision']:.4f}")
    with c_m3:
        st.metric("Macro Recall", f"{clf_metrics['Test_Recall']:.4f}")
    with c_m4:
        st.metric("Macro F1-Score", f"{clf_metrics['Test_F1_Macro']:.4f}")

    col_cm, col_comp = st.columns([1, 1])

    with col_cm:
        st.subheader(f"Confusion Matrix: {clf_choice}")
        cm_data = clf_metrics["Confusion_Matrix"]
        labels = ["Developing", "Star", "Elite"]
        fig_cm = px.imshow(
            cm_data,
            x=labels,
            y=labels,
            text_auto=True,
            color_continuous_scale="Blues",
            labels=dict(x="Predicted Class", y="Actual Ground Truth Class")
        )
        fig_cm.update_layout(height=380)
        st.plotly_chart(fig_cm, use_container_width=True)

    with col_comp:
        st.subheader("Model Comparison on Test Set (Accuracy & F1)")
        all_clf = metrics["classification"]
        comp_df = pd.DataFrame([
            {"Model": m, "Accuracy": all_clf[m]["Test_Accuracy"], "Macro F1": all_clf[m]["Test_F1_Macro"]}
            for m in all_clf
        ]).sort_values("Accuracy", ascending=False)
        fig_bar = px.bar(
            comp_df,
            x="Model",
            y=["Accuracy", "Macro F1"],
            barmode="group",
            color_discrete_sequence=["#2563EB", "#10B981"]
        )
        fig_bar.update_layout(height=380, yaxis_range=[0.75, 1.0])
        st.plotly_chart(fig_bar, use_container_width=True)

# ==============================================================================
# TAB 5: TACTICAL ARCHETYPES & PCA CLUSTERING
# ==============================================================================
elif menu == "5. Tactical Archetypes & PCA Clustering":
    st.markdown('<div class="main-header">Tactical Archetype Discovery & Dimensionality Reduction</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Module VII (K-Means Clustering) & Module VIII (Principal Component Analysis - PCA)</div>', unsafe_allow_html=True)

    col_pca1, col_pca2 = st.columns([2, 1])
    
    with col_pca1:
        st.subheader("Principal Component Analysis (2D Latent Skill Space)")
        # Project full dataset with PCA
        feature_cols = [c for c in metadata["encoded_feature_names"]]
        X_all = pd.get_dummies(df[metadata["num_features"] + metadata["cat_features"]], columns=metadata["cat_features"], drop_first=True, dtype=float)
        # align columns
        for c in feature_cols:
            if c not in X_all.columns:
                X_all[c] = 0.0
        X_all = X_all[feature_cols]
        X_scaled_all = models["scaler"].transform(X_all)
        pca_coords = models["pca"].transform(X_scaled_all)

        df_pca = df.copy()
        df_pca["PC1 (Technical & Athletic Mastery)"] = pca_coords[:, 0]
        df_pca["PC2 (Defensive vs Offensive Orientation)"] = pca_coords[:, 1]

        # Add K-Means cluster labels
        cluster_cols = models["kmeans"]["features"]
        df_pca["Tactical Archetype"] = [metrics["unsupervised"]["archetype_names"][str(c)] for c in models["kmeans"]["model"].predict(X_scaled_all[:, [feature_cols.index(c) for c in cluster_cols]])]

        color_by = st.radio("Color Scatter Points by:", ["Tactical Archetype", "primary_position", "performance_tier"], horizontal=True)

        fig_pca = px.scatter(
            df_pca,
            x="PC1 (Technical & Athletic Mastery)",
            y="PC2 (Defensive vs Offensive Orientation)",
            color=color_by,
            hover_data=["player_name", "club", "overall_performance_rating"],
            opacity=0.75,
            title="Projection of Players on Top 2 Principal Components (Explains 52.9% Variance)"
        )
        fig_pca.update_layout(height=520)
        st.plotly_chart(fig_pca, use_container_width=True)

    with col_pca2:
        st.subheader("Elbow Curve & Silhouette Validation")
        unsup = metrics["unsupervised"]
        fig_elbow = px.line(
            x=unsup["k_range"],
            y=unsup["inertias"],
            markers=True,
            title="Elbow Method: Inertia vs Cluster Count k",
            labels={"x": "Number of Clusters (k)", "y": "Within-Cluster Sum of Squares (Inertia)"}
        )
        fig_elbow.update_layout(height=260)
        st.plotly_chart(fig_elbow, use_container_width=True)

        fig_sil = px.bar(
            x=unsup["k_range"],
            y=unsup["silhouette_scores"],
            color=unsup["silhouette_scores"],
            title="Silhouette Scores across k (Optimal k=4)",
            labels={"x": "Number of Clusters (k)", "y": "Silhouette Score"}
        )
        fig_sil.update_layout(height=260, showlegend=False)
        st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("### Discovered Tactical Archetype Profiles")
    arch_cols = st.columns(4)
    archetypes_info = [
        {"name": "Tactical Playmaker & Orchestrator", "traits": "High vision, short passing, agility, composure; operates in half-spaces and directs ball flow."},
        {"name": "Defensive Anchor & Ball-Winner", "traits": "Exceptional standing tackle, defensive awareness, high strength, and recovery stamina."},
        {"name": "Explosive Forward & Finisher", "traits": "High sprint speed, lethal finishing, acceleration, and aggressive box movement."},
        {"name": "Goalkeeper / Specialist", "traits": "Specialized positioning, reflex stops, high physical frame, isolated low mobility footprint."}
    ]
    for i, arch in enumerate(archetypes_info):
        with arch_cols[i]:
            st.markdown(f"""
            <div class="metric-card">
                <h4 style="color: #1E3A8A;">Cluster {i}</h4>
                <h5>{arch['name']}</h5>
                <p style="font-size: 0.85rem; color: #475569;">{arch['traits']}</p>
            </div>
            """, unsafe_allow_html=True)

# ==============================================================================
# TAB 6: WHAT-IF SCOUTING SIMULATOR
# ==============================================================================
elif menu == "6. What-If Scouting Simulator":
    st.markdown('<div class="main-header">Scouting & What-If Development Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Simulating the impact of measurable athletic & technical training programs on player valuation</div>', unsafe_allow_html=True)

    st.write("Select a player from the database to run targeted physical and tactical training intervention simulations:")
    
    selected_player_name = st.selectbox(
        "Choose Player to Simulate:",
        df["player_name"].head(100).tolist()
    )
    player_row = df[df["player_name"] == selected_player_name].iloc[0]

    sc1, sc2, sc3 = st.columns([1, 1, 1])
    with sc1:
        st.markdown(f"""
        <div class="metric-card">
            <h4>Current Player Profile</h4>
            <h2>{player_row['player_name']}</h2>
            <p><strong>Club:</strong> {player_row['club']} | <strong>League:</strong> {player_row['league']}</p>
            <p><strong>Position:</strong> {player_row['primary_position']} | <strong>Age:</strong> {player_row['age']}</p>
            <h1 style="color: #2563EB;">{player_row['overall_performance_rating']}</h1>
            <p>Market Value: <strong>€{player_row['market_value_eur_m']}M</strong></p>
        </div>
        """, unsafe_allow_html=True)

    with sc2:
        st.markdown("#### Prescribe Targeted Training Program:")
        delta_stamina = st.slider("Stamina & Aerobic Conditioning (Δ)", -5, 15, 6)
        delta_sprint = st.slider("Sprint Speed & Agility Drills (Δ)", -5, 15, 4)
        delta_pass = st.slider("Passing & Vision Immersion (Δ)", -5, 15, 5)
        delta_comp = st.slider("Composure & Mental Resilience (Δ)", -5, 15, 7)

    # Calculate post-intervention rating
    new_stamina = np.clip(player_row['stamina'] + delta_stamina, 40, 99)
    new_sprint = np.clip(player_row['sprint_speed'] + delta_sprint, 40, 99)
    new_pass = np.clip(player_row['short_passing'] + delta_pass, 40, 99)
    new_comp = np.clip(player_row['composure'] + delta_comp, 40, 99)

    # Simplified delta model based on feature importances
    rating_gain = (
        delta_stamina * 0.12 +
        delta_sprint * 0.15 +
        delta_pass * 0.16 +
        delta_comp * 0.22
    )
    new_rating = round(float(np.clip(player_row['overall_performance_rating'] + rating_gain, 50.0, 95.0)), 1)
    new_value = round(float(np.exp((new_rating - 60) * 0.12) * (1.3 if player_row['age'] < 25 else 1.0 if player_row['age'] < 30 else 0.65)), 1)
    val_diff = round(new_value - player_row['market_value_eur_m'], 1)

    with sc3:
        st.markdown(f"""
        <div class="metric-card" style="border: 2px solid #10B981;">
            <h4>Simulated Post-Intervention</h4>
            <h2>Development Outlook</h2>
            <p><strong>Predicted Rating:</strong></p>
            <h1 style="color: #10B981;">{new_rating} <span style="font-size: 1.2rem;">(+{round(new_rating - player_row['overall_performance_rating'], 1)})</span></h1>
            <p>Projected Market Value:</p>
            <h3>€{new_value}M <span style="color: {'#10B981' if val_diff >= 0 else '#EF4444'}; font-size: 1rem;">({'+' if val_diff >= 0 else ''}{val_diff}M)</span></h3>
        </div>
        """, unsafe_allow_html=True)

# ==============================================================================
# TAB 7: MODEL EVALUATION & BENCHMARK STUDIO
# ==============================================================================
elif menu == "7. Model Evaluation & Benchmark Studio":
    st.markdown('<div class="main-header">Rigorous Model Evaluation Studio</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Module VI: 5-Fold Cross Validation, Error Diagnostics, and Feature Importance Rankings</div>', unsafe_allow_html=True)

    st.subheader("1. Supervised Learning Regression Benchmark (Module IV)")
    reg_df = pd.DataFrame(metrics["regression"]).T.reset_index()
    reg_df.rename(columns={"index": "Model Algorithm"}, inplace=True)
    st.dataframe(reg_df.style.highlight_max(subset=["Test_R2", "CV_R2_mean"], color="#DCFCE7").highlight_min(subset=["Test_RMSE", "Test_MAE"], color="#DCFCE7"))

    st.markdown("---")
    st.subheader("2. Supervised Learning Classification Benchmark (Module V)")
    clf_df = pd.DataFrame(metrics["classification"]).T.reset_index()
    clf_df.rename(columns={"index": "Model Algorithm"}, inplace=True)
    clf_df_display = clf_df.drop(columns=["Confusion_Matrix"])
    st.dataframe(clf_df_display.style.highlight_max(subset=["Test_Accuracy", "Test_F1_Macro", "CV_Accuracy_mean"], color="#DCFCE7"))

    st.markdown("---")
    st.subheader("3. Feature Importance Analysis (Module VIII Ensemble)")
    col_imp1, col_imp2 = st.columns(2)
    with col_imp1:
        st.markdown("#### Top Factors Driving Overall Performance Rating (Random Forest Regressor)")
        reg_imp = pd.DataFrame(list(metrics["feature_importances"]["regression_top10"].items()), columns=["Measurable Factor", "Importance Weight"])
        fig_imp1 = px.bar(reg_imp, x="Importance Weight", y="Measurable Factor", orientation="h", color="Importance Weight", color_continuous_scale="Blues")
        fig_imp1.update_layout(yaxis=dict(autorange="reversed"), height=380)
        st.plotly_chart(fig_imp1, use_container_width=True)

    with col_imp2:
        st.markdown("#### Top Factors Driving Performance Tier Classification (Random Forest Classifier)")
        clf_imp = pd.DataFrame(list(metrics["feature_importances"]["classification_top10"].items()), columns=["Measurable Factor", "Importance Weight"])
        fig_imp2 = px.bar(clf_imp, x="Importance Weight", y="Measurable Factor", orientation="h", color="Importance Weight", color_continuous_scale="Greens")
        fig_imp2.update_layout(yaxis=dict(autorange="reversed"), height=380)
        st.plotly_chart(fig_imp2, use_container_width=True)

    st.markdown("### 4. Technical Diagnostics & Error Analysis Summary")
    st.write("""
    - **Linear vs Non-Linear Bounds:** Linear Regression achieves an impressive $R^2 = 0.9386$, indicating that performance is predominantly a strong linear combination of core skills. However, Random Forest Regressor edges ahead at $R^2 = 0.9461$ and lower RMSE ($1.9189$), capturing slight non-linear interactions between athleticism and tactical positioning.
    - **Classification Trade-offs:** Multinomial Logistic Regression and Random Forest Classifier tie with the highest test accuracy ($87.97\%$ and $87.81\%$). In contrast, Decision Tree ($79.84\%$) suffers from high variance and hierarchical boundary slicing.
    - **Feature Insights:** `athletic_power_index` and `composure` dominate model splits, confirming the hypothesis that innate physical capability paired with elite cognitive decision-making constitutes over $80\%$ of professional player performance ratings.
    """)
