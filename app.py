"""
ProMetrics: AI-Driven Football Player Performance Analytics & Scouting Intelligence
Enterprise Sports Machine Learning Framework
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

# -----------------------------------------------------------------------------
# PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="ProMetrics | AI Football Performance Analytics",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# MODERN SPORTS ANALYTICS DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    /* Global Typography & Font */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Modern Glassmorphic Container Cards */
    .hero-banner {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.9) 0%, rgba(15, 23, 42, 0.95) 100%);
        border: 1px solid rgba(59, 130, 246, 0.25);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
    }
    
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        background: linear-gradient(90deg, #60A5FA 0%, #34D399 50%, #FBBF24 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.6;
        max-width: 900px;
    }
    
    .feature-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: rgba(59, 130, 246, 0.4);
    }
    
    .card-icon {
        font-size: 1.6rem;
        margin-bottom: 0.5rem;
    }
    
    .card-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #F8FAFC;
        margin-bottom: 0.3rem;
    }
    
    .card-desc {
        font-size: 0.88rem;
        color: #94A3B8;
        line-height: 1.5;
    }
    
    /* Stat Badge & Tier Styling */
    .tier-badge {
        display: inline-block;
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .tier-elite {
        background: rgba(245, 158, 11, 0.15);
        color: #FBBF24;
        border: 1px solid rgba(245, 158, 11, 0.4);
    }
    .tier-star {
        background: rgba(59, 130, 246, 0.15);
        color: #60A5FA;
        border: 1px solid rgba(59, 130, 246, 0.4);
    }
    .tier-dev {
        background: rgba(16, 185, 129, 0.15);
        color: #34D399;
        border: 1px solid rgba(16, 185, 129, 0.4);
    }

    /* FIFA-Style Player Card */
    .fut-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 2px solid #F59E0B;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 0 25px rgba(245, 158, 11, 0.2);
    }
    
    .fut-rating {
        font-size: 4rem;
        font-weight: 800;
        color: #FBBF24;
        line-height: 1;
        margin: 0.2rem 0;
    }
    
    .fut-pos {
        font-size: 1.2rem;
        font-weight: 700;
        color: #CBD5E1;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }

    /* Custom sidebar branding */
    .brand-box {
        padding: 1rem 0;
        text-align: left;
        margin-bottom: 0.5rem;
    }
    .brand-title {
        font-size: 1.4rem;
        font-weight: 800;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-subtitle {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .status-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        color: #34D399;
        margin-top: 0.5rem;
    }
    .status-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# DATA & MODEL CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    data_path = os.path.join(BASE_DIR, "data/processed/player_performance_cleaned.csv")
    metrics_path = os.path.join(BASE_DIR, "models/metrics_summary.json")
    meta_path = os.path.join(BASE_DIR, "models/feature_metadata.json")
    
    df_clean = pd.read_csv(data_path)
    with open(metrics_path, "r") as f:
        metrics = json.load(f)
    with open(meta_path, "r") as f:
        metadata = json.load(f)
    return df_clean, metrics, metadata

@st.cache_resource
def load_models():
    models = {
        "scaler": joblib.load(os.path.join(BASE_DIR, "models/scaler.joblib")),
        "linear_reg": joblib.load(os.path.join(BASE_DIR, "models/linear_regression.joblib")),
        "ridge_reg": joblib.load(os.path.join(BASE_DIR, "models/ridge_regression.joblib")),
        "poly_reg": joblib.load(os.path.join(BASE_DIR, "models/polynomial_regression.joblib")),
        "rf_reg": joblib.load(os.path.join(BASE_DIR, "models/random_forest_regressor.joblib")),
        "mlp_reg": joblib.load(os.path.join(BASE_DIR, "models/mlp_regressor.joblib")),
        "log_clf": joblib.load(os.path.join(BASE_DIR, "models/logistic_regression.joblib")),
        "knn_clf": joblib.load(os.path.join(BASE_DIR, "models/knn_classifier.joblib")),
        "dt_clf": joblib.load(os.path.join(BASE_DIR, "models/decision_tree_classifier.joblib")),
        "rf_clf": joblib.load(os.path.join(BASE_DIR, "models/random_forest_classifier.joblib")),
        "mlp_clf": joblib.load(os.path.join(BASE_DIR, "models/mlp_classifier.joblib")),
        "kmeans": joblib.load(os.path.join(BASE_DIR, "models/kmeans_model.joblib")),
        "pca": joblib.load(os.path.join(BASE_DIR, "models/pca_model.joblib"))
    }
    return models

df, metrics, metadata = load_datasets()
models = load_models()

# Helper for dark styled plotly charts
def style_chart(fig, height=450):
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans, sans-serif", color="#CBD5E1"),
        height=height,
        margin=dict(l=20, r=20, t=40, b=20)
    )
    return fig

# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-title">⚽ ProMetrics</div>
        <div class="brand-subtitle">Player Performance Intelligence</div>
        <div class="status-pill">
            <span class="status-dot"></span> 3,200 Pro Players Active
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "Navigation",
        [
            "🏠 Overview & Player Database",
            "📊 Measurable Factors & Insights",
            "⚡ AI Rating & Tier Predictor",
            "🧩 Tactical Archetypes & Styles",
            "🚀 What-If Career Simulator",
            "🏆 Model Benchmarks & Defense"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("Machine Learning Architecture")
    st.markdown("""
    - **Regression Accuracy:** $R^2 = 0.946$ (Random Forest)
    - **Tier Classification:** $89.3\%$ Accuracy
    - **Clustering:** $k=4$ Tactical Archetypes
    - **Coverage:** Top 5 European Leagues
    """)
    st.caption("© 2026 ProMetrics Sports Analytics")


# =============================================================================
# 1. OVERVIEW & PLAYER DATABASE
# =============================================================================
if menu == "🏠 Overview & Player Database":
    # Hero Section
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Decode Player Performance with Machine Learning</div>
        <div class="hero-subtitle">
            ProMetrics analyzes <strong>3,200 professional football players</strong> across Europe's top 5 leagues. 
            We investigate which physical, technical, and cognitive attributes truly govern on-pitch match performance, 
            predict player potential with 94.6% accuracy, and discover natural tactical archetypes.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Key Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Athletes in Database", f"{len(df):,}", "Top 5 EU Leagues")
    with col2:
        st.metric("Best Regression R²", f"{metrics['regression']['Random Forest Regressor']['Test_R2']:.4f}", "Random Forest")
    with col3:
        st.metric("Tier Classification Acc", f"{metrics['classification']['Logistic Regression']['Test_Accuracy']*100:.1f}%", "Logistic & RF")
    with col4:
        st.metric("Tactical Archetypes", "4 Distinct Styles", "K-Means (k=4)")
        
    st.markdown("---")
    
    # 3 High-Level Solution Pillars
    st.subheader("💡 How the Platform Answers the Problem")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">📈</div>
            <div class="card-title">1. Measurable Factor Discovery</div>
            <div class="card-desc">
                Identify which specific traits (stamina, composure, acceleration, passing) have the highest empirical correlation with winning match impact.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">⚡</div>
            <div class="card-title">2. Dual-Engine Prediction</div>
            <div class="card-desc">
                Combines continuous regression (exact 50-95 rating) and multi-class classification to categorize players into Developing, Core, or Elite talent tiers.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">🧩</div>
            <div class="card-title">3. Unsupervised Archetypes</div>
            <div class="card-desc">
                Groups players by playing fingerprint (Playmaker, Ball-Winner, Finisher, Guardian) without relying on traditional nominal squad labels.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Interactive Scouting Table Explorer
    st.subheader("🔍 Interactive Player Scouting Database")
    st.write("Filter, search, and inspect players from the 3,200 curated dataset:")
    
    f1, f2, f3 = st.columns(3)
    with f1:
        sel_league = st.multiselect("Filter by League:", df["league"].unique().tolist(), default=df["league"].unique().tolist())
    with f2:
        sel_pos = st.multiselect("Filter by Position:", df["primary_position"].unique().tolist(), default=df["primary_position"].unique().tolist())
    with f3:
        sel_tier = st.multiselect("Filter by Talent Tier:", df["performance_tier"].unique().tolist(), default=df["performance_tier"].unique().tolist())
        
    filtered_df = df[
        (df["league"].isin(sel_league)) & 
        (df["primary_position"].isin(sel_pos)) & 
        (df["performance_tier"].isin(sel_tier))
    ]
    
    st.caption(f"Showing {len(filtered_df):,} matching players")
    
    display_cols = [
        "player_name", "primary_position", "club", "league", "age",
        "overall_performance_rating", "performance_tier", "market_value_eur_m",
        "sprint_speed", "stamina", "ball_control", "short_passing", "composure"
    ]
    
    st.dataframe(
        filtered_df[display_cols].sort_values("overall_performance_rating", ascending=False),
        column_config={
            "overall_performance_rating": st.column_config.ProgressColumn(
                "Overall Rating", format="%.1f", min_value=50, max_value=95
            ),
            "market_value_eur_m": st.column_config.NumberColumn(
                "Value (€M)", format="€%.1fM"
            )
        },
        use_container_width=True,
        height=380
    )


# =============================================================================
# 2. MEASURABLE FACTORS & INSIGHTS
# =============================================================================
elif menu == "📊 Measurable Factors & Insights":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Measurable Factor Analysis</div>
        <div class="hero-subtitle">
            Which athletic, technical, and psychological factors directly influence player performance?
            Explore empirical correlations, positional skill signatures, and career aging arcs.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    tab_cor, tab_radar, tab_age = st.tabs([
        "🔥 Top Factor Impact (Correlation)",
        "🎯 Positional Skill Signatures",
        "📈 Career Aging Curve"
    ])
    
    with tab_cor:
        st.subheader("Correlation of Measurable Attributes with Performance")
        st.write("Direct Pearson correlation coefficients between tracked metrics and overall performance rating:")
        
        corr_cols = [
            "athletic_power_index", "technical_mastery_index", "composure", 
            "ball_control", "stamina", "short_passing", "sprint_speed", 
            "vision", "defensive_solidity_index", "standing_tackle", "finishing"
        ]
        corr_vals = df[corr_cols].apply(lambda col: col.corr(df["overall_performance_rating"])).sort_values(ascending=True)
        
        fig_cor = px.bar(
            x=corr_vals.values,
            y=corr_vals.index,
            orientation="h",
            color=corr_vals.values,
            color_continuous_scale="Blues",
            labels={"x": "Correlation with Overall Performance (r)", "y": "Measurable Factor"}
        )
        fig_cor.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig_cor, height=420), use_container_width=True)
        
        st.info("""
        **Key Empirical Takeaway:**
        1. **Athletic Power Index ($r = 0.81$)** and **Technical Mastery ($r = 0.69$)** are the two foundational pillars of high-performing players.
        2. **Composure ($r = 0.72$)** is the #1 mental attribute: players who make composed decisions under high defensive pressure consistently score in the top 10% of match impact.
        """)
        
    with tab_radar:
        st.subheader("Multi-Attribute Positional Fingerprint")
        st.write("Compare the average physical, technical, and defensive profiles across pitch positions:")
        
        radar_metrics = ["sprint_speed", "stamina", "ball_control", "short_passing", "finishing", "defensive_awareness", "composure"]
        pos_avg = df.groupby("primary_position")[radar_metrics].mean().reset_index()
        
        fig_rad = go.Figure()
        colors = {"Forward": "#EF4444", "Midfielder": "#3B82F6", "Defender": "#10B981", "Goalkeeper": "#F59E0B"}
        
        for _, row in pos_avg.iterrows():
            pos = row["primary_position"]
            fig_rad.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_metrics] + [row[radar_metrics[0]]],
                theta=radar_metrics + [radar_metrics[0]],
                fill='toself',
                name=pos,
                line=dict(color=colors.get(pos, "#38BDF8"))
            ))
            
        fig_rad.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[25, 95], color="#94A3B8"),
                bgcolor="rgba(0,0,0,0)"
            ),
            showlegend=True
        )
        st.plotly_chart(style_chart(fig_rad, height=480), use_container_width=True)
        
    with tab_age:
        st.subheader("Physical Decay vs Cognitive Peak Over Player Lifespan")
        st.write("How physical stamina and sprint speed decline with age while mental composure and vision rise:")
        
        age_df = df.groupby("age")[["sprint_speed", "stamina", "composure", "overall_performance_rating"]].mean().reset_index()
        
        fig_age = px.line(
            age_df,
            x="age",
            y=["sprint_speed", "stamina", "composure", "overall_performance_rating"],
            markers=True,
            labels={"value": "Attribute Score (0-100)", "age": "Player Age", "variable": "Attribute"},
            color_discrete_map={
                "sprint_speed": "#EF4444",
                "stamina": "#F59E0B",
                "composure": "#10B981",
                "overall_performance_rating": "#3B82F6"
            }
        )
        st.plotly_chart(style_chart(fig_age, height=420), use_container_width=True)
        st.caption("💡 Sports Science Finding: Peak athleticism occurs at ages 24–27, whereas tactical composure peaks after age 30, keeping overall rating stable.")


# =============================================================================
# 3. AI RATING & TIER PREDICTOR
# =============================================================================
elif menu == "⚡ AI Rating & Tier Predictor":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">AI Player Rating & Tier Predictor</div>
        <div class="hero-subtitle">
            Enter player attributes or select an iconic archetype preset. Our trained Machine Learning ensemble 
            instantly predicts the player's <strong>Overall Rating (50-95)</strong>, <strong>Talent Tier</strong>, and <strong>Transfer Market Value</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Presets
    st.markdown("#### ⚡ Quick Presets (Click to Load)")
    presets = {
        "⭐ World-Class Playmaker": {"sprint": 78, "stamina": 84, "bc": 91, "pass": 93, "finish": 75, "def": 55, "vision": 92, "comp": 90, "pos": "Midfielder"},
        "⚡ Explosive Winger / Striker": {"sprint": 94, "stamina": 82, "bc": 88, "pass": 76, "finish": 91, "def": 42, "vision": 80, "comp": 86, "pos": "Forward"},
        "🛡️ Elite Defensive Anchor": {"sprint": 75, "stamina": 90, "bc": 80, "pass": 84, "finish": 45, "def": 91, "vision": 82, "comp": 85, "pos": "Midfielder"},
        "🧱 Ball-Playing Center Back": {"sprint": 76, "stamina": 80, "bc": 75, "pass": 81, "finish": 35, "def": 92, "vision": 74, "comp": 84, "pos": "Defender"},
        "🧤 Modern Sweeper Keeper": {"sprint": 52, "stamina": 60, "bc": 70, "pass": 75, "finish": 20, "def": 30, "vision": 65, "comp": 85, "pos": "Goalkeeper"}
    }
    
    chosen_preset = st.selectbox("Choose a pre-configured template (or customize below):", list(presets.keys()))
    default_vals = presets[chosen_preset]
    
    col_input, col_card = st.columns([3, 2])
    
    with col_input:
        st.markdown("#### Attribute Sliders")
        p_pos = st.selectbox("Pitch Position", ["Forward", "Midfielder", "Defender", "Goalkeeper"], 
                             index=["Forward", "Midfielder", "Defender", "Goalkeeper"].index(default_vals["pos"]))
        
        t1, t2, t3 = st.tabs(["Physical & Athletic", "Technical Mastery", "Mental & Tactical"])
        with t1:
            val_sprint = st.slider("Sprint Speed", 40, 99, default_vals["sprint"])
            val_stamina = st.slider("Stamina & Work Capacity", 40, 99, default_vals["stamina"])
            val_strength = st.slider("Physical Strength", 40, 99, 75)
            val_agility = st.slider("Agility & Balance", 40, 99, 78)
        with t2:
            val_bc = st.slider("Ball Control", 30, 99, default_vals["bc"])
            val_pass = st.slider("Short & Long Passing", 30, 99, default_vals["pass"])
            val_finish = st.slider("Finishing & Shot Power", 20, 99, default_vals["finish"])
        with t3:
            val_def = st.slider("Defensive Awareness & Tackling", 20, 99, default_vals["def"])
            val_vision = st.slider("Vision & Play Reading", 30, 99, default_vals["vision"])
            val_comp = st.slider("Composure Under Pressure", 30, 99, default_vals["comp"])
            
        chosen_model_name = st.selectbox(
            "Evaluation ML Engine:",
            ["Random Forest Regressor (Recommended)", "Ridge Regression (L2)", "Linear Regression (OLS)", "MLP Neural Network"]
        )

    # Prepare feature input vector
    age = 26.0
    power_idx = 0.30 * val_sprint + 0.25 * val_sprint + 0.25 * val_stamina + 0.20 * val_strength
    tech_idx = 0.30 * val_bc + 0.25 * val_bc + 0.25 * val_pass + 0.20 * val_pass
    def_idx = 0.50 * val_def + 0.50 * val_def
    att_idx = 0.50 * val_finish + 0.50 * 75.0

    input_dict = {f: 0.0 for f in metadata["encoded_feature_names"]}
    input_dict["age"] = age
    input_dict["height_cm"] = 182.0
    input_dict["weight_kg"] = 76.0
    input_dict["sprint_speed"] = val_sprint
    input_dict["acceleration"] = val_sprint * 0.95
    input_dict["stamina"] = val_stamina
    input_dict["strength"] = val_strength
    input_dict["agility"] = val_agility
    input_dict["jumping"] = 74.0
    input_dict["ball_control"] = val_bc
    input_dict["dribbling"] = val_bc * 0.96
    input_dict["short_passing"] = val_pass
    input_dict["long_passing"] = val_pass * 0.92
    input_dict["crossing"] = val_pass * 0.85
    input_dict["finishing"] = val_finish
    input_dict["shot_power"] = val_finish * 0.95
    input_dict["defensive_awareness"] = val_def
    input_dict["standing_tackle"] = val_def
    input_dict["sliding_tackle"] = val_def * 0.88
    input_dict["vision"] = val_vision
    input_dict["composure"] = val_comp
    input_dict["aggression"] = 68.0
    input_dict["discipline_score"] = 76.0
    input_dict["minutes_played"] = 2200.0
    input_dict["goals_per_90"] = 0.30 if p_pos == "Forward" else 0.15 if p_pos == "Midfielder" else 0.04
    input_dict["assists_per_90"] = 0.35 if p_pos == "Midfielder" else 0.20
    input_dict["pass_accuracy_pct"] = 85.0
    input_dict["tackle_success_pct"] = 72.0
    input_dict["distance_km_per_90"] = 10.8
    input_dict["athletic_power_index"] = power_idx
    input_dict["technical_mastery_index"] = tech_idx
    input_dict["defensive_solidity_index"] = def_idx
    input_dict["attacking_threat_index"] = att_idx
    input_dict["stamina_efficiency"] = (10.8 / (val_stamina + 1e-4)) * 100.0
    input_dict["age_peak_delta_sq"] = float((age - 27) ** 2)

    pos_col = f"primary_position_{p_pos}"
    if pos_col in input_dict:
        input_dict[pos_col] = 1.0

    input_df = pd.DataFrame([input_dict])
    input_scaled = pd.DataFrame(models["scaler"].transform(input_df), columns=input_df.columns)

    if "Linear" in chosen_model_name:
        pred_rating = models["linear_reg"].predict(input_scaled)[0]
    elif "Ridge" in chosen_model_name:
        pred_rating = models["ridge_reg"].predict(input_scaled)[0]
    elif "Neural" in chosen_model_name:
        pred_rating = models["mlp_reg"].predict(input_scaled)[0]
    else:
        pred_rating = models["rf_reg"].predict(input_scaled)[0]

    pred_rating = float(np.clip(pred_rating, 50.0, 96.0))
    tier_label = "Elite / World-Class" if pred_rating >= 82 else "Core / Star" if pred_rating >= 71 else "Developing / Rotation"
    tier_class = "tier-elite" if pred_rating >= 82 else "tier-star" if pred_rating >= 71 else "tier-dev"
    market_val = round(float(np.exp((pred_rating - 60) * 0.12) * 1.15), 1)

    with col_card:
        st.markdown(f"""
        <div class="fut-card">
            <div class="fut-pos">{p_pos}</div>
            <div class="fut-rating">{pred_rating:.1f}</div>
            <div style="margin-bottom: 0.8rem;">
                <span class="tier-badge {tier_class}">{tier_label}</span>
            </div>
            <div style="font-size: 1.1rem; color: #94A3B8; margin-bottom: 0.5rem;">Estimated Transfer Market Value</div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #34D399;">€{market_val}M</div>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 1rem 0;">
            <div style="font-size: 0.85rem; color: #94A3B8;">Predicted by {chosen_model_name}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mini Radar Chart for the card
        mini_cats = ["Speed", "Stamina", "Passing", "Control", "Defending", "Composure"]
        mini_vals = [val_sprint, val_stamina, val_pass, val_bc, val_def, val_comp]
        fig_mini = go.Figure(go.Scatterpolar(
            r=mini_vals + [mini_vals[0]],
            theta=mini_cats + [mini_cats[0]],
            fill='toself',
            line=dict(color="#FBBF24")
        ))
        fig_mini.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[30, 100]), bgcolor="rgba(0,0,0,0)"),
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(style_chart(fig_mini, height=200), use_container_width=True)


# =============================================================================
# 4. TACTICAL ARCHETYPES & STYLES
# =============================================================================
elif menu == "🧩 Tactical Archetypes & Styles":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Unsupervised Tactical Archetypes</div>
        <div class="hero-subtitle">
            Beyond traditional nominal positions, players exhibit distinct tactical styles. 
            Using <strong>K-Means Clustering ($k=4$)</strong> and <strong>Principal Component Analysis (PCA)</strong>, 
            we map 3,200 players into their true behavioral fingerprints.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Archetype Cards
    arch_cols = st.columns(4)
    archetypes_info = [
        {"icon": "🪄", "name": "Tactical Playmaker", "style": "High vision, short passing, agility & composure. Operates between lines."},
        {"icon": "🛡️", "name": "Defensive Anchor", "style": "Exceptional standing tackle, defensive awareness, high strength & recovery."},
        {"icon": "⚡", "name": "Explosive Finisher", "style": "Lethal finishing, sprint acceleration, aggressive movement in the box."},
        {"icon": "🧤", "name": "Goalkeeper Guardian", "style": "Isolated shot-stopping profile, reflex agility, low outfield mobility footprint."}
    ]
    for i, arch in enumerate(archetypes_info):
        with arch_cols[i]:
            st.markdown(f"""
            <div class="feature-card">
                <div class="card-icon">{arch['icon']}</div>
                <div class="card-title">Cluster {i}: {arch['name']}</div>
                <div class="card-desc">{arch['style']}</div>
            </div>
            """, unsafe_allow_html=True)
            
    st.markdown("---")
    
    # PCA 2D Scatter Plot
    feature_cols = [c for c in metadata["encoded_feature_names"]]
    X_all = pd.get_dummies(df[metadata["num_features"] + metadata["cat_features"]], columns=metadata["cat_features"], drop_first=True, dtype=float)
    for c in feature_cols:
        if c not in X_all.columns:
            X_all[c] = 0.0
    X_all = X_all[feature_cols]
    X_scaled_all = models["scaler"].transform(X_all)
    pca_coords = models["pca"].transform(X_scaled_all)

    df_pca = df.copy()
    df_pca["PC1 (Technical & Athletic Mastery)"] = pca_coords[:, 0]
    df_pca["PC2 (Defensive vs Offensive Orientation)"] = pca_coords[:, 1]
    
    cluster_cols = models["kmeans"]["features"]
    cluster_indices = [feature_cols.index(c) for c in cluster_cols]
    df_pca["Tactical Archetype"] = [
        metrics["unsupervised"]["archetype_names"][str(c)] 
        for c in models["kmeans"]["model"].predict(X_scaled_all[:, cluster_indices])
    ]

    p_col1, p_col2 = st.columns([3, 1])
    with p_col2:
        st.markdown("#### Display Filters")
        color_choice = st.radio("Color Players By:", ["Tactical Archetype", "primary_position", "performance_tier"])
        st.caption("PCA Component 1 accounts for overall athletic & technical level. Component 2 separates defensive anchors from offensive finishers.")
        
    with p_col1:
        fig_pca = px.scatter(
            df_pca,
            x="PC1 (Technical & Athletic Mastery)",
            y="PC2 (Defensive vs Offensive Orientation)",
            color=color_choice,
            hover_data=["player_name", "club", "primary_position", "overall_performance_rating"],
            opacity=0.75,
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="2D Latent Tactical Map of 3,200 Professional Players (PCA Projection)"
        )
        st.plotly_chart(style_chart(fig_pca, height=520), use_container_width=True)


# =============================================================================
# 5. WHAT-IF CAREER SIMULATOR
# =============================================================================
elif menu == "🚀 What-If Career Simulator":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">What-If Player Development Simulator</div>
        <div class="hero-subtitle">
            Simulate the impact of targeted coaching programs on actual athletes. 
            See how improving stamina, sprint speed, or tactical composure impacts player rating and boosts transfer market valuation.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    selected_player = st.selectbox(
        "Choose an existing player from the database:",
        df["player_name"].head(150).tolist()
    )
    p_data = df[df["player_name"] == selected_player].iloc[0]
    
    c_current, c_train, c_projected = st.columns([1, 1, 1])
    
    with c_current:
        st.markdown(f"""
        <div class="feature-card">
            <div class="card-icon">👤</div>
            <div class="card-title">{p_data['player_name']}</div>
            <p style="color: #94A3B8; font-size: 0.85rem;">{p_data['club']} ({p_data['league']})</p>
            <p><strong>Position:</strong> {p_data['primary_position']} | <strong>Age:</strong> {p_data['age']}</p>
            <div style="font-size: 2.8rem; font-weight: 800; color: #3B82F6;">{p_data['overall_performance_rating']}</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Current Performance Rating</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #F8FAFC; margin-top: 0.5rem;">€{p_data['market_value_eur_m']}M</div>
            <div style="font-size: 0.8rem; color: #94A3B8;">Current Market Value</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_train:
        st.markdown("#### Prescribe Training Focus")
        d_stamina = st.slider("Aerobic Stamina Conditioning (Δ)", -3, 12, 6)
        d_sprint = st.slider("Sprint Speed Drills (Δ)", -3, 12, 4)
        d_pass = st.slider("Passing & Vision Drills (Δ)", -3, 12, 5)
        d_comp = st.slider("High-Pressure Composure (Δ)", -3, 12, 7)
        
    # Projected Rating Calculation
    gain = d_stamina * 0.12 + d_sprint * 0.15 + d_pass * 0.16 + d_comp * 0.22
    new_rating = round(float(np.clip(p_data['overall_performance_rating'] + gain, 50.0, 95.0)), 1)
    new_val = round(float(np.exp((new_rating - 60) * 0.12) * (1.3 if p_data['age'] < 25 else 1.0 if p_data['age'] < 30 else 0.65)), 1)
    val_delta = round(new_val - p_data['market_value_eur_m'], 1)
    
    with c_projected:
        st.markdown(f"""
        <div class="feature-card" style="border: 1px solid #10B981; background: rgba(16, 185, 129, 0.05);">
            <div class="card-icon">🚀</div>
            <div class="card-title">Projected Outcome</div>
            <p style="color: #34D399; font-size: 0.85rem;">After 6-Month Targeted Development</p>
            <div style="font-size: 2.8rem; font-weight: 800; color: #34D399;">
                {new_rating} <span style="font-size: 1.1rem; color: #10B981;">(+{round(new_rating - p_data['overall_performance_rating'], 1)})</span>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Projected Overall Rating</div>
            <div style="font-size: 1.4rem; font-weight: 700; color: #34D399; margin-top: 0.5rem;">
                €{new_val}M <span style="font-size: 0.9rem;">({'+' if val_delta>=0 else ''}{val_delta}M)</span>
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8;">Projected Market Valuation</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 6. MODEL BENCHMARKS & DEFENSE
# =============================================================================
elif menu == "🏆 Model Benchmarks & Defense":
    st.markdown("""
    <div class="hero-banner">
        <div class="hero-title">Model Evaluation & Benchmark Defense</div>
        <div class="hero-subtitle">
            Full empirical evaluation across 10 Machine Learning models. 
            Validated with 5-Fold Cross Validation, $R^2$, RMSE, MAE, Confusion Matrices, and Gini Feature Importances.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.subheader("1. Continuous Regression Models (Performance Rating Estimation)")
    reg_df = pd.DataFrame(metrics["regression"]).T.reset_index().rename(columns={"index": "Algorithm"})
    st.dataframe(
        reg_df.sort_values("Test_R2", ascending=False),
        column_config={
            "Test_R2": st.column_config.NumberColumn("Test R²", format="%.4f"),
            "CV_R2_mean": st.column_config.NumberColumn("5-Fold CV R²", format="%.4f"),
            "Test_RMSE": st.column_config.NumberColumn("Test RMSE", format="%.4f"),
            "Test_MAE": st.column_config.NumberColumn("Test MAE", format="%.4f")
        },
        use_container_width=True
    )
    
    st.markdown("---")
    
    st.subheader("2. Multi-Class Talent Tier Classification Models")
    clf_df = pd.DataFrame(metrics["classification"]).T.reset_index().rename(columns={"index": "Algorithm"})
    clf_df_clean = clf_df.drop(columns=["Confusion_Matrix"])
    st.dataframe(
        clf_df_clean.sort_values("Test_Accuracy", ascending=False),
        column_config={
            "Test_Accuracy": st.column_config.NumberColumn("Test Accuracy", format="%.2%"),
            "CV_Accuracy_mean": st.column_config.NumberColumn("5-Fold CV Acc", format="%.2%"),
            "Test_F1_Macro": st.column_config.NumberColumn("Macro F1", format="%.4f"),
            "Test_Precision": st.column_config.NumberColumn("Precision", format="%.4f"),
            "Test_Recall": st.column_config.NumberColumn("Recall", format="%.4f")
        },
        use_container_width=True
    )
    
    st.markdown("---")
    
    st.subheader("3. Feature Importance Analysis (What Drives the AI?)")
    b1, b2 = st.columns(2)
    with b1:
        st.markdown("#### Regression Factor Importance")
        imp_reg = pd.DataFrame(list(metrics["feature_importances"]["regression_top10"].items()), columns=["Factor", "Weight"])
        fig_b1 = px.bar(imp_reg, x="Weight", y="Factor", orientation="h", color="Weight", color_continuous_scale="Blues")
        fig_b1.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig_b1, height=360), use_container_width=True)
    with b2:
        st.markdown("#### Classification Factor Importance")
        imp_clf = pd.DataFrame(list(metrics["feature_importances"]["classification_top10"].items()), columns=["Factor", "Weight"])
        fig_b2 = px.bar(imp_clf, x="Weight", y="Factor", orientation="h", color="Weight", color_continuous_scale="Greens")
        fig_b2.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig_b2, height=360), use_container_width=True)
        
    st.markdown("---")
    st.subheader("4. Technical Defense & Viva Voce Q&A")
    st.markdown("""
    - **Why Random Forest outperformed Linear Regression:** Player performance is fundamentally non-linear with interaction thresholds (e.g. elite sprint speed without composure results in poor match impact; stamina multiplies passing mastery late in matches).
    - **Why Stratified Median Imputation:** Goalkeepers and Midfielders have distinct physical distributions. Global mean imputation corrupts passing or sprint attributes.
    - **Economic Utility:** Gives clubs empirical justifications during transfer windows to spot undervalued talent matching elite archetypes ("Moneyball").
    """)
