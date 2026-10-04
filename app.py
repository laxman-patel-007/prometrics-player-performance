"""
CricMetrics Pro: AI-Powered Cricket Player Performance Analytics & Auction War Room
Enterprise Decision Support Platform for Franchise Management, Scouting & Player Valuation
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
# 1. PAGE CONFIGURATION
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="CricMetrics Pro | Franchise Analytics & Auction War Room",
    page_icon="🏏",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. ENTERPRISE SPORTS ANALYTICS DESIGN SYSTEM (CSS)
# -----------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hero Banner for Franchise War Room */
    .hero-banner {
        background: linear-gradient(135deg, rgba(15, 23, 42, 0.95) 0%, rgba(30, 41, 59, 0.9) 50%, rgba(15, 23, 42, 0.98) 100%);
        border: 1px solid rgba(245, 158, 11, 0.3);
        border-radius: 16px;
        padding: 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -5px rgba(0, 0, 0, 0.4);
    }
    
    .hero-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F59E0B 0%, #FBBF24 35%, #38BDF8 70%, #34D399 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        line-height: 1.6;
        max-width: 950px;
    }

    .org-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.5rem;
        background: rgba(245, 158, 11, 0.12);
        border: 1px solid rgba(245, 158, 11, 0.35);
        padding: 0.3rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 700;
        color: #FBBF24;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    
    /* Feature Card */
    .feature-card {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.09);
        border-radius: 14px;
        padding: 1.25rem;
        height: 100%;
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .feature-card:hover {
        transform: translateY(-2px);
        border-color: rgba(245, 158, 11, 0.5);
    }
    
    .card-icon {
        font-size: 1.6rem;
        margin-bottom: 0.4rem;
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
    
    /* Auction War Room Player Card */
    .auction-card {
        background: linear-gradient(145deg, #1E293B, #0F172A);
        border: 2px solid #F59E0B;
        border-radius: 18px;
        padding: 1.8rem;
        text-align: center;
        box-shadow: 0 0 30px rgba(245, 158, 11, 0.25);
    }
    
    .auction-rating {
        font-size: 4rem;
        font-weight: 800;
        color: #FBBF24;
        line-height: 1;
        margin: 0.2rem 0;
    }
    
    .auction-price {
        font-size: 2.3rem;
        font-weight: 800;
        color: #34D399;
        margin: 0.3rem 0;
    }
    
    .tier-badge {
        display: inline-block;
        padding: 0.3rem 0.85rem;
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

    /* Sidebar Branding */
    .brand-box {
        padding: 0.8rem 0;
        margin-bottom: 0.5rem;
    }
    .brand-title {
        font-size: 1.35rem;
        font-weight: 800;
        color: #F8FAFC;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .brand-subtitle {
        font-size: 0.78rem;
        color: #94A3B8;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .live-status {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(16, 185, 129, 0.1);
        border: 1px solid rgba(16, 185, 129, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.72rem;
        color: #34D399;
        margin-top: 0.4rem;
    }
    .live-dot {
        width: 7px;
        height: 7px;
        background-color: #10B981;
        border-radius: 50%;
    }
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# -----------------------------------------------------------------------------
# 3. ASSET CACHING
# -----------------------------------------------------------------------------
@st.cache_data
def load_datasets():
    data_path = os.path.join(BASE_DIR, "data/processed/cricket_players_clean.csv")
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

def style_chart(fig, height=440):
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
# 4. SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div class="brand-box">
        <div class="brand-title">🏏 CricMetrics Pro</div>
        <div class="brand-subtitle">Franchise Analytics & War Room</div>
        <div class="live-status">
            <span class="live-dot"></span> 17 IPL Seasons (2008-2024)
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    menu = st.radio(
        "War Room Navigation",
        [
            "🏛️ War Room & Roster Intel",
            "📊 Telemetry & Factor Impact",
            "⚡ AI Rating & Auction Valuation",
            "🧩 Tactical Archetypes & 2D Map",
            "🚀 What-If Franchise Simulator",
            "🏆 Model Benchmarks & Defense"
        ],
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.caption("Franchise Intelligence Overview")
    st.markdown("""
    - **Total Players:** 619 Pro Cricketers
    - **Total Deliveries:** 260,920 Balls
    - **ML Rating Accuracy:** $R^2 = 0.9789$
    - **Tier Classification:** $95.16\%$ Accuracy
    - **Archetypes:** 5 Tactical Clusters
    """)
    st.caption("Enterprise Sports Analytics Suite")


# =============================================================================
# 1. WAR ROOM & ROSTER INTEL
# =============================================================================
if menu == "🏛️ War Room & Roster Intel":
    st.markdown("""
    <div class="hero-banner">
        <div class="org-pill">IPL Franchise Analytics & Auction War Room</div>
        <div class="hero-title">Cricket Player Performance & Scouting Intelligence</div>
        <div class="hero-subtitle">
            An enterprise decision-support platform analyzing <strong>260,920 deliveries across 1,095 IPL matches (2008–2024)</strong>. 
            Investigate measurable batting, bowling, and clutch factors to accurately predict player ratings ($R^2 = 0.9789$), 
            evaluate talent tiers, discover tactical archetypes, and optimize franchise auction valuations.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 4 Key Metrics Bar
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Qualified Cricketers", f"{len(df):,}", "17 IPL Seasons")
    with col2:
        st.metric("Continuous Rating R²", f"{metrics['regression']['Random Forest Regressor']['Test_R2']:.4f}", "Random Forest")
    with col3:
        st.metric("Tier Classification Acc", f"{metrics['classification']['K-Nearest Neighbors (KNN)']['Test_Accuracy']*100:.1f}%", "KNN & RF (95%+)")
    with col4:
        st.metric("Tactical Archetypes", "5 Distinct Roles", "K-Means (k=5)")
        
    st.markdown("---")
    
    # 3 Strategic Pillars
    st.subheader("💡 Franchise Decision-Making Pillars")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">⚡</div>
            <div class="card-title">1. Impact Over Volume</div>
            <div class="card-desc">
                Quantifies true T20 value: death overs strike rate, dot ball percentage, boundary frequency, and clutch match-winning awards rather than pure aggregate runs.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">💰</div>
            <div class="card-title">2. AI Auction Valuation</div>
            <div class="card-desc">
                Empirical market pricing engine estimating fair auction value (₹ Crores) to eliminate overpaying heuristics and detect undervalued marquee talent.
            </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="feature-card">
            <div class="card-icon">🧩</div>
            <div class="card-title">3. Unsupervised Archetypes</div>
            <div class="card-desc">
                Identifies 5 tactical playing styles (e.g. Death-Over Finisher, Mystery Spinner, Dual All-Rounder) to balance squad synergy and replacement depth.
            </div>
        </div>
        """, unsafe_allow_html=True)
        
    st.markdown("---")
    
    # Interactive Scouting Explorer
    st.subheader("🔍 Interactive Franchise Player Database")
    st.write("Filter, rank, and inspect players from the 17-season IPL database:")
    
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        sel_role = st.multiselect("Playing Role:", df["primary_role"].unique().tolist(), default=df["primary_role"].unique().tolist())
    with f2:
        sel_tier = st.multiselect("Talent Tier:", df["performance_tier"].unique().tolist(), default=df["performance_tier"].unique().tolist())
    with f3:
        min_matches = st.slider("Minimum Matches Played:", 3, 200, 15)
    with f4:
        search_query = st.text_input("Search Player by Name:", placeholder="e.g. Kohli, Bumrah, Dhoni")
        
    filtered = df[
        (df["primary_role"].isin(sel_role)) &
        (df["performance_tier"].isin(sel_tier)) &
        (df["matches_played"] >= min_matches)
    ]
    if search_query:
        filtered = filtered[filtered["player_name"].str.contains(search_query, case=False, na=False)]
        
    st.caption(f"Displaying {len(filtered)} players matching filters")
    
    display_cols = [
        "player_name", "primary_role", "matches_played", "total_runs", "batting_strike_rate", 
        "batting_average", "wickets_taken", "economy_rate", "death_overs_strike_rate", 
        "overall_performance_rating", "performance_tier", "estimated_auction_val_cr"
    ]
    
    st.dataframe(
        filtered[display_cols].sort_values("overall_performance_rating", ascending=False),
        column_config={
            "overall_performance_rating": st.column_config.ProgressColumn(
                "Performance Rating", format="%.1f", min_value=50, max_value=95
            ),
            "estimated_auction_val_cr": st.column_config.NumberColumn(
                "Est. Auction (₹ Cr)", format="₹%.1f Cr"
            ),
            "batting_strike_rate": st.column_config.NumberColumn("Bat SR", format="%.1f"),
            "economy_rate": st.column_config.NumberColumn("Econ Rate", format="%.2f")
        },
        use_container_width=True,
        height=380
    )


# =============================================================================
# 2. TELEMETRY & MEASURABLE FACTORS
# =============================================================================
elif menu == "📊 Telemetry & Factor Impact":
    st.markdown("""
    <div class="hero-banner">
        <div class="org-pill">Sports Science & Match Impact Telemetry</div>
        <div class="hero-title">Measurable Factor Analysis</div>
        <div class="hero-subtitle">
            What measurable traits truly drive winning performance in modern franchise cricket? 
            Investigate empirical correlations, multi-skill role signatures, and phase-specific impact.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    t1, t2, t3 = st.tabs([
        "🔥 Correlation with Overall Rating",
        "🎯 Role Multi-Skill Radars",
        "📈 Strike Rate vs Average Trade-off"
    ])
    
    with t1:
        st.subheader("Correlation of Tracked Metrics with Overall Performance Rating")
        st.write("Pearson correlation coefficients measuring association with player performance:")
        
        corr_cols = [
            'batting_impact_index', 'bowling_impact_index', 'clutch_match_winner_index',
            'matches_played', 'death_overs_strike_rate', 'boundary_run_pct', 
            'batting_strike_rate', 'total_runs', 'wickets_taken', 'dot_ball_bowled_pct'
        ]
        corr_vals = df[corr_cols].apply(lambda col: col.corr(df["overall_performance_rating"])).sort_values(ascending=True)
        
        fig_cor = px.bar(
            x=corr_vals.values,
            y=corr_vals.index,
            orientation="h",
            color=corr_vals.values,
            color_continuous_scale="Viridis",
            labels={"x": "Correlation with Overall Performance (r)", "y": "Measurable Metric"}
        )
        fig_cor.update_layout(coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig_cor, height=420), use_container_width=True)
        
        st.info("""
        **Executive Insights:**
        1. **Clutch Match-Winner Index ($r = 0.84$)** has the strongest singular correlation with overall rating, demonstrating that winning match-day awards and producing 50+ scores or 3+ wicket hauls dominates franchise value.
        2. **Death Overs Strike Rate ($r = 0.62$)** and **Boundary Run % ($r = 0.58$)** severely outshine nominal batting average in modern T20 decision-making.
        """)
        
    with t2:
        st.subheader("Multi-Attribute Playing Role Signatures")
        st.write("Hexagonal radar comparison of skills across primary playing roles:")
        
        radar_cols = [
            'batting_strike_rate', 'boundary_run_pct', 'death_overs_strike_rate', 
            'dot_ball_bowled_pct', 'batting_impact_index', 'bowling_impact_index'
        ]
        # Normalize radar metrics 0-100 for visual comparison
        role_avg = df.groupby("primary_role")[radar_cols].mean().reset_index()
        
        fig_rad = go.Figure()
        role_colors = {
            "Specialist Batter": "#38BDF8", 
            "All-Rounder": "#F59E0B", 
            "Specialist Bowler": "#10B981", 
            "Bowling Specialist": "#A855F7",
            "Squad Batter": "#94A3B8"
        }
        
        for _, row in role_avg.iterrows():
            role = row["primary_role"]
            fig_rad.add_trace(go.Scatterpolar(
                r=[row[m] for m in radar_cols] + [row[radar_cols[0]]],
                theta=radar_cols + [radar_cols[0]],
                fill='toself',
                name=role,
                line=dict(color=role_colors.get(role, "#FBBF24"))
            ))
            
        fig_rad.update_layout(
            polar=dict(radialaxis=dict(visible=True, color="#94A3B8"), bgcolor="rgba(0,0,0,0)"),
            showlegend=True
        )
        st.plotly_chart(style_chart(fig_rad, height=480), use_container_width=True)
        
    with t3:
        st.subheader("Batting Strike Rate vs Average Matrix")
        st.write("Categorizing batters into T20 quadrants (Elite Aggressors vs Anchors):")
        
        batters_only = df[df["total_runs"] >= 300]
        fig_scatter = px.scatter(
            batters_only,
            x="batting_strike_rate",
            y="batting_average",
            color="performance_tier",
            size="total_runs",
            hover_data=["player_name", "total_runs", "primary_role", "estimated_auction_val_cr"],
            color_discrete_map={
                "Elite / Marquee": "#F59E0B",
                "Core / Star": "#3B82F6",
                "Developing / Squad": "#10B981"
            },
            title="Batting Strike Rate vs Batting Average (Min 300 Runs)"
        )
        fig_scatter.add_vline(x=135, line_dash="dash", line_color="red", annotation_text="T20 Baseline SR (135)")
        fig_scatter.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="T20 Baseline Avg (30)")
        st.plotly_chart(style_chart(fig_scatter, height=480), use_container_width=True)


# =============================================================================
# 3. AI RATING & AUCTION VALUATION
# =============================================================================
elif menu == "⚡ AI Rating & Auction Valuation":
    st.markdown("""
    <div class="hero-banner">
        <div class="org-pill">AI Valuation Engine</div>
        <div class="hero-title">Player Rating & Auction Purse Estimator</div>
        <div class="hero-subtitle">
            Input player performance metrics or select an iconic player profile preset. 
            Our trained Machine Learning models estimate the player's <strong>Overall Performance Rating (50-95)</strong>, 
            <strong>Talent Tier</strong>, and <strong>Fair Market Auction Valuation (₹ Crores)</strong>.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Quick Presets
    st.markdown("#### ⚡ Quick Presets (Click to Load Iconic Profiles)")
    presets = {
        "👑 Virat Kohli (Elite Master Chaser & Anchor)": {
            "runs": 8000, "sr": 131.0, "avg": 38.0, "bound": 58.0, "death_sr": 180.0,
            "wkts": 4, "econ": 8.8, "dot_bowl": 30.0, "death_econ": 11.5, "matches": 250, "mom": 17, "role": "Specialist Batter"
        },
        "⚡ Heinrich Klaasen / Andre Russell (Lethal Death Finisher)": {
            "runs": 2800, "sr": 168.0, "avg": 32.0, "bound": 74.0, "death_sr": 215.0,
            "wkts": 65, "econ": 9.1, "dot_bowl": 35.0, "death_econ": 10.2, "matches": 110, "mom": 12, "role": "All-Rounder"
        },
        "🎯 Jasprit Bumrah (Elite Death-Over Pacer)": {
            "runs": 75, "sr": 95.0, "avg": 10.0, "bound": 25.0, "death_sr": 110.0,
            "wkts": 168, "econ": 7.3, "dot_bowl": 48.0, "death_econ": 7.8, "matches": 135, "mom": 11, "role": "Specialist Bowler"
        },
        "🪄 Sunil Narine / Rashid Khan (Mystery Spin All-Rounder)": {
            "runs": 1500, "sr": 155.0, "avg": 18.0, "bound": 68.0, "death_sr": 175.0,
            "wkts": 180, "econ": 6.7, "dot_bowl": 45.0, "death_econ": 8.4, "matches": 175, "mom": 15, "role": "All-Rounder"
        },
        "🦁 Hardik Pandya (Pace Bowling All-Rounder)": {
            "runs": 2500, "sr": 146.0, "avg": 30.0, "bound": 62.0, "death_sr": 185.0,
            "wkts": 60, "econ": 8.9, "dot_bowl": 36.0, "death_econ": 10.5, "matches": 130, "mom": 9, "role": "All-Rounder"
        }
    }
    
    preset_choice = st.selectbox("Choose a pre-configured template (or customize below):", list(presets.keys()))
    default_vals = presets[preset_choice]
    
    col_input, col_card = st.columns([3, 2])
    
    with col_input:
        st.markdown("#### Performance Metric Inputs")
        p_role = st.selectbox("Playing Role", ["Specialist Batter", "All-Rounder", "Specialist Bowler", "Bowling Specialist", "Squad Batter"],
                              index=["Specialist Batter", "All-Rounder", "Specialist Bowler", "Bowling Specialist", "Squad Batter"].index(default_vals["role"]))
        
        t_bat, t_bowl, t_match = st.tabs(["Batting Telemetry", "Bowling Telemetry", "Experience & Clutch"])
        
        with t_bat:
            val_runs = st.slider("Total Career Runs", 0, 8500, default_vals["runs"])
            val_sr = st.slider("Batting Strike Rate", 80.0, 210.0, default_vals["sr"])
            val_avg = st.slider("Batting Average", 5.0, 50.0, default_vals["avg"])
            val_bound = st.slider("Boundary Run %", 10.0, 85.0, default_vals["bound"])
            val_death_sr = st.slider("Death Overs Strike Rate (Overs 16-20)", 90.0, 250.0, default_vals["death_sr"])
            
        with t_bowl:
            val_wkts = st.slider("Wickets Taken", 0, 220, default_vals["wkts"])
            val_econ = st.slider("Economy Rate", 5.5, 12.5, default_vals["econ"])
            val_dot_bowl = st.slider("Dot Ball Bowled %", 15.0, 55.0, default_vals["dot_bowl"])
            val_death_econ = st.slider("Death Overs Economy (Overs 16-20)", 6.0, 14.0, default_vals["death_econ"])
            
        with t_match:
            val_matches = st.slider("Matches Played", 5, 260, default_vals["matches"])
            val_mom = st.slider("Player of the Match Awards", 0, 25, default_vals["mom"])
            
        chosen_reg_name = st.selectbox(
            "Evaluation Model Engine:",
            ["Random Forest Regressor (Recommended - 97.9% R²)", "Ridge Regression (L2)", "Linear Regression (OLS)", "MLP Neural Network"]
        )

    # Compute composite indices matching pipeline
    bat_score = (
        min(val_avg / 45.0, 1.5) * 35.0 +
        min(val_sr / 160.0, 1.5) * 35.0 +
        min(val_bound / 75.0, 1.5) * 15.0 +
        min(val_death_sr / 200.0, 1.5) * 15.0
    )
    econ_factor = max((11.0 - val_econ) / 4.0, 0.0) * 40.0
    bowl_sr_val = (val_matches * 24 / max(val_wkts, 1)) if val_wkts > 0 else 50.0
    bowl_sr_factor = max((35.0 - bowl_sr_val) / 18.0, 0.0) * 35.0
    dots_factor = min(val_dot_bowl / 50.0, 1.5) * 25.0
    bowl_score = (econ_factor + bowl_sr_factor + dots_factor) if val_wkts >= 5 else 15.0
    clutch_idx = min(val_mom * 4.0, 40) + min((val_runs // 250) * 2.5, 30) + min((val_wkts // 15) * 3.0, 30)

    # Build input row
    input_dict = {f: 0.0 for f in metadata["encoded_feature_names"]}
    input_dict["matches_played"] = val_matches
    input_dict["total_runs"] = val_runs
    input_dict["balls_faced"] = (val_runs / max(val_sr, 1)) * 100
    input_dict["batting_average"] = val_avg
    input_dict["batting_strike_rate"] = val_sr
    input_dict["fours"] = int(val_runs * 0.10)
    input_dict["sixes"] = int(val_runs * 0.04)
    input_dict["boundary_run_pct"] = val_bound
    input_dict["dot_ball_faced_pct"] = 40.0
    input_dict["highest_score"] = min(int(val_runs * 0.15) + 30, 125)
    input_dict["thirties"] = int(val_runs / 180)
    input_dict["fifties"] = int(val_runs / 350)
    input_dict["death_overs_strike_rate"] = val_death_sr
    input_dict["overs_bowled"] = val_matches * 3.5 if val_wkts >= 15 else 5.0
    input_dict["wickets_taken"] = val_wkts
    input_dict["economy_rate"] = val_econ
    input_dict["bowling_strike_rate"] = bowl_sr_val
    input_dict["bowling_average"] = (val_wkts * 28.0) if val_wkts > 0 else 60.0
    input_dict["dot_ball_bowled_pct"] = val_dot_bowl
    input_dict["three_plus_wickets"] = int(val_wkts / 12)
    input_dict["death_overs_economy"] = val_death_econ
    input_dict["player_of_match_awards"] = val_mom
    input_dict["batting_impact_index"] = bat_score
    input_dict["bowling_impact_index"] = bowl_score
    input_dict["clutch_match_winner_index"] = clutch_idx

    role_col = f"primary_role_{p_role}"
    if role_col in input_dict:
        input_dict[role_col] = 1.0

    input_df = pd.DataFrame([input_dict])
    input_scaled = pd.DataFrame(models["scaler"].transform(input_df), columns=input_df.columns)

    if "Linear" in chosen_reg_name:
        pred_rating = models["linear_reg"].predict(input_scaled)[0]
    elif "Ridge" in chosen_reg_name:
        pred_rating = models["ridge_reg"].predict(input_scaled)[0]
    elif "Neural" in chosen_reg_name:
        pred_rating = models["mlp_reg"].predict(input_scaled)[0]
    else:
        pred_rating = models["rf_reg"].predict(input_scaled)[0]

    pred_rating = float(np.clip(pred_rating, 50.0, 95.0))
    tier_label = "Elite / Marquee" if pred_rating >= 80.0 else "Core / Star" if pred_rating >= 68.0 else "Developing / Squad"
    tier_class = "tier-elite" if pred_rating >= 80.0 else "tier-star" if pred_rating >= 68.0 else "tier-dev"
    
    # Auction purse estimate
    auc_base = np.exp((pred_rating - 58.0) * 0.09) * 1.5
    if p_role == "All-Rounder":
        auc_base *= 1.25
    if val_mom >= 10:
        auc_base *= 1.15
    pred_auction = round(float(np.clip(auc_base, 0.5, 24.5)), 1)

    with col_card:
        st.markdown(f"""
        <div class="auction-card">
            <div style="font-size: 0.9rem; font-weight: 700; color: #94A3B8; text-transform: uppercase;">{p_role}</div>
            <div class="auction-rating">{pred_rating:.1f}</div>
            <div style="margin-bottom: 0.6rem;">
                <span class="tier-badge {tier_class}">{tier_label}</span>
            </div>
            <div style="font-size: 0.9rem; color: #94A3B8; margin-top: 0.5rem;">Estimated IPL Auction Purse Valuation</div>
            <div class="auction-price">₹{pred_auction} Cr</div>
            <hr style="border-color: rgba(255,255,255,0.1); margin: 0.8rem 0;">
            <div style="font-size: 0.82rem; color: #94A3B8;">Engine: {chosen_reg_name}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Mini Radar
        radar_m = ["Batting SR", "Boundary %", "Death SR", "Wkts", "Dot Bowl %", "Clutch"]
        radar_vals = [
            min(val_sr / 180.0 * 100, 100),
            min(val_bound / 75.0 * 100, 100),
            min(val_death_sr / 210.0 * 100, 100),
            min(val_wkts / 150.0 * 100, 100),
            min(val_dot_bowl / 45.0 * 100, 100),
            min(clutch_idx / 80.0 * 100, 100)
        ]
        fig_mini = go.Figure(go.Scatterpolar(
            r=radar_vals + [radar_vals[0]],
            theta=radar_m + [radar_m[0]],
            fill='toself',
            line=dict(color="#FBBF24")
        ))
        fig_mini.update_layout(
            polar=dict(radialaxis=dict(visible=False, range=[0, 100]), bgcolor="rgba(0,0,0,0)"),
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            showlegend=False
        )
        st.plotly_chart(style_chart(fig_mini, height=200), use_container_width=True)


# =============================================================================
# 4. TACTICAL ARCHETYPES & 2D MAP
# =============================================================================
elif menu == "🧩 Tactical Archetypes & 2D Map":
    st.markdown("""
    <div class="hero-banner">
        <div class="org-pill">Unsupervised Clustering & Archetype Discovery</div>
        <div class="hero-title">Tactical Playing Styles & 2D Map</div>
        <div class="hero-subtitle">
            Beyond traditional playing roles, athletes exhibit distinct tactical fingerprints. 
            Using <strong>K-Means Clustering ($k=5$)</strong> and <strong>Principal Component Analysis (PCA)</strong>, 
            we map all 619 cricketers into their natural playing style clusters.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # 5 Archetype Cards
    arch_cols = st.columns(5)
    archetypes_info = [
        {"icon": "👑", "name": "Top-Order Anchor", "style": "High average, controlled powerplay pacing, deep innings construction (e.g. Kohli, Warner, Dhawan)."},
        {"icon": "🎯", "name": "Pace Spearhead", "style": "Yorker accuracy, low death overs economy, early wicket breakthroughs (e.g. Bumrah, Malinga, B Kumar)."},
        {"icon": "⚡", "name": "Death Finisher", "style": "Lethal death strike rate (>180), high boundary %, explosive strike power (e.g. Russell, Klaasen, Dhoni)."},
        {"icon": "🪄", "name": "Mystery Spin Wizard", "style": "High dot ball %, miserly economy in middle overs, deceived LBWs & bowled (e.g. Narine, Chahal, Rashid)."},
        {"icon": "🦁", "name": "Dual All-Rounder", "style": "Elite two-way contribution, late-order hitting + reliable 4-over quota (e.g. Jadeja, Hardik, Bravo)."}
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
    df_pca["PC1 (T20 Match Impact & Volume)"] = pca_coords[:, 0]
    df_pca["PC2 (Batting vs Bowling Bias)"] = pca_coords[:, 1]
    
    cluster_cols = models["kmeans"]["features"]
    cluster_indices = [feature_cols.index(c) for c in cluster_cols]
    df_pca["Tactical Archetype"] = [
        metrics["unsupervised"]["archetype_names"][str(c)] 
        for c in models["kmeans"]["model"].predict(X_scaled_all[:, cluster_indices])
    ]

    p_col1, p_col2 = st.columns([3, 1])
    with p_col2:
        st.markdown("#### Map Filters")
        color_choice = st.radio("Color Players By:", ["Tactical Archetype", "primary_role", "performance_tier"])
        st.caption("PC1 isolates overall T20 performance and career experience. PC2 separates specialist pacers/spinners from top-order batters.")
        
    with p_col1:
        fig_pca = px.scatter(
            df_pca,
            x="PC1 (T20 Match Impact & Volume)",
            y="PC2 (Batting vs Bowling Bias)",
            color=color_choice,
            hover_data=["player_name", "primary_role", "total_runs", "wickets_taken", "overall_performance_rating", "estimated_auction_val_cr"],
            opacity=0.8,
            color_discrete_sequence=px.colors.qualitative.Bold,
            title="2D Latent Tactical Map of 619 IPL Players (PCA Projection)"
        )
        st.plotly_chart(style_chart(fig_pca, height=520), use_container_width=True)


# =============================================================================
# 5. WHAT-IF FRANCHISE SIMULATOR
# =============================================================================
elif menu == "🚀 What-If Franchise Simulator":
    st.markdown("""
    <div class="hero-banner">
        <div class="org-pill">Franchise Development & Auction Simulator</div>
        <div class="hero-title">What-If Player Development & Valuation Simulator</div>
        <div class="hero-subtitle">
            Select any real cricketer from the database to simulate targeted training interventions. 
            Analyze how optimizing death-overs boundary hitting or economy rate boosts the player's 
            overall rating and enhances franchise auction market value.
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    selected_player = st.selectbox(
        "Choose an existing player from the database:",
        df["player_name"].tolist()
    )
    p_data = df[df["player_name"] == selected_player].iloc[0]
    
    c_current, c_train, c_projected = st.columns([1, 1, 1])
    
    with c_current:
        st.markdown(f"""
        <div class="feature-card">
            <div class="card-icon">👤</div>
            <div class="card-title">{p_data['player_name']}</div>
            <p style="color: #94A3B8; font-size: 0.85rem;">Role: {p_data['primary_role']}</p>
            <p><strong>Matches:</strong> {p_data['matches_played']} | <strong>Runs:</strong> {p_data['total_runs']} | <strong>Wickets:</strong> {p_data['wickets_taken']}</p>
            <div style="font-size: 2.8rem; font-weight: 800; color: #F59E0B;">{p_data['overall_performance_rating']}</div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Current Performance Rating</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #34D399; margin-top: 0.5rem;">₹{p_data['estimated_auction_val_cr']} Cr</div>
            <div style="font-size: 0.8rem; color: #94A3B8;">Current Auction Valuation</div>
        </div>
        """, unsafe_allow_html=True)
        
    with c_train:
        st.markdown("#### Prescribe Targeted Training Focus")
        d_sr = st.slider("Death Overs Strike Rate Boost (Δ)", -10.0, 35.0, 15.0)
        d_bound = st.slider("Boundary Hitting Frequency (Δ %)", -5.0, 15.0, 6.0)
        d_econ = st.slider("Economy Rate Improvement (Δ RPO reduction)", -2.0, 2.0, 0.8)
        d_dots = st.slider("Dot Ball Bowling Mastery (Δ %)", -5.0, 15.0, 8.0)
        
    # Projected Rating Calculation
    gain = (d_sr * 0.08) + (d_bound * 0.12) + (d_econ * 1.5) + (d_dots * 0.10)
    new_rating = round(float(np.clip(p_data['overall_performance_rating'] + gain, 50.0, 95.0)), 1)
    
    # Valuation delta
    base_new_val = np.exp((new_rating - 58.0) * 0.09) * 1.5
    if p_data['primary_role'] == 'All-Rounder':
        base_new_val *= 1.25
    if p_data['player_of_match_awards'] >= 10:
        base_new_val *= 1.15
    new_val = round(float(np.clip(base_new_val, 0.5, 24.5)), 1)
    val_delta = round(new_val - p_data['estimated_auction_val_cr'], 1)
    
    with c_projected:
        st.markdown(f"""
        <div class="feature-card" style="border: 1px solid #10B981; background: rgba(16, 185, 129, 0.05);">
            <div class="card-icon">🚀</div>
            <div class="card-title">Projected Outcome</div>
            <p style="color: #34D399; font-size: 0.85rem;">Post-Camp Development Outlook</p>
            <div style="font-size: 2.8rem; font-weight: 800; color: #34D399;">
                {new_rating} <span style="font-size: 1.1rem; color: #10B981;">(+{round(new_rating - p_data['overall_performance_rating'], 1)})</span>
            </div>
            <div style="font-size: 0.85rem; color: #94A3B8;">Projected Overall Rating</div>
            <div style="font-size: 1.5rem; font-weight: 700; color: #34D399; margin-top: 0.5rem;">
                ₹{new_val} Cr <span style="font-size: 0.95rem;">({'+' if val_delta>=0 else ''}{val_delta} Cr)</span>
            </div>
            <div style="font-size: 0.8rem; color: #94A3B8;">Projected Auction Valuation</div>
        </div>
        """, unsafe_allow_html=True)


# =============================================================================
# 6. MODEL BENCHMARKS & DEFENSE
# =============================================================================
elif menu == "🏆 Model Benchmarks & Defense":
    st.markdown("""
    <div class="hero-banner">
        <div class="org-pill">Empirical Model Governance</div>
        <div class="hero-title">Model Evaluation Benchmarks & Technical Defense</div>
        <div class="hero-subtitle">
            Comprehensive evaluation across 10 Machine Learning algorithms. 
            Validated with 5-Fold Cross Validation, $R^2$, RMSE, MAE, Confusion Matrices, and Gini Feature Importance rankings.
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
        st.markdown("#### Regression Factor Importance (Random Forest)")
        imp_reg = pd.DataFrame(list(metrics["feature_importances"]["regression_top10"].items()), columns=["Factor", "Weight"])
        fig_b1 = px.bar(imp_reg, x="Weight", y="Factor", orientation="h", color="Weight", color_continuous_scale="Viridis")
        fig_b1.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig_b1, height=360), use_container_width=True)
    with b2:
        st.markdown("#### Classification Factor Importance (Random Forest)")
        imp_clf = pd.DataFrame(list(metrics["feature_importances"]["classification_top10"].items()), columns=["Factor", "Weight"])
        fig_b2 = px.bar(imp_clf, x="Weight", y="Factor", orientation="h", color="Weight", color_continuous_scale="Cividis")
        fig_b2.update_layout(yaxis=dict(autorange="reversed"), coloraxis_showscale=False)
        st.plotly_chart(style_chart(fig_b2, height=360), use_container_width=True)
        
    st.markdown("---")
    st.subheader("4. Technical Defense & Viva Voce Q&A for Organizations")
    st.markdown("""
    - **Q1: Why did Random Forest Regressor achieve an exceptional $R^2 = 0.9789$ and RMSE of $1.25$?**  
      *A:* Cricket performance has complex non-linear thresholds: a strike rate of 150+ in death overs is exponentially more valuable than a strike rate of 120 in powerplays; similarly, bowling economy below 7.5 in death overs has an outsized win contribution. Random Forest effortlessly partitions these multi-dimensional boundary conditions.
    - **Q2: Why is the Clutch Match-Winner Index so influential in player valuation?**  
      *A:* Franchise cricket is outcome-driven. Players who routinely deliver in clutch situations (earning Player of the Match awards and scoring match-winning fifties) have demonstrated psychological resilience under extreme franchise pressure.
    - **Q3: How does this system provide real economic value to an IPL franchise?**  
      *A:* During IPL mega-auctions with ₹100+ Crore purse limits, emotional bidding wars often result in overpaying for brand names. CricMetrics Pro gives franchise directors quantitative valuation anchors to recruit high-performing, undervalued tactical archetypes ("Moneyball for Cricket").
    """)
