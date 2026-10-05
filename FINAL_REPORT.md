# Case Study: Player Performance Analysis
# Final Academic & Technical Project Report

**Project Title:**  
## CricMetrics Pro: AI-Powered Multi-Task Cricket Performance Analytics, Talent Tier Classification & Fair Market Valuation

**System Domain:** Professional Sports Analytics & Machine Learning Engineering  
**Application Focus:** Quantitative Player Evaluation, T20 Match Impact, Regression Valuation, Talent Classification & Archetype Clustering  
**Dataset Foundation:** Official 17-Season Real IPL Ball-by-Ball Telemetry (2008–2024, 260,920 Deliveries, 1,095 Matches, 619 Qualified Players)  
**Technology Stack:** Python 3.9+, Scikit-Learn, Pandas, NumPy, Plotly, Streamlit  

---

## Deliverables Mapping & Table of Contents

1. [Deliverable 1: Problem Definition & Sponsoring Stakeholder Motivation](#1-deliverable-1-problem-definition--sponsoring-stakeholder-motivation)
2. [Deliverable 2: Dataset Documentation & Data Quality Observations](#2-deliverable-2-dataset-documentation--data-quality-observations)
3. [Deliverable 3: Exploratory Data Analysis (EDA) & Measurable Factors](#3-deliverable-3-exploratory-data-analysis-eda--measurable-factors)
4. [Deliverable 4: Preprocessing & Mathematical Feature Preparation](#4-deliverable-4-preprocessing--mathematical-feature-preparation)
5. [Deliverable 5: Model Development (Regression, Classification, Clustering & PCA)](#5-deliverable-5-model-development-regression-classification-clustering--pca)
6. [Deliverable 6: Evaluation Measures, Benchmark Results & Error Analysis](#6-deliverable-6-evaluation-measures-benchmark-results--error-analysis)
7. [Deliverable 7: Functional Streamlit Web Application Architecture](#7-deliverable-7-functional-streamlit-web-application-architecture)
8. [Deliverable 8: Viva Voce Examination Questions & Model Answers](#8-deliverable-8-viva-voce-examination-questions--model-answers)

---

## 1. Deliverable 1: Problem Definition & Sponsoring Stakeholder Motivation

### 1.1 Problem Statement
> *"A sports organization wants to investigate measurable factors associated with player performance. (With Proper Justification)"*

In modern franchise cricket (specifically the Indian Premier League, with ₹100–120 Crore auction budgets), selecting and pricing players is one of the highest-stakes financial decisions in global sports. Traditionally, franchises relied on:
- **Recency Bias:** Overvaluing a player who had one great week in a domestic tournament.
- **Volume Over Impact:** Valuing pure career aggregate runs or wickets rather than situational efficiency (death-overs strike rate, boundary frequency, economy rate).
- **Auction Bidding Wars:** Emotional overpaying resulting in severe salary cap depletion.

### 1.2 Multi-Task Machine Learning Solution
CricMetrics Pro addresses this by establishing an objective, empirical multi-task system:
1. **Supervised Regression:** Predicts continuous **Overall Performance Rating (50.0–95.0)** and estimates **Fair Market Auction Purse (₹ Crores)** to benchmark value before auction day.
2. **Supervised Classification:** Predicts discrete **Talent Tiers** (Elite / Marquee, Core / Star, Developing / Squad) with calibrated multi-class probability scores.
3. **Unsupervised Clustering (K-Means, $k=5$):** Groups cricketers into tactical playing archetypes without subjective labeling.
4. **Dimensionality Reduction (PCA):** Projects 26 dimensional metrics down to 2 components for visual cluster inspection.

---

## 2. Deliverable 2: Dataset Documentation & Data Quality Observations

### 2.1 Dataset Provenance & Overview
The dataset is constructed from **260,920 legal ball-by-ball delivery event logs across 1,095 IPL matches spanning 17 seasons (2008–2024)**, covering **619 qualified professional cricketers**:

| Metric | Specification |
| :--- | :--- |
| **Total Deliveries Analyzed** | 260,920 ball-by-ball delivery records |
| **Total Matches Analyzed** | 1,095 IPL matches across 17 tournaments |
| **Total Qualified Players ($N$)** | 619 professional cricketers ($\ge 3$ matches) |
| **Engineered Features** | 26 quantitative factors + categorical role |
| **Primary Target Variables** | `overall_performance_rating` (Continuous) & `performance_tier_code` (0, 1, 2) |

### 2.2 Data Quality Observations
1. **Division-by-Zero Imputation:** Pure specialist bowlers who never batted have undefined batting averages. Imputed with 0.0 with role indicator preservation. Pure specialist batters who never bowled have undefined economy rates. Imputed with league-worst penalty (12.0) to prevent false perfection.
2. **Sample Size Disparity:** Uncapped domestic players with $< 3$ appearances were filtered out to eliminate non-representative noise.
3. **Outlier Boundaries:** Handled extreme strike rates for tailenders who faced 1 ball and hit a six ($SR = 600.0$) by capping at domain realistic limits ($300.0$).

---

## 3. Deliverable 3: Exploratory Data Analysis (EDA) & Measurable Factors

### 3.1 Strongest Measurable Drivers of Performance
Pearson correlation analysis ($r$) between measured factors and overall rating:
- **Clutch Match-Winner Index ($r = 0.84$):** Demonstrates that frequency of Player of the Match awards and 50+ scores / 3+ wicket hauls dominates match impact.
- **Death Overs Strike Rate ($r = 0.62$):** Hitting boundaries in overs 16–20 correlates far higher with team wins than early-innings stat accumulation.
- **Boundary Run % ($r = 0.58$):** Percentage of runs scored exclusively in 4s and 6s heavily outstrips nominal batting average ($r = 0.44$).
- **Dot Ball Bowled % ($r = 0.52$):** Building dot-ball pressure is the primary driver of bowling economy and wicket creation.

### 3.2 T20 Strategic Quadrants
Batting Strike Rate vs Batting Average analysis shows 4 distinct player clusters:
- **High SR + High Avg:** Elite Marquee Aggressors (Kohli, Klaasen, Russell).
- **High SR + Moderate Avg:** Death-Over Finishers & Pinch Hitters.
- **Moderate SR + High Avg:** Top-Order Anchors & Accumulators.
- **Low SR + Low Avg:** Squad Rotation & Tailenders.

---

## 4. Deliverable 4: Preprocessing & Mathematical Feature Preparation

1. **Categorical Encoding:** One-Hot Encoding (`pd.get_dummies`) with `drop_first=True` applied to `primary_role` to prevent multicollinearity in linear models.
2. **Feature Scaling (StandardScaler):** Scaled all 26 numerical variables using $z = \frac{x - \mu}{\sigma}$ to ensure equal distance weighting in KNN, Ridge, PCA, and K-Means.
3. **Stratified Partitioning:** 80% Training ($n = 495$) / 20% Testing ($n = 124$) using **Stratified Split** on `performance_tier` to maintain exact class proportions.

---

## 5. Deliverable 5: Model Development (Regression, Classification, Clustering & PCA)

### 5.1 Supervised Regression Models (Rating & Fair Valuation)
1. **Multiple Linear Regression (Baseline):** Fits an ordinary least squares hyper-plane through the scaled 26-dimensional feature space.
2. **Polynomial Regression (Degree 2 with Ridge Regularization):** Captures non-linear feature interactions (e.g., strike rate $\times$ boundary frequency) with $L_2$ regularization ($\alpha = 10.0$) to prevent overfitting.
3. **Random Forest Regressor:** Ensemble of 150 bagged decision trees with maximum depth of 10, averaging tree predictions.

### 5.2 Supervised Classification Models (Talent Tiers)
1. **K-Nearest Neighbors (KNN, $k=7$):** Distance-weighted Euclidean voting among the 7 most similar historical cricketer profiles.
2. **Random Forest Classifier:** 150 bootstrap-aggregated decision trees with Gini impurity splitting.
3. **Multinomial Logistic Regression:** Softmax cross-entropy boundary separating the 3 tiers.
4. **Decision Tree Classifier (CART):** Interpretable white-box hierarchical decision tree (max depth 6, min samples split 10).

### 5.3 Unsupervised Tactical Archetypes (K-Means Clustering, $k=5$)
Using 10 domain traits, K-Means discovers 5 tactical playing styles:
- **Cluster 0:** Top-Order Anchor & Accumulator
- **Cluster 1:** Powerplay & Middle-Overs Pace Specialist
- **Cluster 2:** Death-Overs Finisher & Power Hitter
- **Cluster 3:** Defensive Middle-Overs Economy Spinner
- **Cluster 4:** Elite Dual-Threat All-Rounder

### 5.4 Dimensionality Reduction (PCA, 2D)
Principal Component Analysis projects the 26 scaled features into a 2D plane:
- **PC 1 ($31.2\%$ variance):** Career Volume & All-Round Dual Impact.
- **PC 2 ($14.5\%$ variance):** Batting Strike Rate vs Bowling Economy Specialization.
- **Total Variance Explained:** $45.7\%$.

---

## 6. Deliverable 6: Evaluation Measures, Benchmark Results & Error Analysis

### 6.1 Supervised Regression Benchmark Table

| Model Algorithm | Test $R^2$ Score | 5-Fold CV $R^2$ | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) |
| :--- | :---: | :---: | :---: | :---: |
| **Polynomial Regression (Degree 2)** | **0.9941** | **0.9956** | **0.5147** | **0.6648** |
| **Random Forest Regressor** | **0.9780** | **0.9522** | **0.9412** | **1.2837** |
| **Multiple Linear Regression** | 0.9490 | 0.8797 | 1.4820 | 1.9534 |

### 6.2 Supervised Classification Benchmark Table

| Classifier Algorithm | Test Accuracy | 5-Fold Stratified CV | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN, k=7)** | **95.16%** | 90.30% | **0.9521** | **0.9504** | **0.9501** |
| **Random Forest Classifier** | **94.35%** | **94.75%** | **0.9468** | **0.9452** | **0.9459** |
| **Multinomial Logistic Regression** | 87.90% | 92.53% | 0.8872 | 0.8841 | 0.8853 |
| **Decision Tree Classifier (CART)** | 87.90% | 89.90% | 0.8835 | 0.8802 | 0.8812 |

### 6.3 Real-World Limitations & Error Analysis
1. **Uncapped Player Volatility:** Players with $< 5$ matches have high sample variance.
2. **Injury & Biomechanics:** On-field telemetry cannot measure physical stamina or recurrence of bowling injuries.
3. **Pitch & Ground Geometry:** A strike rate of 140 at a slow spin venue (Chepauk) represents higher true value than 150 at a high-altitude ground with short boundaries (Chinnaswamy).

---

## 7. Deliverable 7: Functional Streamlit Web Application Architecture

The application is structured into 6 tabs that directly map to the project deliverables:
1. **🏛️ Deliverable 1: Problem Definition** (Formulated title, problem definition, business justification, KPI cards)
2. **📋 Deliverable 2: Dataset & Preprocessing** (Interactive database, data dictionary, quality steps)
3. **📊 Deliverable 3: Exploratory Analysis (EDA)** (Factor correlations, multi-skill role radars, quadrant matrices)
4. **⚡ Deliverable 4: AI Valuation Engine** (1-click presets for iconic players, live regression & classification predictions)
5. **🧩 Deliverable 5: Tactical Archetypes & PCA** (2D latent space scatter plot with cluster inspection)
6. **🏆 Deliverable 6: Benchmarks & Evaluation** (Side-by-side leaderboards, confusion matrices, limitations)

---

## 8. Deliverable 8: Viva Voce Examination Questions & Model Answers

### Q1: Why did you formulate both a Regression task and a Classification task?
**Answer:** In professional sports management, decision-makers need two different outputs. When negotiating salary and auction purse limits, they need a continuous number (e.g. Fair Market Value of ₹14.5 Crore or Rating of 88.2)—which is Regression. When planning squad hierarchy and foreign player quotas, they need discrete operational categories (Elite vs Core vs Squad)—which is Classification.

### Q2: Why is Polynomial Regression performing so well ($R^2 = 0.9941$)?
**Answer:** Cricket performance is inherently multiplicative. A batter with both high strike rate AND high boundary percentage produces an exponential impact on winning probability, not merely a linear sum. Degree 2 polynomial interaction terms capture these metric synergies naturally.

### Q3: Why does K-Nearest Neighbors achieve the highest raw test accuracy ($95.16\%$)?
**Answer:** Cricket scouting operates on peer comparison. When assessing an uncapped all-rounder, scouts compare them to historical prototypes (e.g., "Does this player play like young Hardik Pandya or Shane Watson?"). KNN mathematically computes Euclidean distance to those exact historical player profiles.

### Q4: Why is Random Forest preferred for production deployment despite KNN's slightly higher test accuracy?
**Answer:** Random Forest achieved higher 5-Fold Cross-Validation accuracy ($94.75\%$ vs $90.30\%$) and zero inference latency. KNN requires storing and scanning the entire training dataset at query time ($O(N \cdot D)$), whereas Random Forest evaluates pre-computed decision trees ($O(\text{trees} \cdot \text{depth})$) in milliseconds.

### Q5: How was data leakage prevented during preprocessing?
**Answer:** We performed the 80/20 train/test split **before** fitting the `StandardScaler`. The scaler was fitted exclusively on `X_train`, and then used to transform `X_test` and all incoming live simulator inputs.
