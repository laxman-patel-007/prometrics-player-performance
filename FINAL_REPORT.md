# Case Study no. 102: Player Performance Analysis
# Final Academic & Organizational Project Report

**Project Title:**  
## CricMetrics Pro: Multi-Dimensional Cricket Player Performance Analysis, Tactical Archetype Discovery, and Auction Valuation in Modern T20 Cricket

**System Domain:** Professional Sports Analytics & Machine Learning Engineering  
**Application Focus:** Quantitative Player Evaluation, T20 Match Impact, Talent Tier Classification & Franchise Auction Valuation  
**Dataset Foundation:** Official 17-Season Real IPL Ball-by-Ball Telemetry (2008–2024, 260,920 Deliveries, 1,095 Matches, 619 Qualified Players)  
**Technology Stack:** Python 3.9+, Scikit-Learn, Pandas, NumPy, Plotly, Streamlit  

---

## Table of Contents
1. [Problem Definition & Real-World Organizational Motivation](#1-problem-definition--real-world-organizational-motivation)
2. [Dataset Provenance, Feature Dictionary & Data Quality](#2-dataset-provenance-feature-dictionary--data-quality)
3. [Exploratory Data Analysis (EDA) & Domain Observations](#3-exploratory-data-analysis-eda--domain-observations)
4. [Data Preprocessing & Feature Engineering](#4-data-preprocessing--feature-engineering)
5. [Supervised Learning: Talent Tier Classification Models](#5-supervised-learning-talent-tier-classification-models)
6. [Unsupervised Learning: Tactical Archetype Clustering](#6-unsupervised-learning-tactical-archetype-clustering)
7. [Dimensionality Reduction & PCA Analysis](#7-dimensionality-reduction--pca-analysis)
8. [What-If Development & Auction Valuation Simulator](#8-what-if-development--auction-valuation-simulator)
9. [Viva Voce Examination Guide: Model Answers](#9-viva-voce-examination-guide-model-answers)

---

## 1. Problem Definition & Real-World Organizational Motivation

### 1.1 Organizational Problem Statement
> *"A sports organization wants to investigate measurable factors associated with player performance. (With Proper Justification)"*

In modern professional sports organizations—specifically Indian Premier League (IPL) franchises and national cricket boards (BCCI, Cricket Australia, ECB)—hundreds of crores of rupees are committed during mega-auctions with strict purse caps (e.g., ₹100–120 Crore squad budgets). Historically, cricket player acquisition was fraught with subjective heuristics, superstar brand bias, and over-indexing on nominal career run or wicket aggregates rather than contextual match impact.

A modern cricket organization requires an objective, measurable, and machine-learning-driven framework to:
1. **Dissect and Quantify Performance Drivers:** Identify which measurable batting (Strike Rate, Boundary %, Death Overs SR), bowling (Economy Rate, Dot Ball %, Death Overs Economy), and clutch factors truly dictate match wins.
2. **Classify Strategic Talent Tiers (Classification):** Categorize players into actionable cohorts:
   - **Tier 0:** *Developing / Squad Rotation* (Rating $< 68.0$)
   - **Tier 1:** *Core / Franchise Star* ($68.0 \le \text{Rating} < 80.0$)
   - **Tier 2:** *Elite / Marquee Pillar* ($\text{Rating} \ge 80.0$)
3. **Discover Tactical Archetypes (Unsupervised Clustering):** Group players by their true playing fingerprint (e.g., separating Death-Over Finishers from Top-Order Anchors, and Mystery Spinners from Pace Spearheads).
4. **Simulate Player Development & Auction Valuation (What-If Analysis):** Provide franchise directors and coaching staff with an empirical tool to project how targeted improvements in death-overs hitting or economy rate boost ratings and fair market auction valuations (₹ Crores).

---

## 2. Dataset Provenance, Feature Dictionary & Data Quality

### 2.1 Dataset Overview & Provenance
The dataset is aggregated from **260,920 real deliveries across 1,095 IPL matches (2008–2024)**, covering **619 qualified professional cricketers** who have competed in at least 3 IPL matches:

| Metric | Specification |
| :--- | :--- |
| **Total Deliveries Analyzed** | 260,920 ball-by-ball delivery records |
| **Total Matches Analyzed** | 1,095 IPL matches spanning 17 seasons |
| **Total Qualified Players ($N$)** | 619 professional cricketers |
| **Raw & Engineered Features** | 31 quantitative attributes + metadata |
| **Target Variables** | `overall_performance_rating` (Continuous) & `performance_tier_code` (0, 1, 2) |

### 2.2 Feature Dictionary

| Variable Name | Category | Type | Domain Significance |
| :--- | :--- | :--- | :--- |
| `player_name` | Identifier | String | Official player identity (e.g., V Kohli, JJ Bumrah, MS Dhoni) |
| `primary_role` | Categorical | Nominal | Playing specialization (Specialist Batter, All-Rounder, Specialist Bowler) |
| `matches_played` | Experience | Integer | Total career IPL match appearances |
| `total_runs` | Batting | Integer | Total runs scored |
| `batting_average` | Batting | Continuous | Career batting average ($\frac{\text{Runs}}{\text{Dismissals}}$) |
| `batting_strike_rate` | Batting | Continuous | Career batting strike rate ($\frac{\text{Runs}}{\text{Balls Faced}} \times 100$) |
| `boundary_run_pct` | Batting | Continuous | Percentage of runs scored through boundaries |
| `death_overs_strike_rate`| Batting | Continuous | Strike rate in death overs (overs 16–20) |
| `wickets_taken` | Bowling | Integer | Total wickets taken |
| `economy_rate` | Bowling | Continuous | Runs conceded per 6 balls bowled |
| `bowling_strike_rate` | Bowling | Continuous | Balls bowled per wicket taken |
| `dot_ball_bowled_pct` | Bowling | Continuous | Percentage of dot balls bowled ($\frac{\text{Dots}}{\text{Balls Bowled}} \times 100$) |
| `death_overs_economy` | Bowling | Continuous | Economy rate during overs 16–20 |
| `player_of_match_awards` | Clutch | Integer | Career Player of the Match awards won |
| `batting_impact_index` | Composite | Continuous | Weighted composite metric of batting effectiveness (0–100) |
| `bowling_impact_index` | Composite | Continuous | Weighted composite metric of bowling effectiveness (0–100) |
| `clutch_match_winner_index` | Composite | Continuous | Weighted metric of match-winning performances |
| `overall_performance_rating`| Target (Reg) | Continuous | True performance rating ($50.0 \le y \le 95.0$) |
| `performance_tier` | Target (Clf) | Categorical | Talent tier: Developing / Squad, Core / Star, Elite / Marquee |
| `estimated_auction_val_cr`| Financial | Continuous | Fair market auction valuation in ₹ Crores |

---

## 3. Exploratory Data Analysis (EDA) & Domain Observations

1. **Clutch Match-Winning Dominance:**  
   `clutch_match_winner_index` exhibits the strongest direct correlation with overall rating ($r = 0.84$), confirming that match-turning contributions (Player of the Match awards and 50+ scores) are far more predictive of franchise success than volume accumulation alone.
2. **Death Overs Mastery as a Key Differentiator:**  
   In T20 cricket, `death_overs_strike_rate` ($r = 0.62$) and `death_overs_economy` correlate significantly higher with match outcomes than overall strike rate or economy, representing the high-pressure phase of matches.
3. **The Strike Rate vs. Average Trade-off:**  
   Scatter plot analysis reveals that batters with $\text{SR} > 145$ and $\text{Average} > 30$ constitute the top $5\%$ of elite T20 assets (e.g., AB de Villiers, Heinrich Klaasen, Andre Russell, David Warner).

---

## 4. Data Preprocessing & Feature Engineering

1. **Stratification & Normalization:** Continuous metrics were transformed using Scikit-Learn's `StandardScaler` ($z = \frac{x - \mu}{\sigma}$).
2. **One-Hot Encoding:** Categorical feature `primary_role` was one-hot encoded into binary indicator variables with `drop_first=True`.
3. **Train/Test Splitting:** A stratified 80/20 train/test split (495 training instances, 124 test instances) was performed preserving class proportions across the three talent tiers.

---

## 5. Supervised Learning: Talent Tier Classification Models

We formulated the multi-class classification problem predicting talent tiers (Developing=0, Star=1, Elite=2):

| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | $0.9030 \pm 0.0221$ | **$95.16\%$** | $0.9587$ | **$0.9421$** | **$0.9501$** |
| **Random Forest Classifier** | **$0.9475 \pm 0.0142$** | $94.35\%$ | **$0.9496$** | $0.9426$ | $0.9459$ |
| **Logistic Regression** | $0.9253 \pm 0.0180$ | $87.90\%$ | $0.9067$ | $0.8694$ | $0.8853$ |
| **Decision Tree Classifier** | $0.8990 \pm 0.0195$ | $87.90\%$ | $0.8931$ | $0.8706$ | $0.8812$ |

---

## 6. Unsupervised Learning: Tactical Archetype Clustering

Using K-Means Clustering on multi-dimensional skill vectors with Elbow and Silhouette validation ($k=5$):

### The 5 Discovered Tactical Archetypes:
1. **Cluster 0: Tactical Anchor & Top-Order Accumulator:** High batting average ($>35$), controlled powerplay strike rate ($125-135$), deep match batting (e.g. Virat Kohli, Shikhar Dhawan, David Warner, KL Rahul).
2. **Cluster 1: High-Impact Pace Spearhead & Death Bowler:** High dot ball % ($>40\%$), low death overs economy ($<8.5$), elite yorker precision (e.g. Jasprit Bumrah, Lasith Malinga, Bhuvneshwar Kumar).
3. **Cluster 2: Explosive Death-Over Finisher & Boundary Hitter:** Extreme death overs strike rate ($>180$), boundary % ($>70\%$), power hitter profile (e.g. Andre Russell, Heinrich Klaasen, MS Dhoni, Kieron Pollard).
4. **Cluster 3: Mystery / Control Spin Maestro:** High dot ball bowling ($>42\%$), restrictive economy rate in middle overs ($<7.2$), deceived dismissals (e.g. Sunil Narine, Yuzvendra Chahal, Rashid Khan, R Ashwin).
5. **Cluster 4: Elite Dual-Threat All-Rounder:** Balanced high batting and bowling impact indices, reliable 4-over bowling quota + finishing punch (e.g. Ravindra Jadeja, Hardik Pandya, DJ Bravo, Shane Watson).

---

## 7. Dimensionality Reduction & PCA Analysis

Principal Component Analysis (PCA) projected the 29-dimensional space into 2 principal components explaining **60.79% of total variance**:
- **PC1 (T20 Match Impact & Volume):** Captures total matches, clutch awards, and multi-skill volume.
- **PC2 (Batting vs. Bowling Orientation):** Neatly separates specialist pacers and spinners (negative values) from pure top-order batters (positive values), placing all-rounders in the central equilibrium.

---

## 8. What-If Development & Auction Valuation Simulator

The system models how targeted interventions enhance player rating and auction valuation in ₹ Crores:

$$\text{Valuation} = \min\left(24.5, \max\left(0.5, \exp\left((\hat{y} - 58.0) \times 0.09\right) \times 1.5 \times \text{Multiplier}\right)\right)$$

Where All-Rounders receive an auction premium multiplier of $1.25\times$.

---

## 9. Viva Voce Examination Guide: Model Answers

1. **Q: Why did you transition from nominal career runs to rate and phase metrics?**  
   *A:* In modern T20 franchise cricket, a batter scoring 40 runs off 20 balls in the death overs has a vastly higher win probability contribution than a batter scoring 50 off 45 balls in the middle overs. Evaluating rate metrics (Death Overs SR, Boundary %, Dot Ball %) eliminates the volume distortion of older players.
2. **Q: Why does KNN achieve 95.16% accuracy in tier classification?**  
   *A:* In normalized multidimensional sports feature spaces, elite players (Bumrah, Kohli, Russell) form dense, distinct geometric clusters in proximity to other elite benchmarks, allowing distance-weighted nearest neighbors to establish accurate boundaries.
3. **Q: How does this system prevent franchise overspending in IPL auctions?**  
   *A:* CricMetrics Pro provides quantitative fair-value anchors derived from 17 seasons of longitudinal ball-by-ball data, enabling franchise directors to identify undervalued tactical archetypes ("Moneyball").

---
© 2026 CricMetrics Pro | Enterprise Sports Analytics Suite
