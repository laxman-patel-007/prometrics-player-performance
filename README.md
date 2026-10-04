# CricMetrics Pro: Cricket Player Performance Analysis & Auction War Room
### Case Study no. 102 | Enterprise Sports Machine Learning Framework

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit App](https://img.shields.io/badge/Streamlit-Live%20Dashboard-FF4B4B.svg)](http://localhost:8501)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.6-orange.svg)](https://scikit-learn.org/)
[![Dataset: Real IPL 2008-2024](https://img.shields.io/badge/Dataset-260k%20IPL%20Deliveries-green.svg)](https://cricsheet.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Assigned Problem Statement:**  
> *"A sports organization wants to investigate measurable factors associated with player performance. (With Proper Justification)"*  
> **Organization Application:** Professional T20 Cricket Franchise Analytics & Auction War Room (IPL / Global Franchise Management, Talent Scouting, and Player Valuation).

---

## 🌟 Executive Overview & Solution Highlights

CricMetrics Pro is an end-to-end Machine Learning decision support platform built on **260,920 real deliveries across 1,095 IPL matches (2008–2024)**, covering **619 qualified professional cricketers**:

| Component | Technical Focus | Project Implementation |
| :--- | :--- | :--- |
| **Problem Formulation** | Franchise Economics & Scouting | Evidence-based T20 player valuation, death-overs impact, and auction budget allocation. |
| **Data Foundation** | Real IPL Ball-by-Ball Data | Aggregated 17 seasons of ball-by-ball deliveries (`deliveries_2008_2024.csv`, `matches_2008_2024.csv`). |
| **Feature Engineering** | Batting, Bowling & Clutch Indices | Batting SR, Death Overs (16-20) SR, Boundary %, Economy Rate, Dot Ball %, Player of the Match awards. |
| **Continuous Regression** | Rating Prediction ($R^2 = 0.9789$) | Random Forest Regressor, Linear Regression (OLS), Polynomial interaction, MLP. |
| **Talent Classification** | Tier Categorization ($95.16\%$ Acc) | K-Nearest Neighbors (KNN), Random Forest Classifier, Logistic Regression, Decision Tree, MLP. |
| **Tactical Clustering** | Unsupervised Style Discovery | K-Means Clustering ($k=5$) discovering Anchors, Death Finishers, Pace Spearheads, Mystery Spinners, and All-Rounders. |
| **Dimensionality Reduction** | Latent 2D Tactical Map | Principal Component Analysis (PCA) mapping 619 players into 2D skill space (60.8% variance explained). |
| **Auction War Room** | Enterprise Streamlit UI | High-contrast executive dashboard with instant player search, one-click iconic presets, and what-if development simulator. |

---

## 🚀 Quickstart Guide

### 1. Environment & Dependencies
```bash
# Clone / navigate to project directory
cd /Users/laxmanpatel/Desktop/ML

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open **[http://localhost:8501](http://localhost:8501)** to access the 6-in-1 War Room platform:
1. 🏛️ **War Room & Roster Intel:** Search and filter 619 cricketers with real-time rating and valuation progress bars.
2. 📊 **Telemetry & Factor Impact:** Pearson correlation with performance, role multi-skill radars, and SR vs Average quadrants.
3. ⚡ **AI Rating & Auction Valuation:** Sliders + 1-Click Iconic Presets (Kohli, Bumrah, Russell, Narine, Pandya) + gleaming Auction Card with Purse Valuation.
4. 🧩 **Tactical Archetypes & 2D Map:** PCA 2D latent scatter plot mapping every real cricketer.
5. 🚀 **What-If Franchise Simulator:** Prescribe training drills to boost death-overs strike rate or economy, watching ratings jump and auction valuations surge in ₹ Crores.
6. 🏆 **Model Benchmarks & Defense:** Full 10-model leaderboards, cross-validation metrics, confusion matrices, and organizational Q&A.

### 3. Open the Jupyter Notebook
```bash
jupyter notebook notebooks/cricket_player_performance_analysis.ipynb
```

### 4. Re-run Pipeline Scripts
```bash
# Aggregate raw ball-by-ball deliveries into player-level clean dataset
python src/cricket_data_pipeline.py

# Train and serialize all 12+ Machine Learning models to models/
python src/train_cricket_models.py

# Regenerate clean Jupyter Notebook
python src/create_cricket_notebook.py
```

---

## 📊 Experimental Results Summary

### Supervised Continuous Regression Benchmark (Predicting Overall Rating 50–95)
| Model Architecture | 5-Fold CV $R^2$ (Mean $\pm$ Std) | Test MAE | Test RMSE | Test $R^2$ Score |
| :--- | :---: | :---: | :---: | :---: |
| **Random Forest Regressor** | **$0.9528 \pm 0.0129$** | **$0.9024$** | **$1.2562$** | **$0.9789$** |
| **Linear Regression (OLS)** | $0.8797 \pm 0.0371$ | $1.5364$ | $1.9534$ | $0.9490$ |
| **Polynomial Regression (Deg 2)** | $0.8739 \pm 0.0343$ | $1.6563$ | $2.6142$ | $0.9087$ |
| **MLP Regressor (Neural Net)** | $0.4503 \pm 0.1375$ | $4.2246$ | $5.8058$ | $0.5495$ |

### Supervised Classification Benchmark (Predicting Talent Tier)
| Model Architecture | 5-Fold CV Accuracy | Test Accuracy | Macro Precision | Macro Recall | Macro F1-Score |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **K-Nearest Neighbors (KNN)** | $0.9030 \pm 0.0221$ | **$95.16\%$** | $0.9587$ | **$0.9421$** | **$0.9501$** |
| **Random Forest Classifier** | **$0.9475 \pm 0.0142$** | $94.35\%$ | **$0.9496$** | $0.9426$ | $0.9459$ |
| **MLP Classifier (Neural Net)** | $0.8141 \pm 0.0384$ | $92.74\%$ | $0.9405$ | $0.9177$ | $0.9287$ |
| **Logistic Regression** | $0.9253 \pm 0.0180$ | $87.90\%$ | $0.9067$ | $0.8694$ | $0.8853$ |
| **Decision Tree Classifier** | $0.8990 \pm 0.0195$ | $87.90\%$ | $0.8931$ | $0.8706$ | $0.8812$ |

### Unsupervised Learning: Tactical Archetypes ($k=5$)
- **Cluster 0:** *Tactical Anchor & Top-Order Accumulator* (Kohli, Warner, Dhawan, KL Rahul)
- **Cluster 1:** *High-Impact Pace Spearhead & Death Bowler* (Bumrah, Malinga, B Kumar, Shami)
- **Cluster 2:** *Explosive Death-Over Finisher & Boundary Hitter* (Russell, Klaasen, Dhoni, Pollard)
- **Cluster 3:** *Mystery / Control Spin Maestro* (Narine, Chahal, Rashid Khan, Ashwin)
- **Cluster 4:** *Elite Dual-Threat All-Rounder* (Jadeja, Hardik Pandya, DJ Bravo, Watson)

---

## 🏆 Franchise Governance & Viva Voce Q&A

1. **Why is Random Forest superior to linear models for cricket?**  
   T20 performance exhibits non-linear threshold dynamics: a death overs strike rate > 180 is exponentially more valuable than a middle overs strike rate of 125; a death economy < 8.0 RPO carries massive win equity. Decision tree ensembles naturally isolate these complex non-linear interaction surfaces.
2. **How does this system support franchise auction economics?**  
   IPL teams frequently fall prey to emotional bidding wars for famous names. CricMetrics Pro uses 17 seasons of ball-by-ball telemetry to produce objective fair market valuations, enabling franchises to recruit high-performing, undervalued tactical archetypes ("Moneyball").
3. **What is the organizational utility of the 5 archetypes?**  
   When marquee international players are injured or unavailable, franchises can query the exact tactical cluster to identify like-for-like tactical replacements rather than relying on generic squad positions.

---
© 2026 CricMetrics Pro | Enterprise Sports Analytics Suite
