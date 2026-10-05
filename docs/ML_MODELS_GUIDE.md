# CricMetrics Pro: Complete Machine Learning Models Guide
### Easy-to-Understand Guide to Every Model Used in This Project (What, When, Why, How)

> **About this Guide:** This document explains all the Machine Learning (ML) models used in the **CricMetrics Pro** cricket player performance system. It is written in simple, clear, and easy-to-understand English so that anyone—students, evaluators, coaches, and team managers—can easily understand how each model works, why it was chosen, and how it performs.

---

## 1. Quick Summary of All Models

In this project, we analyze **619 real IPL cricketers** across **17 seasons (2008–2024)** using **260,920 deliveries**. We evaluate **9 Machine Learning models** across 4 core machine learning tasks:

| ML Task | Goal / Output | Models Implemented | Top Performer |
| :--- | :--- | :--- | :--- |
| **1. Continuous Regression** | Predict continuous Overall Rating ($50.0 - 95.0$) & Fair Auction Purse (₹ Crores) | • Random Forest Regressor<br>• Polynomial Regression (Deg 2)<br>• Linear Regression (OLS) | **Polynomial Regression** ($R^2 = 0.9941$, RMSE = $0.6648$)<br>& **Random Forest** ($R^2 = 0.9780$, RMSE = $1.2837$) |
| **2. Multi-Class Classification** | Categorize player into 3 Talent Tiers:<br>• Elite / Marquee (Tier 1)<br>• Core / Star (Tier 2)<br>• Developing / Squad (Tier 0) | • K-Nearest Neighbors (KNN)<br>• Random Forest Classifier<br>• Logistic Regression<br>• Decision Tree (CART) | **K-Nearest Neighbors (KNN)** ($95.16\%$ Test Acc, $0.9501$ Macro F1)<br>& **Random Forest** ($94.35\%$ Test Acc, $94.75\%$ 5-Fold CV) |
| **3. Unsupervised Clustering** | Group players into natural playing styles without using predefined human labels | • K-Means Clustering ($k=5$) | **5 Tactical Archetypes** (Anchor, Finisher, Fast Bowler, Spinner, All-Rounder) |
| **4. Dimensionality Reduction** | Compress 29 complex statistics into an interpretable 2D tactical map | • Principal Component Analysis (PCA) | **2D Latent Map** ($60.79\%$ Total Variance Explained) |
| **5. Data Preprocessing** | Clean, encode, and scale features fairly | • StandardScaler ($Z$-score scaling)<br>• One-Hot Encoding | **StandardScaler** ($\mu = 0, \sigma = 1$) |

---

## 2. Data Preparation Tools

Before feeding cricket numbers into machine learning models, we must prepare the data so algorithms can compute distances and weights fairly.

### 2.1 StandardScaler ($Z$-Score Normalization)
- **WHAT is it?**  
  A mathematical tool that rescales all numerical features so that their mean is 0 and standard deviation is 1:
  $$z = \frac{x - \mu}{\sigma}$$
- **WHEN is it used?**  
  Immediately after splitting the dataset into training (80%) and testing (20%). It is fitted **only on training data** and then used to scale test data and real-time user inputs in the web app.
- **WHY do we need it?**  
  In cricket, `total_runs` can be thousands (e.g. 8,000 runs for Virat Kohli), while `economy_rate` is small (e.g. 7.2 runs per over). If we do not scale them, distance-based models (KNN, K-Means, PCA) will pay 99% of their attention to runs and completely ignore bowling economy. Scaling ensures every skill contributes proportionally.
- **HOW is it implemented?**  
  Implemented via Scikit-Learn `StandardScaler()` and saved to disk as `models/scaler.joblib`.

---

### 2.2 One-Hot Encoding
- **WHAT is it?**  
  A transformation that converts categorical text (like `'All-Rounder'` or `'Specialist Bowler'`) into orthogonal binary indicator columns (0 or 1).
- **WHEN is it used?**  
  During the data pipeline transformation in `src/train_cricket_models.py` and `app.py`.
- **WHY do we need it?**  
  Algorithms cannot read text strings. If we labeled roles as `1 = Batter, 2 = All-Rounder, 3 = Bowler`, algorithms would assume a Bowler is "three times" a Batter. One-hot encoding creates separate switches (`primary_role_Specialist Bowler`, etc.) with `drop_first=True` to avoid the **dummy variable trap** (multicollinearity).
- **HOW is it implemented?**  
  Using Pandas `pd.get_dummies(df, columns=['primary_role'], drop_first=True)`.

---

## 3. Supervised Learning: Continuous Regression Models

Regression models predict a continuous target value: an **Overall Performance Rating ($50.0 - 95.0$)** that powers fair auction purse estimation in ₹ Crores.

### 3.1 Random Forest Regressor — 🏆 PRODUCTION REGRESSION ENGINE
- **WHAT is it?**  
  An ensemble of **150 randomized decision regression trees** that each predict a player's continuous rating. The final rating is the average prediction across all 150 trees.
- **WHEN is it used?**  
  Powers the **Live AI Valuation Engine (Tab 4)** and the **Regression Feature Importance Leaderboard (Tab 6)**.
- **WHY do we need it?**  
  1. **Non-Linear Interactions:** Handles complex cricket relationships (e.g. high strike rate is great, but only when paired with decent average).
  2. **Robust Against Overfitting:** Averaging 150 bootstrapped trees prevents any single outlier from skewing ratings.
  3. **High Generalization:** Achieved $95.22\%$ 5-fold cross-validation $R^2$.
- **HOW does it work & performance?**  
  - Initialized with `RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)`.
  - **Test $R^2$ Score:** `0.9780` (explains $97.8\%$ of rating variance on unseen players).
  - **5-Fold CV $R^2$:** `0.9522 ± 0.0128` (highly consistent across folds).
  - **Test RMSE:** `1.2837` points | **Test MAE:** `0.9162` points.
  - **Top Factors:** Clutch Index ($57.4\%$), Batting Impact Index ($20.6\%$), Matches Played ($4.4\%$).

---

### 3.2 Polynomial Regression (Degree 2) — 🏆 TOP STATISTICAL FIT
- **WHAT is it?**  
  An extension of linear regression that adds squared feature terms ($x_i^2$) and interaction pairs ($x_i \cdot x_j$) to capture curved, non-linear relationships.
- **WHEN is it used?**  
  Benchmarked in Tab 6 as the high-capacity non-linear polynomial baseline.
- **WHY do we need it?**  
  Cricket performance follows diminishing returns or exponential growth: hitting boundaries at the death yields non-linear increases in match win probabilities.
- **HOW does it work & performance?**  
  - Created using `PolynomialFeatures(degree=2, include_bias=False)` piped into `Ridge(alpha=100.0)`.
  - **Test $R^2$ Score:** `0.9941` (highest mathematical fit).
  - **5-Fold CV $R^2$:** `0.9956 ± 0.0012`.
  - **Test RMSE:** `0.6648` points | **Test MAE:** `0.3647` points.

---

### 3.3 Linear Regression (Ordinary Least Squares - OLS)
- **WHAT is it?**  
  A fundamental parametric model that calculates a weighted sum of all input features to predict rating:
  $$\hat{y} = \beta_0 + \sum_{j=1}^{p} \beta_j x_j$$
- **WHEN is it used?**  
  Used as the baseline benchmark in Tab 6 to verify whether simpler linear relationships can explain rating.
- **WHY do we need it?**  
  Direct interpretability: each coefficient $\beta_j$ represents the marginal increase in player rating per standard deviation increase in that feature.
- **HOW does it work & performance?**  
  - Fit analytically via Scikit-Learn `LinearRegression()`.
  - **Test $R^2$ Score:** `0.9490`.
  - **5-Fold CV $R^2$:** `0.8797 ± 0.0371`.
  - **Test RMSE:** `1.9534` points | **Test MAE:** `1.5364` points.

---

## 4. Supervised Learning: Multi-Class Classification Models

Classification models categorize each cricketer into one of **3 discrete talent tiers**:
- **Tier 1 (Elite / Marquee):** High-impact franchise anchors and match winners.
- **Tier 2 (Core / Star):** Reliable starting XI performers and role specialists.
- **Tier 0 (Developing / Squad):** Emerging young prospects and backup squad depth.

---

### 4.1 K-Nearest Neighbors (KNN, $k=7$) — 🏆 TOP TEST ACCURACY
- **WHAT is it?**  
  A non-parametric model that classifies a player by finding their **7 closest historical peers** in 29-dimensional space and taking a distance-weighted majority vote.
- **WHEN is it used?**  
  Evaluated in Tab 6 as the top test accuracy model for talent tier classification.
- **WHY do we need it?**  
  1. **Top Accuracy:** Reached **$95.16\%$ test accuracy** and **$0.9501$ Macro F1**.
  2. **Mirrors Real Cricket Scouting:** When talent scouts evaluate a domestic prospect, they naturally ask: *"Who does this player look like in the historical record?"* (e.g. matching Andre Russell or Jasprit Bumrah).
- **HOW does it work & performance?**  
  - Computes Euclidean distance: $d(p, q) = \sqrt{\sum (p_i - q_i)^2}$.
  - **Test Accuracy:** `95.16%` | **Macro F1:** `0.9501` | **Precision:** `0.9601` | **Recall:** `0.9421`.
  - **5-Fold CV Accuracy:** `90.30% ± 3.97%`.

---

### 4.2 Random Forest Classifier — 🏆 PRODUCTION TIER ENGINE
- **WHAT is it?**  
  An ensemble of **150 randomized decision trees** voting on a player's tier.
- **WHEN is it used?**  
  Powers the **Live AI Valuation Engine (Tab 4)** tier classification and **Classification Feature Importance Leaderboard (Tab 6)**.
- **WHY do we need it?**  
  1. **High Cross-Validation Stability:** Achieved **$94.75\%$ 5-fold CV score** with minimal variance ($\pm 2.25\%$).
  2. **Gini Impurity Feature Importance:** Explicitly reveals which measurable factors drive elite tier categorization.
- **HOW does it work & performance?**  
  - Initialized with `RandomForestClassifier(n_estimators=150, max_depth=10, random_state=42)`.
  - **Test Accuracy:** `94.35%` | **5-Fold CV Score:** `94.75%` | **Macro F1:** `0.9459`.
  - **Top Factors:** Clutch Index ($12.8\%$), Batting Impact Index ($11.6\%$), Player of Match Awards ($7.8\%$), Matches Played ($6.8\%$).

---

### 4.3 Decision Tree Classifier (CART)
- **WHAT is it?**  
  A hierarchical flowchart of logical "IF-THEN" rules splitting data on Gini impurity.
- **WHEN is it used?**  
  Benchmarked in Tab 6 for maximum human interpretability.
- **WHY do we need it?**  
  Enables coaching staff and non-technical executives to audit exact decision thresholds without black-box opacity.
- **HOW does it work & performance?**  
  - Fit with `DecisionTreeClassifier(max_depth=5, random_state=42)`.
  - **Test Accuracy:** `87.90%` | **5-Fold CV Score:** `89.90%` | **Macro F1:** `0.8812`.

---

### 4.4 Logistic Regression (Multinomial / Softmax)
- **WHAT is it?**  
  A linear classification model that maps feature combinations into calibrated class probabilities that sum to 100%.
- **WHEN is it used?**  
  Provides the probability distribution displayed in the Live AI Valuation Engine (Tab 4).
- **WHY do we need it?**  
  Franchise directors managing bidding caps need risk percentages (e.g. $85\%$ Elite vs. $15\%$ Core) rather than rigid binary labels.
- **HOW does it work & performance?**  
  - Trained via `LogisticRegression(max_iter=1000, random_state=42)`.
  - **Test Accuracy:** `87.90%` | **5-Fold CV Score:** `92.53%` | **Macro F1:** `0.8853`.

---

## 5. Unsupervised Learning: Clustering (Tactical Archetypes)

Unlike classification, clustering does not use human labels. It automatically groups athletes by multidimensional skill fingerprints.

### 5.1 K-Means Clustering ($k=5$)
- **WHAT is it?**  
  An iterative clustering algorithm that partitions 619 cricketers into **5 tactical archetypes** by minimizing within-cluster sum of squares (inertia):
  $$\mathcal{J} = \sum_{i=1}^{k} \sum_{x \in S_i} ||x - \mu_i||^2$$
- **WHEN is it used?**  
  Powers **Tab 5 (Tactical Archetypes & Roster Balance)**.
- **WHY do we need it?**  
  Nominal labels ("Batter" or "Bowler") fail modern franchise recruitment:
  - Virat Kohli and Andre Russell are both "Batters", but Kohli anchors innings while Russell finishes in death overs.
  - Jasprit Bumrah and Yuzvendra Chahal are both "Bowlers", but Bumrah bowls 145 km/h yorkers while Chahal controls middle overs with leg-spin.
  - K-Means automatically discovered these **5 tactical archetypes**:
    1. **Cluster 0: Top-Order Anchor & Accumulator** (high average, builds partnerships).
    2. **Cluster 1: Powerplay & Middle-Overs Pace Specialist** (wicket-taking seam bowling).
    3. **Cluster 2: Death-Overs Finisher & Power Hitter** (strike rate $>150$, high boundary %).
    4. **Cluster 3: Defensive Middle-Overs Economy Spinner** (low economy rate, high dot ball %).
    5. **Cluster 4: Elite Dual-Threat All-Rounder** (high contributions with both bat and ball).
- **HOW does it work?**  
  - Validated using the Elbow Method and Silhouette Analysis ($s = 0.2676$).
  - Serialized as `models/kmeans_model.joblib`.

---

## 6. Dimensionality Reduction: Visualizing High-Dimensional Data

### 6.1 Principal Component Analysis (PCA, 2D Latent Space)
- **WHAT is it?**  
  An orthogonal linear transformation that projects 29 performance metrics onto 2 uncorrelated principal components ($PC_1, PC_2$).
- **WHEN is it used?**  
  Powers the interactive 2D Tactical Scatter Plot in **Tab 5**.
- **WHY do we need it?**  
  Human decision-makers cannot comprehend a 29-dimensional space. PCA compresses the data into an intuitive 2D tactical map where coaching staff can instantly see tactical proximity between any two players.
- **HOW does it work & performance?**  
  - Derived via Singular Value Decomposition (SVD) of the covariance matrix.
  - **$PC_1$ (X-axis, $38.16\%$ variance):** Measures career volume, longevity, and overall match contribution.
  - **$PC_2$ (Y-axis, $22.63\%$ variance):** Captures skill orientation (bowling-dominant at top, batting-dominant at bottom, dual-threat in middle).
  - **Total Variance Explained:** `60.79%` across the first two components.
  - Serialized as `models/pca_model.joblib`.

---

## 7. Model Evaluation & Validation Rigor

To prevent data leakage and guarantee that models generalize to future tournaments:
1. **80/20 Stratified Split:** 495 training players (80%) and 124 held-out test players (20%) with identical tier proportions.
2. **5-Fold Stratified Cross-Validation:** Every model is evaluated across 5 distinct validation slices to prove score stability.
3. **Comprehensive Metric Suite:**
   - **Regression:** $R^2$ (variance explained), Root Mean Squared Error (RMSE), Mean Absolute Error (MAE).
   - **Classification:** Accuracy, Precision, Recall, Macro F1-Score, and Full Confusion Matrices.

---

## 8. Master Model Leaderboard

| Model | Task | Test Metric | 5-Fold Cross-Validation | Key Strength / Role in System |
| :--- | :--- | :---: | :---: | :--- |
| **Polynomial Reg (Deg 2)** | Continuous Rating ($50-95$) | **$R^2 = 0.9941$**<br>(RMSE: 0.6648) | **$0.9956 \pm 0.0012$** | Mathematical precision modeling curved feature synergies. |
| **Random Forest Regressor** | Continuous Rating ($50-95$) | **$R^2 = 0.9780$**<br>(RMSE: 1.2837) | $0.9522 \pm 0.0128$ | Production rating engine & continuous feature rankings. |
| **Linear Regression (OLS)** | Continuous Rating ($50-95$) | $R^2 = 0.9490$<br>(RMSE: 1.9534) | $0.8797 \pm 0.0371$ | Parametric baseline measuring direct linear marginal impacts. |
| **K-Nearest Neighbors (KNN)** | Talent Tier Classification | **$95.16\%$ Acc**<br>(F1: 0.9501) | $0.9030 \pm 0.0397$ | Top test accuracy; identifies nearest historical player comparables. |
| **Random Forest Classifier** | Talent Tier Classification | $94.35\%$ Acc<br>(F1: 0.9459) | **$0.9475 \pm 0.0225$** | Top CV stability; production tier classification & Gini rankings. |
| **Logistic Regression** | Talent Tier Classification | $87.90\%$ Acc<br>(F1: 0.8853) | $0.9253 \pm 0.0310$ | Outputs calibrated auction risk probabilities ($0-100\%$). |
| **Decision Tree (CART)** | Talent Tier Classification | $87.90\%$ Acc<br>(F1: 0.8812) | $0.8990 \pm 0.0293$ | Transparent "IF-THEN" rule tree for coaching audits. |
| **K-Means Clustering** | Tactical Playing Styles | **$k = 5$ Archetypes**<br>($s = 0.2676$) | Elbow / Silhouette | Groups players by true tactical capability beyond nominal roles. |
| **PCA (2 Components)** | 2D Visualization | **$60.79\%$ Variance** | SVD Eigenvalues | Projects 29-dimensional performance into an interpretable 2D map. |

---

### Key Takeaway for Viva & Presentation Defense
> **Question:** *"Why did you use both regression and classification models in this system?"*  
> **Answer:**  
> *"In a sports franchise front office, decision-makers require two distinct levels of precision:  
> 1. **Classification (KNN & Random Forest Classifier)** solves squad composition by grouping players into 3 broad talent tiers (Elite, Core, Developing) for roster depth and retention limits.  
> 2. **Regression (Random Forest Regressor & Polynomial Regression)** solves financial valuation by computing a precise, continuous FIFA-style rating ($50.0 - 95.0$) that directly calculates a fair auction purse in ₹ Crores, ensuring the franchise never overpays in bidding wars."*
