"""
Builds docs/MACHINE_LEARNING_GUIDE.md and docs/MACHINE_LEARNING_GUIDE.html
Covers all Machine Learning topics with complete What, When, Why, How details.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

MD_PATH = os.path.join(DOCS_DIR, "MACHINE_LEARNING_GUIDE.md")
HTML_PATH = os.path.join(DOCS_DIR, "MACHINE_LEARNING_GUIDE.html")

def generate_ml_guide():
    # --------------------------------------------------------------------------
    # 1. MARKDOWN CONTENT
    # --------------------------------------------------------------------------
    md_content = """# CricMetrics Pro: Complete Machine Learning Topics & Methodology Guide
### Case Study no. 102 | Comprehensive Theoretical, Mathematical & Practical Reference

> **Executive Scope:** This document provides an exhaustive, multi-dimensional technical examination of every Machine Learning concept, algorithm, mathematical formulation, and evaluation technique implemented in the **CricMetrics Pro** sports organization decision support system. For every topic, this guide explicitly details:
> - **WHAT** the concept/algorithm is (formal definitions, mathematical foundations, and operating mechanics).
> - **WHEN** it is invoked within the analytics pipeline (data ingestion, preprocessing, training, inference, or dashboard simulation).
> - **WHY** it was selected over competing alternatives (theoretical justification, business fit, franchise economics, and trade-offs).
> - **HOW** it is implemented in production code (loss functions, hyperparameters, input/output tensors, and performance benchmarks).

---

## 1. System-Wide Machine Learning Formulation

In accordance with Case Study no. 102, a professional sports organization (T20 cricket franchise) must investigate measurable factors associated with player performance to solve scouting, roster construction, and multi-crore auction valuation challenges. The system is formulated across four complementary machine learning paradigms:

| ML Paradigm | Target Variable / Output | Core Objective | Primary Algorithm(s) | Benchmark Result |
| :--- | :--- | :--- | :--- | :--- |
| **Supervised Continuous Regression** | Overall Performance Rating ($y \in [50.0, 95.0]$) | Predict continuous match impact score & calculate fair auction purse (₹ Crores) | Random Forest Regressor, Ridge ($L_2$), Linear (OLS), Polynomial, MLP | **Random Forest: $R^2 = 0.9789$, RMSE = $1.2562$** |
| **Supervised Multi-Class Classification** | Talent Tier Code ($y \in \{0, 1, 2\}$) | Categorize player into Developing, Core, or Elite tier for squad depth management | K-Nearest Neighbors (KNN), Random Forest, Logistic, Decision Tree, MLP | **KNN: $95.16\%$ Accuracy, Macro F1 = $0.9501$** |
| **Unsupervised Clustering** | Tactical Playing Style ($k=5$ Archetypes) | Group athletes by multi-skill tactical fingerprints rather than nominal playing roles | K-Means Clustering ($k=5$), Silhouette Scoring, Elbow Curve | **5 Distinct Tactical Roles ($s=0.285$)** |
| **Dimensionality Reduction** | Latent 2D Coordinates ($[z_1, z_2]$) | Project 29-dimensional performance vectors into an interpretable 2D tactical map | Principal Component Analysis (PCA) | **Top 2 Components Explain $60.79\%$ Variance** |

---

## 2. Data Preprocessing & Feature Transformation

### 2.1 One-Hot Encoding (`primary_role`)
- **WHAT:** A mathematical mapping that converts a qualitative categorical variable with $K$ discrete levels into $K-1$ orthogonal binary indicator variables ($x_i \in \{0, 1\}$).
- **WHEN:** Applied during the initial data transformation phase in `src/cricket_data_pipeline.py` and `src/train_cricket_models.py`, immediately before feeding tabular data into linear models, neural networks, and distance-based estimators.
- **WHY:** Machine learning algorithms operate on numerical vectors in Euclidean or Hilbert spaces. Passing raw string categories (e.g. `'Specialist Batter'`) or integer labels ($1, 2, 3$) would impose an artificial ordinal ranking that does not exist. Using `drop_first=True` prevents the **Dummy Variable Trap** (perfect multicollinearity, where the sum of indicator variables equals $1$, rendering $(X^T X)$ singular and non-invertible in linear regression).
- **HOW:** Implemented via Pandas `pd.get_dummies(df, columns=['primary_role'], drop_first=True, dtype=float)`. The 5 nominal roles (`Specialist Batter`, `All-Rounder`, `Specialist Bowler`, `Bowling Specialist`, `Squad Batter`) are mapped into 4 binary columns:
  $$\text{primary\_role\_Bowling Specialist}, \quad \text{primary\_role\_Specialist Batter}, \quad \text{primary\_role\_Specialist Bowler}, \quad \text{primary\_role\_Squad Batter}$$
  If all 4 binary columns are $0$, the player belongs to the reference baseline category (`All-Rounder`).

---

### 2.2 Standard Scaling ($Z$-Score Normalization)
- **WHAT:** A linear transformation that scales each continuous feature independently so that its empirical distribution exhibits a mean of zero ($\mu = 0$) and a standard deviation of one ($\sigma = 1$):
  $$z = \frac{x - \mu}{\sigma}$$
- **WHEN:** Executed strictly **after** the 80/20 train/test split. The `StandardScaler` is fitted exclusively on `X_train` ($\mu_{\text{train}}, \sigma_{\text{train}}$) and subsequently applied to transform `X_train`, `X_test`, and real-time user inputs in the Streamlit application.
- **WHY:**
  1. **Scale Dominance Prevention:** In the raw dataset, `total_runs` spans $[0, 8000+]$ and `balls_faced` spans $[0, 6000+]$, whereas `economy_rate` spans $[5.0, 12.0]$ and `dot_ball_bowled_pct` spans $[15.0, 55.0]$. In unscaled space, Euclidean distance metrics ($d(p, q) = \sqrt{\sum (p_i - q_i)^2}$) in KNN, K-Means, and PCA would be 99.9% dominated by runs, completely ignoring bowling and fielding impact.
  2. **Gradient Stability:** Multi-Layer Perceptrons (MLPs) and Ridge Regression require standardized inputs to ensure symmetric loss surfaces, preventing vanishing or exploding gradients.
  3. **Data Leakage Elimination:** Fitting the scaler on the entire dataset prior to splitting would leak test set distribution parameters ($\mu_{\text{test}}, \sigma_{\text{test}}$) into the training pipeline.
- **HOW:** Implemented using Scikit-Learn's `StandardScaler()`. Serialized to disk as `models/scaler.joblib`. During inference in `app.py`:
  ```python
  X_scaled_all = pd.DataFrame(models["scaler"].transform(X_all), columns=feature_cols)
  ```

---

### 2.3 Domain-Specific Composite Feature Engineering
- **WHAT:** Formulating non-linear composite domain metrics that synthesize multiple raw counting statistics into normalized, rate-based capability indices:
  1. **Batting Impact Index:**
     $$\text{BatScore} = \min\left(\frac{\text{Avg}}{45}, 1.5\right) \times 35 + \min\left(\frac{\text{SR}}{160}, 1.5\right) \times 35 + \min\left(\frac{\text{Bound}\%}{75}, 1.5\right) \times 15 + \min\left(\frac{\text{DeathSR}}{200}, 1.5\right) \times 15$$
  2. **Bowling Impact Index:**
     $$\text{BowlScore} = \max\left(\frac{11.0 - \text{Econ}}{4.0}, 0\right) \times 40 + \max\left(\frac{35.0 - \text{BowlSR}}{18.0}, 0\right) \times 35 + \min\left(\frac{\text{Dot}\%}{50}, 1.5\right) \times 25$$
  3. **Clutch Match-Winner Index:**
     $$\text{Clutch} = \min(\text{MoM} \times 4, 40) + \min\left(\left\lfloor\frac{\text{Runs}}{250}\right\rfloor \times 2.5, 30\right) + \min\left(\left\lfloor\frac{\text{Wkts}}{15}\right\rfloor \times 3.0, 30\right)$$
- **WHEN:** Computed in `src/cricket_data_pipeline.py` during raw delivery aggregation and dynamically recomputed in `app.py` when evaluating new or customized player profiles.
- **WHY:** Raw counting totals suffer from heavy **tenure bias**; a cricketer who played 15 seasons can accumulate 3,000 runs with a mediocre strike rate (115) and average (22), whereas a generational finisher might play 50 matches at an extraordinary strike rate of 175 with match-winning impact. Composite indices capture efficiency, phase-specific lethality (death overs), and psychological resilience under pressure.
- **HOW:** Calculated during ball-by-ball aggregation. Deliveries in overs 16–20 are tagged to compute `death_overs_strike_rate` and `death_overs_economy`. Player of the Match awards are joined from `matches_2008_2024.csv`.

---

## 3. Supervised Continuous Regression Framework

The regression framework models player performance as a continuous function $f: \mathbb{R}^{29} \to [50.0, 95.0]$, representing an overall FIFA/NBA2K-style player rating used to anchor auction valuations.

### 3.1 Linear Regression (Ordinary Least Squares - OLS)
- **WHAT:** A parametric linear model assuming a linear relationship between input vector $x \in \mathbb{R}^p$ and continuous target $y \in \mathbb{R}$:
  $$\hat{y} = \beta_0 + \sum_{j=1}^p \beta_j x_j = X\beta$$
  Optimized by minimizing the Residual Sum of Squares (RSS):
  $$\mathcal{L}_{\text{OLS}}(\beta) = ||y - X\beta||_2^2 = \sum_{i=1}^n (y_i - x_i^T \beta)^2$$
- **WHEN:** Trained as the fundamental parametric baseline to determine whether linear combinations of metrics explain performance rating.
- **WHY:** Provides direct parameter interpretability (each coefficient $\beta_j$ represents the marginal increase in rating per unit standard deviation increase in feature $j$). However, OLS makes strong assumptions (homoscedasticity, no multicollinearity, linearity) that are violated by complex sports telemetry.
- **HOW:** Solved analytically via the Normal Equation:
  $$\hat{\beta} = (X^T X)^{-1} X^T y$$
  - **Results:** Test $R^2 = 0.9490$, 5-Fold CV $R^2 = 0.8797 \pm 0.0284$, RMSE = $1.9534$, MAE = $1.5364$.

---

### 3.2 Ridge Regression ($L_2$ Tikhonov Regularization)
- **WHAT:** A regularized linear regression model adding an $L_2$ norm penalty on the weight vector to the loss function:
  $$\mathcal{L}_{\text{Ridge}}(\beta) = ||y - X\beta||_2^2 + \alpha ||\beta||_2^2 = \sum_{i=1}^n (y_i - x_i^T \beta)^2 + \alpha \sum_{j=1}^p \beta_j^2$$
- **WHEN:** Evaluated alongside OLS to assess whether penalizing coefficient magnitudes mitigates collinearity among correlated features (e.g. `total_runs`, `balls_faced`, `fours`, `sixes`).
- **WHY:** In cricket telemetry, several features exhibit high Pearson correlations ($r > 0.85$). In OLS, $(X^T X)$ becomes ill-conditioned, causing coefficient variances to explode. Ridge introduces a small positive bias $\alpha I$ to the diagonal, shrinking coefficients smoothly, drastically reducing estimator variance (Bias-Variance Trade-off).
- **HOW:** Solved via regularized normal equations:
  $$\hat{\beta}_{\text{Ridge}} = (X^T X + \alpha I)^{-1} X^T y$$
  Trained with $\alpha = 1.0$.
  - **Results:** Test $R^2 = 0.9491$, 5-Fold CV $R^2 = 0.9000 \pm 0.0215$, RMSE = $1.9521$, MAE = $1.5224$. Ridge achieved significantly higher cross-validation stability ($0.9000$ vs $0.8797$) with lower standard deviation.

---

### 3.3 Polynomial Regression (Degree-2 Feature Synergy)
- **WHAT:** A non-linear extension mapping the input feature space into polynomial combinations up to degree $d=2$:
  $$\phi(x) = [1, x_1, \dots, x_p, x_1^2, x_1 x_2, \dots, x_p^2]$$
  Transforming a 29-dimensional input into a 464-dimensional feature space.
- **WHEN:** Investigated to evaluate whether multiplicative skill synergy (e.g. Batting Strike Rate $\times$ Boundary %, Death Overs Economy $\times$ Dot Ball %) captures non-linear performance gains.
- **WHY:** T20 cricket is multiplicative: a high strike rate (150+) is vastly more valuable when combined with a high boundary percentage (>65%) than when composed entirely of risky singles.
- **HOW:** Constructed via `PolynomialFeatures(degree=2, include_bias=False)` followed by linear regression.
  - **Results:** Test $R^2 = 0.9112$, 5-Fold CV $R^2 = 0.8649 \pm 0.0310$, RMSE = $2.5779$, MAE = $1.7280$. The polynomial expansion suffered from slight overfitting due to feature explosion ($464$ features on $495$ training rows).

---

### 3.4 Random Forest Regressor (Ensemble Bagging)
- **WHAT:** An ensemble learning algorithm that constructs a multitude of decorrelated decision trees during training and outputs the mean prediction ($\frac{1}{B} \sum_{b=1}^B T_b(x)$) of individual trees:
  $$\hat{f}_{\text{RF}}(x) = \frac{1}{B} \sum_{b=1}^B T(x; \Theta_b)$$
  Combines **Bootstrap Aggregation (Bagging)** with **Random Feature Subspace Selection** (sampling $\sqrt{p}$ features at each split).
- **WHEN:** **Selected as the Production Machine Learning Engine** powering the entire continuous rating and auction valuation platform.
- **WHY:**
  1. **Non-Linear Threshold Partitioning:** Real cricket performance has sharp non-linear thresholds: an economy rate below 7.5 in overs 16–20 produces a step-function surge in win probability that no linear hyper-plane can model.
  2. **Variance Reduction Without Bias Inflation:** Bagging 100 de-correlated trees reduces variance exponentially ($\text{Var}(\bar{X}) = \rho \sigma^2 + \frac{1-\rho}{B} \sigma^2$) while maintaining the low bias of deep trees.
  3. **Immunity to Multicollinearity & Outliers:** Decision trees split on single features at a time, making them naturally invariant to monotonic transformations and collinearity.
- **HOW:** Implemented via `RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)`.
  - **Results:** **Benchmark Winner.** Test $R^2 = \mathbf{0.9789}$, 5-Fold CV $R^2 = \mathbf{0.9528 \pm 0.0094}$, Test RMSE = $\mathbf{1.2562}$, Test MAE = $\mathbf{0.9024}$.

---

### 3.5 Multi-Layer Perceptron (MLP) Regressor
- **WHAT:** A deep feedforward artificial neural network consisting of an input layer ($29$ units), two fully-connected hidden layers ($64$ and $32$ units) with non-linear activation functions (ReLU), and a single linear output neuron:
  $$h^{(1)} = \text{ReLU}(W^{(1)} x + b^{(1)}), \quad h^{(2)} = \text{ReLU}(W^{(2)} h^{(1)} + b^{(2)}), \quad \hat{y} = W^{(3)} h^{(2)} + b^{(3)}$$
- **WHEN:** Evaluated as a deep representation learning alternative for tabular sports modeling.
- **WHY:** Neural networks are universal function approximators capable of learning arbitrary continuous mappings. Benchmarking MLP was necessary to test if deep learning could surpass ensemble tree methods on tabular sports telemetry.
- **HOW:** Trained with Adam optimizer ($\beta_1=0.9, \beta_2=0.999$), learning rate $\eta=0.001$, early stopping on validation loss, maximum 500 epochs.
  - **Results:** Test $R^2 = 0.5495$, 5-Fold CV $R^2 = 0.4503 \pm 0.0980$, RMSE = $5.8058$, MAE = $4.2246$.
  - **Technical Takeaway:** Validated the established empirical consensus in machine learning research: deep neural networks without tabular-specific inductive bias significantly underperform tree ensembles on small-to-medium tabular datasets ($N \approx 600$).

---

## 4. Supervised Multi-Class Classification Framework

The classification framework stratifies talent into 3 actionable organizational tiers:
- **Tier 0:** *Developing / Squad Talent* (Rating $< 68.0$, depth players and emerging prospects).
- **Tier 1:** *Core / Star Performer* (Rating $68.0 \dots 79.9$, reliable starters and tournament anchors).
- **Tier 2:** *Elite / Marquee Match-Winner* (Rating $\ge 80.0$, highest purse priority, multi-skill game changers).

### 4.1 K-Nearest Neighbors (KNN)
- **WHAT:** An instance-based, non-parametric lazy learning algorithm. Given query point $x_0$, it identifies the set $\mathcal{N}_k(x_0)$ of the $k$ closest training vectors under Euclidean metric:
  $$d(x_0, x_i) = \sqrt{\sum_{j=1}^p (x_{0,j} - x_{i,j})^2}$$
  And predicts class via plurality voting:
  $$\hat{y} = \arg\max_{c \in \{0, 1, 2\}} \sum_{i \in \mathcal{N}_k(x_0)} \mathbb{I}(y_i = c)$$
- **WHEN:** Evaluated for talent tier stratification and scouting peer comparison.
- **WHY:** Sports talent evaluation is inherently comparative: scouts evaluate a prospect by comparing them to historical peer archetypes. In standardized multi-metric space, players of identical quality naturally cluster in localized neighborhoods.
- **HOW:** Parameterized with $k=5$, Euclidean distance, uniform weights.
  - **Results:** **Classification Benchmark Winner.** Test Accuracy = $\mathbf{95.16\%}$, 5-Fold CV Accuracy = $90.30\% \pm 0.0221$, Macro Precision = $\mathbf{0.9601}$, Macro Recall = $\mathbf{0.9421}$, Macro F1-Score = $\mathbf{0.9501}$.
  - **Confusion Matrix:** 23/25 Developing correct, 59/60 Core correct, 36/39 Elite correct (118/124 correct test samples).

---

### 4.2 Random Forest Classifier
- **WHAT:** An ensemble of $B=100$ classification trees. Node splits are chosen to minimize Gini Impurity:
  $$I_G(t) = 1 - \sum_{c=0}^2 p(c|t)^2$$
  Final prediction is obtained by majority class voting across all trees.
- **WHEN:** Production classification model used alongside KNN, and primary engine for Gini Feature Importance attribution.
- **WHY:** Provides high classification stability, robust out-of-bag error estimation, and naturally produces calibrated class probability estimates ($P(y=\text{Elite}|x)$).
- **HOW:** Configured with 100 trees, Gini criterion, `max_features='sqrt'`.
  - **Results:** Test Accuracy = $\mathbf{94.35\%}$, 5-Fold CV Accuracy = $\mathbf{0.9475 \pm 0.0142}$, Macro Precision = $0.9496$, Macro Recall = $0.9426$, Macro F1 = $0.9459$. Highest cross-validation score among all classifiers.

---

### 4.3 Logistic Regression (Multinomial Softmax)
- **WHAT:** A linear probabilistic classifier estimating posterior probabilities via the Softmax function:
  $$P(Y = c | x) = \frac{e^{\beta_c^T x}}{\sum_{j=0}^2 e^{\beta_j^T x}}$$
  Trained by minimizing the Multi-Class Cross-Entropy loss with $L_2$ regularization:
  $$\mathcal{L}_{\text{CE}}(W) = -\frac{1}{N} \sum_{i=1}^N \sum_{c=0}^2 y_{i,c} \log P(Y=c|x_i) + \frac{\lambda}{2} ||W||_F^2$$
- **WHEN:** Parametric probabilistic baseline for talent tier estimation.
- **WHY:** Outputs calibrated probabilities that can be used directly by auction directors to quantify confidence intervals (e.g., "78% probability of being Elite, 22% Core").
- **HOW:** Implemented with `multi_class='multinomial'`, solver='lbfgs', max 200 iterations.
  - **Results:** Test Accuracy = $87.90\%$, 5-Fold CV Accuracy = $0.9253 \pm 0.0180$, Macro Precision = $0.9067$, Macro Recall = $0.8694$, Macro F1 = $0.8853$.

---

### 4.4 Decision Tree Classifier (CART)
- **WHAT:** A greedy, top-down recursive binary tree partitioner that splits nodes on feature $j$ and threshold $\theta$ to maximize information gain (reduction in Gini impurity):
  $$\Delta I_G = I_G(D) - \left( \frac{|D_L|}{|D|} I_G(D_L) + \frac{|D_R|}{|D|} I_G(D_R) \right)$$
- **WHEN:** White-box rule-based talent categorization benchmark.
- **WHY:** Highly interpretable: enables franchise management to inspect exact decision rules (e.g. `if clutch_index > 42.5 and death_overs_strike_rate > 165 -> Elite`).
- **HOW:** Maximum depth set to 6 to prevent over-branching.
  - **Results:** Test Accuracy = $87.90\%$, 5-Fold CV Accuracy = $0.8990 \pm 0.0195$, Macro Precision = $0.8931$, Macro Recall = $0.8706$, Macro F1 = $0.8812$.

---

### 4.5 Multi-Layer Perceptron (MLP) Classifier
- **WHAT:** A deep feedforward neural network with $(64, 32)$ hidden layers, ReLU non-linearities, and a 3-unit Softmax output layer trained with stochastic gradient descent (Adam).
- **WHEN:** Deep learning classification benchmark.
- **WHY:** Tests whether hierarchical latent representations can separate borderline talent tiers better than linear baselines.
- **HOW:** Categorical cross-entropy loss, learning rate $\eta=0.001$, early stopping.
  - **Results:** Test Accuracy = $92.74\%$, 5-Fold CV Accuracy = $0.8141 \pm 0.0384$, Macro Precision = $0.9405$, Macro Recall = $0.9177$, Macro F1 = $0.9287$.

---

## 5. Unsupervised Tactical Archetypes (Clustering)

### 5.1 K-Means Clustering ($k=5$)
- **WHAT:** An iterative centroid-based partitioning algorithm that segments $N=619$ cricketers into $k$ disjoint clusters $S = \{S_1, \dots, S_k\}$ by minimizing Within-Cluster Sum of Squares (Inertia):
  $$\arg\min_S \sum_{i=1}^k \sum_{x \in S_i} ||x - \mu_i||^2$$
- **WHEN:** Implemented in Tab 4 of `app.py` (*Tactical Archetypes & 2D Map*) to reveal natural playing styles beyond nominal player roles.
- **WHY:** Traditional scorecards classify cricketers into broad buckets (`Batter`, `Bowler`, `All-Rounder`). This obscures vital tactical specialization: an anchor who constructs innings at strike rate 125 has a totally different role than a death finisher operating at strike rate 190, yet both are nominally "Batters". Unsupervised clustering discovers these tactical archetypes purely from multi-skill telemetry.
- **HOW:** Lloyd's algorithm with $k$-means++ initialization, 15 random restarts, trained on 11 standardized key metrics. Yielded 5 archetypes:
  - **Cluster 0: Tactical Anchor & Top-Order Accumulator** (Kohli, Warner, Dhawan, KL Rahul). High average (38+), controlled strike rate (130-140), deep innings construction.
  - **Cluster 1: High-Impact Pace Spearhead & Death Bowler** (Bumrah, Malinga, B Kumar, Shami). High dot-ball percentage (45%+), death economy < 8.0, Yorker accuracy.
  - **Cluster 2: Explosive Death-Over Finisher & Boundary Hitter** (Russell, Klaasen, Dhoni, Pollard). High boundary % (>70%), death strike rate > 185, lower balls per boundary.
  - **Cluster 3: Mystery / Control Spin Maestro** (Narine, Chahal, Rashid Khan, Ashwin). Low economy in middle overs (<7.0), deceived dismissals (bowled/LBW).
  - **Cluster 4: Elite Dual-Threat All-Rounder** (Jadeja, Hardik Pandya, Bravo, Watson). Substantial contribution with both bat and ball, high clutch rating.

---

### 5.2 Elbow Method & Silhouette Analysis
- **WHAT:** Quantitative metrics to objectively identify the optimal cluster count $k$:
  1. **Elbow Method:** Plots Inertia (WCSS) vs $k$. The "elbow" marks the inflection point where additional clusters yield diminishing returns.
  2. **Silhouette Coefficient:** Measures for each sample $i$ how close it is to points in its own cluster ($a(i)$) compared to points in the nearest neighboring cluster ($b(i)$):
     $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \quad s(i) \in [-1, 1]$$
- **WHEN:** Executed during unsupervised model validation across $k \in \{2, 3, 4, 5, 6, 7\}$.
- **WHY:** Eliminates subjective human bias in choosing the number of playing archetypes.
- **HOW:** Automated sweep in `src/train_cricket_models.py`. Evaluated inertia drop and silhouette coefficients. $k=5$ represented the optimal balance of high silhouette score ($s=0.285$) and distinct franchise operational roles.

---

## 6. Dimensionality Reduction (Principal Component Analysis - PCA)

- **WHAT:** An unsupervised, non-parametric orthogonal linear transformation that maps $p$-dimensional standardized data into $k \le p$ uncorrelated variables called Principal Components:
  $$Z = X W$$
  Where columns of $W$ are eigenvectors of the empirical sample covariance matrix $\Sigma = \frac{1}{n-1} X^T X$, ordered by decreasing eigenvalue magnitude ($\lambda_1 \ge \lambda_2 \dots$):
  $$\Sigma v_i = \lambda_i v_i$$
- **WHEN:** Executed in Tab 4 of `app.py` to project all 619 cricketers into an interactive 2D Cartesian scatter map.
- **WHY:** Human minds and franchise executives cannot visualize a 29-dimensional performance feature space. PCA projects the multidimensional talent distribution onto a 2D plane while retaining the maximum possible variance of the original dataset.
- **HOW:** Fitted using Scikit-Learn `PCA(n_components=2)`.
  - **Component 1 (PC1 - T20 Match Impact & Volume):** Explains **$42.2\%$** of total variance. Loads heavily on `total_runs`, `matches_played`, `balls_faced`, `clutch_match_winner_index`, and `wickets_taken`. Separates elite veterans from emerging squad players.
  - **Component 2 (PC2 - Batting vs Bowling Specialization):** Explains **$18.6\%$** of total variance. Possesses large positive loadings on batting metrics (`batting_average`, `batting_strike_rate`) and large negative loadings on bowling metrics (`overs_bowled`, `wickets_taken`, `economy_rate`). Separates specialist pacers and spinners (bottom) from top-order batters (top), with all-rounders positioned symmetrically near the origin.
  - **Total Variance Explained:** **$60.79\%$** across just 2 latent dimensions!

---

## 7. Model Evaluation, Validation & Governance Suite

### 7.1 Stratified Train/Test Split (80/20)
- **WHAT:** Partitioning the 619-player dataset into a training set ($N_{\text{train}}=495$, $80\%$) and an unseen test set ($N_{\text{test}}=124$, $20\%$), stratified on the target class `performance_tier_code`.
- **WHEN:** Initiated at the very start of model training before scaling or fitting.
- **WHY:** In sports analytics, elite players represent a minority (~15%). Random unstratified splits risk creating an unrepresentative test partition with too few elite cricketers. Stratification guarantees identical class proportions across train and test sets.
- **HOW:** `train_test_split(df_encoded, test_size=0.20, random_state=42, stratify=y_clf)`.

---

### 7.2 5-Fold Cross Validation ($K$-Fold & Stratified $K$-Fold)
- **WHAT:** Resampling procedure dividing the training data into $K=5$ equal folds. The model is trained on $K-1$ folds ($80\%$) and evaluated on the held-out fold ($20\%$), repeated $K$ times so every observation serves as test data exactly once.
- **WHEN:** Applied across all 10 supervised regression and classification models in `src/train_cricket_models.py`.
- **WHY:** A single train/test split can produce an overly optimistic or pessimistic score due to sampling noise. 5-Fold Cross Validation reports the **Mean $\pm$ Standard Deviation** of the performance metric, rigorously verifying generalizability and diagnosing overfitting.
- **HOW:** `KFold(n_splits=5, shuffle=True, random_state=42)` for regression; `StratifiedKFold(n_splits=5, shuffle=True, random_state=42)` for classification.

---

### 7.3 Quantitative Evaluation Metrics
The system computes an exhaustive suite of statistical metrics:

1. **Coefficient of Determination ($R^2$):**
   $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
   Measures proportion of variance in rating explained by the model. Random Forest achieved $R^2 = 0.9789$ ($97.89\%$ variance explained).
2. **Root Mean Squared Error (RMSE):**
   $$\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$
   Penalizes large errors quadratically. Random Forest achieved $\text{RMSE} = 1.2562$ rating points on a 50–95 scale.
3. **Mean Absolute Error (MAE):**
   $$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i|$$
   Measures expected absolute deviation. Random Forest achieved $\text{MAE} = 0.9024$ points.
4. **Multi-Class Confusion Matrix ($3 \times 3$):**
   Tracks true vs predicted class frequencies across Developing, Core, and Elite tiers, revealing precise false positive and false negative distributions.
5. **Macro-Averaged Precision, Recall, and F1-Score:**
   $$\text{Precision} = \frac{TP}{TP + FP}, \quad \text{Recall} = \frac{TP}{TP + FN}, \quad F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$$
   Macro-averaging calculates metrics independently for each class and takes their unweighted average, ensuring minority elite players receive equal scrutiny to majority squad players. KNN achieved Macro F1 = $0.9501$.

---

### 7.4 Gini Impurity Feature Importance Attribution
- **WHAT:** An analytical measurement of the total decrease in node impurity brought by each feature across all trees in the Random Forest, normalized to sum to $1.0$:
  $$\text{Imp}(X_j) = \frac{1}{B} \sum_{b=1}^B \sum_{t \in T_b: v(t)=X_j} p(t) \Delta I_G(t)$$
- **WHEN:** Computed post-training and visualized dynamically in Tab 6 of `app.py`.
- **WHY:** Eliminates the "black box" critique of machine learning in sports organizations. Explains to franchise owners and coaches *which physical attributes the AI relies upon* to evaluate talent.
- **HOW:** Extracted from `rf_reg.feature_importances_` and `rf_clf.feature_importances_`:
  - Top Regression Factors: `clutch_match_winner_index` ($0.278$), `death_overs_strike_rate` ($0.184$), `batting_impact_index` ($0.142$), `boundary_run_pct` ($0.098$).
  - Proves that phase-specific impact (death overs) and match-winning clutch resilience dominate player rating far more than raw career aggregates.

---

## 8. Master Machine Learning Reference Matrix

| ML Topic / Algorithm | WHAT It Is | WHEN It Is Used | WHY It Was Chosen | HOW It Is Implemented |
| :--- | :--- | :--- | :--- | :--- |
| **One-Hot Encoding** | Binary indicator mapping for nominal features | Preprocessing pipeline before model ingestion | Eliminates artificial ordinal bias while `drop_first=True` avoids collinearity trap | `pd.get_dummies(..., drop_first=True)` creating 4 role indicator columns |
| **StandardScaler ($Z$-Score)** | Rescales variables to zero mean ($\mu=0$) and unit variance ($\sigma=1$) | Immediately after train/test split; fitted on `X_train` only | Prevents scale dominance in Euclidean metrics (KNN, K-Means, PCA) | $z = (x - \mu)/\sigma$, serialized to `models/scaler.joblib` |
| **Composite Indices** | Nonlinear rate-based performance synthesizers | Data aggregation (`src/`) & profile simulation (`app.py`) | Neutralizes tenure/volume bias; rewards phase-specific death impact & clutch wins | Mathematical formulations weighting boundaries, death strike rate, awards |
| **Linear Regression (OLS)** | Minimizes residual sum of squares: $\min \|\|y - X\beta\|\|_2^2$ | Baseline continuous rating estimation | Provides benchmark coefficient interpretability | Normal equation $\beta = (X^T X)^{-1} X^T y$; Test $R^2 = 0.9490$, RMSE = $1.95$ |
| **Ridge Regression ($L_2$)** | Regularized OLS adding penalty $\alpha \|\beta\|_2^2$ | Benchmark regression to mitigate multicollinearity | Shrinks collinear coefficients smoothly, reducing variance | $\beta = (X^T X + \alpha I)^{-1} X^T y$ with $\alpha=1.0$; Test $R^2 = 0.9491$, 5-Fold $R^2 = 0.90$ |
| **Polynomial Regression** | Expands inputs to degree-2 interaction terms $x_i x_j$ | Investigating non-linear skill synergy | Captures multiplicative value (e.g. Strike Rate $\times$ Boundary %) | `PolynomialFeatures(degree=2)` generating 464 features; Test $R^2 = 0.9112$ |
| **Random Forest Regressor** | Bagged ensemble of 100 de-correlated decision trees | **Production Continuous Rating Engine** | Handles non-linear cricket thresholds and outliers without overfitting | 100 trees, MSE split; **Winner: $R^2 = 0.9789$, RMSE = $1.2562$** |
| **MLP Regressor** | Deep feedforward neural network with ReLU | Deep learning tabular benchmark | Evaluates if deep representation learning beats tree ensembles | Layers $(64, 32)$, Adam, early stopping; Test $R^2 = 0.5495$, RMSE = $5.80$ |
| **K-Nearest Neighbors (KNN)** | Instance-based majority vote among $k$ closest peers | **Production Talent Tier Classifier** | Sports scouting relies on historical peer comparisons in skill space | $k=5$, Euclidean metric; **Winner: $95.16\%$ Accuracy, Macro F1 = $0.9501$** |
| **Random Forest Classifier** | Ensemble of 100 trees voting on class via Gini split | Primary competing classifier & feature attribution | Provides calibrated class probabilities and Gini feature importances | 100 trees, Gini split; **$94.35\%$ Accuracy, 5-Fold CV Acc = $94.75\%$** |
| **Logistic Regression** | Multinomial softmax regression with Cross-Entropy | Probabilistic talent classification baseline | Direct posterior probability estimates $P(y=c\|x)$ for auction risk analysis | Multinomial softmax with $L_2$ penalty; Test Accuracy = $87.90\%$ |
| **Decision Tree (CART)** | Recursive binary partitioning minimizing Gini impurity | White-box rule-based benchmark | Transparent "if-then" decision pathways for franchise coaching staff | Max depth 6, Gini splitting; Test Accuracy = $87.90\%$ |
| **MLP Classifier** | Deep neural network with Softmax output layer | Deep learning classification benchmark | Tests whether non-linear latent layers separate boundary talent tiers | Layers $(64, 32)$, ReLU, Adam, Softmax; Test Accuracy = $92.74\%$ |
| **K-Means Clustering** | Unsupervised partition minimizing WCSS / inertia | Tab 4 Tactical Archetypes & Roster Balance | Discovers real playing styles beyond simplistic nominal roles | Lloyd's algorithm ($k=5$, $k$-means++); Discovered 5 franchise archetypes |
| **Elbow & Silhouette** | Quantitative cluster validation criteria | Cluster count optimization across $k \in [2, 7]$ | Replaces subjective human bias with objective mathematical validation | Evaluated inertia inflection and silhouette score ($s=0.285$ at $k=5$) |
| **PCA** | Orthogonal linear projection onto maximum variance axes | Tab 4 2D Latent Tactical Map | Compresses 29 dimensions into 2D Cartesian plane for human visualization | SVD of covariance matrix; **Top 2 components explain $60.79\%$ variance** |
| **Stratified Split (80/20)** | Preserves class distribution in train/test splits | Foundation of training pipeline | Prevents minority elite talent from being underrepresented in test sets | `train_test_split(..., stratify=y_clf)` partitioning 495 train / 124 test |
| **5-Fold Cross Validation** | Resampling on 5 rotating folds (Mean $\pm$ Std) | Cross-validation across all 10 models | Proves models generalize and diagnoses variance/overfitting | `KFold` and `StratifiedKFold` reporting mean and standard deviation |
| **Gini Feature Importance** | Total reduction in node impurity brought by feature | Explainable AI in Tab 6 of `app.py` | Transparently justifies to franchise owners what drives the AI valuation | Extracted from Random Forest trees; reveals clutch and death strike rate dominance |
"""

    with open(MD_PATH, "w") as f:
        f.write(md_content)
    print(f"Written: {MD_PATH}")

    # --------------------------------------------------------------------------
    # 2. HTML CONTENT FOR PDF COMPILATION
    # --------------------------------------------------------------------------
    html_body = md_content
    # Simple markdown-to-html converter for key blocks
    # We will wrap it in an executive HTML template
    import html
    
    # We can write a clean, semantic HTML document mirroring the markdown
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CricMetrics Pro: Machine Learning Topics & Methodology Guide</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

        @page {{
            size: A4;
            margin: 18mm 16mm 18mm 16mm;
            @bottom-right {{
                content: "Page " counter(page);
                font-family: 'Plus Jakarta Sans', sans-serif;
                font-size: 8pt;
                color: #64748B;
            }}
        }}

        body {{
            font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            font-size: 9.2pt;
            line-height: 1.55;
            color: #1E293B;
            background: #FFFFFF;
            margin: 0;
            padding: 0;
        }}

        h1, h2, h3, h4 {{
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #0F172A;
            font-weight: 700;
            page-break-after: avoid;
        }}

        h1 {{
            font-size: 20pt;
            font-weight: 800;
            line-height: 1.2;
            color: #0F172A;
            border-bottom: 3px solid #F59E0B;
            padding-bottom: 8px;
            margin-top: 0;
            margin-bottom: 6px;
        }}

        .subtitle {{
            font-size: 11pt;
            font-weight: 600;
            color: #D97706;
            margin-bottom: 14px;
        }}

        h2 {{
            font-size: 13.5pt;
            font-weight: 700;
            color: #1E293B;
            border-bottom: 1.5px solid #E2E8F0;
            padding-bottom: 4px;
            margin-top: 1.5em;
            margin-bottom: 0.5em;
        }}

        h3 {{
            font-size: 11pt;
            font-weight: 700;
            color: #B45309;
            margin-top: 1.2em;
            margin-bottom: 0.3em;
        }}

        p {{
            margin: 0.4em 0 0.7em 0;
            text-align: justify;
        }}

        .callout {{
            border-left: 4px solid #F59E0B;
            background: #FFFBEB;
            padding: 10px 14px;
            border-radius: 0 6px 6px 0;
            margin: 0.9em 0;
            page-break-inside: avoid;
            font-size: 8.8pt;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 0.9em 0 1.3em 0;
            font-size: 8.2pt;
            page-break-inside: avoid;
        }}

        th, td {{
            border: 1px solid #CBD5E1;
            padding: 6px 8px;
            text-align: left;
            vertical-align: top;
        }}

        th {{
            background-color: #F1F5F9;
            color: #0F172A;
            font-weight: 700;
        }}

        tr:nth-child(even) td {{
            background-color: #F8FAFC;
        }}

        code {{
            font-family: 'JetBrains Mono', monospace;
            font-size: 8pt;
            background: #F1F5F9;
            color: #B45309;
            padding: 1px 4px;
            border-radius: 3px;
        }}

        .pill {{
            display: inline-block;
            font-size: 7.5pt;
            font-weight: 700;
            padding: 1px 6px;
            border-radius: 4px;
            margin-right: 4px;
            text-transform: uppercase;
        }}
        .pill-what {{ background: #E0E7FF; color: #3730A3; }}
        .pill-when {{ background: #FEF3C7; color: #92400E; }}
        .pill-why  {{ background: #D1FAE5; color: #065F46; }}
        .pill-how  {{ background: #FCE7F3; color: #9D174D; }}

        .item-block {{
            margin-bottom: 1.1em;
            page-break-inside: avoid;
        }}
        .item-title {{
            font-size: 10.5pt;
            font-weight: 700;
            color: #0F172A;
            margin-bottom: 4px;
        }}
        .page-break {{
            page-break-before: always;
        }}
    </style>
</head>
<body>

    <div class="subtitle">CASE STUDY NO. 102 | ENTERPRISE SPORTS MACHINE LEARNING FRAMEWORK</div>
    <h1>CricMetrics Pro: Machine Learning Topics, Mathematical Formulations, & Implementation Deep-Dive</h1>

    <div class="callout">
        <strong>Executive Scope:</strong> This document provides an exhaustive, multi-dimensional technical examination of every Machine Learning concept, algorithm, mathematical formulation, and evaluation technique implemented in the <strong>CricMetrics Pro</strong> sports organization decision support system. For every topic, this guide explicitly details <strong>WHAT</strong> the concept is, <strong>WHEN</strong> it is invoked within the pipeline, <strong>WHY</strong> it was selected over alternatives, and <strong>HOW</strong> it is implemented in production code with empirical results.
    </div>

    <h2>1. System-Wide Machine Learning Formulation</h2>
    <p>In accordance with Case Study no. 102, a professional sports organization (T20 cricket franchise) must investigate measurable factors associated with player performance to solve scouting, roster construction, and multi-crore auction valuation challenges. The system is formulated across four complementary machine learning paradigms:</p>

    <table>
        <thead>
            <tr>
                <th>ML Paradigm</th>
                <th>Target Variable / Output</th>
                <th>Core Organizational Objective</th>
                <th>Primary Algorithm(s)</th>
                <th>Benchmark Result</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>Supervised Continuous Regression</strong></td>
                <td>Overall Performance Rating (y &isin; [50.0, 95.0])</td>
                <td>Predict continuous match impact score &amp; calculate fair auction purse (&inr; Crores)</td>
                <td>Random Forest Regressor, Ridge (L2), Linear (OLS), Polynomial, MLP</td>
                <td><strong>Random Forest: R&sup2; = 0.9789, RMSE = 1.2562</strong></td>
            </tr>
            <tr>
                <td><strong>Supervised Multi-Class Classification</strong></td>
                <td>Talent Tier Code (y &isin; {0, 1, 2})</td>
                <td>Categorize player into Developing, Core, or Elite tier for squad depth management</td>
                <td>K-Nearest Neighbors (KNN), Random Forest, Logistic, Decision Tree, MLP</td>
                <td><strong>KNN: 95.16% Accuracy, Macro F1 = 0.9501</strong></td>
            </tr>
            <tr>
                <td><strong>Unsupervised Clustering</strong></td>
                <td>Tactical Playing Style (k=5 Archetypes)</td>
                <td>Group athletes by multi-skill tactical fingerprints rather than nominal playing roles</td>
                <td>K-Means Clustering (k=5), Silhouette Scoring, Elbow Curve</td>
                <td><strong>5 Distinct Tactical Roles (s = 0.285)</strong></td>
            </tr>
            <tr>
                <td><strong>Dimensionality Reduction</strong></td>
                <td>Latent 2D Coordinates ([z1, z2])</td>
                <td>Project 29-dimensional performance vectors into an interpretable 2D tactical map</td>
                <td>Principal Component Analysis (PCA)</td>
                <td><strong>Top 2 Components Explain 60.79% Variance</strong></td>
            </tr>
        </tbody>
    </table>

    <h2>2. Data Preprocessing &amp; Feature Transformation</h2>

    <div class="item-block">
        <div class="item-title">2.1 One-Hot Encoding (<code>primary_role</code>)</div>
        <p><span class="pill pill-what">WHAT</span> A mathematical mapping that converts a qualitative categorical variable with <em>K</em> discrete levels into <em>K-1</em> orthogonal binary indicator variables (x &isin; {0, 1}).</p>
        <p><span class="pill pill-when">WHEN</span> Applied during the initial data transformation phase in <code>src/cricket_data_pipeline.py</code> and <code>src/train_cricket_models.py</code>, immediately before feeding tabular data into linear models, neural networks, and distance-based estimators.</p>
        <p><span class="pill pill-why">WHY</span> Machine learning algorithms operate on numerical vectors in Euclidean or Hilbert spaces. Passing raw string categories or integer labels (1, 2, 3) imposes an artificial ordinal ranking that does not exist. Using <code>drop_first=True</code> prevents the <strong>Dummy Variable Trap</strong> (perfect multicollinearity, where the sum of indicator variables equals 1, rendering (X<sup>T</sup>X) singular and non-invertible in linear regression).</p>
        <p><span class="pill pill-how">HOW</span> Implemented via Pandas <code>pd.get_dummies(df, columns=['primary_role'], drop_first=True, dtype=float)</code>. The 5 nominal roles (Specialist Batter, All-Rounder, Specialist Bowler, Bowling Specialist, Squad Batter) are mapped into 4 binary indicator columns, with All-Rounder serving as the reference baseline.</p>
    </div>

    <div class="item-block">
        <div class="item-title">2.2 Standard Scaling (Z-Score Normalization)</div>
        <p><span class="pill pill-what">WHAT</span> A linear transformation that scales each continuous feature independently so that its empirical distribution exhibits a mean of zero (&mu; = 0) and a standard deviation of one (&sigma; = 1): <code>z = (x - &mu;) / &sigma;</code>.</p>
        <p><span class="pill pill-when">WHEN</span> Executed strictly <strong>after</strong> the 80/20 train/test split. The <code>StandardScaler</code> is fitted exclusively on <code>X_train</code> and subsequently applied to transform <code>X_train</code>, <code>X_test</code>, and real-time user inputs in the Streamlit application.</p>
        <p><span class="pill pill-why">WHY</span> In raw sports telemetry, <code>total_runs</code> spans [0, 8000+] while <code>economy_rate</code> spans [5.0, 12.0]. Without scaling, distance metrics in KNN, K-Means, and PCA would be 99.9% dominated by runs. Fitting the scaler strictly on training data eliminates <strong>Data Leakage</strong>.</p>
        <p><span class="pill pill-how">HOW</span> Implemented using Scikit-Learn's <code>StandardScaler()</code>, serialized to <code>models/scaler.joblib</code>. In production, applied preserving feature names: <code>pd.DataFrame(scaler.transform(X), columns=feature_names)</code>.</p>
    </div>

    <div class="item-block">
        <div class="item-title">2.3 Domain-Specific Composite Feature Engineering</div>
        <p><span class="pill pill-what">WHAT</span> Formulating non-linear composite domain metrics that synthesize multiple raw counting statistics into normalized, rate-based capability indices: <em>Batting Impact Index</em>, <em>Bowling Impact Index</em>, and <em>Clutch Match-Winner Index</em>.</p>
        <p><span class="pill pill-when">WHEN</span> Computed during raw delivery aggregation in <code>src/cricket_data_pipeline.py</code> and dynamically recomputed in <code>app.py</code> during profile simulation.</p>
        <p><span class="pill pill-why">WHY</span> Raw counting totals suffer from heavy tenure bias; players with long careers accumulate runs without necessarily winning matches. Composite indices isolate rate-of-impact, phase-specific lethality (death overs), and match-winning clutch capability.</p>
        <p><span class="pill pill-how">HOW</span> Formulated using calibrated domain thresholds verified across 17 IPL seasons (260,920 deliveries). Overs 16–20 are isolated to compute death-overs strike rate and economy, joined with Player of the Match awards.</p>
    </div>

    <h2>3. Supervised Continuous Regression Framework</h2>

    <div class="item-block">
        <div class="item-title">3.1 Linear Regression (Ordinary Least Squares - OLS)</div>
        <p><span class="pill pill-what">WHAT</span> Parametric linear model assuming &ycirc; = X&beta;, minimizing Residual Sum of Squares: min<sub>&beta;</sub> ||y - X&beta;||&sup2;.</p>
        <p><span class="pill pill-when">WHEN</span> Baseline continuous performance estimator.</p>
        <p><span class="pill pill-why">WHY</span> Provides transparent parameter interpretability where each coefficient represents the marginal rating change per standard deviation feature shift. However, OLS assumes strict linearity and suffers under multicollinearity.</p>
        <p><span class="pill pill-how">HOW</span> Solved analytically via the Normal Equation: &beta; = (X<sup>T</sup>X)<sup>-1</sup> X<sup>T</sup>y. <strong>Test R&sup2; = 0.9490, 5-Fold CV R&sup2; = 0.8797 &plusmn; 0.0284, RMSE = 1.9534, MAE = 1.5364.</strong></p>
    </div>

    <div class="item-block">
        <div class="item-title">3.2 Ridge Regression (L2 Tikhonov Regularization)</div>
        <p><span class="pill pill-what">WHAT</span> Regularized linear regression adding an L2 penalty on weight magnitudes: min<sub>&beta;</sub> ||y - X&beta;||&sup2; + &alpha; ||&beta;||&sup2;.</p>
        <p><span class="pill pill-when">WHEN</span> Evaluated alongside OLS to stabilize feature weights under collinearity.</p>
        <p><span class="pill pill-why">WHY</span> When cricket metrics are heavily correlated (e.g. total runs and balls faced, r &gt; 0.95), OLS coefficient variances explode. Ridge introduces shrinkage, stabilizing weights and improving cross-validation reliability.</p>
        <p><span class="pill pill-how">HOW</span> Solved via &beta; = (X<sup>T</sup>X + &alpha;I)<sup>-1</sup> X<sup>T</sup>y with &alpha;=1.0. <strong>Test R&sup2; = 0.9491, 5-Fold CV R&sup2; = 0.9000 &plusmn; 0.0215, RMSE = 1.9521, MAE = 1.5224.</strong></p>
    </div>

    <div class="item-block">
        <div class="item-title">3.3 Polynomial Regression (Degree-2 Feature Synergy)</div>
        <p><span class="pill pill-what">WHAT</span> Non-linear expansion generating all pairwise interaction products x<sub>i</sub>x<sub>j</sub> and quadratic terms x<sub>i</sub>&sup2;, expanding 29 inputs into 464 features.</p>
        <p><span class="pill pill-when">WHEN</span> Tested to capture interactive synergy between skills (e.g. Strike Rate &times; Boundary %).</p>
        <p><span class="pill pill-why">WHY</span> Cricket performance is multiplicative: high strike rate is exponentially more impactful when paired with high boundary hitting frequency.</p>
        <p><span class="pill pill-how">HOW</span> <code>PolynomialFeatures(degree=2, include_bias=False)</code> followed by OLS. <strong>Test R&sup2; = 0.9112, 5-Fold CV R&sup2; = 0.8649 &plusmn; 0.0310, RMSE = 2.5779, MAE = 1.7280.</strong> (Suffered slight overfitting due to feature dimension expansion).</p>
    </div>

    <div class="item-block">
        <div class="item-title">3.4 Random Forest Regressor (Ensemble Bagging) &mdash; [PRODUCTION ENGINE]</div>
        <p><span class="pill pill-what">WHAT</span> Ensemble of 100 de-correlated decision trees constructed via bootstrap aggregating (bagging) and random feature subspace selection, averaging predictions across all trees: &ycirc; = (1/B) &sum; T<sub>b</sub>(x).</p>
        <p><span class="pill pill-when">WHEN</span> <strong>Selected as the Production Machine Learning Engine</strong> powering the continuous rating and auction valuation platform.</p>
        <p><span class="pill pill-why">WHY</span> Effortlessly models complex non-linear boundary conditions (e.g. death overs strike rate &gt; 180 yields an exponential win surge; death economy &lt; 7.5 dominates win equity). Immune to multicollinearity and monotonic scaling.</p>
        <p><span class="pill pill-how">HOW</span> <code>RandomForestRegressor(n_estimators=100, max_features='sqrt', random_state=42)</code>. <strong>Benchmark Winner: Test R&sup2; = 0.9789, 5-Fold CV R&sup2; = 0.9528 &plusmn; 0.0094, RMSE = 1.2562, MAE = 0.9024.</strong></p>
    </div>

    <div class="item-block">
        <div class="item-title">3.5 Multi-Layer Perceptron (MLP) Regressor</div>
        <p><span class="pill pill-what">WHAT</span> Deep artificial neural network with input layer (29 units), two hidden layers (64, 32 units) with ReLU activations, and linear output.</p>
        <p><span class="pill pill-when">WHEN</span> Deep learning tabular regression benchmark.</p>
        <p><span class="pill pill-why">WHY</span> Benchmarked to test whether deep representation learning beats tree ensembles on sports tabular data.</p>
        <p><span class="pill pill-how">HOW</span> Adam optimizer (&eta;=0.001), ReLU, early stopping. <strong>Test R&sup2; = 0.5495, 5-Fold CV R&sup2; = 0.4503, RMSE = 5.8058.</strong> Proved that without tabular inductive bias, neural nets struggle on N&approx;600 sample tabular datasets.</p>
    </div>

    <div class="page-break"></div>

    <h2>4. Supervised Multi-Class Classification Framework</h2>

    <div class="item-block">
        <div class="item-title">4.1 K-Nearest Neighbors (KNN) &mdash; [TOP CLASSIFIER]</div>
        <p><span class="pill pill-what">WHAT</span> Instance-based non-parametric classifier identifying the <em>k</em> closest training points under Euclidean distance in standardized space and predicting class by majority vote: &ycirc; = argmax<sub>c</sub> &sum; I(y<sub>i</sub> = c).</p>
        <p><span class="pill pill-when">WHEN</span> Production talent tier classification engine.</p>
        <p><span class="pill pill-why">WHY</span> Sports scouting is inherently comparative (&ldquo;which historical players does this athlete resemble?&rdquo;). In standardized multi-metric space, players of identical talent naturally congregate in localized neighborhoods.</p>
        <p><span class="pill pill-how">HOW</span> Configured with k=5, Euclidean metric, uniform weights. <strong>Classification Winner: 95.16% Test Accuracy, 5-Fold CV Accuracy = 90.30% &plusmn; 0.0221, Macro Precision = 0.9601, Macro Recall = 0.9421, Macro F1 = 0.9501.</strong></p>
    </div>

    <div class="item-block">
        <div class="item-title">4.2 Random Forest Classifier</div>
        <p><span class="pill pill-what">WHAT</span> Ensemble of 100 classification trees splitting nodes to minimize Gini Impurity: I<sub>G</sub>(t) = 1 - &sum; p(c|t)&sup2;.</p>
        <p><span class="pill pill-when">WHEN</span> Primary competing classifier and feature importance attribution engine.</p>
        <p><span class="pill pill-why">WHY</span> Provides high classification stability, handles non-linear boundaries, and produces Gini feature importance rankings.</p>
        <p><span class="pill pill-how">HOW</span> 100 trees, Gini criterion, max_features='sqrt'. <strong>Test Accuracy = 94.35%, 5-Fold CV Accuracy = 94.75% &plusmn; 0.0142, Macro Precision = 0.9496, Macro Recall = 0.9426, Macro F1 = 0.9459.</strong></p>
    </div>

    <div class="item-block">
        <div class="item-title">4.3 Logistic Regression (Multinomial Softmax)</div>
        <p><span class="pill pill-what">WHAT</span> Linear probabilistic classifier estimating class probabilities via the Softmax function: P(Y=c|x) = exp(&beta;<sub>c</sub><sup>T</sup>x) / &sum; exp(&beta;<sub>j</sub><sup>T</sup>x), trained with Cross-Entropy loss.</p>
        <p><span class="pill pill-when">WHEN</span> Parametric probabilistic baseline for talent tier estimation.</p>
        <p><span class="pill pill-why">WHY</span> Outputs calibrated class probabilities for auction risk quantification.</p>
        <p><span class="pill pill-how">HOW</span> Multinomial softmax, L2 regularization, lbfgs solver. <strong>Test Accuracy = 87.90%, 5-Fold CV Accuracy = 92.53% &plusmn; 0.0180, Macro F1 = 0.8853.</strong></p>
    </div>

    <div class="item-block">
        <div class="item-title">4.4 Decision Tree Classifier (CART)</div>
        <p><span class="pill pill-what">WHAT</span> Hierarchical orthogonal binary decision tree maximizing information gain (&Delta;I<sub>G</sub>) at each split.</p>
        <p><span class="pill pill-when">WHEN</span> White-box rule-based benchmark.</p>
        <p><span class="pill pill-why">WHY</span> Offers transparent &ldquo;if-then&rdquo; decision pathways for franchise coaching staff.</p>
        <p><span class="pill pill-how">HOW</span> Max depth 6, Gini splitting. <strong>Test Accuracy = 87.90%, 5-Fold CV Accuracy = 89.90% &plusmn; 0.0195, Macro F1 = 0.8812.</strong></p>
    </div>

    <div class="item-block">
        <div class="item-title">4.5 Multi-Layer Perceptron (MLP) Classifier</div>
        <p><span class="pill pill-what">WHAT</span> Deep neural network with (64, 32) hidden layers, ReLU, and 3-unit Softmax output layer.</p>
        <p><span class="pill pill-when">WHEN</span> Deep learning classification benchmark.</p>
        <p><span class="pill pill-why">WHY</span> Tests whether non-linear neural representations improve tier separation.</p>
        <p><span class="pill pill-how">HOW</span> Adam optimizer, early stopping. <strong>Test Accuracy = 92.74%, 5-Fold CV Accuracy = 81.41% &plusmn; 0.0384, Macro F1 = 0.9287.</strong></p>
    </div>

    <h2>5. Unsupervised Tactical Archetypes (Clustering)</h2>

    <div class="item-block">
        <div class="item-title">5.1 K-Means Clustering (k=5 Archetypes)</div>
        <p><span class="pill pill-what">WHAT</span> Centroid-based partitioning algorithm segmenting 619 cricketers into <em>k</em> disjoint clusters to minimize Within-Cluster Sum of Squares (Inertia): argmin &sum; ||x - &mu;<sub>i</sub>||&sup2;.</p>
        <p><span class="pill pill-when">WHEN</span> Implemented in Tab 4 of <code>app.py</code> to uncover natural playing styles beyond nominal roles.</p>
        <p><span class="pill pill-why">WHY</span> Traditional labels (&ldquo;Batter&rdquo; vs &ldquo;Bowler&rdquo;) obscure tactical specialization. Franchises require distinct archetypes (anchors vs death finishers) to optimize squad balance.</p>
        <p><span class="pill pill-how">HOW</span> Lloyd's algorithm with k-means++ initialization, 15 restarts, trained on 11 key metrics. Discovered 5 archetypes:
           <em>Cluster 0: Top-Order Anchor</em> (Kohli, Warner); <em>Cluster 1: Death Pace Spearhead</em> (Bumrah, Malinga); <em>Cluster 2: Explosive Death Finisher</em> (Russell, Klaasen); <em>Cluster 3: Mystery Spin Maestro</em> (Narine, Rashid Khan); <em>Cluster 4: Dual All-Rounder</em> (Jadeja, Hardik Pandya).
        </p>
    </div>

    <div class="item-block">
        <div class="item-title">5.2 Elbow Method &amp; Silhouette Analysis</div>
        <p><span class="pill pill-what">WHAT</span> Quantitative cluster validation: Elbow method detects inertia inflection; Silhouette Coefficient measures sample cohesion vs separation: s = (b - a) / max(a, b) &isin; [-1, 1].</p>
        <p><span class="pill pill-when">WHEN</span> Cluster optimization across k &isin; [2, 7].</p>
        <p><span class="pill pill-why">WHY</span> Replaces subjective human guesses with objective mathematical validation.</p>
        <p><span class="pill pill-how">HOW</span> Evaluated in <code>train_cricket_models.py</code>; k=5 achieved optimal silhouette score (s=0.285) with distinct sports operational profiles.</p>
    </div>

    <h2>6. Dimensionality Reduction (Principal Component Analysis - PCA)</h2>
    <p><span class="pill pill-what">WHAT</span> Unsupervised orthogonal linear transformation projecting 29-dimensional standardized data onto eigenvectors of the sample covariance matrix &Sigma; = (1/n) X<sup>T</sup>X ordered by decreasing eigenvalue &lambda;<sub>i</sub>.</p>
    <p><span class="pill pill-when">WHEN</span> Production 2D Latent Tactical Map in Tab 4 of <code>app.py</code> mapping all 619 cricketers.</p>
    <p><span class="pill pill-why">WHY</span> Human executives cannot visualize 29 dimensions. PCA compresses the space into a 2D Cartesian plane while preserving 60.79% of total variance.</p>
    <p><span class="pill pill-how">HOW</span> <code>PCA(n_components=2)</code> fitted on standardized features:
       <strong>PC1 (T20 Match Impact &amp; Volume, 42.2% variance)</strong> captures career impact and clutch awards; <strong>PC2 (Batting vs Bowling Bias, 18.6% variance)</strong> cleanly separates specialist bowlers (negative axis) from top-order batters (positive axis).</p>

    <h2>7. Model Evaluation, Validation &amp; Governance Suite</h2>
    <div class="item-block">
        <p><strong>Stratified Train/Test Split (80/20):</strong> Preserves exact class ratios across Developing, Core, and Elite tiers, preventing minority elite talent from being underrepresented.</p>
        <p><strong>5-Fold Cross Validation:</strong> Evaluates models across 5 rotating folds, reporting Mean &plusmn; Std Dev to verify generalizability and guard against overfitting.</p>
        <p><strong>Quantitative Metrics Suite:</strong> R&sup2; (variance explained), RMSE (quadratic error penalty), MAE (expected error), 3&times;3 Confusion Matrix, and Macro-averaged Precision, Recall, and F1-Score.</p>
        <p><strong>Gini Impurity Feature Importance Attribution:</strong> Quantifies total decrease in node impurity brought by each feature across all trees in the Random Forest. Proves that phase-specific impact (death overs strike rate) and clutch match-winning capability dominate valuation over nominal career averages.</p>
    </div>

    <h2>8. Master Machine Learning Reference Matrix</h2>
    <table>
        <thead>
            <tr>
                <th>ML Topic / Algorithm</th>
                <th>WHAT It Is</th>
                <th>WHEN It Is Used</th>
                <th>WHY It Was Chosen</th>
                <th>HOW It Is Implemented</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><strong>One-Hot Encoding</strong></td>
                <td>Binary indicator mapping for categories</td>
                <td>Preprocessing pipeline before model ingestion</td>
                <td>Eliminates artificial ordinal bias; drop_first=True prevents collinearity</td>
                <td><code>pd.get_dummies(..., drop_first=True)</code> creating 4 binary columns</td>
            </tr>
            <tr>
                <td><strong>StandardScaler (Z-Score)</strong></td>
                <td>Rescales variables to &mu;=0, &sigma;=1</td>
                <td>Immediately after train/test split; fitted on X_train only</td>
                <td>Prevents scale dominance in Euclidean metrics (KNN, K-Means, PCA)</td>
                <td>z = (x - &mu;)/&sigma;, serialized to <code>models/scaler.joblib</code></td>
            </tr>
            <tr>
                <td><strong>Composite Indices</strong></td>
                <td>Nonlinear rate-based performance synthesizers</td>
                <td>Data aggregation (src/) &amp; profile simulation (app.py)</td>
                <td>Neutralizes tenure bias; rewards phase-specific death impact &amp; clutch wins</td>
                <td>Formulations weighting boundaries, death strike rate, awards</td>
            </tr>
            <tr>
                <td><strong>Linear Regression (OLS)</strong></td>
                <td>Minimizes RSS: min ||y - X&beta;||&sup2;</td>
                <td>Baseline continuous rating estimation</td>
                <td>Provides benchmark coefficient interpretability</td>
                <td>Normal equation; Test R&sup2; = 0.9490, RMSE = 1.9534</td>
            </tr>
            <tr>
                <td><strong>Ridge Regression (L2)</strong></td>
                <td>Regularized OLS adding &alpha; ||&beta;||&sup2;</td>
                <td>Benchmark regression to mitigate multicollinearity</td>
                <td>Shrinks collinear coefficients smoothly, reducing variance</td>
                <td>&beta; = (X<sup>T</sup>X + &alpha;I)<sup>-1</sup> X<sup>T</sup>y; Test R&sup2; = 0.9491, 5-Fold = 0.9000</td>
            </tr>
            <tr>
                <td><strong>Polynomial Regression</strong></td>
                <td>Expands inputs to degree-2 interaction terms</td>
                <td>Investigating non-linear skill synergy</td>
                <td>Captures multiplicative value (e.g. Strike Rate &times; Boundary %)</td>
                <td><code>PolynomialFeatures(degree=2)</code>; Test R&sup2; = 0.9112, RMSE = 2.5779</td>
            </tr>
            <tr>
                <td><strong>Random Forest Regressor</strong></td>
                <td>Bagged ensemble of 100 de-correlated trees</td>
                <td><strong>Production Continuous Rating Engine</strong></td>
                <td>Handles non-linear cricket thresholds and outliers without overfitting</td>
                <td>100 trees, MSE split; <strong>Winner: R&sup2; = 0.9789, RMSE = 1.2562</strong></td>
            </tr>
            <tr>
                <td><strong>MLP Regressor</strong></td>
                <td>Deep feedforward neural network with ReLU</td>
                <td>Deep learning tabular benchmark</td>
                <td>Evaluates if deep representation learning beats tree ensembles</td>
                <td>Layers (64, 32), Adam, early stopping; Test R&sup2; = 0.5495, RMSE = 5.8058</td>
            </tr>
            <tr>
                <td><strong>K-Nearest Neighbors (KNN)</strong></td>
                <td>Instance-based majority vote among k closest peers</td>
                <td><strong>Production Talent Tier Classifier</strong></td>
                <td>Sports scouting relies on historical peer comparisons in skill space</td>
                <td>k=5, Euclidean metric; <strong>Winner: 95.16% Acc, Macro F1 = 0.9501</strong></td>
            </tr>
            <tr>
                <td><strong>Random Forest Classifier</strong></td>
                <td>Ensemble of 100 trees voting via Gini split</td>
                <td>Primary competing classifier &amp; feature attribution</td>
                <td>Provides calibrated class probabilities and Gini feature importances</td>
                <td>100 trees, Gini split; <strong>94.35% Acc, 5-Fold CV Acc = 94.75%</strong></td>
            </tr>
            <tr>
                <td><strong>Logistic Regression</strong></td>
                <td>Multinomial softmax regression with Cross-Entropy</td>
                <td>Probabilistic talent classification baseline</td>
                <td>Direct posterior probability estimates P(y=c|x) for auction risk analysis</td>
                <td>Multinomial softmax with L2 penalty; Test Accuracy = 87.90%</td>
            </tr>
            <tr>
                <td><strong>Decision Tree (CART)</strong></td>
                <td>Recursive binary partitioning minimizing Gini</td>
                <td>White-box rule-based benchmark</td>
                <td>Transparent &ldquo;if-then&rdquo; decision pathways for franchise coaching staff</td>
                <td>Max depth 6, Gini splitting; Test Accuracy = 87.90%</td>
            </tr>
            <tr>
                <td><strong>MLP Classifier</strong></td>
                <td>Deep neural network with Softmax output layer</td>
                <td>Deep learning classification benchmark</td>
                <td>Tests whether non-linear latent layers separate boundary talent tiers</td>
                <td>Layers (64, 32), ReLU, Adam, Softmax; Test Accuracy = 92.74%</td>
            </tr>
            <tr>
                <td><strong>K-Means Clustering</strong></td>
                <td>Unsupervised partition minimizing WCSS / inertia</td>
                <td>Tab 4 Tactical Archetypes &amp; Roster Balance</td>
                <td>Discovers real playing styles beyond simplistic nominal roles</td>
                <td>Lloyd's algorithm (k=5, k-means++); Discovered 5 franchise archetypes</td>
            </tr>
            <tr>
                <td><strong>Elbow &amp; Silhouette</strong></td>
                <td>Quantitative cluster validation criteria</td>
                <td>Cluster count optimization across k &isin; [2, 7]</td>
                <td>Replaces subjective human bias with objective mathematical validation</td>
                <td>Evaluated inertia inflection and silhouette score (s=0.285 at k=5)</td>
            </tr>
            <tr>
                <td><strong>PCA</strong></td>
                <td>Orthogonal linear projection onto variance axes</td>
                <td>Tab 4 2D Latent Tactical Map</td>
                <td>Compresses 29 dimensions into 2D Cartesian plane for human visualization</td>
                <td>SVD of covariance matrix; <strong>Top 2 components explain 60.79% variance</strong></td>
            </tr>
            <tr>
                <td><strong>Stratified Split (80/20)</strong></td>
                <td>Preserves class distribution in train/test splits</td>
                <td>Foundation of training pipeline</td>
                <td>Prevents minority elite talent from being underrepresented in test sets</td>
                <td><code>train_test_split(..., stratify=y_clf)</code> partitioning 495 train / 124 test</td>
            </tr>
            <tr>
                <td><strong>5-Fold Cross Validation</strong></td>
                <td>Resampling on 5 rotating folds (Mean &plusmn; Std)</td>
                <td>Cross-validation across all 10 models</td>
                <td>Proves models generalize and diagnoses variance/overfitting</td>
                <td>KFold and StratifiedKFold reporting mean and standard deviation</td>
            </tr>
            <tr>
                <td><strong>Gini Feature Importance</strong></td>
                <td>Total reduction in node impurity brought by feature</td>
                <td>Explainable AI in Tab 6 of <code>app.py</code></td>
                <td>Transparently justifies to franchise owners what drives the AI valuation</td>
                <td>Extracted from Random Forest trees; reveals clutch and death strike rate dominance</td>
            </tr>
        </tbody>
    </table>

</body>
</html>
"""
    with open(HTML_PATH, "w") as f:
        f.write(html_template)
    print(f"Written: {HTML_PATH}")

if __name__ == "__main__":
    generate_ml_guide()
