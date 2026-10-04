"""
Builds docs/PROJECT_ARCHITECTURE_GUIDE.md and docs/PROJECT_ARCHITECTURE_GUIDE.html
Covers every single file in the project repository with complete What, How, When, Why details.
"""

import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOCS_DIR = os.path.join(BASE_DIR, "docs")
os.makedirs(DOCS_DIR, exist_ok=True)

MD_PATH = os.path.join(DOCS_DIR, "PROJECT_ARCHITECTURE_GUIDE.md")
HTML_PATH = os.path.join(DOCS_DIR, "PROJECT_ARCHITECTURE_GUIDE.html")

def generate_project_guide():
    # --------------------------------------------------------------------------
    # 1. MARKDOWN CONTENT
    # --------------------------------------------------------------------------
    md_content = """# CricMetrics Pro: Complete File-by-File Repository Architecture Guide
### Enterprise Sports Analytics Suite | Technical Blueprint & File Encyclopedia

> **Executive Scope:** This document delivers an exhaustive, component-by-component architectural reference for every file and directory in the **CricMetrics Pro** codebase. For every single file, this specification details:
> - **WHAT** the file is (file type, primary role, interfaces, and dependencies).
> - **HOW** it works internally (functions, classes, data structures, algorithms, and execution flow).
> - **WHEN** it is executed or utilized (pipeline stage, deployment lifecycle, or runtime inference).
> - **WHY** it is necessary (system design justification, performance rationale, and organizational function).

---

## 1. Repository Directory Structure & File Hierarchy

```text
/Users/laxmanpatel/Desktop/ML
├── app.py                                   # Production Streamlit 6-in-1 War Room Web Dashboard
├── requirements.txt                         # Production Python dependencies and pinned versions
├── README.md                                # Public GitHub repository documentation and executive overview
├── FINAL_REPORT.md                          # Comprehensive academic/organizational defense and viva voce dossier
├── .gitignore                               # Git exclusions (ignoring heavy raw data, preserving models & clean CSVs)
├── .streamlit/
│   └── config.toml                          # Streamlit server config, brand gold theme tokens, error settings
├── data/
│   ├── processed/
│   │   └── cricket_players_clean.csv        # Qualified 619-player analytical dataset with 27 engineered features
│   └── raw/
│       ├── matches_2008_2024.csv            # 1,095 IPL match metadata (venues, teams, results, MoM awards)
│       └── deliveries_2008_2024.csv         # 260,920 ball-by-ball deliveries (17 seasons, 69MB local dataset)
├── models/
│   ├── scaler.joblib                        # Fitted StandardScaler object for 29 normalized features
│   ├── random_forest_regressor.joblib       # Production continuous rating model (R² = 0.9789, RMSE = 1.25)
│   ├── ridge_regression.joblib              # L2 regularized linear benchmark model (R² = 0.9491)
│   ├── linear_regression.joblib             # Ordinary Least Squares parametric baseline (R² = 0.9490)
│   ├── polynomial_regression.joblib         # Degree-2 polynomial interaction model (R² = 0.9112)
│   ├── mlp_regressor.joblib                 # Deep neural network regression benchmark (R² = 0.5495)
│   ├── knn_classifier.joblib                # Top talent tier classifier (95.16% Test Accuracy)
│   ├── random_forest_classifier.joblib      # Production tier classifier & Gini feature importance engine (94.35%)
│   ├── logistic_regression.joblib           # Multinomial softmax tier classification baseline (87.90%)
│   ├── decision_tree_classifier.joblib      # Rule-based CART tier classifier (87.90%)
│   ├── mlp_classifier.joblib                # Deep neural network tier classifier (92.74%)
│   ├── kmeans_model.joblib                  # K-Means clustering model (k=5 tactical archetypes)
│   ├── pca_model.joblib                     # Principal Component Analysis model (2D projection, 60.8% variance)
│   ├── feature_metadata.json                # Feature names, schemas, categorical encodings, tier definitions
│   └── metrics_summary.json                 # Complete benchmark evaluation metrics and confusion matrices
├── notebooks/
│   └── cricket_player_performance_analysis.ipynb # 22-cell end-to-end data science study & interactive report
├── src/
│   ├── cricket_data_pipeline.py             # Raw delivery aggregation & multi-skill feature engineering
│   ├── train_cricket_models.py              # Model training, 5-fold cross-validation & artifact serialization
│   ├── create_cricket_notebook.py           # Programmatic JSON generator for reproducible Jupyter notebook
│   └── generate_all_docs.py                 # Automated PDF documentation generator and builder
└── docs/
    ├── MACHINE_LEARNING_GUIDE.md & .pdf     # In-depth guide to all ML topics (What, When, Why, How)
    └── PROJECT_ARCHITECTURE_GUIDE.md & .pdf # File-by-file architectural blueprint (This document)
```

---

## 2. File-by-File Technical Deep Dive

### 2.1 User Interface & Application Layer

#### `app.py` (Streamlit Web Dashboard)
- **WHAT:** The enterprise web application serving as the primary interactive War Room interface for franchise executives, coaches, and sports data analysts (990+ lines of Python).
- **HOW:** Built using Streamlit, Plotly Express/Graph Objects, and Pandas. Utilizes `@st.cache_data` and `@st.cache_resource` to load `cricket_players_clean.csv` and all 13 ML models into RAM instantly. Renders 6 dedicated modules via sidebar radio navigation:
  1. *🏛️ War Room & Roster Intel:* Interactive scouting database with role, tier, experience filters, and live search.
  2. *📊 Telemetry & Factor Impact:* Normalized 0–100 hexagonal skill radar charts, Pearson correlation bar charts, and Strike Rate vs Average quadrant matrices.
  3. *⚡ AI Rating & Auction Valuation:* Interactive sliders with 1-click iconic presets (Kohli, Bumrah, Russell, Narine, Pandya), estimating performance rating, talent tier, and auction purse (₹ Crores) with a custom CSS Player Card.
  4. *🧩 Tactical Archetypes & 2D Map:* 2D PCA latent scatter projection mapping all 619 cricketers, colored by K-Means cluster, role, or tier.
  5. *🚀 What-If Franchise Simulator:* Development simulator forecasting rating and purse changes resulting from targeted death-overs training drills.
  6. *🏆 Model Benchmarks & Defense:* 10-model leaderboards (Regression $R^2$, RMSE, MAE; Classification Accuracy, F1), interactive Confusion Matrix heatmaps, and Gini feature importances.
- **WHEN:** Continuously executed in real-time when accessed locally (`http://localhost:8501`) or deployed in the cloud on Streamlit Community Cloud.
- **WHY:** Bridges complex machine learning algorithms and organizational decision-makers. Franchise directors require intuitive visual exploration rather than raw command-line outputs.

---

#### `.streamlit/config.toml` (Dashboard Theme & Server Configuration)
- **WHAT:** Streamlit's official configuration file governing theme tokens, server parameters, and client behavior.
- **HOW:** Formatted in TOML with dedicated sections:
  ```toml
  [theme]
  primaryColor = "#F59E0B"
  backgroundColor = "#0F172A"
  secondaryBackgroundColor = "#1E293B"
  textColor = "#F8FAFC"
  font = "sans serif"

  [client]
  showErrorDetails = true

  [server]
  headless = true
  enableCORS = false
  enableXsrfProtection = false
  ```
- **WHEN:** Loaded automatically by Streamlit at server startup before `app.py` begins executing.
- **WHY:** Enforces a premium enterprise dark theme (Slate Navy `#0F172A` and Franchise Gold `#F59E0B`) across all operating systems and browsers, regardless of client dark/light mode defaults. `showErrorDetails = true` ensures transparent diagnostics during live deployment.

---

### 2.2 Data Ingestion & Pipeline Layer (`src/` and `data/`)

#### `src/cricket_data_pipeline.py` (Data Pipeline Engine)
- **WHAT:** An end-to-end data processing script that aggregates raw delivery records into player-level career profiles with engineered T20 metrics.
- **HOW:** Reads `data/raw/deliveries_2008_2024.csv` (260,920 balls) and `data/raw/matches_2008_2024.csv` (1,095 matches).
  1. Computes batting statistics: `matches_played`, `total_runs`, `balls_faced`, `batting_average`, `batting_strike_rate`, `fours`, `sixes`, `boundary_run_pct`, `dot_ball_faced_pct`, `highest_score`, `thirties`, `fifties`.
  2. Isolates overs 16–20 to compute `death_overs_strike_rate`.
  3. Computes bowling statistics: `overs_bowled`, `wickets_taken`, `economy_rate`, `bowling_strike_rate`, `bowling_average`, `dot_ball_bowled_pct`, `three_plus_wickets`, and `death_overs_economy`.
  4. Merges `player_of_match_awards` from match records.
  5. Implements qualification filtering (minimum 3 matches or 20 balls faced/bowled), qualifying **619 professional cricketers**.
  6. Computes composite indices (`batting_impact_index`, `bowling_impact_index`, `clutch_match_winner_index`).
  7. Derives ground-truth continuous ratings (`overall_performance_rating` 50–95) and discrete tiers (`performance_tier_code`).
  8. Serializes output to `data/processed/cricket_players_clean.csv`.
- **WHEN:** Executed when updating the database with new match data or recalculating feature definitions (`python src/cricket_data_pipeline.py`).
- **WHY:** Aggregating 260,920 raw deliveries into structured player-level features transforms unstructured event logs into clean ML-ready tabular vectors.

---

#### `data/processed/cricket_players_clean.csv` (Cleaned Qualified Dataset)
- **WHAT:** The curated tabular dataset containing 619 qualified professional IPL cricketers and 32 columns (150 KB).
- **HOW:** Stored in CSV format with 25 numerical metrics, categorical role labels, continuous rating targets, discrete talent tiers, and estimated auction values.
- **WHEN:** Loaded at runtime by `app.py` via `@st.cache_data` and by `src/train_cricket_models.py` during training.
- **WHY:** By keeping this cleaned dataset lightweight (150 KB), GitHub commits remain fast and Streamlit Cloud deployments cold-boot in seconds without needing to parse the heavy 69 MB raw delivery logs on every container restart.

---

#### `data/raw/matches_2008_2024.csv` (IPL Match Summaries)
- **WHAT:** Tabular records of all 1,095 IPL matches played between 2008 and 2024 (218 KB).
- **HOW:** Contains match metadata: `match_id`, `season`, `city`, `date`, `team1`, `team2`, `toss_winner`, `winner`, `player_of_match`, `venue`.
- **WHEN:** Loaded by `src/cricket_data_pipeline.py` to extract `player_of_match` awards for clutch index calculation.
- **WHY:** Match-winning awards represent independent umpire/commentator assessments of high-pressure impact, providing an external ground-truth validation of clutch capability.

---

#### `data/raw/deliveries_2008_2024.csv` (Raw Ball-by-Ball Logs)
- **WHAT:** Granular delivery-by-delivery event data covering 260,920 deliveries across 17 IPL seasons (69 MB).
- **HOW:** Each row records ball-level events: `match_id`, `inning`, `batting_team`, `bowling_team`, `over`, `ball`, `batter`, `bowler`, `batsman_runs`, `extra_runs`, `is_wicket`, `dismissal_kind`.
- **WHEN:** Ingested once by `src/cricket_data_pipeline.py` to produce the cleaned player dataset. Ignored by `.gitignore` to keep cloud repositories lean.
- **WHY:** Granular ball-by-ball delivery data is mandatory to compute situational metrics (e.g. death overs strike rate in overs 16–20 and dot ball percentages) that cannot be found on standard season aggregate scorecards.

---

### 2.3 Machine Learning Pipeline & Serialized Artifacts (`models/` and `src/`)

#### `src/train_cricket_models.py` (Model Training & Evaluation Script)
- **WHAT:** The master training script that fits, cross-validates, evaluates, and serializes all 13 machine learning models and metadata (318 lines of Python).
- **HOW:**
  1. Loads `cricket_players_clean.csv`.
  2. Applies one-hot encoding with `drop_first=True`.
  3. Executes an 80/20 train/test split stratified on `performance_tier_code`.
  4. Fits `StandardScaler` on `X_train` and transforms both sets.
  5. Trains and cross-validates 5 regression models (Linear, Ridge, Polynomial, Random Forest, MLP) via 5-Fold CV.
  6. Trains and cross-validates 5 classification models (KNN, Random Forest, Logistic, Decision Tree, MLP) via Stratified 5-Fold CV.
  7. Performs K-Means clustering ($k=2 \dots 7$) with Elbow and Silhouette scoring, saving optimal $k=5$.
  8. Fits 2-component PCA on standardized features.
  9. Extracts Gini feature importance rankings from tree ensembles.
  10. Serializes all 13 model binaries via `joblib.dump()` and metrics to JSON.
- **WHEN:** Executed during the offline training and model governance cycle (`python src/train_cricket_models.py`).
- **WHY:** Centralizes all model training, hyperparameter configuration, and validation into a reproducible, audited script.

---

#### Serialized Machine Learning Artifacts in `models/`

| Artifact Path | Format | Serialized Object & Hyperparameters | Role & Purpose |
| :--- | :---: | :--- | :--- |
| `models/scaler.joblib` | Joblib Binary | `StandardScaler(mean_, var_, scale_)` | Normalizes 29 raw features to $\mu=0, \sigma=1$ during training and live inference. |
| `models/random_forest_regressor.joblib` | Joblib Binary | `RandomForestRegressor(n_estimators=100, max_features='sqrt')` | **Production Rating Engine:** Generates overall player rating ($R^2 = 0.9789$, RMSE = $1.25$). |
| `models/ridge_regression.joblib` | Joblib Binary | `Ridge(alpha=1.0)` | Regularized linear regression benchmark for collinear feature stabilization. |
| `models/linear_regression.joblib` | Joblib Binary | `LinearRegression()` | Ordinary Least Squares baseline benchmark model. |
| `models/polynomial_regression.joblib` | Joblib Binary | `LinearRegression()` fitted on `PolynomialFeatures(degree=2)` | Degree-2 non-linear interaction model capturing skill synergy. |
| `models/mlp_regressor.joblib` | Joblib Binary | `MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu')` | Deep neural network continuous regression benchmark. |
| `models/knn_classifier.joblib` | Joblib Binary | `KNeighborsClassifier(n_neighbors=5, metric='euclidean')` | **Production Tier Classifier:** Categorizes players into talent tiers ($95.16\%$ Accuracy). |
| `models/random_forest_classifier.joblib` | Joblib Binary | `RandomForestClassifier(n_estimators=100, criterion='gini')` | Production ensemble classifier & Gini feature importance attribution engine ($94.35\%$ Acc). |
| `models/logistic_regression.joblib` | Joblib Binary | `LogisticRegression(multi_class='multinomial', penalty='l2')` | Multinomial softmax probabilistic classification baseline ($87.90\%$ Acc). |
| `models/decision_tree_classifier.joblib` | Joblib Binary | `DecisionTreeClassifier(max_depth=6, criterion='gini')` | White-box rule-based talent classification benchmark ($87.90\%$ Acc). |
| `models/mlp_classifier.joblib` | Joblib Binary | `MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu')` | Deep neural network multi-class tier classification benchmark ($92.74\%$ Acc). |
| `models/kmeans_model.joblib` | Joblib Binary | Dictionary: `{'model': KMeans(n_clusters=5), 'features': cluster_features}` | Unsupervised clustering model discovering 5 tactical playing styles. |
| `models/pca_model.joblib` | Joblib Binary | `PCA(n_components=2, random_state=42)` | Dimensionality reduction engine projecting 29 metrics onto 2D latent coordinates ($60.8\%$ var). |
| `models/feature_metadata.json` | JSON | Feature lists, column orders, categorical role maps, talent tier definitions | Guarantees exact column ordering and schema consistency between training and web app. |
| `models/metrics_summary.json` | JSON | Complete numerical benchmark tables, CV scores, confusion matrices, top 10 factors | Feeds live leaderboard tables and confusion matrix heatmaps in Tab 6 of `app.py`. |

---

### 2.4 Research, Development & Automation Scripts

#### `notebooks/cricket_player_performance_analysis.ipynb` (Jupyter Research Notebook)
- **WHAT:** A verified 22-cell data science notebook documenting the complete experimental workflow from exploratory data analysis to model benchmarking.
- **HOW:** Contains executable code cells paired with Markdown explanations:
  - *Cells 1–4:* Problem formulation, data ingestion, null handling, statistical summaries.
  - *Cells 5–8:* Exploratory Data Analysis, correlation heatmaps, role distributions.
  - *Cells 9–12:* Preprocessing, one-hot encoding, stratified train/test split, standard scaling.
  - *Cells 13–15:* 5-model regression benchmark (Linear, Ridge, Poly, Random Forest, MLP) with 5-Fold CV.
  - *Cells 16–18:* 5-model classification benchmark (KNN, RF, Logistic, Decision Tree, MLP) with confusion matrices.
  - *Cells 19–20:* K-Means clustering ($k=5$), elbow method, silhouette analysis.
  - *Cells 21–22:* PCA 2D latent projection and technical defense Q&A.
- **WHEN:** Executed interactively by data scientists during offline research (`jupyter notebook`).
- **WHY:** Provides an interactive, cell-by-cell sandbox for exploratory verification, visualization, and peer review.

---

#### `src/create_cricket_notebook.py` (Reproducible Notebook Generator)
- **WHAT:** A Python utility script that programmatically synthesizes `cricket_player_performance_analysis.ipynb` as pure JSON.
- **HOW:** Assembles notebook cells, markdown narratives, and code snippets into standard `nbformat` v4 structure and saves the `.ipynb` file.
- **WHEN:** Executed whenever notebook code or narrative documentation needs automated re-generation (`python src/create_cricket_notebook.py`).
- **WHY:** Guarantees 100% reproducibility of the research notebook without manual copy-pasting across environments.

---

#### `src/generate_all_docs.py`, `src/build_ml_guide.py`, `src/build_project_guide.py` (Documentation Suite)
- **WHAT:** Automated documentation generators that construct comprehensive technical guides in Markdown, publication-styled HTML, and convert them to PDFs.
- **HOW:** Programmatically compiles technical text, tables, formulas, and CSS styles, then invokes Google Chrome headless (`--headless --disable-gpu --print-to-pdf`) to render vector-sharp PDF files.
- **WHEN:** Executed when updating system documentation or exporting stakeholder dossiers.
- **WHY:** Provides franchise executives with immediate, beautifully formatted printable PDF documentation alongside version-controlled Markdown files.

---

### 2.5 Governance, Environment & Repository Infrastructure

#### `requirements.txt` (Dependency Specifications)
- **WHAT:** Pinned production dependency manifest for the Python environment.
- **HOW:** Specifies tested versions:
  ```text
  streamlit>=1.35.0
  scikit-learn>=1.4.0
  pandas>=2.1.0
  numpy>=1.26.0
  matplotlib>=3.8.0
  seaborn>=0.13.0
  plotly>=5.18.0
  joblib>=1.3.0
  scipy>=1.11.0
  ```
- **WHEN:** Utilized by `pip install -r requirements.txt` during local virtual environment setup and automatically parsed by Streamlit Community Cloud during build.
- **WHY:** Guarantees deterministic, reproducible dependency resolution across local macOS workstations, Linux servers, and cloud containers.

---

#### `.gitignore` (Git Exclusion Manifest)
- **WHAT:** Git rules preventing untracked, heavy, or temporary files from polluting version control.
- **HOW:** Contains:
  ```text
  .venv/
  __pycache__/
  *.pyc
  .DS_Store
  .ipynb_checkpoints/
  data/raw/deliveries_*.csv
  ```
- **WHEN:** Evaluated by `git` during every `git add` and `git status` operation.
- **WHY:** Crucial for cloud repository health: ignores the 69 MB raw delivery CSV (`deliveries_2008_2024.csv`) while ensuring the cleaned analytical dataset (`cricket_players_clean.csv`) and model binaries (`models/*.joblib`) are tracked. This keeps git pushes instant and prevents hitting GitHub's 100 MB file limit.

---

#### `README.md` (Executive Repository Overview)
- **WHAT:** The public GitHub repository landing page and developer quickstart guide (118 lines of Markdown).
- **HOW:** Features project badges, problem statement formulation, executive highlights, architecture tables, experimental results leaderboards, and setup instructions.
- **WHEN:** Rendered automatically by GitHub upon visiting the repository.
- **WHY:** Serves as the primary public entry point for franchise stakeholders, open-source reviewers, and technical auditors.

---

#### `FINAL_REPORT.md` (Academic & Franchise Defense Dossier)
- **WHAT:** An exhaustive 10-section project thesis and organization defense document (250+ lines of Markdown).
- **HOW:** Structured into formal case study sections:
  1. *Executive Summary & Organizational Problem Definition*
  2. *Data Sourcing, Verification & Preprocessing Decisions*
  3. *Exploratory Telemetry & Factor Correlation Analysis*
  4. *Supervised Continuous Regression Benchmark*
  5. *Supervised Multi-Class Classification Benchmark*
  6. *Unsupervised Clustering & Tactical Archetype Discovery*
  7. *Dimensionality Reduction & 2D Latent Representation*
  8. *Streamlit Application Architecture & Decision Support Design*
  9. *Comprehensive Comparison Matrix*
  10. *Viva Voce Examination Guide: Model Answers for Technical Defense*
- **WHEN:** Used as the definitive written artifact for technical defense, academic evaluation, and franchise executive handover.
- **WHY:** Fulfills all compliance requirements of Case Study no. 102 with rigorous mathematical justification and evidence.

---

## 3. End-to-End System Execution Flow Matrix

The following matrix illustrates how every file interacts across the five operational phases of the project lifecycle:

| Component / File | Phase 1: Ingestion & Feature Engineering | Phase 2: Offline Training & Serialization | Phase 3: Research & Notebook Validation | Phase 4: Local & Cloud Web Deployment | Phase 5: Stakeholder Handover & Defense |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `data/raw/deliveries_2008_2024.csv` | **Primary Input** | - | - | - | - |
| `data/raw/matches_2008_2024.csv` | **Primary Input** | - | - | - | - |
| `src/cricket_data_pipeline.py` | **Executes ETL** | - | - | - | - |
| `data/processed/cricket_players_clean.csv` | **Output** | **Input** | **Input** | **Input (Cached)** | - |
| `src/train_cricket_models.py` | - | **Executes Training** | - | - | - |
| `models/*.joblib` & `*.json` | - | **Outputs** | **Inputs** | **Inputs (Cached)** | - |
| `notebooks/*.ipynb` | - | - | **Interactive Execution** | - | **Research Proof** |
| `src/create_cricket_notebook.py` | - | - | **Synthesizes Notebook** | - | - |
| `app.py` | - | - | - | **Executes App** | **Live Demo** |
| `.streamlit/config.toml` | - | - | - | **Configures UI** | - |
| `requirements.txt` | **Builds Env** | **Builds Env** | **Builds Env** | **Cloud Build** | - |
| `.gitignore` | **Filters Git** | **Filters Git** | **Filters Git** | **Filters Git** | - |
| `README.md` | - | - | - | - | **Executive Overview** |
| `FINAL_REPORT.md` | - | - | - | - | **Formal Dossier** |
| `docs/MACHINE_LEARNING_GUIDE.*` | - | - | - | - | **ML Encyclopedia** |
| `docs/PROJECT_ARCHITECTURE_GUIDE.*` | - | - | - | - | **System Blueprint** |
"""

    with open(MD_PATH, "w") as f:
        f.write(md_content)
    print(f"Written: {MD_PATH}")

    # --------------------------------------------------------------------------
    # 2. HTML CONTENT FOR PDF COMPILATION
    # --------------------------------------------------------------------------
    html_template = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>CricMetrics Pro: Complete File-by-File Repository Architecture Guide</title>
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

        pre {{
            background: #0F172A;
            color: #F8FAFC;
            padding: 10px 12px;
            border-radius: 6px;
            font-family: 'JetBrains Mono', monospace;
            font-size: 7.8pt;
            line-height: 1.35;
            page-break-inside: avoid;
            margin: 0.8em 0;
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
        .pill-how  {{ background: #FCE7F3; color: #9D174D; }}
        .pill-when {{ background: #FEF3C7; color: #92400E; }}
        .pill-why  {{ background: #D1FAE5; color: #065F46; }}

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
    <h1>CricMetrics Pro: Complete File-by-File Repository Architecture Guide</h1>

    <div class="callout">
        <strong>Executive Scope:</strong> This document delivers an exhaustive, component-by-component architectural reference for every file and directory in the <strong>CricMetrics Pro</strong> codebase. For every single file, this specification details <strong>WHAT</strong> the file is, <strong>HOW</strong> it works internally, <strong>WHEN</strong> it is executed or utilized, and <strong>WHY</strong> it is necessary in the organizational architecture.
    </div>

    <h2>1. Repository Directory Structure &amp; File Hierarchy</h2>
    <pre>
/Users/laxmanpatel/Desktop/ML
|-- app.py                                   # Production Streamlit 6-in-1 War Room Web Dashboard
|-- requirements.txt                         # Production Python dependencies and pinned versions
|-- README.md                                # Public GitHub repository documentation and executive overview
|-- FINAL_REPORT.md                          # Comprehensive academic/organizational defense dossier
|-- .gitignore                               # Git exclusions (ignoring heavy raw data, preserving models & clean CSVs)
|-- .streamlit/
|   `-- config.toml                          # Streamlit server config, brand gold theme tokens, error settings
|-- data/
|   |-- processed/
|   |   `-- cricket_players_clean.csv        # Qualified 619-player analytical dataset with 27 engineered features
|   `-- raw/
|       |-- matches_2008_2024.csv            # 1,095 IPL match metadata (venues, teams, results, MoM awards)
|       `-- deliveries_2008_2024.csv         # 260,920 ball-by-ball deliveries (17 seasons, 69MB local dataset)
|-- models/
|   |-- scaler.joblib                        # Fitted StandardScaler object for 29 normalized features
|   |-- random_forest_regressor.joblib       # Production continuous rating model (R2 = 0.9789, RMSE = 1.25)
|   |-- ridge_regression.joblib              # L2 regularized linear benchmark model (R2 = 0.9491)
|   |-- linear_regression.joblib             # Ordinary Least Squares parametric baseline (R2 = 0.9490)
|   |-- polynomial_regression.joblib         # Degree-2 polynomial interaction model (R2 = 0.9112)
|   |-- mlp_regressor.joblib                 # Deep neural network regression benchmark (R2 = 0.5495)
|   |-- knn_classifier.joblib                # Top talent tier classifier (95.16% Test Accuracy)
|   |-- random_forest_classifier.joblib      # Production tier classifier & Gini feature importance engine (94.35%)
|   |-- logistic_regression.joblib           # Multinomial softmax tier classification baseline (87.90%)
|   |-- decision_tree_classifier.joblib      # Rule-based CART tier classifier (87.90%)
|   |-- mlp_classifier.joblib                # Deep neural network tier classifier (92.74%)
|   |-- kmeans_model.joblib                  # K-Means clustering model (k=5 tactical archetypes)
|   |-- pca_model.joblib                     # Principal Component Analysis model (2D projection, 60.8% variance)
|   |-- feature_metadata.json                # Feature names, schemas, categorical encodings, tier definitions
|   `-- metrics_summary.json                 # Complete benchmark evaluation metrics and confusion matrices
|-- notebooks/
|   `-- cricket_player_performance_analysis.ipynb # 22-cell end-to-end data science study & interactive report
|-- src/
|   |-- cricket_data_pipeline.py             # Raw delivery aggregation & multi-skill feature engineering
|   |-- train_cricket_models.py              # Model training, 5-fold cross-validation & artifact serialization
|   |-- create_cricket_notebook.py           # Programmatic JSON generator for reproducible Jupyter notebook
|   `-- generate_all_docs.py                 # Automated PDF documentation generator and builder
`-- docs/
    |-- MACHINE_LEARNING_GUIDE.md & .pdf     # In-depth guide to all ML topics (What, When, Why, How)
    `-- PROJECT_ARCHITECTURE_GUIDE.md & .pdf # File-by-file architectural blueprint (This document)
    </pre>

    <h2>2. File-by-File Technical Deep Dive</h2>

    <h3>2.1 User Interface &amp; Application Layer</h3>

    <div class="item-block">
        <div class="item-title"><code>app.py</code> &mdash; Production Streamlit 6-in-1 War Room Dashboard</div>
        <p><span class="pill pill-what">WHAT</span> The primary web application serving as the interactive decision-support war room for franchise executives, coaches, and sports data analysts (990+ lines of Python).</p>
        <p><span class="pill pill-how">HOW</span> Uses Streamlit, Plotly Express/Graph Objects, and Pandas. Utilizes <code>@st.cache_data</code> and <code>@st.cache_resource</code> to load dataset and 13 ML models into RAM. Organizes intelligence across 6 modules: <em>War Room &amp; Roster Intel</em>, <em>Telemetry &amp; Factor Impact</em> (radar charts and correlations), <em>AI Rating &amp; Auction Valuation</em> (sliders, presets, purse calculator), <em>Tactical Archetypes &amp; 2D Map</em> (PCA coordinates and K-Means clusters), <em>What-If Simulator</em>, and <em>Model Benchmarks &amp; Defense</em>.</p>
        <p><span class="pill pill-when">WHEN</span> Continuously executed in real-time when accessed locally (<code>localhost:8501</code>) or hosted on Streamlit Community Cloud.</p>
        <p><span class="pill pill-why">WHY</span> Bridges complex machine learning algorithms and executive decision-makers. Franchise directors require real-time visual scenario analysis rather than command-line outputs.</p>
    </div>

    <div class="item-block">
        <div class="item-title"><code>.streamlit/config.toml</code> &mdash; Theme &amp; Server Configuration</div>
        <p><span class="pill pill-what">WHAT</span> Streamlit configuration file defining visual design tokens, server ports, and client error handling.</p>
        <p><span class="pill pill-how">HOW</span> Configures primaryColor to franchise gold (<code>#F59E0B</code>), backgroundColor to dark slate (<code>#0F172A</code>), secondaryBackgroundColor to navy (<code>#1E293B</code>), and sets <code>showErrorDetails = true</code>.</p>
        <p><span class="pill pill-when">WHEN</span> Automatically read by Streamlit at server initialization before app script execution.</p>
        <p><span class="pill pill-why">WHY</span> Enforces consistent enterprise dark-mode branding across all devices regardless of user browser defaults.</p>
    </div>

    <h3>2.2 Data Ingestion &amp; Pipeline Layer</h3>

    <div class="item-block">
        <div class="item-title"><code>src/cricket_data_pipeline.py</code> &mdash; Delivery Aggregator &amp; Feature Engine</div>
        <p><span class="pill pill-what">WHAT</span> Data ingestion and transformation script aggregating 260,920 raw deliveries into 619 qualified player profiles with 27 engineered metrics.</p>
        <p><span class="pill pill-how">HOW</span> Parses <code>deliveries_2008_2024.csv</code> and <code>matches_2008_2024.csv</code>. Calculates batting averages, strike rates, death overs (16–20) strike rates, bowling economies, dot ball percentages, and joins Player of the Match awards. Filters players with minimum 3 matches, computes composite indices, and serializes clean CSV output.</p>
        <p><span class="pill pill-when">WHEN</span> Executed when updating the database with new match data or re-engineering feature definitions (<code>python src/cricket_data_pipeline.py</code>).</p>
        <p><span class="pill pill-why">WHY</span> Transforms unstructured granular ball logs into clean, normalized tabular ML vectors.</p>
    </div>

    <div class="item-block">
        <div class="item-title"><code>data/processed/cricket_players_clean.csv</code> &mdash; Qualified Analytical Dataset</div>
        <p><span class="pill pill-what">WHAT</span> Structured CSV dataset of 619 qualified professional cricketers across 32 columns (150 KB).</p>
        <p><span class="pill pill-how">HOW</span> Contains 25 numerical metrics, primary role labels, performance ratings, tier codes, and auction values.</p>
        <p><span class="pill pill-when">WHEN</span> Loaded by <code>app.py</code> during web dashboard runtime and by <code>train_cricket_models.py</code> during training.</p>
        <p><span class="pill pill-why">WHY</span> Lightweight footprint (150 KB) ensures instant git pushes and sub-second container cold-boots on Streamlit Cloud.</p>
    </div>

    <div class="item-block">
        <div class="item-title"><code>data/raw/matches_2008_2024.csv</code> &mdash; Match Metadata</div>
        <p><span class="pill pill-what">WHAT</span> Match summaries for all 1,095 IPL matches played from 2008 to 2024 (218 KB).</p>
        <p><span class="pill pill-how">HOW</span> Contains match dates, venues, participating teams, toss decisions, winners, and Player of the Match awardees.</p>
        <p><span class="pill pill-when">WHEN</span> Ingested by <code>cricket_data_pipeline.py</code> to join match-winning awards.</p>
        <p><span class="pill pill-why">WHY</span> Player of the Match awards provide empirical evidence of clutch resilience under tournament pressure.</p>
    </div>

    <div class="item-block">
        <div class="item-title"><code>data/raw/deliveries_2008_2024.csv</code> &mdash; Ball-by-Ball Logs</div>
        <p><span class="pill pill-what">WHAT</span> Granular ball-by-ball delivery event logs covering 260,920 deliveries across 17 IPL seasons (69 MB).</p>
        <p><span class="pill pill-how">HOW</span> Contains over-by-over batsman runs, bowler names, dismissals, and extra runs.</p>
        <p><span class="pill pill-when">WHEN</span> Processed once by the data pipeline script. Ignored by <code>.gitignore</code> to keep git repositories fast and lean.</p>
        <p><span class="pill pill-why">WHY</span> Necessary to compute granular phase-specific metrics (death overs strike rate) not found on standard season scorecards.</p>
    </div>

    <div class="page-break"></div>

    <h3>2.3 Machine Learning Pipeline &amp; Serialized Models</h3>

    <div class="item-block">
        <div class="item-title"><code>src/train_cricket_models.py</code> &mdash; Model Training &amp; Governance Script</div>
        <p><span class="pill pill-what">WHAT</span> Training script that trains, validates with 5-fold cross-validation, and serializes all 13 ML models.</p>
        <p><span class="pill pill-how">HOW</span> Executes one-hot encoding, stratified 80/20 train/test split, standard scaling, trains 5 regressors (Linear, Ridge, Poly, RF, MLP), trains 5 classifiers (KNN, RF, Logistic, Decision Tree, MLP), trains K-Means (k=5) and PCA, and dumps all artifacts to <code>models/</code> via joblib and JSON.</p>
        <p><span class="pill pill-when">WHEN</span> Executed during offline model training cycles (<code>python src/train_cricket_models.py</code>).</p>
        <p><span class="pill pill-why">WHY</span> Centralizes model training and governance into a deterministic, reproducible, audited script.</p>
    </div>

    <table>
        <thead>
            <tr>
                <th>Artifact Path in <code>models/</code></th>
                <th>Format</th>
                <th>Serialized Object &amp; Architecture</th>
                <th>Role in CricMetrics Pro</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>models/scaler.joblib</code></td>
                <td>Joblib Binary</td>
                <td>StandardScaler(mean_, var_, scale_)</td>
                <td>Normalizes 29 raw features to &mu;=0, &sigma;=1 during training and inference.</td>
            </tr>
            <tr>
                <td><code>models/random_forest_regressor.joblib</code></td>
                <td>Joblib Binary</td>
                <td>RandomForestRegressor(100 trees, max_features='sqrt')</td>
                <td><strong>Production Rating Engine:</strong> Continuous rating prediction (R&sup2; = 0.9789, RMSE = 1.25).</td>
            </tr>
            <tr>
                <td><code>models/ridge_regression.joblib</code></td>
                <td>Joblib Binary</td>
                <td>Ridge(alpha=1.0)</td>
                <td>L2 regularized linear regression benchmark for collinear feature stabilization.</td>
            </tr>
            <tr>
                <td><code>models/linear_regression.joblib</code></td>
                <td>Joblib Binary</td>
                <td>LinearRegression()</td>
                <td>Ordinary Least Squares baseline benchmark model.</td>
            </tr>
            <tr>
                <td><code>models/polynomial_regression.joblib</code></td>
                <td>Joblib Binary</td>
                <td>LinearRegression() on PolynomialFeatures(degree=2)</td>
                <td>Degree-2 non-linear interaction model capturing skill synergy.</td>
            </tr>
            <tr>
                <td><code>models/mlp_regressor.joblib</code></td>
                <td>Joblib Binary</td>
                <td>MLPRegressor(layers=(64, 32), activation='relu')</td>
                <td>Deep neural network continuous regression benchmark.</td>
            </tr>
            <tr>
                <td><code>models/knn_classifier.joblib</code></td>
                <td>Joblib Binary</td>
                <td>KNeighborsClassifier(k=5, metric='euclidean')</td>
                <td><strong>Production Tier Classifier:</strong> Categorizes players into talent tiers (95.16% Acc).</td>
            </tr>
            <tr>
                <td><code>models/random_forest_classifier.joblib</code></td>
                <td>Joblib Binary</td>
                <td>RandomForestClassifier(100 trees, criterion='gini')</td>
                <td>Production ensemble classifier &amp; Gini feature importance engine (94.35% Acc).</td>
            </tr>
            <tr>
                <td><code>models/logistic_regression.joblib</code></td>
                <td>Joblib Binary</td>
                <td>LogisticRegression(multinomial, L2)</td>
                <td>Multinomial softmax probabilistic classification baseline (87.90% Acc).</td>
            </tr>
            <tr>
                <td><code>models/decision_tree_classifier.joblib</code></td>
                <td>Joblib Binary</td>
                <td>DecisionTreeClassifier(max_depth=6, Gini)</td>
                <td>White-box rule-based talent classification benchmark (87.90% Acc).</td>
            </tr>
            <tr>
                <td><code>models/mlp_classifier.joblib</code></td>
                <td>Joblib Binary</td>
                <td>MLPClassifier(layers=(64, 32), Softmax)</td>
                <td>Deep neural network multi-class tier classification benchmark (92.74% Acc).</td>
            </tr>
            <tr>
                <td><code>models/kmeans_model.joblib</code></td>
                <td>Joblib Binary</td>
                <td>KMeans(n_clusters=5, init='k-means++')</td>
                <td>Unsupervised clustering model discovering 5 tactical playing styles.</td>
            </tr>
            <tr>
                <td><code>models/pca_model.joblib</code></td>
                <td>Joblib Binary</td>
                <td>PCA(n_components=2, random_state=42)</td>
                <td>Dimensionality reduction engine projecting 29 metrics onto 2D latent map (60.8% var).</td>
            </tr>
            <tr>
                <td><code>models/feature_metadata.json</code></td>
                <td>JSON Text</td>
                <td>Feature schemas, column orders, tier mappings</td>
                <td>Guarantees exact column ordering and schema consistency between training and app.</td>
            </tr>
            <tr>
                <td><code>models/metrics_summary.json</code></td>
                <td>JSON Text</td>
                <td>Benchmark tables, CV scores, confusion matrices</td>
                <td>Feeds live leaderboard tables and confusion matrix heatmaps in Tab 6 of app.py.</td>
            </tr>
        </tbody>
    </table>

    <h3>2.4 Research, Development &amp; Documentation Layer</h3>

    <div class="item-block">
        <div class="item-title"><code>notebooks/cricket_player_performance_analysis.ipynb</code> &mdash; Interactive Jupyter Study</div>
        <p><span class="pill pill-what">WHAT</span> An executed 22-cell data science notebook documenting the complete experimental research study.</p>
        <p><span class="pill pill-how">HOW</span> Contains interactive cells for data cleaning, correlation heatmaps, model training, 5-fold CV benchmarks, confusion matrices, and PCA plots.</p>
        <p><span class="pill pill-when">WHEN</span> Executed interactively by data scientists during offline research and peer review.</p>
        <p><span class="pill pill-why">WHY</span> Provides transparent, step-by-step visual proof of experimental methodologies and results.</p>
    </div>

    <div class="item-block">
        <div class="item-title"><code>src/create_cricket_notebook.py</code> &mdash; Automated Notebook Generator</div>
        <p><span class="pill pill-what">WHAT</span> Python utility script programmatically generating <code>cricket_player_performance_analysis.ipynb</code> as JSON.</p>
        <p><span class="pill pill-how">HOW</span> Assembles cells, markdown narratives, and code into standard nbformat v4 structure.</p>
        <p><span class="pill pill-when">WHEN</span> Executed when research notebook narrative needs automated re-generation.</p>
        <p><span class="pill pill-why">WHY</span> Guarantees 100% notebook reproducibility without manual editing.</p>
    </div>

    <div class="item-block">
        <div class="item-title"><code>requirements.txt</code> &mdash; Python Dependency Manifest</div>
        <p><span class="pill pill-what">WHAT</span> Pinned Python package dependencies manifest (Streamlit, Scikit-Learn, Pandas, Plotly, Joblib).</p>
        <p><span class="pill pill-how">HOW</span> Read by pip during environment installation and parsed by Streamlit Community Cloud.</p>
        <p><span class="pill pill-when">WHEN</span> Used during environment setup and container build.</p>
        <p><span class="pill pill-why">WHY</span> Ensures deterministic, reproducible dependency resolution across workstations and cloud servers.</p>
    </div>

    <div class="item-block">
        <div class="item-title"><code>.gitignore</code> &mdash; Git Exclusion Rules</div>
        <p><span class="pill pill-what">WHAT</span> Git exclusion manifest preventing temporary or oversized files from polluting version control.</p>
        <p><span class="pill pill-how">HOW</span> Ignores <code>.venv/</code>, <code>__pycache__/</code>, and <code>data/raw/deliveries_*.csv</code> (69MB).</p>
        <p><span class="pill pill-when">WHEN</span> Checked by git on every commit and status command.</p>
        <p><span class="pill pill-why">WHY</span> Prevents hitting GitHub's 100MB file limit while ensuring clean datasets and trained models remain tracked.</p>
    </div>

    <div class="item-block">
        <div class="item-title"><code>README.md</code> &amp; <code>FINAL_REPORT.md</code> &mdash; Executive Documentation</div>
        <p><span class="pill pill-what">WHAT</span> Public repository landing page and comprehensive formal academic defense dossier.</p>
        <p><span class="pill pill-how">HOW</span> Contains executive highlights, quickstart commands, formal methodology proofs, and viva voce model answers.</p>
        <p><span class="pill pill-when">WHEN</span> Read by GitHub visitors, technical auditors, and organizational stakeholders.</p>
        <p><span class="pill pill-why">WHY</span> Documents the system's compliance with Case Study no. 102 and demonstrates organizational return on investment.</p>
    </div>

    <h2>3. End-to-End System Execution Flow Matrix</h2>
    <table>
        <thead>
            <tr>
                <th>Component / File</th>
                <th>Phase 1: Ingestion &amp; ETL</th>
                <th>Phase 2: Training &amp; Serialization</th>
                <th>Phase 3: Notebook Research</th>
                <th>Phase 4: Cloud Deployment</th>
                <th>Phase 5: Executive Handover</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td><code>data/raw/deliveries_2008_2024.csv</code></td>
                <td><strong>Primary Input</strong></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>data/raw/matches_2008_2024.csv</code></td>
                <td><strong>Primary Input</strong></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>src/cricket_data_pipeline.py</code></td>
                <td><strong>Executes ETL</strong></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>data/processed/cricket_players_clean.csv</code></td>
                <td><strong>Output</strong></td>
                <td><strong>Input</strong></td>
                <td><strong>Input</strong></td>
                <td><strong>Input (Cached)</strong></td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>src/train_cricket_models.py</code></td>
                <td>-</td>
                <td><strong>Executes Training</strong></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>models/*.joblib</code> &amp; <code>*.json</code></td>
                <td>-</td>
                <td><strong>Outputs</strong></td>
                <td><strong>Inputs</strong></td>
                <td><strong>Inputs (Cached)</strong></td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>notebooks/*.ipynb</code></td>
                <td>-</td>
                <td>-</td>
                <td><strong>Interactive Execution</strong></td>
                <td>-</td>
                <td><strong>Research Proof</strong></td>
            </tr>
            <tr>
                <td><code>app.py</code></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td><strong>Executes App</strong></td>
                <td><strong>Live Demo</strong></td>
            </tr>
            <tr>
                <td><code>.streamlit/config.toml</code></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td><strong>Configures UI</strong></td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>requirements.txt</code></td>
                <td><strong>Builds Env</strong></td>
                <td><strong>Builds Env</strong></td>
                <td><strong>Builds Env</strong></td>
                <td><strong>Cloud Build</strong></td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>.gitignore</code></td>
                <td><strong>Filters Git</strong></td>
                <td><strong>Filters Git</strong></td>
                <td><strong>Filters Git</strong></td>
                <td><strong>Filters Git</strong></td>
                <td>-</td>
            </tr>
            <tr>
                <td><code>README.md</code></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td><strong>Executive Overview</strong></td>
            </tr>
            <tr>
                <td><code>FINAL_REPORT.md</code></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td><strong>Formal Dossier</strong></td>
            </tr>
            <tr>
                <td><code>docs/MACHINE_LEARNING_GUIDE.*</code></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td><strong>ML Encyclopedia</strong></td>
            </tr>
            <tr>
                <td><code>docs/PROJECT_ARCHITECTURE_GUIDE.*</code></td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td>-</td>
                <td><strong>System Blueprint</strong></td>
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
    generate_project_guide()
