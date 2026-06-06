# 🏆 FIFA World Cup 2026 — ML Prediction Pipeline

A end-to-end machine learning project that predicts international football team qualification for the **2026 FIFA World Cup** using structured team-level features across ranking, form, squad quality, historical pedigree, and coaching data.

---

## Project Structure

```
├── wc2026_ml_dataset.csv          # Raw dataset (100 teams × 30 features)
├── 01_eda.ipynb                   # Exploratory Data Analysis
├── 02_feature_engineering.ipynb   # Feature Engineering & Selection
├── 03_modelling.ipynb             # Modelling, Evaluation & Predictions
│
├── wc2026_features.csv            # Engineered dataset (output of notebook 02)
├── wc2026_features_scaled.csv     # StandardScaler version for linear models
├── selected_features.json         # Final feature list for notebook 03
│
├── wc2026_model_results.csv       # CV comparison across all 6 models
├── wc2026_final_predictions.csv   # Qualification probability for all 100 teams
└── wc2026_feature_importances.csv # Feature importance from best tree model
```

---

## Dataset

**File:** `wc2026_ml_dataset.csv`  
**Shape:** 100 teams × 30 columns  
**Target:** `qualified_for_wc2026` (binary: 1 = qualified, 0 = did not qualify)  
**Class balance:** 52% qualified / 48% not — near-balanced, no resampling needed

### Feature Groups

| Group | Features |
|---|---|
| Identity | country, confederation |
| Ranking | fifa_rank, fifa_rank_tier, fifa_points, elo_rating |
| Squad Quality | squad_value_total_m, players_top5_leagues, top_scorer_goals, squad_depth_score, avg_player_age |
| Coach | coach, coach_tenure_days, coach_success_rate |
| Form & Results | avg_goals_scored/conceded_last10, win/draw rates (5/10/20), home/away win rates, clean_sheet_rate, avg_opponent_elo_last10, xg_difference, unbeaten_streak |
| History | world_cup_appearances, world_cup_titles, continental_titles |

---

## Notebooks

### `01_eda.ipynb` — Exploratory Data Analysis

Covers the full landscape of the dataset before any modelling:

- Dataset overview, dtypes, and missing value audit
- Target variable distribution and class balance across confederations
- Univariate distributions for all numeric features, split by qualification status
- Bivariate analysis with Mann-Whitney U significance tests
- Full correlation matrix and multicollinearity audit (|r| > 0.85 pairs)
- Confederation deep-dive: box plots and radar profiles
- Outlier detection using IQR fence and Z-score methods
- Key findings and feature engineering hypotheses feeding into notebook 02

### `02_feature_engineering.ipynb` — Feature Engineering & Selection

Transforms the raw dataset into a model-ready feature matrix:

**15 engineered features across 5 groups:**

| Group | Features | Rationale |
|---|---|---|
| Ranking (Relative) | rank_vs_conf, elo_vs_conf, rank_ratio | Absolute FIFA rank is misleading across confederations |
| Form & Momentum | form_trajectory, form_consistency, peak_age_form | Recent trend matters more than long-run average |
| Attack / Defence | goal_ratio, defensive_reliability, clutch_factor | Win rates alone don't capture *how* teams win |
| Squad Quality | quality_adjusted_wins, team_strength_index, value_per_top_player | Composite signals reduce correlated sub-features |
| History & Coaching | historical_pedigree, experience_score, coach_efficiency | Institutional experience shapes big-tournament performance |

**Feature selection pipeline (3-step):**
1. Correlation filter — drops features with |r| > 0.92 against any other feature
2. VIF filter — drops features with Variance Inflation Factor > 15
3. Permutation importance — drops features that do not improve Random Forest ROC-AUC

**Outputs:** `wc2026_features.csv`, `wc2026_features_scaled.csv`, `selected_features.json`

### `03_modelling.ipynb` — Modelling, Evaluation & Predictions

Full modelling pipeline from baseline to explainability:

**Models evaluated (5-Fold Stratified CV):**
- Logistic Regression (L2)
- Random Forest
- XGBoost / Gradient Boosting
- Extra Trees
- SVM with RBF kernel
- K-Nearest Neighbours
- Soft-voting ensemble of the top 3 tuned models

**Tuning:** RandomizedSearchCV (60 iterations) for Random Forest and XGBoost/GBM

**Evaluation:**
- Confusion matrix, ROC curve, Precision-Recall curve
- Predicted probability distributions and calibration curve
- Error analysis — which teams were misclassified and why
- Per-confederation accuracy and AUC breakdown

**Explainability (SHAP):**
- Global feature importance (mean |SHAP|)
- Beeswarm plot showing direction and magnitude per feature
- Dependence plots for top 4 features
- Waterfall plots for individual team predictions

**Outputs:** full ranking of all 100 teams by predicted qualification probability with confidence bands (Strong YES → Strong NO)

---

## Setup & Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost shap statsmodels
```

### Running the pipeline

Run the notebooks in order:

```
01_eda.ipynb  →  02_feature_engineering.ipynb  →  03_modelling.ipynb
```

Notebook 03 will automatically fall back to inline feature engineering from the raw CSV if the output files from notebook 02 are not present.

### Python version

Python 3.8 or higher recommended.

---

## Key Results

| Metric | Score |
|---|---|
| Primary metric | ROC-AUC |
| Validation strategy | Stratified K-Fold (k=5) |
| Final model | Best of: tuned RF, tuned XGBoost, or soft ensemble |
| Predictions output | `wc2026_final_predictions.csv` |

Predictions include a `Confidence` band for each team:

| Band | Probability range |
|---|---|
| Strong YES | > 0.70 |
| Lean YES | 0.55 – 0.70 |
| Uncertain | 0.45 – 0.55 |
| Lean NO | 0.30 – 0.45 |
| Strong NO | < 0.30 |

---

## Design Decisions

**Why Stratified K-Fold over a simple hold-out?**  
With only 100 samples, a single hold-out test set introduces high variance in performance estimates. K-Fold uses all data for validation while stratification preserves the class ratio in every fold.

**Why relative ranking features?**  
A team ranked 30th in UEFA faces far stronger competition than a team ranked 30th in OFC. `rank_vs_conf` and `elo_vs_conf` capture this confederation context that raw FIFA rank cannot.

**Why log1p transforms?**  
Features like `world_cup_appearances`, `coach_tenure_days`, and `squad_value_total_m` are right-skewed with a long tail. Log transformation reduces this skew and stabilises variance for distance-based and linear models.

**Why a soft-voting ensemble?**  
On small datasets, individual models are sensitive to the specific train/val split. A soft ensemble averages out this instability, often improving generalisation without additional hyperparameter tuning.

---

*FIFA World Cup 2026 Analytics & Prediction Pipeline*
