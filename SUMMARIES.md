# Project Summaries: Anomaly Detection - Medical Claims Analysis

## Overview
This document contains all stage and phase summaries extracted from the `medical_fraud_analysis_analytics.ipynb` notebook, along with a comprehensive list of figures and tables.

---

## Phase 1: Data Preprocessing & Feature Engineering

### Stage 1 Summary: Data Loading & Grain Confirmation

**What was done:**
- Loaded CSV dataset with ~100K rows, 76 columns
- Confirmed CLM_ID uniqueness (one row = one claim)
- Confirmed SEGMENT = 1 (no multi-segment claims)
- Confirmed DESYNPUF_ID repeats (multiple claims per beneficiary)

**Key findings:**
- Grain: Claim-level (CLM_ID unique)
- No aggregation needed for modeling

**Files saved:**
- `/content/drive/MyDrive/Anomaly_Det/outputs/phase1/grain_confirmation_report.txt`

**Next step:** Data dictionary creation + missingness analysis

---

### Stage 2 Summary: Data Dictionary & Missingness

**What was done:**
- Created data dictionary with all 76 columns
- Computed missingness % for each column
- Plotted top 30 columns with most missing values

**Key findings:**
- To be filled after inspection: X columns with >50% missing
- To be filled: Key columns with 0% missing

**Files saved:**
- `/outputs/phase1/data_dictionary.csv`
- `/outputs/phase1/plots/missingness_top30.png`

**Next step:** Data cleaning (impossible values, duplicates, outliers)

---

### Stage 3 Summary: Data Quality & Cleaning

**What was done:**
- Detected impossible date combinations (THRU < FROM)
- Checked for negative payment amounts
- Applied cleaning rules (dropped invalid rows)
- Saved cleaned dataset

**Key findings:**
- [0] rows with impossible dates removed
- [587] rows with negative payments removed
- Final dataset: [174418] rows

**Files saved:**
- `/outputs/phase1/data/cleaned_data.parquet`
- `/outputs/phase1/data_cleaning_log.txt`

**Next step:** Exploratory Data Analysis (EDA)

---

### Stage 4 Summary: Exploratory Data Analysis (EDA)

**What was done:**
- Analyzed monetary distributions (normal + log scale)
- Analyzed claim duration distribution
- Analyzed temporal patterns (monthly volume)

**Key findings:**
- Payment amounts: heavy right tail (expected)
- Typical claim duration: [1.42] days
- Temporal pattern: [claim volume varies by month, showing periods of higher and lower activity throughout 2010]

**Files saved:**
- `/outputs/phase1/plots/eda_monetary_distributions.png`
- `/outputs/phase1/plots/eda_duration_distribution.png`
- `/outputs/phase1/plots/eda_monthly_volume.png`

**Next step:** Feature engineering (monetary, temporal, coding, provider features)

---

### Stage 5 Summary: Feature Engineering v1

**What was done:**
- Created 14 engineered features across 4 categories:
  - Monetary: log_payment, payment_zscore, high_cost_flag, zero_payment_flag
  - Temporal: claim_duration, month, quarter, day_of_week, weekend_flag
  - Coding: diagnosis_count, procedure_count, total_code_count
  - Provider: provider_claim_volume, provider_avg_payment, high_volume_provider_flag
- Imputed missing values with median
- Generated correlation heatmap

**Key findings:**
- Feature table: [174,418] rows x 15 features
- No NaNs remaining
- 174,418 rows and 16 columns (which includes 14 engineered features plus CLM_ID and DESYNPUF_ID)
- High correlation detected between month and quarter (0.968)

**Files saved:**
- `/outputs/phase1/data/features_v1.parquet`
- `/outputs/phase1/plots/feature_correlation_v1.png`

**Next step:** Phase 2: Train/test split + synthetic anomaly injection

---

### Phase 1 Summary: Data Preprocessing & Feature Engineering

**Key Actions:**
- Loaded raw data and confirmed its grain (CLM_ID uniqueness)
- Created a comprehensive data dictionary and analyzed missing values
- Cleaned data by identifying and removing rows with impossible values (e.g., negative payments)
- Performed Exploratory Data Analysis (EDA) on monetary, temporal, and duration features
- Engineered 14 new features (monetary, temporal, coding, provider) and assembled them into a feature table
- Analyzed feature correlations using a heatmap

**Key Results:**
- Confirmed data grain at the claim level; SEGMENT was constant
- Identified several columns with high missingness (>50%), but critical IDs and payment fields were complete
- Removed 587 rows due to negative payment amounts, resulting in 174,418 clean rows
- Observed highly skewed monetary distributions, short claim durations (median 0 days), and varying monthly claim volumes
- Generated a feature table (`features_v1`) with 174,418 rows and 16 columns (including IDs and 14 engineered features)
- Found a high correlation between 'month' and 'quarter', which was expected

---

## Phase 2: Synthetic Anomaly Injection & Model Evaluation

### Stage 6 Summary: Train/Test Split

**What was done:**
- Split feature table into 70/30 train/test
- Saved train and test sets separately
- Preserved ID columns for traceability

**Key findings:**
- Train: [122092] rows
- Test: [52326] rows

**Files saved:**
- `/outputs/phase2/data/X_train.parquet`
- `/outputs/phase2/data/X_test_clean.parquet`
- `/outputs/phase2/data/ids_train.parquet`
- `/outputs/phase2/data/ids_test.parquet`

**Next step:** Synthetic anomaly injection (3% in test set only)

---

### Stage 7 Summary: Synthetic Anomaly Injection

**What was done:**
- Injected exactly 3% anomalies into test set only
- Anomaly types: 30% monetary, 20% temporal, 30% coding, 20% duplicates
- Created ground-truth labels (y_test: 0=normal, 1=anomaly)
- Saved injection metadata for traceability

**Key findings:**
- Test set size: [52326] rows
- Anomalies injected: [1569] rows (3.00%)
- Train set: clean (no anomalies)

**Files saved:**
- `/outputs/phase2/data/X_test_injected.parquet`
- `/outputs/phase2/data/y_test.parquet`
- `/outputs/phase2/injection_metadata.json`

**Next step:** Model training (Isolation Forest, LOF, One-Class SVM)

---

### Stage 8 Summary: Model Training

**What was done:**
- Trained 3 unsupervised anomaly detection models on clean train set
- Models: Isolation Forest, LOF, One-Class SVM
- Generated anomaly scores for test set (higher = more anomalous)

**Key findings:**
- All models trained successfully
- Score ranges captured for each model

**Files saved:**
- `/outputs/phase2/models/model_if.pkl`
- `/outputs/phase2/models/model_lof.pkl`
- `/outputs/phase2/models/model_ocsvm.pkl`
- `/outputs/phase2/models/scores_if.npy`
- `/outputs/phase2/models/scores_lof.npy`
- `/outputs/phase2/models/scores_ocsvm.npy`

**Next step:** Model evaluation (PR-AUC, Precision@K, Recall@K)

---

### Stage 9 Summary: Model Evaluation

**What was done:**
- Computed evaluation metrics for all 3 models
- Metrics: PR-AUC, ROC-AUC, Precision@K, Recall@K, F1@K
- Generated PR curves comparison plot
- Created model comparison table

**Key findings:**
- Best model by PR-AUC: [LOF] (PR-AUC = 0.818)
- Best Precision@K: LOF (P@K = 0.799)
- All models meet target: PR-AUC ≥ 0.70? No (Isolation Forest's PR-AUC was 0.042)
- Other 2 models meet target: PR-AUC ≥ 0.70? Yes - LOF (PR-AUC = 0.818) and OCSVM (PR-AUC = 0.793)

**Files saved:**
- `/outputs/phase2/metrics/evaluation_metrics.csv`
- `/outputs/phase2/metrics/pr_curves_comparison.png`
- `/outputs/phase2/metrics/model_comparison_table.png`

**Next step:** Phase 3: Generate case files for expert review

---

## Phase 3: Case File Generation for Expert Review

### Stage 10 Summary: Case File Generation

**What was done:**
- Selected top 30 highest-scoring claims from best model
- Selected 10 random normal claims for comparison
- Shuffled 40 cases to blind reviewers
- Created human-readable case files with key features

**Key findings:**
- 40 case files generated (30 anomalies + 10 normals)
- Case files include: CLM_ID, payment, duration, codes, provider info, anomaly score

**Files saved:**
- `/outputs/phase3/case_files_for_review.xlsx`
- `/outputs/phase3/case_files_for_review.csv`

**Next step:**
- Expert review sessions (outside technical scope - provide files to domain experts)

---

## Final Summary: Project Complete

**All phases completed:**

✅ Phase 1: Data prep, EDA, feature engineering (15 features)

✅ Phase 2: Train/test split, 3% injection, 3 models trained, evaluation complete

✅ Phase 3: 40 case files generated for expert review

**Key outcomes:**
- Best model: LOF
- Best PR-AUC: 0.818
- Precision@K: 0.799
- Target met: PR-AUC ≥ 0.70? Yes (for LOF and OCSVM)

**Total deliverables:**
- 35-55 output files across 3 phases
- All plots, tables, metrics ready for thesis writing

**Files saved:**
- `/outputs/final_metadata.json` (complete project metadata)

**Ready for:**
- Thesis writing (Methodology, Results, Discussion chapters)
- Expert review sessions (using case files)

---

## List of Figures

1. **missingness_top30.png** - Top 30 Columns by Missing Values (Stage 2)
2. **eda_monetary_distributions.png** - CLM_PMT_AMT Distribution (normal + log scale) (Stage 4)
3. **eda_duration_distribution.png** - Claim Duration Distribution (Stage 4)
4. **eda_monthly_volume.png** - Claim Volume by Month (2010) (Stage 4)
5. **feature_correlation_v1.png** - Feature Correlation Heatmap (Stage 5)
6. **pr_curves_comparison.png** - Precision-Recall Curves Model Comparison (Stage 9)
7. **model_comparison_table.png** - Model Comparison Table as image (Stage 9)

---

## List of Tables

1. **Data Dictionary Table** (Stage 2)
   - 77 rows × 5 columns
   - Columns: column, dtype, missing_count, missing_pct, usage_category
   - File: `/outputs/phase1/data_dictionary.csv`

2. **Descriptive Statistics Table** (Stage 1)
   - Generated via `df.describe()`
   - Contains statistical summary of numeric columns

3. **Model Comparison Table** (Stage 9)
   - Metrics comparison: IF, LOF, OCSVM
   - Columns: model, PR_AUC, ROC_AUC, Precision@K, Recall@K, F1@K
   - File: `/outputs/phase2/metrics/evaluation_metrics.csv`

4. **Case Files for Review** (Stage 10)
   - 40 cases (30 anomalies + 10 normals)
   - Key columns: CASE_ID, CLM_ID, DESYNPUF_ID, log_payment, claim_duration, diagnosis_count, procedure_count, provider_claim_volume, anomaly_score, anomaly_percentile, payment_interpretation
   - Files: 
     - `/outputs/phase3/case_files_for_review.xlsx`
     - `/outputs/phase3/case_files_for_review.csv`

---

## Notes

- All file paths are relative to the base directory: `/content/drive/MyDrive/Anomaly_Det/outputs/`
- Random seed used throughout: 42
- Date completed: 2026-02-07
- Dataset source: Kaggle - CMS DE-SynPUF 2010 Outpatient Claims
