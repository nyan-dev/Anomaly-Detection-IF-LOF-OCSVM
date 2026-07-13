# Baseline Comparison of IF, LOF, and OCSVM on Medicare Claims

> **Role in the research series:** This repo is the **baseline algorithm comparison** project.  
> For the full end-to-end pipeline see [`Anomaly-Detection-Medicare-Claims`](https://github.com/nyan-dev/Anomaly-Detection-Medicare-Claims) · For the advanced ensemble extension see [`Anomaly-Detection-Ensemble`](https://github.com/nyan-dev/Anomaly-Detection-Ensemble)

---

## Overview

This project compares three classic unsupervised anomaly detection algorithms — **Isolation Forest (IF)**, **Local Outlier Factor (LOF)**, and **One-Class SVM (OCSVM)** — on synthetic Medicare outpatient claims data. It serves as the foundational baseline study for a Master's thesis on unsupervised anomaly detection in healthcare billing.

The goal is to systematically evaluate how each algorithm performs under both controlled evaluation (semi-synthetic anomaly injection) and real-world conditions (hyperparameter stability analysis), and to identify which algorithm combinations are best suited for ensemble consensus detection.

---

## Problem Statement

Healthcare billing fraud and anomalous claim patterns are difficult to detect due to:
- No ground-truth labels in real-world claims data
- High class imbalance (anomalies are rare, ~1–5%)
- Heterogeneous feature distributions (monetary, temporal, provider-level)

This project addresses these challenges using **unsupervised methods** that require no labelled anomalies, evaluated via a semi-synthetic injection protocol that provides controlled ground truth.

---

## Dataset

| Property | Details |
|---|---|
| Source | [CMS DE-SynPUF 2010 Outpatient Claims](https://www.kaggle.com/datasets/kukulauren/cms-desynpuf-2010-outpatient-claims) (Kaggle) |
| Type | **Fully synthetic** — safe for public research, no real patient data |
| Size | ~175,000 claims after cleaning |
| Features used | 14 engineered features (monetary, temporal, provider-level, code counts) |

> See [`data/README.md`](data/README.md) for full dataset notes and feature engineering details.

---

## Algorithms Compared

| Algorithm | Type | Key Hyperparameters Tuned |
|---|---|---|
| Isolation Forest (IF) | Tree-based | `n_estimators`, `contamination`, `max_features` |
| Local Outlier Factor (LOF) | Density-based | `n_neighbors`, `contamination`, `metric` |
| One-Class SVM (OCSVM) | Kernel-based | `kernel`, `nu`, `gamma` |

---

## Methodology

### Phase 1 — Data Preprocessing & Feature Engineering
- Data loading, grain confirmation, beneficiary profiling
- Data dictionary and missingness analysis
- Quality checks: impossible values, invalid dates, outlier clipping
- EDA: monetary distributions, temporal patterns, provider behaviour
- 14 engineered features: log-transformed payments, claim duration, procedure/diagnosis code counts, provider anomaly rates

### Phase 2 — Semi-Synthetic Evaluation (Notebook 1)
- 80/20 train/test split with stratification
- **3% synthetic anomaly injection** into test set only (train stays clean)
- Anomaly patterns injected: inflated payments, unusual code combinations, high-frequency short-duration claims
- Models trained on clean data, evaluated on injected test set
- Metrics: **PR-AUC, ROC-AUC, Precision@K, Recall@K, F1@K**

### Phase 3 — Hyperparameter Tuning & Stability Analysis (Notebook 2)
- Grid search over hyperparameter space on the real (non-injected) dataset
- Evaluates anomaly rate stability and flagged-case overlap across configurations
- Defines the **IF ∩ OCSVM ensemble consensus** set for high-confidence anomaly identification
- Top consensus anomalies prepared as case files for domain expert review

---

## Key Results

| Model | ROC-AUC | PR-AUC | Precision@100 | Recall@100 |
|---|---|---|---|---|
| Isolation Forest | — | — | — | — |
| Local Outlier Factor | — | — | — | — |
| One-Class SVM | — | — | — | — |
| IF ∩ OCSVM Ensemble | — | — | — | — |

> Results table will be populated after full evaluation run. See notebooks for current outputs.

---

## Repository Structure

```
Anomaly-Detection-IF-LOF-OCSVM/
├── medical_fraud_analysis_analytics.ipynb   # Notebook 1: EDA, preprocessing, semi-synthetic evaluation
├── anomaly-models-tuning.ipynb              # Notebook 2: Hyperparameter tuning & ensemble consensus
├── Outputs/                                  # Saved figures, result CSVs, case files
├── Tuning/                                   # Grid search logs and stability reports
├── SUMMARIES.md                              # Plain-English summary of findings per experiment
├── data/
│   └── README.md                             # Dataset provenance, feature engineering notes
├── requirements.txt                          # Python dependencies
└── README.md
```

---

## How to Run

### 1. Clone the repository
```bash
git clone https://github.com/nyan-dev/Anomaly-Detection-IF-LOF-OCSVM.git
cd Anomaly-Detection-IF-LOF-OCSVM
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Download the dataset
Download the CMS DE-SynPUF 2010 Outpatient Claims dataset from [Kaggle](https://www.kaggle.com/datasets/kukulauren/cms-desynpuf-2010-outpatient-claims) and place the CSV file in the project root directory.

### 4. Run notebooks in order
```
1. medical_fraud_analysis_analytics.ipynb   ← Start here (EDA + evaluation)
2. anomaly-models-tuning.ipynb              ← Hyperparameter tuning + ensemble
```

---

## Engineered Features (14 total)

| Category | Features |
|---|---|
| Monetary | Log-transformed payment, reimbursement ratio, deductible amount |
| Temporal | Claim duration (days), admission month, day-of-week |
| Utilization | Procedure code count, diagnosis code count, claim frequency per beneficiary |
| Provider-level | Provider anomaly rate, claims per provider, average payment per provider |
| Structural | Inpatient/outpatient flag, primary diagnosis group |

---

## Limitations

- Dataset is **fully synthetic** — findings may not directly generalise to real Medicare claims
- Anomaly injection is semi-synthetic; real fraud patterns are more complex and diverse
- LOF does not support a native `predict()` method for unseen data (evaluation only on full dataset)
- OCSVM scales poorly with large datasets — applied on a sampled subset for tuning

---

## Related Repositories

| Repo | Role |
|---|---|
| **This repo** | Baseline algorithm comparison (IF vs LOF vs OCSVM) |
| [`Anomaly-Detection-Medicare-Claims`](https://github.com/nyan-dev/Anomaly-Detection-Medicare-Claims) | Full end-to-end case-study pipeline |
| [`Anomaly-Detection-Ensemble`](https://github.com/nyan-dev/Anomaly-Detection-Ensemble) | Advanced ensemble extension & thesis results |

---

## Citation

If you use this repository or methodology in your research, please cite:

```
Nyan Lynn Htet. (2026). Baseline Comparison of Isolation Forest, Local Outlier Factor,
and One-Class SVM for Anomaly Detection in Medicare Claims.
INTI International University. Master's Thesis Research.
```

---

## Author

**Nyan Lynn Htet** — Master's Researcher, Data Science & Machine Learning  
INTI International University, Malaysia  
[GitHub](https://github.com/nyan-dev)
