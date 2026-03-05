# Medicare Claims Anomaly Detection: IF, LOF, and OCSVM with Ensemble Consensus

Comprehensive unsupervised anomaly detection framework for healthcare billing patterns in Medicare outpatient claims using synthetic data. This repository implements three classic anomaly detection algorithms—Isolation Forest (IF), Local Outlier Factor (LOF), and One-Class SVM (OCSVM)—with systematic hyperparameter tuning and ensemble consensus validation. 

## Dataset

CMS DE-SynPUF 2010 Outpatient Claims dataset from [Kaggle](https://www.kaggle.com/datasets/kukulauren/cms-desynpuf-2010-outpatient-claims) (~175k claims after cleaning). 

## Repository Structure

**Notebook 1: `anomaly-detection-evaluation.ipynb`**  
Semi-synthetic anomaly injection experiments with controlled evaluation protocol. Injects 3% synthetic anomalies into test set and compares detection performance across IF, LOF, and OCSVM using PR-AUC, ROC-AUC, Precision@K, Recall@K, and F1@K metrics. 

**Notebook 2: `anomaly-models-tuning.ipynb`**  
Systematic hyperparameter tuning via grid search on the real (non-injected) dataset. Analyzes stability of anomaly rates and flagged cases across configurations and defines the **IF ∩ OCSVM ensemble consensus** anomaly set for expert review. 

## Research Framework

### Three-Phase Methodology

**Phase 1: Data Preprocessing & Feature Engineering**  
- Stage 1: Data loading, grain confirmation, beneficiary analysis
- Stage 2: Data dictionary creation and missingness visualization
- Stage 3: Data quality checks and cleaning (impossible values, invalid dates)
- Stage 4: Exploratory data analysis (monetary, temporal, distributional patterns)
- Stage 5: Feature engineering (log-transformed payments, claim duration, code counts, provider statistics) 

**Phase 2: Synthetic Anomaly Injection & Model Evaluation**  
- Stage 6: Train/test split with stratification
- Stage 7: Controlled synthetic anomaly injection (3% contamination in test set only)
- Stage 8: Model training on clean training data
- Stage 9: Comparative evaluation using multiple metrics
- **Ensemble consensus:** IF ∩ OCSVM intersection defines high-confidence anomaly set 

**Phase 3: Case File Generation for Expert Review**  
- Stage 10: Top-ranked consensus anomalies prepared as human-readable case files for domain expert validation with inter-rater reliability assessment 

## Key Features

- **14 engineered features** covering monetary patterns, reimbursement structure, temporal utilization, and provider-level behavior 
- **Ensemble approach:** Conservative decision-level ensemble using IF ∩ OCSVM agreement to identify high-confidence anomalies 
- **Reproducible pipeline:** Fully documented preprocessing, feature engineering, model training, and evaluation stages 
- **Public synthetic dataset:** Uses CMS DE-SynPUF data, enabling replication without privacy concerns 

## Research Objectives

1. **RO1:** Develop a reproducible unsupervised anomaly detection pipeline on public Medicare-like claims data 
2. **RO2:** Compare effectiveness and stability of IF, LOF, and OCSVM under controlled and real-world evaluation protocols 
3. **RO3:** Assess practical plausibility of detected anomalies through expert review of consensus cases 

## Requirements

- Python 3.8+
- scikit-learn, pandas, numpy, matplotlib, seaborn
- Jupyter Notebook/Lab

## Citation

If you use this repository or methodology in your research, please cite:

```
Nyan Lynn Htet. (2026). Medicare Claims Anomaly Detection: Unsupervised Machine Learning 
Framework with Ensemble Consensus. INTI International University. Master's Thesis.
```

