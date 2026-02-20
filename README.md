# Anomaly-Detection-IF-LOF-OCSVM
anomaly detection and comparison of IF v LOF v OCSVM in Medicare Claim Synthetic data (CMS)
-----

dataset: https://www.kaggle.com/datasets/kukulauren/cms-desynpuf-2010-outpatient-claims


tuning all 3 models: anomaly-models-tuning.ipynb


Here is the overview of the project, broken down by phase:

Phase 1: Data Preprocessing & Feature Engineering
---
**Stage 1:** Data Loading & Grain Confirmation: Loaded the raw data, confirmed that each row represents a unique claim (the 'grain'), and checked for beneficiary repetitions.<br>
Stage 2: Data Dictionary & Missingness Analysis: Created a data dictionary for all columns and visualized missing values to understand data completeness.<br>
Stage 3: Data Quality & Cleaning: Checked for impossible values (e.g., negative payments, impossible dates) and cleaned the data by removing problematic rows.<br>
Stage 4: Exploratory Data Analysis (EDA): Explored distributions of key monetary and temporal features, and looked for temporal patterns.<br>
Stage 5: Feature Engineering v1: Created new features from existing ones (e.g., log-transformed payments, claim duration, code counts, provider statistics) and checked for correlations.<br>

Phase 2: Synthetic Anomaly Injection & Model Evaluation
---
Stage 6: Train/Test Split: Divided the cleaned, engineered data into training (for model learning) and testing (for evaluation) sets.<br>
Stage 7: Synthetic Anomaly Injection: Artificially added a known percentage of anomalies (3%) into only the test set to create a ground truth for evaluation.<br>
Stage 8: Model Training: Trained three unsupervised anomaly detection models (Isolation Forest, Local Outlier Factor, One-Class SVM) on the clean training data.<br>
Stage 9: Model Evaluation: Calculated performance metrics (PR-AUC, ROC-AUC, Precision@K, Recall@K, F1@K) for each model using the injected test set and compared their effectiveness.<br>

Phase 3: Case File Generation for Expert Review
---
Stage 10: Case File Generation: Selected the top-scoring anomalous claims and some normal claims from the best-performing model to create human-readable case files for review by domain experts.<br>
This project aimed to build and evaluate anomaly detection models for healthcare claims, simulating a real-world scenario where anomalies are rare and need to be flagged for human review.

----
