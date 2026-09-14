# Machine Learning for Anti-Money-Laundering (AML) Detection

**MSc Business Analytics Dissertation — Aston University (2025–2026)**
Author: Mahendra Varma Gottumukkala

A machine-learning framework for detecting money-laundering transactions, benchmarked against a traditional rule-based system and made explainable with SHAP reason codes. Built on the IBM synthetic AML dataset (~31.9M transactions) with a focus on the metrics that actually matter for financial-crime detection: **recall** (catching laundering) and **AUPRC** (ranking quality under extreme class imbalance).

> **Note on scope.** This is an academic benchmarking study on a *synthetic* dataset. It is not a deployed or production counter-fraud system. Results should be read as a methodological comparison, not as operational performance figures.

---

## Problem

Money laundering is rare, adversarial, and expensive to investigate. Two properties make it a hard machine-learning problem:

- **Extreme class imbalance** — laundering makes up under 1% of transactions, so accuracy is meaningless (a model predicting "all legitimate" scores ~99%).
- **Asymmetric costs** — a missed laundering case (false negative) is far costlier than a false alert, but every false alert consumes scarce investigator time.

The regulatory priority is therefore **high recall at a manageable alert volume**, with **explainable** alerts an investigator can act on. This project evaluates whether machine learning improves on legacy rule-based screening against those goals.

## Dataset

- **Source:** [IBM Transactions for Anti-Money Laundering (AML)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) (Kaggle), HI-Medium split.
- **Scale:** 31,898,238 transactions; 18,391 labelled laundering (~0.06%).
- **Research sample:** all 18,391 laundering transactions + 2,000,000 sampled legitimate transactions = **2,018,391 rows (0.91% positive)**, persisted to Parquet. Keeping every positive while down-sampling the negatives preserves the signal while making iteration tractable in memory.

## Approach

```
Raw transactions (32M)
        │  label from pattern file · enrich with account metadata · smart-sample
        ▼
Research sample (2.02M)  ──►  Feature engineering (FATF typologies)
        │                              │
        │                              ▼
        │                     14 behavioural risk features
        ▼                              │
Rule-based baseline  ◄─────────────────┤
        │                              ▼
        │                     5-model benchmark (recall-first, ranked by AUPRC)
        │                              │
        ▼                              ▼
   Comparison  ◄────────────  Champion + SHAP explainability
```

### Feature engineering (mapped to FATF typologies)

| Feature | Laundering typology it targets |
|---|---|
| `Sender_24h_Count`, `Sender_24h_Volume` | Velocity / **placement** |
| `Amt_vs_7d_Avg` | Deviation from baseline / **structuring** |
| `Unique_Receivers_24h` | Fan-out / **layering** (mule behaviour) |
| `Rapid_Exit_Risk` | Pass-through / rapid movement of funds |
| `Is_Round_1000` | Round-number payoff pattern |
| `Pay_*`, `Is_Interbank` | Channel and cross-institution risk |

### Modelling

Five classifiers — Logistic Regression, Decision Tree, Random Forest, XGBoost, LightGBM — trained on a stratified 60/20/20 split with class-imbalance handling (`scale_pos_weight` / `class_weight='balanced'`). For each model, the decision threshold is set to the highest value that still holds **recall ≥ 75%**, reflecting the regulatory preference for detection over precision. Models are ranked by **AUPRC**, the correct primary metric for imbalanced detection.

## Results

### Model comparison

![Comparative model performance (AUPRC)](figures/figure_4_4_model_comparison.png)

| Model | AUPRC | Recall | Precision | Threshold |
|---|---|---|---|---|
| **XGBoost (champion)** | **0.1168** | 75.0% | 8.93% | 0.690 |
| LightGBM | 0.1122 | 75.2% | 8.93% | 0.702 |
| Decision Tree | 0.1091 | 75.6% | 8.34% | 0.708 |
| Logistic Regression | 0.0557 | 75.0% | 1.05% | 0.500 |
| Random Forest | 0.0487 | 100% | 0.91% | 0.000 |

XGBoost is the champion on AUPRC. At the 75%-recall operating point its precision is **8.93%** — modest in absolute terms, which is expected for a <1% base rate, but a large improvement on the legacy baseline below.

### Machine learning vs. the rule-based baseline

![ML vs rule-based system](figures/figure_4_5_ml_vs_rbs.png)

| System | Recall | Precision | Alert volume |
|---|---|---|---|
| Traditional rule-based system | 36.4% | 1.24% | 538,390 alerts (26.7% of all transactions) |
| XGBoost (75%-recall operating point) | 75.0% | 8.93% | risk-ranked queue |

The rule-based system flags **more than a quarter of all transactions** yet catches only ~36% of laundering. The ML approach roughly **doubles recall**, and because its output is a *risk-ranked* queue, precision at the top of that queue is far higher than the rule-based system's flat 1.24%:

| Review tier | Precision @ K |
|---|---|
| Top 100 alerts | 28.0% |
| Top 1,000 alerts | 24.1% |
| Top 5,000 alerts | 14.1% |

This is the practical win for an investigations team: focus Enhanced Due Diligence on the highest-scoring cases, where roughly 1 in 4 of the top 1,000 is a true laundering case versus roughly 1 in 80 under the rule-based system.

### Explainability (SHAP)

Because regulators require automated alerts to be explainable, the champion model is paired with SHAP. Two artefacts are produced by `src/explainability.py`:

- a **global summary** ranking the behavioural drivers of risk, and
- a **per-case waterfall** that deconstructs a single detected transaction into the exact factors that raised its score — effectively the skeleton of a Suspicious Activity Report (SAR) narrative.

Top global drivers observed: `Amt_vs_7d_Avg`, `Sender_24h_Count`, `Rapid_Exit_Risk`, and payment-channel flags. Both plots are regenerated when the pipeline is run.

## Repository structure

```
aml-detection/
├── README.md
├── requirements.txt
├── LICENSE
├── src/
│   ├── data_preparation.py     # load, label, enrich, smart-sample
│   ├── feature_engineering.py  # FATF-typology behavioural features
│   ├── train_models.py         # 5-model benchmark, recall-first thresholds
│   ├── rule_based_baseline.py  # legacy RBS for comparison
│   ├── explainability.py       # SHAP global + local (SAR) plots
│   └── run_pipeline.py         # end-to-end runner
├── figures/                    # generated charts
├── results/                    # benchmark tables (CSV)
└── notebooks/                  # original Colab export (reference)
```

## Running it

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Add Kaggle credentials (kaggle.json) so kagglehub can download the dataset
#    https://www.kaggle.com/docs/api

# 3. Run the full pipeline (data prep is skipped if smart_sample.parquet exists)
python src/run_pipeline.py
```

Individual stages can also be run standalone, e.g. `python src/train_models.py`.

## Methodology notes & limitations

- **Synthetic data.** The IBM dataset is generated, not real transactions; absolute performance will not transfer directly to production data.
- **Labels from pattern indices.** Laundering labels are applied by matching row indices from the dataset's pattern file, with a 0-/1-based offset check.
- **Metric choice.** AUPRC and recall are reported rather than accuracy or ROC-AUC, both of which are misleading under this level of imbalance.
- **Precision is low by design.** Holding recall at 75% on a <1% base rate necessarily yields low precision; the operational answer is risk-ranked top-K review (Enhanced Due Diligence on the highest-scoring alerts) rather than binary blocking.

## Tech stack

Python · pandas · scikit-learn · XGBoost · LightGBM · SHAP · matplotlib
