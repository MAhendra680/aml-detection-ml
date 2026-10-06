# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Sample → Gold, then benchmark 5 models with MLflow
# MAGIC
# MAGIC 1. **Smart sample:** keep every laundering transaction and add 2,000,000 random normal ones, as in the dissertation.
# MAGIC 2. **Split:** stratified 60 / 20 / 20 train / validation / test.
# MAGIC 3. **Train** Logistic Regression, Decision Tree, Random Forest, XGBoost and LightGBM with the dissertation's settings.
# MAGIC 4. **Recall-first threshold:** the highest threshold that still catches at least 75% of laundering.
# MAGIC 5. **Log everything to MLflow** (one parent run, one child run per model) and save the champion.
# MAGIC
# MAGIC > **Change from the Colab version.** Colab picked the 75%-recall threshold and the champion on the **test** set,
# MAGIC > so the test score was no longer an unbiased estimate. Here both are chosen on the **validation** set, which
# MAGIC > Colab created but never used, and the test set is only scored once.

# COMMAND ----------

# MAGIC %pip install xgboost lightgbm --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "aml", "Catalog")
dbutils.widgets.text("n_negatives", "2000000", "Normal transactions to sample")
dbutils.widgets.text("target_recall", "0.75", "Target recall")
dbutils.widgets.dropdown("include_interbank", "false", ["false", "true"], "Add is_interbank feature")

CATALOG = dbutils.widgets.get("catalog")
N_NEG = int(dbutils.widgets.get("n_negatives"))
TARGET_RECALL = float(dbutils.widgets.get("target_recall"))
SEED = 42

FEATURES = ["amount_paid", "sender_24h_count", "sender_24h_volume", "amt_vs_7d_avg",
            "unique_receivers_24h", "is_round_1000", "rapid_exit_risk",
            "pay_ach", "pay_bitcoin", "pay_cash", "pay_cheque", "pay_credit_card",
            "pay_reinvestment", "pay_wire"]                 # the dissertation's 14 features
if dbutils.widgets.get("include_interbank") == "true":
    FEATURES.append("is_interbank")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1 · Smart sample → `gold.training_set`

# COMMAND ----------

from pyspark.sql import functions as F

silver = spark.table(f"{CATALOG}.silver.features")
pos = silver.filter("label = 1")
neg_all = silver.filter("label = 0")
n_neg_all = neg_all.count()

# Oversample slightly, then take exactly N_NEG in a seeded random order
neg = (neg_all.sample(fraction=min(1.0, N_NEG * 1.02 / n_neg_all), seed=SEED)
              .orderBy(F.rand(SEED)).limit(N_NEG))

gold = pos.unionByName(neg).select("txn_id", "typology", "label", *FEATURES)
gold.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.gold.training_set")

gold = spark.table(f"{CATALOG}.gold.training_set")
display(gold.groupBy("label").count())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 · Into pandas and split 60 / 20 / 20
# MAGIC About 2M rows × 15 columns fits comfortably in memory, so the models train in scikit-learn as in the dissertation.

# COMMAND ----------

import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

pdf = gold.toPandas().sort_values("txn_id").reset_index(drop=True)   # fixed order → reproducible split
X = pdf[FEATURES].astype("float64").fillna(0)
y = pdf["label"].astype(int)

X_train, X_rem, y_train, y_rem = train_test_split(X, y, test_size=0.4, stratify=y, random_state=SEED)
X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.5, stratify=y_rem, random_state=SEED)
print(f"train {len(X_train):,} | val {len(X_val):,} | test {len(X_test):,} | positive rate {y.mean():.4%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 · Models and metrics

# COMMAND ----------

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (average_precision_score, roc_auc_score, precision_recall_curve,
                             precision_score, recall_score, confusion_matrix)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

scale_w = (len(y_train) - y_train.sum()) / max(y_train.sum(), 1)

MODELS = {
    # Scaled here; the Colab version fed raw amounts (millions) straight into LR, which handicapped it
    "Logistic Regression": make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000)),
    "Decision Tree": DecisionTreeClassifier(class_weight="balanced", max_depth=12, random_state=SEED),
    "Random Forest": RandomForestClassifier(class_weight="balanced", n_estimators=100, n_jobs=-1, random_state=SEED),
    "XGBoost": XGBClassifier(n_estimators=150, max_depth=6, scale_pos_weight=scale_w,
                             tree_method="hist", random_state=SEED, n_jobs=-1),
    "LightGBM": LGBMClassifier(n_estimators=150, scale_pos_weight=scale_w, importance_type="gain",
                               random_state=SEED, verbose=-1),
}

def threshold_for_recall(y_true, p, target):
    """Highest threshold whose recall is still >= target (best precision at that recall)."""
    _, rec, thr = precision_recall_curve(y_true, p)
    ok = np.where(rec[:-1] >= target)[0]
    return float(thr[ok[-1]]) if len(ok) else 0.5

def evaluate(y_true, p, thr, prefix):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    order = np.argsort(-p)
    yt = np.asarray(y_true)[order]
    m = {
        f"{prefix}_auprc": average_precision_score(y_true, p),
        f"{prefix}_roc_auc": roc_auc_score(y_true, p),
        f"{prefix}_recall": recall_score(y_true, pred, zero_division=0),
        f"{prefix}_precision": precision_score(y_true, pred, zero_division=0),
        f"{prefix}_alerts": int(tp + fp), f"{prefix}_tp": int(tp), f"{prefix}_fp": int(fp), f"{prefix}_fn": int(fn),
    }
    for k in (100, 1000, 5000):
        m[f"{prefix}_precision_at_{k}"] = float(yt[:k].mean()) if len(yt) >= k else float("nan")
    return m

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4 · Train and track with MLflow
# MAGIC Results appear in the **Experiments** page (left sidebar) under `aml-detection`. Tick the child runs and
# MAGIC click **Compare** to get a side-by-side chart of the five models.

# COMMAND ----------

import mlflow, mlflow.sklearn, time

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/aml-detection")

results, probs_test, run_ids = [], {}, {}
with mlflow.start_run(run_name="benchmark") as parent:
    mlflow.log_params({"n_features": len(FEATURES), "features": ",".join(FEATURES),
                       "n_train": len(X_train), "n_val": len(X_val), "n_test": len(X_test),
                       "positive_rate": round(float(y.mean()), 6), "target_recall": TARGET_RECALL,
                       "split": "stratified 60/20/20", "threshold_chosen_on": "validation"})
    for name, model in MODELS.items():
        with mlflow.start_run(run_name=name, nested=True) as run:
            t0 = time.time()
            model.fit(X_train, y_train)
            train_s = time.time() - t0

            p_val = model.predict_proba(X_val)[:, 1]
            thr = threshold_for_recall(y_val, p_val, TARGET_RECALL)
            p_test = model.predict_proba(X_test)[:, 1]

            metrics = {**evaluate(y_val, p_val, thr, "val"), **evaluate(y_test, p_test, thr, "test"),
                       "threshold": thr, "train_seconds": train_s}
            mlflow.log_params({k: v for k, v in model.get_params().items()
                               if isinstance(v, (int, float, str, bool)) and v is not None})
            mlflow.log_metrics({k: v for k, v in metrics.items() if v == v})   # skip NaN
            mlflow.sklearn.log_model(model, "model", input_example=X_train.head(5),
                                     serialization_format="cloudpickle")   # readable on old and new MLflow

            results.append({"model": name, "run_id": run.info.run_id, **metrics})
            probs_test[name] = p_test
            print(f"{name:<20} val AUPRC {metrics['val_auprc']:.4f} | test AUPRC {metrics['test_auprc']:.4f} "
                  f"| test recall {metrics['test_recall']:.2%} | test precision {metrics['test_precision']:.2%}")

    bench = pd.DataFrame(results).sort_values("val_auprc", ascending=False).reset_index(drop=True)
    mlflow.log_table(bench.drop(columns=["run_id"]), "benchmark.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5 · Benchmark table (champion = best **validation** AUPRC)

# COMMAND ----------

show_cols = ["model", "val_auprc", "test_auprc", "test_roc_auc", "threshold", "test_recall", "test_precision",
             "test_alerts", "test_fp", "test_precision_at_100", "test_precision_at_1000", "test_precision_at_5000"]
display(bench[show_cols].round(4))

spark.createDataFrame(bench[show_cols + ["run_id"]]).write.mode("overwrite") \
     .option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.gold.benchmark_results")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6 · Save the champion and its scored test set for notebooks 04 and 05

# COMMAND ----------

champ = bench.iloc[0]
print(f"Champion: {champ['model']}  (run {champ['run_id']}, threshold {champ['threshold']:.4f})")

test_out = pdf.loc[X_test.index, ["txn_id", "typology", "label", *FEATURES]].copy()
test_out["prob"] = probs_test[champ["model"]]
spark.createDataFrame(test_out).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{CATALOG}.gold.test_predictions")

spark.createDataFrame([(champ["model"], champ["run_id"], float(champ["threshold"]), ",".join(FEATURES))],
                      "model string, run_id string, threshold double, features string") \
     .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.gold.champion")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Optional · Register the champion in Unity Catalog
# MAGIC Uncomment to version the model as `<catalog>.gold.aml_champion`. It then appears under **Catalog → Models**.

# COMMAND ----------

# mlflow.set_registry_uri("databricks-uc")
# mlflow.register_model(f"runs:/{champ['run_id']}/model", f"{CATALOG}.gold.aml_champion")
