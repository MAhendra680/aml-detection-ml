# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Traditional rule-based baseline vs the ML champion
# MAGIC
# MAGIC The dissertation's four legacy rules. An alert fires if any of them is true:
# MAGIC 1. Amount > 10,000 (BSA-style threshold)
# MAGIC 2. More than 3 transactions from the sender in 24h (velocity)
# MAGIC 3. Exact multiple of 1,000 (round number)
# MAGIC 4. Inter-bank transfer above 5,000
# MAGIC
# MAGIC > **Changes from the Colab version.** In Colab the velocity rule was silently skipped because the
# MAGIC > `Sender_24h_Count` column was missing, and the inter-bank rule never fired because the bank-merge failed.
# MAGIC > Both work here. The rules are scored on the **same test set** as the models, so the comparison is fair.
# MAGIC
# MAGIC The headline comparison is **false positives at the same recall**: how many fewer false alerts the model raises
# MAGIC when it catches as much laundering as the rules do. That is the defensible version of a "false-positive reduction" claim.
# MAGIC
# MAGIC *Caveat: amounts are in mixed currencies, so the 10,000 and 5,000 thresholds are nominal, as in the dissertation.*

# COMMAND ----------

dbutils.widgets.text("catalog", "aml", "Catalog")
CATALOG = dbutils.widgets.get("catalog")

import numpy as np, pandas as pd
from pyspark.sql import functions as F

champ = spark.table(f"{CATALOG}.gold.champion").first()
test = (spark.table(f"{CATALOG}.gold.test_predictions").alias("t")
        .join(spark.table(f"{CATALOG}.silver.features").select("txn_id", F.col("is_interbank").alias("_interbank")),
              "txn_id", "left")
        .toPandas())

rules = pd.DataFrame({
    "large_amount":   test.amount_paid > 10_000,
    "high_velocity":  test.sender_24h_count > 3,
    "round_number":   (test.amount_paid % 1000) == 0,
    "interbank_risk": (test._interbank == 1) & (test.amount_paid > 5_000),
})
test["rbs_alert"] = rules.any(axis=1).astype(int)

# COMMAND ----------

def summarise(name, y, pred):
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    return {"system": name, "alerts": tp + fp, "true_positives": tp, "false_positives": fp,
            "recall": tp / max(tp + fn, 1), "precision": tp / max(tp + fp, 1),
            "alert_rate": (tp + fp) / len(y)}

y, p = test.label.values, test.prob.values
rbs = summarise("Rule-based system (4 rules)", y, test.rbs_alert.values)

# ML at its own 75%-recall threshold (chosen on validation)
ml_own = summarise(f"{champ['model']} @ threshold {champ['threshold']:.3f}", y, (p >= champ["threshold"]).astype(int))

# ML at the SAME recall as the rules: lowest number of alerts that catches as many true positives
k_tp = rbs["true_positives"]
order = np.argsort(-p); cum_tp = np.cumsum(y[order])
n_alerts = int(np.searchsorted(cum_tp, k_tp) + 1) if k_tp > 0 else 0
matched = np.zeros_like(y); matched[order[:n_alerts]] = 1
ml_matched = summarise(f"{champ['model']} @ same recall as rules", y, matched)

comp = pd.DataFrame([rbs, ml_own, ml_matched])
display(comp.round(4))

fp_red = 1 - ml_matched["false_positives"] / max(rbs["false_positives"], 1)
print(f"At equal recall ({rbs['recall']:.1%}), the model raises {fp_red:.1%} fewer false positives than the rules.")

spark.createDataFrame(comp).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{CATALOG}.gold.rbs_comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Which rules fire, and how often are they right?

# COMMAND ----------

display(pd.DataFrame({
    "rule": rules.columns,
    "alerts": [int(rules[c].sum()) for c in rules],
    "precision": [float(y[rules[c].values].mean()) if rules[c].any() else 0.0 for c in rules],
    "recall": [float(rules[c].values[y == 1].mean()) for c in rules],
}).round(4))
