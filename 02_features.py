# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Feature engineering → Silver
# MAGIC
# MAGIC The same behavioural features as the dissertation, rewritten as PySpark window functions and computed over
# MAGIC **all 31.9M transactions**.
# MAGIC
# MAGIC > **Change from the Colab version.** Colab sampled first (all positives + 2M negatives) and computed the
# MAGIC > rolling features on the sample. That removed about 94% of each normal account's history while keeping
# MAGIC > every laundering transaction, so velocity and fan-out were biased by the sampling itself. Here the features
# MAGIC > see every transaction, and sampling happens afterwards in notebook 03.
# MAGIC
# MAGIC | Feature | Stage (FATF) | Definition |
# MAGIC |---|---|---|
# MAGIC | `amount_paid` | – | amount of this transaction |
# MAGIC | `sender_24h_count` | Placement / velocity | sender's transactions in the trailing 24h |
# MAGIC | `sender_24h_volume` | Placement / velocity | sender's total amount in the trailing 24h |
# MAGIC | `amt_vs_7d_avg` | Structuring | amount ÷ sender's trailing 7-day average |
# MAGIC | `unique_receivers_24h` | Layering / fan-out | distinct receivers in the trailing 24h |
# MAGIC | `is_round_1000` | Structuring | amount is an exact multiple of 1,000 |
# MAGIC | `rapid_exit_risk` | Layering / pass-through | 24h volume ÷ this amount |
# MAGIC | `pay_*` (7) | – | one-hot payment format |
# MAGIC | `is_interbank` | – | sender bank ≠ receiver bank *(planned in Colab but dropped by a merge bug; off by default in 03)* |

# COMMAND ----------

dbutils.widgets.text("catalog", "aml", "Catalog")
CATALOG = dbutils.widgets.get("catalog")

# COMMAND ----------

from pyspark.sql import functions as F, Window

DAY = 86400
src = (spark.table(f"{CATALOG}.bronze.transactions")
       .withColumn("ts_sec", F.col("ts").cast("long")))

# pandas rolling('1D') covers (t − 1 day, t], so the window starts 1 second after t − 86400.
# One small difference: Spark counts every row in the same minute, while pandas only counted rows
# sorted before the current one. That matters only for same-minute bursts.
w24 = Window.partitionBy("from_account").orderBy("ts_sec").rangeBetween(-(DAY - 1), 0)
w7d = Window.partitionBy("from_account").orderBy("ts_sec").rangeBetween(-(7 * DAY - 1), 0)

PAY_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]

feats = (src
    .withColumn("sender_24h_count",     F.count("*").over(w24).cast("int"))
    .withColumn("sender_24h_volume",    F.sum("amount_paid").over(w24))
    .withColumn("sender_7d_avg",        F.avg("amount_paid").over(w7d))
    .withColumn("amt_vs_7d_avg",        F.col("amount_paid") / (F.col("sender_7d_avg") + 1e-6))
    .withColumn("unique_receivers_24h", F.size(F.collect_set("to_account").over(w24)))
    .withColumn("is_round_1000",        ((F.col("amount_paid") % 1000) == 0).cast("int"))
    .withColumn("rapid_exit_risk",      F.col("sender_24h_volume") / (F.col("amount_paid") + 1e-6))
    .withColumn("is_interbank",         (F.col("from_bank") != F.col("to_bank")).cast("int"))
)
for p in PAY_FORMATS:
    feats = feats.withColumn("pay_" + p.lower().replace(" ", "_"),
                             (F.col("payment_format") == p).cast("int"))

feats = feats.drop("sender_7d_avg", "ts_sec")

(feats.write.mode("overwrite").option("overwriteSchema", "true")
      .saveAsTable(f"{CATALOG}.silver.features"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Checks
# MAGIC Row count should equal Bronze. Compare feature averages for laundering vs normal transactions to
# MAGIC see which behaviours actually separate the classes.

# COMMAND ----------

s = spark.table(f"{CATALOG}.silver.features")
print("silver rows:", f"{s.count():,}", "| bronze rows:", f"{spark.table(f'{CATALOG}.bronze.transactions').count():,}")

FEATURE_COLS = ["amount_paid", "sender_24h_count", "sender_24h_volume", "amt_vs_7d_avg",
                "unique_receivers_24h", "is_round_1000", "rapid_exit_risk", "is_interbank"]
display(s.groupBy("label").agg(*[F.round(F.avg(c), 3).alias(c) for c in FEATURE_COLS]).orderBy("label"))
