# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Ingest → Bronze
# MAGIC
# MAGIC Loads the 31.9M-row transaction file into a Delta table **with the dataset's own ground-truth label**
# MAGIC (`Is Laundering`, the last column of the CSV), and attaches the laundering *typology*
# MAGIC (FAN-OUT, CYCLE, …) from the patterns file as an extra descriptive column.
# MAGIC
# MAGIC > **Change from the Colab version.** Colab extracted every number in the patterns file with a regex and
# MAGIC > used them as row indices, so the labels pointed at the wrong rows. Here the label comes straight from the
# MAGIC > CSV. The patterns file is only used for typology, joined on the full transaction record.

# COMMAND ----------

dbutils.widgets.text("catalog", "aml", "Catalog")
CATALOG = dbutils.widgets.get("catalog")
VOLUME = f"/Volumes/{CATALOG}/raw/files"

for s in ["bronze", "silver", "gold"]:
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{s}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1 · Transactions with an explicit schema
# MAGIC The CSV header has two columns called `Account` and names with spaces, so we supply our own names.
# MAGIC Bank IDs stay as strings (they are zero-padded, e.g. `00952`), which keeps the typology join exact.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, IntegerType

schema = StructType([
    StructField("ts_raw", StringType()),
    StructField("from_bank", StringType()),
    StructField("from_account", StringType()),
    StructField("to_bank", StringType()),
    StructField("to_account", StringType()),
    StructField("amount_received", DoubleType()),
    StructField("receiving_currency", StringType()),
    StructField("amount_paid", DoubleType()),
    StructField("payment_currency", StringType()),
    StructField("payment_format", StringType()),
    StructField("label", IntegerType()),          # <- "Is Laundering": the real ground truth
])

trans = (spark.read.option("header", True).schema(schema)
         .csv(f"{VOLUME}/HI-Medium_Trans.csv")
         .withColumn("ts", F.to_timestamp("ts_raw", "yyyy/MM/dd HH:mm"))
         .withColumn("txn_id", F.monotonically_increasing_id())
         .drop("ts_raw"))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2 · Typology lookup from the patterns file
# MAGIC Each block looks like `BEGIN LAUNDERING ATTEMPT - FAN-OUT … END LAUNDERING ATTEMPT`, with one full
# MAGIC transaction record per line. We parse the whole record (not just numbers) and remember its typology.

# COMMAND ----------

import csv

rows, typology = [], None
with open(f"{VOLUME}/HI-Medium_Patterns.txt") as f:
    for line in f:
        line = line.strip()
        if line.startswith("BEGIN LAUNDERING ATTEMPT"):
            typology = line.split("-", 1)[1].split(":")[0].strip()
        elif line.startswith("END") or not line:
            continue
        else:
            r = next(csv.reader([line]))
            rows.append(r[:10] + [typology])

pcols = ["ts_raw", "from_bank", "from_account", "to_bank", "to_account",
         "amount_received", "receiving_currency", "amount_paid",
         "payment_currency", "payment_format", "typology"]

keys = ["ts", "from_bank", "from_account", "to_bank", "to_account",
        "amount_paid", "payment_currency", "payment_format"]

typologies = (spark.createDataFrame(rows, pcols)
              .withColumn("ts", F.to_timestamp("ts_raw", "yyyy/MM/dd HH:mm"))
              .withColumn("amount_paid", F.col("amount_paid").cast("double"))
              .select(*keys, "typology")
              .dropDuplicates(keys))          # one typology per record so the join can't duplicate rows

typologies.write.mode("overwrite").saveAsTable(f"{CATALOG}.bronze.laundering_typologies")
print(f"pattern records parsed: {len(rows):,}   unique: {typologies.count():,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3 · Write the Bronze table

# COMMAND ----------

bronze = trans.join(typologies, on=keys, how="left")
bronze.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.bronze.transactions")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4 · Sanity checks
# MAGIC Expected for HI-Medium: **~31.9M rows**, **~35k positives** (about 1 in 900), **0 null timestamps**.
# MAGIC The last check shows how many positives are covered by a named typology — the patterns file only
# MAGIC describes some of the laundering, so this will be below 100%.

# COMMAND ----------

b = spark.table(f"{CATALOG}.bronze.transactions")
stats = b.agg(
    F.count("*").alias("rows"),
    F.sum("label").alias("positives"),
    F.sum(F.col("ts").isNull().cast("int")).alias("null_timestamps"),
    F.sum(F.col("typology").isNotNull().cast("int")).alias("rows_with_typology"),
    F.sum(((F.col("label") == 1) & F.col("typology").isNotNull()).cast("int")).alias("positives_with_typology"),
).first().asDict()
stats["positive_rate"] = stats["positives"] / stats["rows"]
for k, v in stats.items():
    print(f"{k:>24}: {v:,.6f}" if isinstance(v, float) else f"{k:>24}: {v:,}")

# COMMAND ----------

display(b.filter("label = 1").groupBy("typology").count().orderBy(F.desc("count")))
