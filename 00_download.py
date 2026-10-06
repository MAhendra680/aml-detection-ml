# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Download the IBM AML dataset from Kaggle
# MAGIC
# MAGIC Downloads `HI-Medium_Trans.csv` and `HI-Medium_Patterns.txt` straight into a Unity Catalog volume,
# MAGIC so the raw files never touch a laptop and the step is re-runnable.
# MAGIC
# MAGIC **Before running:** paste your Kaggle username and API key into the two widgets at the top.
# MAGIC The key is never written into the notebook, so it can't leak to GitHub.

# COMMAND ----------

# MAGIC %pip install kaggle --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "aml", "Catalog")
dbutils.widgets.text("kaggle_user", "", "Kaggle username")
dbutils.widgets.text("kaggle_key", "", "Kaggle API key")

CATALOG = dbutils.widgets.get("catalog")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Create the landing volume
# MAGIC If `CREATE CATALOG` fails with a storage error, set the `catalog` widget to the catalog your
# MAGIC workspace already has (usually named after the workspace) and re-run — every notebook uses the widget.

# COMMAND ----------

spark.sql(f"CREATE CATALOG IF NOT EXISTS {CATALOG}")
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.raw")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.raw.files")

# COMMAND ----------

import os, subprocess, glob

os.environ["KAGGLE_USERNAME"] = dbutils.widgets.get("kaggle_user")
os.environ["KAGGLE_KEY"] = dbutils.widgets.get("kaggle_key")

DATASET = "ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
DEST = f"/Volumes/{CATALOG}/raw/files"
FILES = ["HI-Medium_Trans.csv", "HI-Medium_Patterns.txt"]

for f in FILES:
    if os.path.exists(f"{DEST}/{f}"):
        print(f"already present: {f}")
        continue
    print(f"downloading {f} ...")
    subprocess.run(["kaggle", "datasets", "download", "-d", DATASET, "-f", f, "-p", DEST], check=True)

# Kaggle zips large files — unzip anything that arrived zipped
for z in glob.glob(f"{DEST}/*.zip"):
    subprocess.run(["unzip", "-o", z, "-d", DEST], check=True)
    os.remove(z)

display(dbutils.fs.ls(DEST))
