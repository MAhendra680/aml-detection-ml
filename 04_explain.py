# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Evaluation figures and SHAP reason codes
# MAGIC
# MAGIC Recreates the dissertation's Chapter 4 figures **from the real run**, and attaches them to the champion's
# MAGIC MLflow run. Nothing is hard-coded or simulated.
# MAGIC
# MAGIC - PR curve with the random baseline
# MAGIC - Threshold trade-off (precision vs recall by threshold)
# MAGIC - Confusion matrix at the chosen threshold
# MAGIC - SHAP global drivers (beeswarm + bar)
# MAGIC - SHAP waterfall for the highest-scoring true positive (the "SAR narrative")
# MAGIC - A **reason-code table**: the top 3 SHAP drivers for each of the top 1,000 alerts → `gold.alert_reason_codes`

# COMMAND ----------

# MAGIC %pip install xgboost lightgbm shap --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "aml", "Catalog")
CATALOG = dbutils.widgets.get("catalog")

import numpy as np, pandas as pd, matplotlib.pyplot as plt, mlflow, mlflow.sklearn, shap
from sklearn.metrics import precision_recall_curve, average_precision_score, confusion_matrix

champ = spark.table(f"{CATALOG}.gold.champion").first()
FEATURES = champ["features"].split(",")
THR, RUN_ID = champ["threshold"], champ["run_id"]
print(f"Champion: {champ['model']} | threshold {THR:.4f} | run {RUN_ID}")

model = mlflow.sklearn.load_model(f"runs:/{RUN_ID}/model")
test = spark.table(f"{CATALOG}.gold.test_predictions").toPandas()
X_test, y_test, p = test[FEATURES].astype("float64"), test["label"].values, test["prob"].values

user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{user}/aml-detection")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Precision-recall curve, threshold trade-off and confusion matrix

# COMMAND ----------

prec, rec, thr = precision_recall_curve(y_test, p)
auprc = average_precision_score(y_test, p)

fig_pr, ax = plt.subplots(figsize=(8, 6))
ax.plot(rec, prec, lw=2.5, label=f"{champ['model']} (AUPRC = {auprc:.4f})")
ax.axhline(y_test.mean(), ls="--", color="grey", label=f"Random baseline ({y_test.mean():.2%})")
i_thr = int(np.searchsorted(thr, THR))          # thresholds are ascending: first point at/above THR
ax.scatter(rec[i_thr], prec[i_thr], color="red", zorder=5, s=80,
           label=f"Chosen threshold {THR:.3f}")
ax.set(xlabel="Recall (detection rate)", ylabel="Precision (alert accuracy)", title="Precision-recall curve (test set)")
ax.legend(); ax.grid(alpha=.3)

fig_tt, ax = plt.subplots(figsize=(9, 5))
ax.plot(thr, prec[:-1], label="Precision"); ax.plot(thr, rec[:-1], label="Recall")
ax.axvline(THR, ls="--", color="red", label=f"Chosen threshold {THR:.3f}")
ax.set(xlabel="Decision threshold", ylabel="Score", title="Precision / recall trade-off by threshold")
ax.legend(); ax.grid(alpha=.3)

cm = confusion_matrix(y_test, (p >= THR).astype(int), labels=[0, 1])
fig_cm, ax = plt.subplots(figsize=(6, 5))
ax.imshow(cm, cmap="Blues")
for (i, j), v in np.ndenumerate(cm):
    ax.text(j, i, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black")
ax.set_xticks([0, 1], ["Normal", "Laundering"]); ax.set_yticks([0, 1], ["Normal", "Laundering"])
ax.set(xlabel="Predicted", ylabel="Actual", title=f"Confusion matrix (threshold {THR:.3f})")
display(fig_pr); display(fig_tt); display(fig_cm)

# COMMAND ----------

# MAGIC %md
# MAGIC ### SHAP: global drivers and one detected case

# COMMAND ----------

X_bg = X_test.sample(min(1000, len(X_test)), random_state=42)
try:
    explainer = shap.TreeExplainer(model)
except Exception:                                   # e.g. Logistic Regression pipeline
    explainer = shap.Explainer(lambda d: model.predict_proba(pd.DataFrame(d, columns=FEATURES))[:, 1], X_bg)

def explain(X):
    ex = explainer(X)
    return ex[:, :, 1] if ex.values.ndim == 3 else ex     # keep the "laundering" class only

def tidy(fig, title):
    fig.set_size_inches(10, 6); fig.suptitle(title); fig.tight_layout()   # stops feature names being cut off
    return fig

ex_bg = explain(X_bg)

plt.figure(); shap.plots.beeswarm(ex_bg, max_display=15, show=False)
fig_bee = tidy(plt.gcf(), "SHAP global risk drivers"); display(fig_bee)

plt.figure(); shap.plots.bar(ex_bg, max_display=15, show=False)
fig_bar = tidy(plt.gcf(), "Mean |SHAP| per feature"); display(fig_bar)

tp = test[(test.label == 1)].sort_values("prob", ascending=False)
case = tp.index[0]
ex_case = explain(X_test.loc[[case]])
plt.figure(); shap.plots.waterfall(ex_case[0], max_display=10, show=False)
fig_wf = tidy(plt.gcf(), f"Why txn {test.loc[case, 'txn_id']} was flagged (score {test.loc[case, 'prob']:.3f})")
display(fig_wf)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Reason codes for the top 1,000 alerts → `gold.alert_reason_codes`
# MAGIC An investigator-facing table: for each alert, the three features that pushed its score up the most.

# COMMAND ----------

top = test.sort_values("prob", ascending=False).head(1000)
sv = explain(X_test.loc[top.index]).values
top3 = np.argsort(-sv, axis=1)[:, :3]

codes = top[["txn_id", "label", "typology", "prob"]].copy()
for r in range(3):
    codes[f"reason_{r+1}"] = [FEATURES[i] for i in top3[:, r]]
    codes[f"reason_{r+1}_shap"] = sv[np.arange(len(sv)), top3[:, r]]

spark.createDataFrame(codes).write.mode("overwrite").option("overwriteSchema", "true") \
     .saveAsTable(f"{CATALOG}.gold.alert_reason_codes")
display(codes.head(20))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Attach every figure and table to the champion's MLflow run

# COMMAND ----------

with mlflow.start_run(run_id=RUN_ID):
    for name, fig in [("pr_curve", fig_pr), ("threshold_tradeoff", fig_tt), ("confusion_matrix", fig_cm),
                      ("shap_beeswarm", fig_bee), ("shap_bar", fig_bar), ("shap_waterfall_top_case", fig_wf)]:
        mlflow.log_figure(fig, f"figures/{name}.png")
    mlflow.log_table(codes, "alert_reason_codes.json")
print("Figures logged. Open the champion run in Experiments → Artifacts → figures/.")
