# Databricks notebook source
# MAGIC %md-sandbox
# MAGIC # Wind Turbine Predictive Maintenance
# MAGIC 
# MAGIC ## The goal
# MAGIC Our goal is to **perform anomaly detection to find damaged wind turbines**.
# MAGIC 
# MAGIC A damaged, single, inactive wind turbine costs energy utility companies **thousands of dollars per day in losses**.
# MAGIC 
# MAGIC 
# MAGIC <img src="https://quentin-demo-resources.s3.eu-west-3.amazonaws.com/images/turbine/turbine_flow.png" width="1150px" />
# MAGIC 
# MAGIC 
# MAGIC <div style="float:right; margin: -10px 50px 0px 50px">
# MAGIC   <img src="https://s3.us-east-2.amazonaws.com/databricks-knowledge-repo-images/ML/wind_turbine/wind_small.png" width="400px" /><br/>
# MAGIC   *Fig. 1 - Sensor location*
# MAGIC </div>
# MAGIC 
# MAGIC ## The data
# MAGIC Our dataset consists of sensor vibration readings. Sensors are located in the gearboxes of wind turbines (See Fig. 1). 
# MAGIC 
# MAGIC ## The plan
# MAGIC We will use a `Gradient Boosted Tree Classification` algorithm to predict which set of vibrations could be indicative of failure.
# MAGIC We'll also use MFLow to track experiments. In particular we will:
# MAGIC - Track model parameters, performance metrics and arbitrary artifacts
# MAGIC - Version and tag models in a governed model registry
# MAGIC - Automatically select the best models
# MAGIC - Package and deploy models for real-time or batch scoring
# MAGIC 
# MAGIC *Data Source Acknowledgement: Data source is provided by NREL**
# MAGIC 
# MAGIC *https://www.nrel.gov/docs/fy12osti/54530.pdf*

# COMMAND ----------

# MAGIC %pip install mlflow==1.14.1

# COMMAND ----------

import re
import mlflow
import mlflow.spark
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

from pyspark.sql.functions import rand, input_file_name, from_json, col
from pyspark.sql.types import *
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
from pyspark.ml.feature import StringIndexer, StandardScaler, VectorAssembler
from pyspark.ml import Pipeline
from pyspark.mllib.evaluation import MulticlassMetrics
from mlflow.utils.file_utils import TempDir

# COMMAND ----------

# reset widgets
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("reset_all_data", "true", ["true", "false"])

# COMMAND ----------

# MAGIC %run ./resources/00-setup $reset_all=$reset_all_data

# COMMAND ----------

# MAGIC %md 
# MAGIC ##Use ML and MLFlow to detect damaged turbine
# MAGIC 
# MAGIC Our data is now ready. We'll now train a model to detect damaged turbines.

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Data Exploration
# MAGIC What do the distributions of sensor readings look like for our turbines? 
# MAGIC 
# MAGIC *Notice the much larger stdev in AN8, AN9 and AN10 for Damaged turbined.*

# COMMAND ----------

dataset = spark.read.load("/mnt/quentin-demo-resources/turbine/gold-data-for-ml")
display(dataset)

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Model Creation: Workflows with Pyspark.ML Pipeline

# COMMAND ----------

import mlflow
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

with mlflow.start_run():
  # Split dataset into train and test
  training, test = dataset.limit(1000).randomSplit([0.9, 0.1], seed = 42)
  
  # Define the classifier and performance metrics
  gbt = GBTClassifier(labelCol="label", featuresCol="features").setMaxIter(5)
  grid = ParamGridBuilder().addGrid(gbt.maxDepth, [3,4,5]).build()

  metrics = MulticlassClassificationEvaluator(metricName="f1")
  cv = CrossValidator(estimator=gbt, estimatorParamMaps=grid, evaluator=metrics, numFolds=2)

  # Define pre-processing pipeline
  featureCols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10"]
  stages = [VectorAssembler(inputCols=featureCols, outputCol="va"),
            StandardScaler(inputCol="va", outputCol="features"),
            StringIndexer(inputCol="status", outputCol="label"), cv]
  pipeline = Pipeline(stages=stages)

  pipelineTrained = pipeline.fit(training)
  
  predictions = pipelineTrained.transform(test)
  metrics = MulticlassMetrics(predictions.select(['prediction', 'label']).rdd)
  
  # Define mlflow artifacts to log with the experiment run
  mlflow.log_metric("precision", metrics.precision(1.0))
  mlflow.log_metric("recall", metrics.recall(1.0))
  mlflow.log_metric("f1", metrics.fMeasure(1.0))
  
  mlflow.spark.log_model(pipelineTrained, "turbine_anomalies")
  mlflow.set_tag("model", "gbt") 
  
  # Add confusion matrix to the model
  labels = pipelineTrained.stages[2].labels
  fig = plt.figure()
  sn.heatmap(pd.DataFrame(metrics.confusionMatrix().toArray()), annot=True, fmt='g', xticklabels=labels, yticklabels=labels)
  plt.suptitle("Turbine Damage Prediction. F1={:.2f}".format(metrics.fMeasure(1.0)), fontsize = 18)
  plt.xlabel("Predicted Labels")
  plt.ylabel("True Labels")
  mlflow.log_figure(fig, "confusion_matrix.png") # needs mlflow version >=1.13.1

# COMMAND ----------

# MAGIC %md ## Saving our model to MLFLow registry

# COMMAND ----------

# DBTITLE 1,Save our new model to the registry as a new version
# Get the best model having the best metrics.AUROC from the registry that fits our search criteria
best_models = mlflow.search_runs(filter_string='tags.model="gbt" and attributes.status = "FINISHED" and metrics.f1 > 0',
                                 order_by=['metrics.f1 DESC'], max_results=1)
model_uri = best_models.iloc[0].artifact_uri
print(f"Model is stored at '{model_uri}'.")

# Register the model
model_registered = mlflow.register_model(best_models.iloc[0].artifact_uri+"/turbine_anomalies", "turbine_anomalies")

# COMMAND ----------

# DBTITLE 1,Flag this version as production ready
client = mlflow.tracking.MlflowClient()
print(f"Registering model version {model_registered.version} as production model.")
client.transition_model_version_stage(name = "turbine_anomalies", version = model_registered.version, stage = "Production", archive_existing_versions=True)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Detecting damaged turbine in a production pipeline

# COMMAND ----------

# DBTITLE 1,Load the model from our registry
model_name = 'turbine_anomalies'
model_from_registry = mlflow.spark.load_model(f'models:/{model_name}/production')

# COMMAND ----------

# Define Spark UDF so we can parallelize model scoring
udf = mlflow.pyfunc.spark_udf(spark, f'models:/{model_name}/production')

# Apply prediction UDF to dataset
predictions = dataset.withColumn("colPrediction", udf(*dataset.select(*featureCols).columns))

# COMMAND ----------

# DBTITLE 1,Compute predictions using our Spark model
# Compute predictions for the first 100 values
prediction = model_from_registry.transform(dataset.limit(100))
display(prediction.select(*featureCols+['prediction']+['ID']+['status']))

# COMMAND ----------

# MAGIC %md
# MAGIC ### We can now explore our prediction in a new dashboard
# MAGIC https://e2-demo-west.cloud.databricks.com/sql/dashboards/92d8ccfa-10bb-411c-b410-274b64b25520-turbine-demo-predictions?o=2556758628403379
