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

# MAGIC %run ./resources/setup $reset_all=$reset_all_data

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

dataset = spark.read.format("parquet").load("/tmp/turbine_demo/data-sources/gold-data-for-ml")
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

client.transition_model_version_stage(name = "turbine_anomalies", 
                                      version = model_registered.version,
                                      stage = "Production", 
                                      archive_existing_versions=True)

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
# MAGIC ### We can now explore our prediction in a new dashboard!
# MAGIC Open SQL Analytics and query away! Some ideas:
# MAGIC https://e2-demo-west.cloud.databricks.com/sql/dashboards/92d8ccfa-10bb-411c-b410-274b64b25520-turbine-demo-predictions?o=2556758628403379

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Explainability
# MAGIC Our Spark model comes with a basic feature importance metric we can use to have a first understanding of our model:

# COMMAND ----------

bestModel = pipelineTrained.stages[-1:][0].bestModel

# Convert numpy.float64 to str for spark.createDataFrame()
weights = map(lambda w: '%.10f' % w, bestModel.featureImportances)
weightedFeatures = spark.createDataFrame(sorted(zip(weights, featureCols), key=lambda x: x[1], reverse=True)).toDF("weight", "feature")
display(weightedFeatures.select("feature", "weight").orderBy("weight", ascending=False))

# COMMAND ----------

# MAGIC %md #### Explaining our model with SHAP
# MAGIC Our model feature importance are is limited (we can't explain a single prediction) and can lead to a surprising result. 
# MAGIC 
# MAGIC You can have a look to [Scott Lundberg blog post](https://towardsdatascience.com/interpretable-machine-learning-with-xgboost-9ec80d148d27) for more details.
# MAGIC 
# MAGIC Using `shap`, we can understand how our model is behaving for a specific row. Let's analyze the importance of each feature for the first row of our dataset.

# COMMAND ----------

import shap
import numpy as np

# We'll need to add shap bundle js to display nice graph
with open(shap.__file__[:shap.__file__.rfind('/')]+"/plots/resources/bundle.js", 'r') as file:
   shap_bundle_js = '<script type="text/javascript">'+file.read()+'</script>'
    
# Build our explainer    
explainer = shap.TreeExplainer(bestModel)

# Let's draw the shap value (~force) of each feature
X = dataset.select(featureCols).limit(1000).toPandas()
shap_values = explainer.shap_values(X, check_additivity=False)
mean_abs_shap = np.absolute(shap_values).mean(axis=0).tolist()
display(spark.createDataFrame(sorted(list(zip(mean_abs_shap, X.columns)), reverse=True)[:6], ["Mean |SHAP|", "Column"]))

# COMMAND ----------

# MAGIC %md The following explanation shows how each feature contributes to "pushing" the model output from the base value (the average model output of the training dataset we passed) to the prediction.
# MAGIC 
# MAGIC Features pushing the prediction higher are shown in red, those pushing the prediction lower appear in blue:

# COMMAND ----------

plot_html = shap.force_plot(explainer.expected_value, shap_values[884,:], X.iloc[884,:], feature_names=X.columns)
displayHTML(shap_bundle_js + plot_html.html())

# COMMAND ----------

# MAGIC %md #### Overview of all features
# MAGIC To get an overview of which features are most important for a model we can plot the SHAP values of every feature for every sample. The plot below sorts features by the sum of its SHAP value magnitudes over all samples, and uses SHAP values to show the distribution of the impact each feature has on the model output. The color represents the feature value (red means high and blue means low). 
# MAGIC 
# MAGIC This reveals for example that big negative `AN5` strongly influence the prediction for a damaged turbine. Negative SHAP value, the purple = data around 0 is stacked to the left, and data with high values - positive or negative - on the right.

# COMMAND ----------

X = dataset.select(featureCols).limit(1000).toPandas()
shap_values = explainer.shap_values(X, check_additivity=False)

# COMMAND ----------

# Summarize the effects of all the features
shap.summary_plot(shap_values, X)

# COMMAND ----------

# MAGIC %md To understand how a single feature effects the output of the model we can plot the SHAP value of that feature vs. the value of the feature for all the examples in a dataset. 
# MAGIC 
# MAGIC Since SHAP values represent a feature's responsibility for a change in the model output, the plot below represents the change in turbine health as `AN9` changes. Vertical dispersion at a single value of `AN9` represents interaction effects with other features. 
# MAGIC 
# MAGIC To help reveal these interactions dependence_plot can selects another feature for coloring. In this case, we realize that `AN9` and `AN3` are linked: purple values (0) are stacked where `AN3=3` (you can try with `interaction_index=None` to remove color).
# MAGIC 
# MAGIC We clearly see from this that rows having a `AN3` close to 0 (no vibration) have a low SHAP value (healthy).

# COMMAND ----------

shap.dependence_plot("AN3", shap_values, X, interaction_index="AN9")
