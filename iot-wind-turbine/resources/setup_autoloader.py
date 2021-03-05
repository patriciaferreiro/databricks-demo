# Databricks notebook source
# MAGIC %run ./00-setup $reset_all=$reset_all

# COMMAND ----------

# MAGIC %fs ls /mnt/quentin-demo-resources/turbine/

# COMMAND ----------

# Remove data in path to start from scratch
dbutils.fs.rm(path+"/turbine/incoming-data-json", True)

# Read json data
(spark.read.format("json").option("inferSchema", "true")
 .load("/mnt/quentin-demo-resources/turbine/incoming-data-json").limit(10).repartition(1)
 .write.format("json").mode("overwrite").save(path+"/turbine/incoming-data-json"))
