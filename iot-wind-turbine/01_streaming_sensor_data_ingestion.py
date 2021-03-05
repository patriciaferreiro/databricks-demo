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
# MAGIC *Data Source Acknowledgement: Data source is provided by NREL*
# MAGIC 
# MAGIC *https://www.nrel.gov/docs/fy12osti/54530.pdf*

# COMMAND ----------

# from pyspark.sql.functions import rand, input_file_name, from_json, col
# from pyspark.sql.types import *

# COMMAND ----------

# Define widgets - 'true' will reset all tables defined this notebook
dbutils.widgets.removeAll()
dbutils.widgets.dropdown("reset_all_data", "true", ["true", "false"])

# COMMAND ----------

# DBTITLE 1,Let's prepare our data first
# MAGIC %run ./resources/setup $reset_all=$reset_all_data

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Bronze layer: ingest data from Kafka
# MAGIC In this case, data is read from a Kafka source that streams the turbine sensor readings. We will write the data to our Data Lake in Delta format **for additional performance and reliability**, but you can use the exact same code to write in other formats.
# MAGIC 
# MAGIC In this section we will:
# MAGIC 
# MAGIC 1. Read raw data in from file system
# MAGIC 2. Register to the metastore so we can query the data with SQL
# MAGIC 3. Read streaming data from Kafka or File System and write it to the Data Lake in Delta format

# COMMAND ----------

# DBTITLE 1,Let's explore what is being delivered by our wind turbines stream...
# MAGIC %sql SELECT * FROM parquet.`/mnt/quentin-demo-resources/turbine/incoming-data` LIMIT 10

# COMMAND ----------

# DBTITLE 1,Create a Delta table for the sensor data
# MAGIC %sql
# MAGIC -- Create delta table with defined schema - note there's no data yet!
# MAGIC CREATE TABLE IF NOT EXISTS turbine_bronze (key double NOT null, value string) USING delta;
# MAGIC   
# MAGIC -- Turn on autocompaction to solve small files issues on your streaming job, that's all you have to do!
# MAGIC ALTER TABLE turbine_bronze SET TBLPROPERTIES ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true);

# COMMAND ----------

# DBTITLE 1,Read and write streaming data from Kafka to our Delta Lake...
# Option 1, read from kinesis directly
# Load stream from Kafka - skip this cell if you don't have an available deployment
bronzeDF = (spark.readStream
                 .format("kafka")
                 .option("kafka.bootstrap.servers", "kafkaserver1:9092, kafkaserver2:9092")
                 .option("subscribe", "turbine")
                 .load())

# Write the output to a delta table
(bronzeDF.selectExpr("CAST(key AS STRING) as key", "CAST(value AS STRING) as value")
         .writeStream
         .format("delta")
         .option("checkpointLocation", f"{path}/turbine/bronze/_checkpoint")
         .option("path", f"{path}/turbine/bronze/data")
         .trigger(once=True)
         .start())

# COMMAND ----------

# DBTITLE 1,...or from a File System
# Option 2, read from files using the cloudFiles source
# This will automatically process new files as they arrive
raw_data_path = "/mnt/quentin-demo-resources/turbine/incoming-data"
bronzeDF = (spark.readStream
                 .format("cloudFiles")
                 .option("cloudFiles.format", "parquet")
                 .schema("value string, key double")
                 .load(raw_data_path))

# Write the output to a delta table
(bronzeDF.writeStream
         .option("ignoreChanges", "true")
         .trigger(processingTime="5 seconds")
         .table("turbine_bronze"))

# COMMAND ----------

# DBTITLE 1,Our raw data is now available in a Delta table, without having small files issues & with great performance!
# MAGIC %sql 
# MAGIC -- Query data as it arrives in real-time! If no records are returned, wait for the stream to start.
# MAGIC SELECT * FROM turbine_bronze

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Silver layer: transform JSON data into tabular table
# MAGIC Now we want to process the raw sensor data so that it's usable by the Data Science and BI teams. We also want to make sure it **complies with our quality standards**, so we will introduce constraints the data needs to meet in order to be "promoted" to the next refinement stage, the Silver layer.
# MAGIC 
# MAGIC For these reasons we will:
# MAGIC 1. Unpack JSON from Bronze Delta table and write it to the Silver layer
# MAGIC 2. Define value constraints to ensure data quality
# MAGIC 3. Enable automatic optimizations for additional performance
# MAGIC 3. Query the resulting data with SQL

# COMMAND ----------

# Define json schema
sensor_cols = ["AN3", "AN4", "AN5", "AN6", "AN7", "AN8", "AN9", "AN10", "SPEED", "TORQUE", "ID"]
jsonSchema = StructType([StructField(col, DoubleType(), False) for col in sensor_cols] + [StructField("TIMESTAMP", TimestampType())])

# Read in data from Bronze delta table, transform it and write it to a Silver table
(spark.readStream.table('turbine_bronze')
      .withColumn("jsonData", from_json(col("value"), jsonSchema))
      .select("jsonData.*")
      .writeStream
      .option("ignoreChanges", "true")
      .format("delta")
      .trigger(processingTime='5 seconds')
      .table("turbine_silver"))

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Add some constraints in our table, to ensure or ID can't be negative (needs DBR 7.5)
# MAGIC ALTER TABLE turbine_silver ADD CONSTRAINT idGreaterThanZero CHECK (id >= 0);
# MAGIC 
# MAGIC -- Enable auto-compaction
# MAGIC ALTER TABLE turbine_silver SET TBLPROPERTIES ('delta.autoOptimize.autoCompact' = true, 'delta.autoOptimize.optimizeWrite' = true);
# MAGIC 
# MAGIC -- Select data
# MAGIC SELECT * FROM turbine_silver;

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Gold layer: join information on Turbine status to add a label to our dataset
# MAGIC We now want to **enrich our sensor data with another dataset**, in this case one containing turbine status information. 
# MAGIC 
# MAGIC To achieve that, we'll:
# MAGIC 1. Create a Gold level table with Turbine status data
# MAGIC 2. Join the Turbine status table with the Sensor readings
# MAGIC 3. Delete wrong values transactionally using Delta
# MAGIC 4. Verify the changes have created a new version of the table

# COMMAND ----------

# MAGIC %sql 
# MAGIC CREATE TABLE IF NOT EXISTS turbine_status_gold (id int, status string) USING delta;
# MAGIC 
# MAGIC -- Copy data from a file location to a Delta table 
# MAGIC COPY INTO turbine_status_gold
# MAGIC   FROM '/mnt/quentin-demo-resources/turbine/status'
# MAGIC   FILEFORMAT = PARQUET;

# COMMAND ----------

# MAGIC %sql SELECT * FROM turbine_status_gold

# COMMAND ----------

# DBTITLE 1,Join data with turbine status (Damaged or Healthy)
# Read in the Silver streaming table as a Spark DF
turbine_stream = spark.readStream.table("turbine_silver")

# Read in the Gold turbine status table from the filesystem
turbine_status = spark.read.table("turbine_status_gold")

# Join both tables
(turbine_stream.join(turbine_status, ['id'], 'left')
               .writeStream
               .option("ignoreChanges", "true")
               .format("delta")
               .trigger(processingTime="5 seconds")
               .table("turbine_gold"))

# COMMAND ----------

# MAGIC %sql SELECT * FROM turbine_gold

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Run DELETE/UPDATE/MERGE with DELTA ! 
# MAGIC We just realized that something is wrong in the data before 2020! Let's DELETE all this data from our gold table as we don't want to have wrong values in our dataset. Delta supports ACID operations and will thus make sure **consumers always read a consistent version of the data**, even while the change is being executed.
# MAGIC 
# MAGIC Additionally, any operations on Delta format are recorded in the transaction log, automatically versioning the data.
# MAGIC This means we will be able to **query the data at any point in the past** (time travel) or **restore a previous version seamlesly**!

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Delete records meeting the predicate condition
# MAGIC DELETE FROM turbine_gold WHERE timestamp < '2020-00-01'

# COMMAND ----------

# MAGIC %sql 
# MAGIC -- Verify the history, you'll see there's a new version of the data for each atomic operation!
# MAGIC DESCRIBE HISTORY turbine_gold

# COMMAND ----------

# MAGIC %sql
# MAGIC -- If needed, we can go back in time to select a specific version or timestamp
# MAGIC SELECT * FROM turbine_gold VERSION AS OF 1;
# MAGIC -- SELECT * FROM turbine_gold TIMESTAMP AS OF '2020-12-01'
# MAGIC 
# MAGIC -- Or restore a given version
# MAGIC -- RESTORE turbine_gold TO TIMESTAMP AS OF '2020-12-01'
# MAGIC 
# MAGIC -- Or clone the table (zero copy)
# MAGIC -- CREATE TABLE turbine_gold_clone [SHALLOW | DEEP] CLONE turbine_gold VERSION AS OF 32

# COMMAND ----------

# MAGIC %md 
# MAGIC ## Our data is ready! Let's create a dashboard to monitor our Turbine plant
# MAGIC 
# MAGIC https://e2-demo-west.cloud.databricks.com/sql/dashboards/a81f8008-17bf-4d68-8c79-172b71d80bf0-turbine-demo?o=2556758628403379
