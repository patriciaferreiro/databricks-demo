# Databricks notebook source
# MAGIC %md
# MAGIC ### Setup notebook
# MAGIC This notebook prepares the environment so the demo is reproducible. You could also execute this cells in your main notebook, but it may be cleaner to do it this way. For production applications, you may prefer to **package and distribute your libraries / install them in the cluster** and then import them normally.
# MAGIC 
# MAGIC **Note**: Use the `%run` magic command to call other Notebooks and simplify your code. Please refer to our docs on the topic for more information.

# COMMAND ----------

# Get user metadata, create database and set as default
current_user = dbutils.notebook.entry_point.getDbutils().notebook().getContext().tags().apply('user')
dbName = re.sub(r'\W+', '_', current_user)
path = f"/Users/{current_user}/demo"
dbutils.widgets.text("path", path, "path")
dbutils.widgets.text("dbName", dbName, "dbName")
print(f"Current user is '{current_user}'.")
print(f"Using path '{path}'.")

# Create a user database, where we will create all tables and views
# Note this is just a logical construct to organise your Data tab i.e. the Hive metastore
spark.sql(f"CREATE DATABASE IF NOT EXISTS {dbName} LOCATION '{path}/turbine/tables'")
spark.sql(f"USE {dbName}")
print(f"Created database '{dbName}' in location '{path}/turbine/tables'.")

# COMMAND ----------

# Reset tables
tables = ["turbine_bronze", "turbine_silver", "turbine_gold", "turbine_power", "turbine_schema_evolution"]
reset_all = dbutils.widgets.get("reset_all") == "true" or any([not spark.catalog._jcatalog.tableExists(table) for table in ["turbine_power"]])
if reset_all:
  print("Reseting data...")
  
  # Drop all tables
  for table in tables:
    spark.sql(f"DROP TABLE IF EXISTS {dbName}.{table}")
  print(f"Tables {tables} have been dropped.")
  
  # Drop database - optional
  # spark.sql(f"DROP DATABASE IF EXISTS {dbName} CASCADE")
  
  spark.sql(f"CREATE DATABASE IF NOT EXISTS {dbName} LOCATION '{path}/tables'")
  # Remove the data on disk
  dbutils.fs.rm(path+"/turbine/bronze/", True)
  dbutils.fs.rm(path+"/turbine/silver/", True)
  dbutils.fs.rm(path+"/turbine/gold/", True)
  dbutils.fs.rm(path+"/turbine/_checkpoint", True)
  
  # Read in turbine raw data and write it to specified path in delta format
  raw_data_schema = "turbine_id bigint, date timestamp, power float, wind_speed float, theoretical_power_curve float, wind_direction float"
  raw_data_path = "/mnt/quentin-demo-resources/turbine/power/raw"
  (spark.read.format("json")
             .schema(raw_data_schema)
             .load(raw_data_path)
             .write.format("delta").mode("overwrite").save(path+"/turbine/power/bronze/data")
  )
  
  # Create delta table from existing data
  spark.sql(f"CREATE TABLE IF NOT EXISTS turbine_power USING DELTA LOCATION '{path}/turbine/power/bronze/data'")
  print(f"Created table 'turbine_power' in path '{path}/turbine/power/bronze/data'.")
else:
  print("Loaded without data reset.")

# Define the default checkpoint location to avoid managing that per stream and making it easier
# In production it can be better to set the location at a stream level
spark.conf.set("spark.sql.streaming.checkpointLocation", path+"/turbine/_checkpoint")

# Allow schema inference for auto loader
spark.conf.set("spark.databricks.cloudFiles.schemaInference.enabled", "true")
