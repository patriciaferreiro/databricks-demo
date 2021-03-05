# Databricks notebook source
def load_csv_as_format(in_path, out_path, out_format='delta'):
    # Reads csv file from path and writes it to output path as the specified format
    spark.read.format('csv').load(in_path).write.format(out_format).mode("overwrite").save(out_path)

# COMMAND ----------

def sample_as_csv(in_path, out_path, in_format='delta', file_name='data_sample', samples=1000, random=False, sample_pctg=0.1):
    # Reads in file from input path in the specified format and writes it to output path as a single csv file
    if random:
        # Take a random sample of <sample_pctg> %
        seed = 42
        spark.read.format(in_format).load(in_path).sample(sample_pctg, seed).repartition(1).write.format('csv').mode("overwrite").save(out_path+file_name)
    else:
        # Take <samples> records
        spark.read.format(in_format).load(in_path).limit(samples).repartition(1).write.format('csv').mode("overwrite").save(out_path+file_name)
