# Databricks notebook source
def load_csv_as_format(in_path, schema, out_path, out_format='delta'):
    # Reads csv file from path and writes it to output path as the specified format
    import pandas as pd
    pdf = pd.read_csv(in_path)
    spark.createDataFrame(pdf, schema).write.format(out_format).mode("overwrite").save(out_path)

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

# COMMAND ----------

# Read datasets in and write in appropiate formats / folders for the main demo to use
out_path = '/tmp/turbine_demo/data-sources'
datasets = ['incoming-data', 'status', 'gold-data-for-ml']
schemas = ['key:double, value:string', 'id:int, status:string',
           'ID:float, AN3:float, AN4:float, AN5:float, AN6:float, AN7:float, AN8:float, AN9:float, AN10:float, SPEED:float, TORQUE:float, TIMESTAMP:string, status:string']

for (dataset, schema) in zip(datasets, schemas):
    load_csv_as_format(in_path=f'./resources/data/{dataset}.csv', schema=schema, out_path=f'{out_path}/{dataset}', out_format='parquet')
    print(f"Processed {dataset} dataset.")
    
print(f"Loaded datasets {datasets} to path '{out_path}'.")
