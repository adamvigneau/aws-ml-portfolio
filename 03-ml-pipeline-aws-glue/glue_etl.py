import sys
from awsglue.transforms import *
from awsglue.utils import getResolvedOptions
from pyspark.context import SparkContext
from awsglue.context import GlueContext
from awsglue.job import Job
from pyspark.sql.functions import *
from pyspark.sql.types import *

# Initialize
args = getResolvedOptions(sys.argv, ['JOB_NAME', 'INPUT_PATH', 'OUTPUT_PATH'])
sc = SparkContext()
glueContext = GlueContext(sc)
spark = glueContext.spark_session
job = Job(glueContext)
job.init(args['JOB_NAME'], args)

# Read raw data
print("Reading raw data from S3...")
df = spark.read.csv(args['INPUT_PATH'], header=True, inferSchema=True)
print(f"Loaded {df.count()} rows")

# Data quality check - log missing values
print("Checking data quality...")
for column_name in df.columns:
    null_count = df.filter(col(column_name).isNull()).count()
    if null_count > 0:
        print(f"  {column_name}: {null_count} missing values")

# Handle missing values
print("Handling missing values...")
df = df.fillna({'Age': df.agg({'Age': 'median'}).first()[0]})
df = df.fillna({'Fare': df.agg({'Fare': 'median'}).first()[0]})
df = df.fillna({'Embarked': 'S'})

# Encode categorical variables
print("Encoding categorical variables...")
df = df.withColumn('Sex', when(col('Sex') == 'male', 0).otherwise(1))

embarked_map = {'C': 0, 'Q': 1, 'S': 2}
df = df.withColumn('Embarked', 
                   when(col('Embarked') == 'C', 0)
                   .when(col('Embarked') == 'Q', 1)
                   .otherwise(2))

# Feature engineering
print("Creating new features...")
df = df.withColumn('FamilySize', col('SibSp') + col('Parch'))
df = df.withColumn('IsAlone', when(col('FamilySize') == 0, 1).otherwise(0))

# Select final features
print("Selecting final features...")
feature_cols = ['Survived', 'Pclass', 'Sex', 'Age', 'Fare', 'Embarked', 'FamilySize', 'IsAlone']
df_final = df.select(feature_cols)

# Data validation
print("Validating processed data...")
assert df_final.count() > 0, "No rows in final dataset!"
assert df_final.filter(col('Survived').isNull()).count() == 0, "Target has nulls!"

print(f"Final dataset: {df_final.count()} rows, {len(df_final.columns)} columns")

# Write to S3 as Parquet
print(f"Writing to {args['OUTPUT_PATH']}...")
df_final.coalesce(1).write.mode('overwrite').parquet(args['OUTPUT_PATH'])

print("ETL job completed successfully!")
job.commit()