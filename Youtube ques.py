# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.functions import *
from pyspark.sql.types import *
from pyspark.sql.window import *

spark=SparkSession.builder.appName('EY').getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### EY Data Engineer Interview Question |PySpark| Split Full Name into First, Middle & Last Names

# COMMAND ----------

data=[('Renu Sri Loganathan'),
      ('Ruba priya')]

schema=StructType([
    StructField('name',StringType(),True)
])

df=spark.createDataFrame(data,schema)

df.display()

df_split=df.withColumn('Firstname',split(col('name'),' ').getItem(0))\
    .withColumn('Middlename',when (size(split(col('name'),' '))==3,split(col('name'),' ').getItem(1)).otherwise(lit('none')))\
        .withColumn('Lastname',when (size(split(col('name'),' '))==3,split(col('name'),' ').getItem(2)).otherwise(split(col('name'),' ').getItem(1)))

df_split.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###EY Data Engineer Interview Questions (Part 2) | Count no of Consonants| PySpark Regex

# COMMAND ----------

data=[('Hello World'),
      ('Spark is amazing'),
      ('spark makes big data fast')]
schema=StructType([
    StructField('name',StringType(),True)
])

df=spark.createDataFrame(data,schema)

df.display()
df_lower=df.withColumn('lowername',lower(col('name')))
df_replace=df_lower.withColumn('coumt',regexp_replace(col('lowername'), 'a|e|i|o|u|\\s',''))
#df_replace=df_lower.withColumn('coumt',regexp_replace(col('lowername'),' [^b-df-hj-np-tv-z]','')) #consonants
df_count=df_replace.withColumn('count',length(col('coumt')))
df_count.display()

# COMMAND ----------

df_split=df.withColumn('split_name',explode(split(col('name'),' ')))
df_grp=df_split.groupBy(col('name')).count()
df_grp.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Crack PwC Data Engineer Interviews| Real-Life PySpark Question with Join, Filter & Rename Column

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, IntegerType, StringType

source_data = [
    (10, "A"),
    (20, "B"),
    (30, "C"),
    (40, "D")
]

schema = StructType([
    StructField("id", IntegerType(), True),
    StructField("name", StringType(), True)
])

df_source = spark.createDataFrame(source_data, schema)

df_source.display()
target_data = [
    (10, "A"),
    (20, "B"),
    (40, "X"),
    (50, "F")
]

df_target = spark.createDataFrame(target_data, schema)

df_target.display()


# COMMAND ----------

df_new=df_source.join(df_target,on='id',how='left_anti').withColumn('result',lit('new in src')).select('id','result')
df_both=df_source.join(df_target,on='id',how='inner').filter(df_source['name'] !=df_target['name'])\
    .withColumn('result',lit('in both')).select('id','result')
df_tar=df_target.join(df_source,on='id',how='left_anti').withColumn('result',lit('new in tar')).select('id','result')

df_new.unionByName(df_both).unionByName(df_tar).display()



# above or below approach
df_outer=df_source.alias('s').join(df_target.alias('t'),on='id',how='outer')\
    .withColumn('result',when( (col('s.id')== col('t.id')) & (col('s.name')!= col('t.name')),'mismatch')\
        .when(col('s.name').isNull(),'new_src').when(col('t.name').isNull(),'new_tgt').otherwise(lit('null')))
    
df_outer.display()


# COMMAND ----------

# MAGIC %md
# MAGIC ###EY Data Engineer Interview Question| PySpark| Get Customer Journey Location |groupBy, Join, Subtract

# COMMAND ----------

data = [
    ('A001', 'Tokyo', 'Seoul'),
    ('A001', 'Singapore', 'Tokyo'),
    ('A001', 'Seoul', 'Bangkok'),
    ('A001', 'Bangkok', 'Manila'),
    ('B002', 'Paris', 'Rome'),
    ('B002', 'Berlin', 'Paris'),
    ('B002', 'Rome', 'Madrid'),
    ('C003', 'Cairo', 'Dubai'),
    ('C003', 'Istanbul', 'Cairo'),
    ('C003', 'Dubai', 'Riyadh'),
    ('C003', 'Riyadh', 'Doha'),
    ('D004', 'Toronto', 'Montreal'),
    ('D004', 'Vancouver', 'Toronto'),
    ('E005', 'Sydney', 'Melbourne')
]


schema = StructType([
    StructField("customer", StringType(), True),
    StructField("start_location", StringType(), True),
    StructField("end_location", StringType(), True)
])


df = spark.createDataFrame(data=data, schema=schema)

print("Original DataFrame:")
df.display()
df_start=df.select('customer','start_location')
df_end=df.select('customer','end_location')

df_final=df_start.subtract(df_end)
df_final1=df_end.subtract(df_start)

df_final2=df_final.alias('s').join(df_final1.alias('t'),on='customer',how='inner')

df_final.display()
df_final1.display()
df_final2.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###EXPLODE AND COLLECT_LIST

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType

data = [
    ("Forest", "Badminton,Cricket,Football"),
    ("Bob", "Football,Football"),
    ("Ace", "Golf,Volleyball")
]

schema = StructType([
    StructField("Name", StringType(), True),
    StructField("Sports", StringType(), True)
])

df = spark.createDataFrame(data, schema)

df.display()
df_explode=df.withColumn('result',explode(split(col('sports'),',')))
df_collectlist=df_explode.groupBy(col('name')).agg(concat_ws(',',collect_set(col('result'))))
df_explode.display()
df_collectlist.display()


# COMMAND ----------

# MAGIC %md
# MAGIC ### KPMG Data Engineer Interview Question –Swap Consecutive Names in PySpark |LAG, LEAD, ORDER BY |Part2

# COMMAND ----------

data = [
    (1,"Alice", 1200, "Q1"),
    (2,"Bob", 900, "Q1"),
    (3,"Charlie", 1500, "Q2"),
    (4,"David", 1700, "Q2"),
    (5,"Eva", 1100, "Q3"),
    (6,"Frank", 220, "Q3"),
    (7,"Grace", 1300, "Q4")
]

df = spark.createDataFrame(data, ["id", "name", "sales", "quarter"])
df.display()
df_win=Window.partitionBy(col('quarter')).orderBy('id')
df_lead_lag=df.withColumn('lead',lead('name').over(df_win)).withColumn('lag',lag('name').over(df_win))

df_result=df_lead_lag.withColumn('swap',coalesce(when(col('id')%2==0,col('lag')).otherwise(col('lead')),col('name'))).select('id','name','swap')
df_result.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###PWC Data Engineer 2025 Most Asked Interview Questions | Grab the HIGHEST PACKAGE

# COMMAND ----------

data = [
    (1),
    (3),
    (5),
    (9)
]
df = spark.createDataFrame(data, ["id"])

df_min=df.select(min('id')).first()[0]
df_max=df.select(max('id')).first()[0]
df_range=spark.range(df_min -1,df_max)
df_range.display()

df_join=df_range.join(df,on='id',how='left_anti')
df_join.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Most Asked Deloitte Data Engineer Interview Questions (Part 3) | LAG, GROUP BY, MIN & Alias

# COMMAND ----------


product_data = [
    (1, 'Laptops', 'Electronics'),
    (2, 'Jeans', 'Clothing'),
    (3, 'Chairs', 'Home Appliances')
]

product_schema = ['product_id', 'product_name', 'category']

product_df = spark.createDataFrame(data=product_data, schema=product_schema)
print("Product DataFrame:")
product_df.display()


sales_data = [
    (1, 2019, 1000.00),
    (1, 2020, 1500.00),
    (1, 2021, 1200.00),
    (2, 2019, 500.00),
    (2, 2020, 700.00),
    (2, 2021, 900.00),
    (3, 2019, 400.00),
    (3, 2020, 450.00),
    (3, 2021, 300.00)
]

sales_schema = ['product_id', 'year', 'total_sales_revenue']

sales_df = spark.createDataFrame(data=sales_data, schema=sales_schema)
print("Sales DataFrame:")
sales_df.display()         

# COMMAND ----------

df_win=Window.partitionBy(col('product_id')).orderBy(col('year'))
df_lag=sales_df.withColumn('previous',lag('total_sales_revenue').over(df_win))\
    .withColumn('compare',when(col('total_sales_revenue') > col('previous'),lit('0'))\
        .when(col('total_sales_revenue') < col('previous'),lit('1'))\
            .when(col('previous').isNull(),'0').otherwise(lit('0')))
df_lag.display()

df_final=df_lag.groupBy(col('product_id')).agg(sum(col('compare')).alias('result')).filter(col('result')== 0)
df_join=product_df.join(df_final,on='product_id',how='inner')
df_join.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###Null and empty

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType
from pyspark.sql import Row

data = [
    ("Kolkata", '', "WB"),
    ('', "Gurgaon", None),
    (None, '', "banaglore")  # assuming "banaglore" is as-is from your data
]

schema = StructType([
    StructField("city1", StringType(), True),
    StructField("city2", StringType(), True),
    StructField("city3", StringType(), True)
])

df = spark.createDataFrame(data, schema)

df.display()

df_result = df.withColumn(
    'result',
    coalesce(
        when(col('city1') != '', col('city1')),
        when(col('city2') != '', col('city2')),
        when(col('city3') != '', col('city3'))
    )
)
df_result.display()


# COMMAND ----------

# MAGIC %md
# MAGIC ###PIVOT AND STACK

# COMMAND ----------

from pyspark.sql.types import StructType, StructField, StringType, IntegerType

# Sample data: sales of products by region
data = [
    ("North", "Apple", 50),
    ("North", "Banana", 30),
    ("South", "Apple", 70),
    ("South", "Banana", 60),
    ("East",  "Apple", 40),
    ("East",  "Banana", 20)
]

schema = StructType([
    StructField("region", StringType(), True),
    StructField("product", StringType(), True),
    StructField("sales", IntegerType(), True)
])

df = spark.createDataFrame(data, schema)

df.display()


# COMMAND ----------

df_pivot=df.groupBy('product').pivot('region').agg(sum(col('sales')))
df_pivot.display()

df_stack=df_pivot.selectExpr('product',
                             "stack(3,'East',east,'north',north,'south',south) as (region,sales)")

df_stack.display()

# COMMAND ----------

# MAGIC %md
# MAGIC #windows,lead,lag,rank,denserank,rownumber,isnull,notisnull,isin
# MAGIC

# COMMAND ----------

df_win=Window.partitionBy(col('employee')).orderBy(desc(col('amount')))
df_window=df.withColumn('Rank',rank().over(df_win)).withColumn('denseRank',dense_rank().over(df_win)).withColumn('Rownum',row_number().over(df_win))\
    .withColumn('lead',coalesce(lead('amount').over(df_win),col('amount'))).withColumn('lag',lag('amount').over(df_win)).withColumn('leadd',lead('amount').over(df_win))
df_window.display()
#df_isnotnull=df_window.filter(col('lead').isNotNull())
df_filtered = df.filter(col("employee").isin("Alice", "Bob"))
df_filtered.display()

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import max_by,min_by, col, struct

spark = SparkSession.builder.getOrCreate()

data = [
    (1, "a", 10),
    (1, "b", 5),
    (2, "c", 8),
    (2, "d", 12)
]

df = spark.createDataFrame(data, ["group_id", "name", "value"])

# Get the 'name' corresponding to the minimum 'value' per group
df.groupBy("group_id").agg(
    min_by("name", "value").alias("min_name"),
    max_by("name", "value").alias("max_name")
).show()


# COMMAND ----------

from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col, when, first

# Window definitions
window_min = Window.partitionBy("group_id").orderBy(col("value").asc())
window_max = Window.partitionBy("group_id").orderBy(col("value").desc())

# Add row numbers
df_rn = df.withColumn("rn_min", row_number().over(window_min)).withColumn("rn_max", row_number().over(window_max))

# Aggregate to get single row per group
df_result = df_rn.groupBy("group_id")\
    .agg(
             first(when(col("rn_min") == 1, col("name")), True).alias("min_name"),
             first(when(col("rn_max") == 1, col("name")), True).alias("max_name")
         )


df_result.show()
