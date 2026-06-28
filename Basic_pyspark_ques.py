# Databricks notebook source
from pyspark.sql import SparkSession
from pyspark.sql.types import*
from pyspark.sql.window import*
from pyspark.sql.functions import*

spark=SparkSession.builder.appName('demo').getOrCreate()

# COMMAND ----------

# Preferred to save time and prevent from wrong datatype
# Manually giving the datatype(inferschema =false)
# inferSchema=True automatically detects column data types.But it takes extra read time and may infer incorrect data types if the data is inconsistent.
# Eg: numeric column with one text value may be inferred as StringType
data = [
    ('Renu', 99, 'coimbatore'),
    ('Ram', 98, 'chennai'),
    (None,7,None)
]

#value = ['Name', 'Mark', 'location']

value=StructType([
    StructField('name',StringType(),True),
    StructField('Mark',IntegerType(),False),
    StructField('Subject',StringType(),True)
])

df = spark.createDataFrame(data, schema=value)

df.display()

# COMMAND ----------

#group by

data = [
    (1, "A", "IT", 5000),
    (2, "B", "IT", 7000),
    (3, "C", "IT", 5000),
    (4, "D", "HR", 4000),
    (5, "E", "HR", 6000),
    (6, "F", "Finance", 8000),
    (7, "G", "Finance", 8000),
    (8, "H", "Finance", 10000),
]

columns = ["EmpID", "EmpName", "Department", "salary"]

df = spark.createDataFrame(data, columns)

df_grp = df.groupBy("Department").agg(
    sum("salary").alias("sum_sal"),
    avg("salary").alias("avg_sal"),
    countDistinct("EmpID").alias("emp_count")
).orderBy(col("sum_sal").desc())

df_grp.display()

# COMMAND ----------

#windows


data = [
    (1, "Arun", "IT", 50000),
    (2, "Bala", "IT", 60000),
    (3, "Chitra", "IT", None),
    (4, "Divya", "HR", 45000),
    (5, "Eshan", "HR", 45000),
    (6, "Farah", "HR", 50000),
    (7, "Gokul", "Finance", 70000),
    (8, "Hari", "Finance", None),
    (9, "Ishita", "Finance", 70000)
]

columns = ["Id", "Name", "Department", "Salary"]

df = spark.createDataFrame(data, columns)
df.display()

window_spec = Window.partitionBy("Department").orderBy("Salary")

df1 = df.withColumn("Lead_Salary", lead("Salary",2).over(window_spec)) \
        .withColumn("Lag_Salary", lag("Salary").over(window_spec)) \
        .withColumn("Rank", rank().over(window_spec)) \
        .withColumn("DenseRank", dense_rank().over(window_spec)) \
        .withColumn("RowNumber", row_number().over(window_spec)) \
        .withColumn("IsNull_Salary", col("Salary").isNull()) \
        .withColumn("IsNotNull_Salary", col("Salary").isNotNull())#.filter(col('Lead_Salary').isNull())# we use isnull/not is null to filter the records like below
        

df1.show()

# COMMAND ----------

#Windows fn 

data = [
    ("HR", 1000),
    ("HR", 1200),
    ("HR", 1500),
    ("HR", 1800),
    ("HR", 2000),
    ("IT", 1100),
    ("IT", 1300),
    ("IT", 1600),
    ("IT", 1900),
    ("IT", 2200)
]

columns = ["Department", "salary"]

df = spark.createDataFrame(data, columns)

win_cumulative = (
    Window
    .partitionBy("Department")
    .orderBy(col("salary"))
    .rowsBetween(Window.unboundedPreceding, Window.currentRow)
)

# Last 3 rows (rolling window)
win_last_3 = (
    Window
    .partitionBy("Department")
    .orderBy(col("salary"))
    .rowsBetween(-2, Window.currentRow)
)

# Current + next 2 rows (forward window)
win_forward = (
    Window
    .partitionBy("Department")
    .orderBy(col("salary"))
    .rowsBetween(0, 2)
)

# Previous 2 + current row
win_prev2 = (
    Window
    .partitionBy("Department")
    .orderBy(col("salary"))
    .rowsBetween(-2, 0)
)

###########################################

df_result = df \
    .withColumn("running_salary", sum("salary").over(win_cumulative)) \
    .withColumn("last_3_sum", sum("salary").over(win_last_3)) \
    .withColumn("forward_3_sum", sum("salary").over(win_forward)) \
    .withColumn("prev2_sum", sum("salary").over(win_prev2))

df_result.show()

# COMMAND ----------

#case stmt /between

data = [
    (1, "A", 45000),
    (2, "B", 47000),
    (3, "C", 49000),
    (4, "D", 50000),
    (5, "E", 52000),
    (6, "F", 60000)
]

columns = ["EmpID", "EmpName", "salary"]

df = spark.createDataFrame(data, columns)

df.display()

#df_val = df.withColumn("comments",
#    when(col("salary") < 47000, "low")
#    .when((col("salary") >= 47000) & (col("salary") <= 50000), "medium")
#    .otherwise("high")
#)

### OR

df_val = df.withColumn("comments",
    when(col("salary") < 47000, "low")
    .when(col("salary").between(47000, 50000), "medium")
    .otherwise("high")
)

df_val.display()

# COMMAND ----------

#DROP duplicate records,fill random values in null value

data = [
    (1, "A", "IBM", "India", 50000),
    (2, "B", "IBM", None, 60000),
    (3, "C", None, "USA", None),
    (4, "D", "TCS", "India", 70000),
    (4, "D", "TCS", "India", 70000),   # duplicate row
    (5, "E", None, None, 80000),
]

columns = ["EmpID", "EmpName", "Company", "Country", "Salary"]

df = spark.createDataFrame(data, columns)

df.display()

# COMMAND ----------

#Drop NA

df_any = df.dropna(how="any")
df_any.display()

df_subset = df.dropna(subset=["Company", "Country"])
df_subset.display()

# COMMAND ----------

#Fill NA

df_mean = df.select(mean(col("Salary"))).first()[0] #same way we can calcute avg,sum
df_fill = df.fillna({"Salary": df_mean,"Country": "loc"})
df_fill.display()

#Harcode the specific value in all empty place irrespective of columns
df_fill_emp = df.fillna("empty")
df_fill_emp.display()

# COMMAND ----------

#Drop the dupicate records

df_distinct = df.distinct()
df_distinct.display()

df_dup_all = df.dropDuplicates()
df_dup_all.display()

# Based on specific columns 
df_dup_subset = df.dropDuplicates(["EmpID", "Company"])
df_dup_subset.display()

# COMMAND ----------

#Pivot and Stack(Transpose)
data = [
    ('Accenture', 'IT', 50000),
    ('Accenture', 'HR', 30000),
    ('Accenture', 'Finance', 40000),
    ('CTS', 'IT', 45000),
    ('CTS', 'HR', 25000),
    ('Infosys', 'IT', 48000),
    ('Infosys', 'Finance', 42000),
    ('TCS', 'IT', 47000),
    ('TCS', 'HR', 26000)
]

columns = ['Company', 'Department', 'Salary']

df_pivot = spark.createDataFrame(data, columns)

df_pivot.show()

# ---------------- PIVOT ----------------
df_p = df_pivot.groupBy("Company") \
    .pivot("Department").agg(sum("Salary"))

df_p.show()

#  STACK 
df_stack = df_p.selectExpr(
    "Company",
    "stack(3, 'Finance', Finance, 'HR', HR, 'IT', IT) as (Department, Salary)"
)

df_stack.show()

# COMMAND ----------

#collectlist and Flatten

data = [
    ("A", [1, 2, 3]),
    ("A", [4]),
    ("B", [5, 6]),
    ("B", [1]),
    ("C", None)
]

schema = StructType([
    StructField("key", StringType(), True),
    StructField("values", ArrayType(IntegerType()), True)
])

df = spark.createDataFrame(data, schema)

df_grp=df.groupBy(col('key')).agg(collect_list(col('values')))
df_grp.display()

df_group=df.groupBy(col('key')).agg(flatten(collect_list(col('values'))))
df_group.display()

# COMMAND ----------

# Exlode outer-->it will inclue null records as well

df_explode=df.withColumn('exp',explode_outer(col('values')))
df_explode.display()

df_collect=df_explode.groupby('key').agg(collect_list(col('exp')))
df_collect.display()

# COMMAND ----------

# AND and OR ,collect_list

data = [
    ("Rahul", 85, 1),
    ("Adarsh", 73, 1),
    ("Riti", 95, 1),
    ("Viraj", 80, 1),
    ("Vimal", 83, 2),
    ("Neha", 77, 2),
    ("Priti", 73, 2),
    ("Himanshi", 85, 2)
]

columns = ["passenger_name", "weight_kg", "lift_id"]

df = spark.createDataFrame(data, columns)
df.show()

df_win=Window.partitionBy(col('lift_id')).orderBy(col('weight_kg'))
df_result=df.withColumn('running_total',sum('weight_kg').over(df_win))\
    .filter(
        ((col("lift_id") == 1) & (col("running_total") <= 300)) | ((col("lift_id") == 2) & (col("running_total") <= 350))
    )
df_result.display()

df_group=df_result.groupBy(col('lift_id')).agg(concat_ws(',',collect_list(col('passenger_name'))))
df_group.display()

# COMMAND ----------

# Array sort will sort the products in specific column like this(Camera,Laptop,TV)

data = [
    ("Electronics", "TV"),
    ("Electronics", "Laptop"),
    ("Electronics", "Camera"),
    ("Grocery", "Rice"),
    ("Grocery", "Wheat"),
    ("Grocery", "Sugar")
]

columns = ["Category", "ProductName"]

df = spark.createDataFrame(data, columns)

result_df = df.groupBy("Category").agg(
        concat_ws(
            ",",
            array_sort(collect_list("ProductName"))
        ).alias("Products")
    )

result_df.show(truncate=False)

# COMMAND ----------

#join,union

products_data = [
    (1, "iPhone", "Apple"),
    (2, "Galaxy", "Samsung"),
    (3, "Pixel", "Google"),
    (4, "Redmi", "Xiaomi")
]

products_cols = ["ProductID", "ProductName", "Brand"]
products_df = spark.createDataFrame(products_data, products_cols)


orders_data = [
    (101, 1, 2),
    (102, 2, 1),
    (103, 3, 5),
    (104, 1, 1),
    (105, 4, 3)
]

orders_cols = ["OrderID", "ProductID", "Quantity"]
orders_df = spark.createDataFrame(orders_data, orders_cols)

new_products_data = [
    (5, "OnePlus", "OnePlus"),
    (6, "iPad", "Apple")
]

products_cols = ["ProductID", "ProductName", "Brand"]
new_products_df = spark.createDataFrame(new_products_data, products_cols)




joined_df = orders_df.join(products_df, "ProductID", "inner")

filtered_df = products_df.filter(
    col("Brand").isin("Apple", "Samsung")
)

union_df = products_df.union(new_products_df)

joined_df.display()
filtered_df.display()
union_df.display()


# COMMAND ----------

#union by name

data1 = [
    (1, "Apple", "iPhone"),
    (2, "Samsung", "Galaxy")]

df1 = spark.createDataFrame(data1, ["ProductID", "Brand", "ProductName"])
df1.show()

data2 = [
    ("OnePlus", 3, "Nord"),
    ("Google", 4, "Pixel")]

df2 = spark.createDataFrame(data2,["Brand", "ProductID", "ProductName"]  # different order
)
df2.show()

final_df = df1.unionByName(df2)
final_df.display()


# COMMAND ----------

# MAGIC %md
# MAGIC # Interview questions
# MAGIC ###IBM

# COMMAND ----------

#Need firstname,Middle name,lastname(Asked in IBM 1 st round)
spark = SparkSession.builder.appName("NameSplit").getOrCreate()

data = [
    ("ABC XYZ",),
    ("ABC PQR XYZ",)
]

df = spark.createDataFrame(data, ['name'])
df.display()

df_name=df.withColumn('firstname',split('name',' ').getItem(0))\
    .withColumn('middle_name',when(size(split('name',' ')) ==3,split('name',' ').getItem(1)).otherwise(lit('none')))\
        .withColumn('last_name',when(size(split('name',' ')) ==3,split('name',' ').getItem(2)).otherwise(split('name',' ').getItem(1)))

df_name.display()

# COMMAND ----------

#Find the customer with the highest number of orders in each city.(Asked in IBM 1 st round)
data = [
    (1, 101, "Amit", "Mumbai", 2000),
    (2, 101, "Amit", "Mumbai", 3000),
    (3, 102, "Priya", "Mumbai", 2500),
    (4, 103, "Rahul", "Pune", 4000),
    (5, 103, "Rahul", "Pune", 3500),
    (6, 104, "Sneha", "Pune", 1500),
    (7, 105, "Karan", "Bangalore", 5000),
    (8, 105, "Karan", "Bangalore", 6000),
    (9, 105, "Karan", "Bangalore", 2000)
]

columns = ["order_id", "customer_id", "customer_name", "city", "amount"]
df = spark.createDataFrame(data, columns)
df.display()

df_count = df.groupBy("city", "customer_name")\
    .agg(count("*").alias("total_count"))

win = Window.partitionBy("city").orderBy(desc("total_count"))

result = df_count.withColumn("rank", dense_rank().over(win))

result.display()

# COMMAND ----------

# give top company based on their total revenue(Asked in IBM 2 nd round)

#calculate the difference between the current year revenue and previous year revenue in sql(using the below dataset) (Asked in IBM 2 nd round)-->use window fn(lag)

from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("CompanyRevenue").getOrCreate()

data = [
    ("IBM", 2021, 9000),
    ("IBM", 2020, 7000),
    ("ACN", 2020, 7000),
    ("ACN", 2019, 5000),
    ("INFY", 2023, 7000),
    ("INFY", 2024, 9000)
]

columns = ["CO_NAME", "YEAR", "REVENUE"]
df = spark.createDataFrame(data, columns)
df.display()


df_grp = df.groupBy("CO_NAME").agg(sum("REVENUE").alias("total_revenue"))
win = Window.orderBy(desc("total_revenue"))
df_rank = df_grp.withColumn("dense_rank",dense_rank().over(win)).filter(col('dense_rank')==1)
df_rank.display()

# COMMAND ----------

#Flatten the structure and how many records will get in an output (Asked in IBM 2 nd round)

data = [
    {
        "order_id": 101,
        "customer": {
            "name": "Alice",
            "email": "alice@example.com"
        },
        "items": [
            {"product": "Laptop", "qty": 1, "price": 75000},
            {"product": "Mouse", "qty": 2, "price": 500}
        ],
        "payment": {
            "method": "UPI",
            "status": "Success"
        }
    },
    {
        "order_id": 102,
        "customer": {
            "name": "Bob",
            "email": "bob@example.com"
        },
        "items": [
            {"product": "Keyboard", "qty": 1, "price": 1200}
        ],
        "payment": {
            "method": "Card",
            "status": "Pending"
        }
    }
]

df = spark.createDataFrame(data)
df.display()

df_exploded = df.withColumn("item", explode(col("items")))

df_result = df_exploded.select(
    col("order_id"),
    col("customer.name").alias("customer_name"),
    col("customer.email").alias("cust_email"),
    col("item.product").alias("item_product"),
    col("item.qty").alias("quantity"),
    col("item.price").alias("price"),
    col("payment.method").alias("pay_method"),
    col("payment.status").alias("pay_status")
)

df_result.display()

# COMMAND ----------

# MAGIC %md
# MAGIC ###EXL

# COMMAND ----------

Orders-->| order_id | item_id | price | address_id |
Address-->| address_id | city |

ques 1:Find the number of orders per city in sql

ques 2: For each order, find the top 3 most expensive items using price

ques 3:
Table A --> a a d b c c
Table B --> NULL NULL D B A A
Give the result for all 4 joins(right,inner,left,outer)

Pyspark
1.Fillna
2.group by ques
3.Both pyspark and sql -->Find the hrs or each employee
Emp_name	Type	date_time
Emp A	Login	07-04-2026 07:00
Emp A	Logout	07-04-2026 12:00
Emp A	Login	07-04-2026 14:00
Emp B	Login	07-04-2026 08:00
Emp C	Login	07-04-2026 07:00
Emp C	Logout	07-04-2026 12:00

# COMMAND ----------

# MAGIC %md
# MAGIC ###PWC Total sales by region where amount > 10000

# COMMAND ----------


df_result = df \
    .filter(col("sales") > 10000) \
    .groupBy("region") \
    .agg(sum("sales").alias("total_sales"))

df_result.display()

# COMMAND ----------

#3rd Highest Salary IN SQL

WITH cte AS (
    SELECT *,
           ROW_NUMBER() OVER (ORDER BY salary DESC) AS rn
    FROM employee
)
SELECT *
FROM cte
WHERE rn = 3;

# COMMAND ----------

data = [
    (1, "flight1", "Delhi", "Hyderabad"),
    (1, "flight2", "Hyderabad", "Kochi"),
    (1, "flight3", "Kochi", "Mangalore"),
    (2, "flight1", "Mumbai", "Ayodhya"),
    (2, "flight2", "Ayodhya", "Gorakhpur")
]

columns = ["cust_id", "flight_id", "origin", "destination"]

df = spark.createDataFrame(data, columns)
df.show()

w = Window.partitionBy("cust_id").orderBy("flight_id")

df_window = df.withColumn("first_origin", first("origin").over(w)) \
              .withColumn("last_destination", last("destination").over(w))

result = df_window.groupBy("cust_id") \
                  .agg(
                      first("first_origin").alias("origin"),
                      last("last_destination").alias("destination")
                  )

result.show()


# COMMAND ----------

# SALTING(Not much imporant for interview ,they ask the definition).Used during data skewness

SALT_BUCKETS = 3
new_df_salted = new_df.withColumn("salt",floor(rand() * SALT_BUCKETS))#gives the random values for each row

existing_df_salted = existing_df.withColumn("salt",explode(array([lit(i) for i in range(SALT_BUCKETS)])))#small dataset give copy of each records 3 time becoz we mentioned sat_ucket is 3 

joined_df = new_df_salted.join(existing_df_salted,on=["Customer_ID", "salt"],how="left")



# COMMAND ----------

##write and read format

#---------- READ FORMATS ----------

# CSV
df_csv = spark.read.format("csv") \
    .option("header", "true") \
    .option("inferSchema", "true") \
    .load("/mnt/data/input.csv")

# PARQUET
df_parquet = spark.read.format("parquet").load("/mnt/data/input_parquet")

# JSON
df_json = spark.read.format("json") \
    .option("multiline", "true") \
    .load("/mnt/data/input.json")

# AVRO
df_avro = spark.read.format("avro").load("/mnt/data/input_avro")

# ORC
df_orc = spark.read.format("orc").load("/mnt/data/input_orc")

# TEXT
df_text = spark.read.format("text").load("/mnt/data/input.txt")

# DELTA (file)
df_delta_file = spark.read.format("delta").load("/mnt/data/input_delta")

# TABLE READS
df_table = spark.read.table("default.customer_table")
df_table_sql = spark.sql("SELECT * FROM default.customer_table")

# ---------- WRITE FORMATS ----------overwrite/append

# CSV
df_csv.write.format("csv") \
    .option("header", "true") \
    .mode("overwrite") \
    .save("/mnt/data/output_csv")

# PARQUET
df_csv.write.format("parquet") \
    .mode("overwrite") \
    .save("/mnt/data/output_parquet")

# JSON
df_csv.write.format("json") \
    .mode("overwrite") \
    .save("/mnt/data/output_json")

# AVRO
df_csv.write.format("avro") \
    .mode("overwrite") \
    .save("/mnt/data/output_avro")

# ORC
df_csv.write.format("orc") \
    .mode("overwrite") \
    .save("/mnt/data/output_orc")

# TEXT
df_csv.selectExpr("CAST(col1 AS STRING)") \
    .write.format("text") \
    .mode("overwrite") \
    .save("/mnt/data/output_text")

# DELTA (file)
df_csv.write.format("delta") \
    .mode("overwrite") \
    .save("/mnt/data/output_delta")

# ---------- TABLE WRITES ----------

# Managed table
df_csv.write.mode("overwrite").saveAsTable("default.customer_managed")

# Delta table
df_csv.write.format("delta") \
    .mode("overwrite") \
    .saveAsTable("default.customer_delta")

# Insert into existing table
df_csv.write.mode("append").insertInto("default.customer_delta")

# ---------- WRITE MODES ----------
# overwrite | append | ignore | error / errorifexists

print("All read & write formats executed in a single cell.")


# COMMAND ----------

#Read modes(Helps to drop/create the new column for invalid records)

path = "data.csv"  # replace with your file path

print("=== PERMISSIVE MODE ===")
df_permissive = (
    spark.read
    .option("header", "true")
    .option("mode", "PERMISSIVE")
    .option("columnNameOfCorruptRecord", "_corrupt_record")
    .csv(path)
)
df_permissive.show(truncate=False)

print("Corrupt rows (PERMISSIVE):")
df_permissive.filter("_corrupt_record IS NOT NULL").show(truncate=False)


print("=== DROPMALFORMED MODE ===")
df_drop = (
    spark.read
    .option("header", "true")
    .option("mode", "DROPMALFORMED")
    .csv(path)
)
df_drop.show(truncate=False)


print("=== FAILFAST MODE ===")
try:
    df_failfast = (
        spark.read
        .option("header", "true")
        .option("mode", "FAILFAST")
        .csv(path)
    )
    df_failfast.show(truncate=False)
except Exception as e:
    print("FAILFAST error:")
    print(e)


# COMMAND ----------

# MAGIC %md
# MAGIC ### **DATE FUNCTIONS**

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    col, to_date, to_timestamp, date_format, datediff, date_add, date_sub,
    add_months, months_between, year, month, dayofmonth, hour, minute, second,
    current_date, current_timestamp, trunc, next_day, weekofyear, unix_timestamp,
    from_unixtime, expr, lit
)

# Initialize Spark
spark = SparkSession.builder.appName("DateFunctionsSeparateDFs").getOrCreate()

# Base DataFrame with sample data
data = [
    ("2023-01-10", "2023-01-15 13:45:30"),
    ("2023-02-20", "2023-02-25 09:15:00"),
    ("2023-03-05", "2023-03-10 22:30:45")
]
columns = ["date_str", "timestamp_str"]
df = spark.createDataFrame(data, columns)
df.display()



# COMMAND ----------

# ---- 1. to_date ----
df_to_date = df.select("date_str", to_date(col("date_str"), "yyyy-MM-dd").alias("date"))
df_to_date.display()

# ---- 2. to_timestamp ----
df_to_timestamp = df.select("timestamp_str", to_timestamp(col("timestamp_str"), "yyyy-MM-dd HH:mm:ss").alias("timestamp"))
df_to_timestamp.display()

# ---- 3. date_format ----
df_date_format = df.select("date_str", date_format(to_date(col("date_str")), "dd/MM/yyyy").alias("formatted_date"))
df_date_format.display()

# ---- 4. datediff ----
df_datediff = df.select(
    "date_str",
    datediff(current_date(), to_date(col("date_str"))).alias("days_diff")
)
df_datediff.display()

# ---- 5. date_add ----
df_date_add = df.select("date_str", date_add(to_date(col("date_str")), 10).alias("date_plus_10"))
df_date_add.display()

# ---- 6. date_sub ----
df_date_sub = df.select("date_str", date_sub(to_date(col("date_str")), 5).alias("date_minus_5"))
df_date_sub.display()

# ---- 7. add_months ----
df_add_months = df.select("date_str", add_months(to_date(col("date_str")), 2).alias("date_plus_2_months"))
df_add_months.display()

# ---- 8. months_between ----
df_months_between = df.select("date_str", months_between(current_date(), to_date(col("date_str"))).alias("months_between"))
df_months_between.display()

# ---- 9. Extract parts ----
df_extract_parts = df.select(
    "date_str",
    year(to_date(col("date_str"))).alias("year"),
    month(to_date(col("date_str"))).alias("month"),
    dayofmonth(to_date(col("date_str"))).alias("day"),
    hour(to_timestamp(col("timestamp_str"))).alias("hour"),
    minute(to_timestamp(col("timestamp_str"))).alias("minute"),
    second(to_timestamp(col("timestamp_str"))).alias("second")
)
df_extract_parts.display()

# ---- 10. current_date and current_timestamp ----
df_current = df.select(current_date().alias("current_date"), current_timestamp().alias("current_timestamp"))
df_current.display()

# ---- 11. trunc ----
df_trunc = df.select(
    "date_str",
    trunc(to_date(col("date_str")), "year").alias("year_start"),
    trunc(to_date(col("date_str")), "month").alias("month_start")
)
df_trunc.display()

# ---- 12. next_day ----
df_next_day = df.select("date_str", next_day(to_date(col("date_str")), "Sun").alias("next_sunday"))
df_next_day.display()

# ---- 13. weekofyear ----
df_weekofyear = df.select("date_str", weekofyear(to_date(col("date_str"))).alias("week_of_year"))
df_weekofyear.display()

# ---- 14. unix_timestamp and from_unixtime ----
df_unix = df.select(
    "timestamp_str",
    unix_timestamp(col("timestamp_str"), "yyyy-MM-dd HH:mm:ss").alias("unix_time"),
    from_unixtime(unix_timestamp(col("timestamp_str"), "yyyy-MM-dd HH:mm:ss")).alias("from_unix_time")
)
df_unix.display()

# ---- 15. expr date arithmetic ----
df_expr = df.select(
    "date_str",
    expr("date_add(to_date(date_str), 7)").alias("date_plus_7"),
    expr("date_sub(to_date(date_str), 3)").alias("date_minus_3")
)
df_expr.display()


# COMMAND ----------

# MAGIC %md
# MAGIC SCD types -->MERGE into(Very important for interview)
# MAGIC
# MAGIC Interview ques for spark and Databricks-->Raja's Data Engineering(1 playlist -144 videos)
# MAGIC
# MAGIC Hadoop/Map Reduce/Spark architecture -->Gowtham SB (2 playlist around 15+ hrs)
# MAGIC
# MAGIC Good To learn
# MAGIC 1.Autoloader-->including(Schema hint,2 modes-file listing)
# MAGIC 2.Checkpointing
# MAGIC 3.collect-->This action will nor prefered in development it leads to OOM issue because it retrive the data from all worker node and gives to our Master node at the same time
# MAGIC 4.optimize+z-ordering vs Liquid clustering
# MAGIC 5.CDC VS Change Data Feed(Table level in delta table)
# MAGIC 6.Broadcast varible
# MAGIC 7.Types of cluster-->job nd Interactive 
# MAGIC 8.Saprk nd databricks Architecture(databricks is built on top of spark)
# MAGIC
# MAGIC
# MAGIC sql important topics
# MAGIC 1.Group by,partition,joins windows fn(Dense_rank),cte
# MAGIC