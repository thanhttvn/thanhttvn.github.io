---
title: Xử lý dữ liệu với Spark Dataframe
author: thanhtt
date: 2021-03-07 11:33:00 +0700
categories: [blog, spark]
tags: [blog, spark]
math: true
mermaid: true
image:
  src: '/assets/img/posts/spark-dataframes.png'
---

Đầu tiên, bạn có biết Spark là gì chưa nhỉ.

Nếu chưa biết Spark là gì thì bạn nên tìm hiểu Spark, cũng như làm quen với nó trước nhé.

Bạn có thể tìm hiểu Spark qua link sau: [Các bài viết liên quan đến Spark](https://vietnamlab.vn/blog/tag/spark/)


<h3> I. Vậy, Spark DataFrame là gì ấy nhỉ ?</h3>


Ngày xửa, ngày xưa, khi Spark ver 1.3 ra đời, Spark đã đẻ thêm tính năng có tên là Spark DataFrame. Vậy nó có gì hay ho nhỉ ?


* Có thể thiết lập [Schema](https://stackoverflow.com/questions/298739/what-is-the-difference-between-a-schema-and-a-table-and-a-database) cho Spark RDD và có thể tạo Object DataFrame.


Chưa thấy hay lắm nhỉ, dùng Spark RDD có sao đâu...
Thế bạn thao tác dữ liệu chỉ sử dụng RDD thấy gặp vấn đề gì phức tạp không ?
Viết code có khó khăn không ? Rồi vấn đề về hiệu năng ?


* Giống như viết SQL, đầy đủ chức năng như select, where ... đặc biệt là join với các DataFrame khác.
* Sử dụng các method như filter, select để trích xuất dữ liệu theo cột, hàng.
* Xử gọn các loại data như Log ... với groupBy → agg
* Thêm 1 cột dễ dàng với UDF(User Defined Function)
* Giống như SQL, Spark DataFrame đã hỗ trợ [Pivot](http://oracletechtalk.blogspot.jp/2015/09/pivot-va-unpivot-phan-1.html) (Spark 1.6 trở lên) rất hữu ích cho việc lập bảng biểu, báo cáo.


Tóm lại là dễ xài, đơn giản hơn RDD mà hiệu suất, khả năng optimized truy vấn tốt hơn RDD.
Chính vì vậy với các trường hợp thông thường, các bạn nên xài DataFrame.


<h3>II. Tạo DataFrame như thế nào nhỉ ?</h3>


Bài viết này sẽ sử dụng file log này để thực hành nhé. [Link dowload](https://drive.google.com/file/d/0B7Z4XTfCeeq_TFJLZmVmczhILUU/view?usp=sharing). Cấu trúc file Log gồm 3 cột : Date, User_ID, Campaign_ID

```python
click.at	user.id	campaign.id
4/27/2015 20:40	144012	Campaign077
4/27/2015 00:27	24485	Campaign063
4/27/2015 00:28	24485	Campaign063
4/27/2015 00:33	24485	Campaign038
4/27/2015 01:00	24485	Campaign063
4/27/2015 16:10	145066	Campaign103
```


**Cách 1: Tạo từ RDD**

Nếu bạn đã có RDD với tên column và type tương ứng (TimestampType, IntegerType, StringType)thì bạn có thể dễ dàng tạo DataFrame bằng

```python
sqlContext.createDataFrame(my_rdd, my_schema)
```

Với `printSchema(), dtypes` sẽ in thông tin của schema
và `count()` trả về số record

Và nếu chỉ muốn in n record đầu tiên thì sử dụng `show(n)`


```python
fields = [StructField("access_time", TimestampType(), True), StructField("userID", IntegerType(), True), StructField("campaignID", StringType(), True)]
schema = StructType(fields)

whole_log_df = sqlContext.createDataFrame(whole_log, schema)
print whole_log_df.count()
print whole_log_df.printSchema()
print whole_log_df.dtypes
print whole_log_df.show(5)

#327430
#root
# |-- access_time: timestamp (nullable = true)
# |-- userID: integer (nullable = true)
# |-- campaignID: string (nullable = true)
#
#[('access_time', 'timestamp'), ('userID', 'int'), ('campaignID', 'string')]
#
#+--------------------+------+-----------+
#|         access_time|userID| campaignID|
#+--------------------+------+-----------+
#|2015-04-27 20:40:...|144012|Campaign077|
#|2015-04-27 00:27:...| 24485|Campaign063|
#|2015-04-27 00:28:...| 24485|Campaign063|
#|2015-04-27 00:33:...| 24485|Campaign038|
#|2015-04-27 01:00:...| 24485|Campaign063|
```

**Cách 2: Tạo trực tiếp từ file CSV**


```python
from pyspark.shell import spark
from pyspark.sql.types import *

# Định nghĩa Schema
struct = StructType([
    StructField('a', StringType(), False),
    StructField('b', StringType(), False),
    StructField('c', StringType(), False)
])

# Tạo DataFrame từ file CSV
df_data = spark.read.csv('click_data_sample', struct)
```


Ngoài cách trên còn rất nhiều cách khác, có thể kể đến là gọi cổ 1 thằng thuộc họ nhà [Spark Package](https://spark-packages.org/) tên là [spark-csv](https://github.com/databricks/spark-csv) ra xài, hỗ trợ nhiều method hữu ích, dễ chơi, dễ trúng thưởng hơn.

Lưu ý là nếu không định nghĩa schema thì tất cả các column sẽ có kiểu string. Nhưng nếu dùng thêm thằng em `inferSchema` thì mọi chuyện sẽ êm xuôi. ^^ Không tin đơn giản như vậy ư, bạn thử như code dưới xem ^^


```python
whole_log_df_2 = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").load("/user/hadoop/click_data_sample.csv")
print whole_log_df_2.printSchema()
print whole_log_df_2.show(5)

#root
# |-- click.at: string (nullable = true)
# |-- user.id: string (nullable = true)
# |-- campaign.id: string (nullable = true)
#
#+-------------------+-------+-----------+
#|           click.at|user.id|campaign.id|
#+-------------------+-------+-----------+
#|2015-04-27 20:40:40| 144012|Campaign077|
#|2015-04-27 00:27:55|  24485|Campaign063|
#|2015-04-27 00:28:13|  24485|Campaign063|
#|2015-04-27 00:33:42|  24485|Campaign038|
#|2015-04-27 01:00:04|  24485|Campaign063|

whole_log_df_3 = sqlContext.read.format("com.databricks.spark.csv").option("header", "true").option("inferSchema", "true").load("click_data_sample.csv")
print whole_log_df_3.printSchema()

#root
# |-- click.at: timestamp (nullable = true)
# |-- user.id: integer (nullable = true)
# |-- campaign.id: string (nullable = true)
```


Nhân tiện đây, đôi lúc tên column không được chuẩn như Lê Duẩn thì bạn có thể dễ dàng thay đổi tên column bằng `withColumnRenamed`.

Và cũng tiện đây, có 1 lưu ý là về cơ bản DataFrame là `imutable`(ông cha ta hay gọi với cái tên thuần Tàu là `bất biến`) nên hễ có thay đổi nội dung của DataFrame thì 1 DataFrame mới sẽ được tạo để lưu trữ những thay đổi đó.


```python
whole_log_df_4 = whole_log_df_3.withColumnRenamed("click.at", "access_time")\
                 .withColumnRenamed("user.id", "userID")\
                 .withColumnRenamed("campaign.id", "campaignID")
print whole_log_df_4.printSchema()

#root
# |-- access_time: timestamp (nullable = true)
# |-- userID: integer (nullable = true)
# |-- campaignID: string (nullable = true)
```


**Cách 3: Giao lưu trực tiếp từ file json**

Bằng cách sử dụng `sqlContext.read.json`. Mỗi dòng của file json sẽ được coi là 1 object. Trong trường hợp object thiếu data thì sẽ null tại đó.


```python
# test_json.json gồm 3 dòng như dưới, dòng cuối không có "campaignID"
#
#{"access_time": "2015-04-27 20:40:40", "userID": "24485", "campaignID": "Campaign063"}
#{"access_time": "2015-04-27 00:27:55", "userID": "24485", "campaignID": "Campaign038"}
#{"access_time": "2015-04-27 00:27:55", "userID": "24485"}

df_json = sqlContext.read.json("test_json.json")
df_json.printSchema()
df_json.show(5)

#root
# |-- access_time: string (nullable = true)
# |-- campaignID: string (nullable = true)
# |-- userID: string (nullable = true)
#
#+-------------------+-----------+------+
#|        access_time| campaignID|userID|
#+-------------------+-----------+------+
#|2015-04-27 20:40:40|Campaign063| 24485|
#|2015-04-27 00:27:55|Campaign038| 24485|
#|2015-04-27 00:27:55|       null| 24485|
#+-------------------+-----------+------+

```


**Cách 4: Giao thông trực tiếp từ parquet**


Nếu bạn chưa biết parquet là gì thì tham khảo tại [đây](https://parquet.apache.org/) nhé. Với rất nhiều ưu điểm, mình sẽ giới thiệu vào 1 bài nào đó ở lúc nào đó nhé. ^^
Và cách đọc, đơn giản như đan rổ là sử dụng `sqlContext.read.parquet`. Như đoạn code bên dưới, cả folder chứa file parquet sẽ được đọc. Quá nhanh gọn lẹ phải không.


```python
sqlContext.read.parquet("/user/hadoop/parquet_folder/")
```


<h3>III. Thao tác với DataFrame </h3>


<h4>1. Thử query bằng SQL</h4>


Bằng cách sử dụng `registerTempTable`, bạn sẽ có một table được tham chiếu đến Dataframe đó, bạn có thể sử dụng tên table này để viết query SQL. Nếu bạn sử dụng `sqlContext.sql('query SQL')` thì giá trị trả về cũng là Dataframe.


Có 1 lưu ý là: Bạn cũng có thể viết subquery nhưng subquery cần được gán Alias, nếu không toạch lô (Syntax error) đấy.


```python
#SQL query

whole_log_df.registerTempTable("whole_log_table")

print sqlContext.sql(" SELECT * FROM whole_log_table where campaignID == 'Campaign047' ").count()
#18081
print sqlContext.sql(" SELECT * FROM whole_log_table where campaignID == 'Campaign047' ").show(5)
#+--------------------+------+-----------+
#|         access_time|userID| campaignID|
#+--------------------+------+-----------+
#|2015-04-27 05:26:...| 14151|Campaign047|
#|2015-04-27 05:26:...| 14151|Campaign047|
#|2015-04-27 05:26:...| 14151|Campaign047|
#|2015-04-27 05:27:...| 14151|Campaign047|
#|2015-04-27 05:28:...| 14151|Campaign047|
#+--------------------+------+-----------+


#Trường hợp thêm biến số vào trong câu SQL
for count in range(1, 3):
    print "Campaign00" + str(count)
    print sqlContext.sql("SELECT count(*) as access_num FROM whole_log_table where campaignID == 'Campaign00" + str(count) + "'").show()

#Campaign001
#+----------+
#|access_num|
#+----------+
#|      2407|
#+----------+
#
#Campaign002
#+----------+
#|access_num|
#+----------+
#|      1674|
#+----------+

#Trường hợp Sub Query：
print sqlContext.sql("SELECT count(*) as first_count FROM (SELECT userID, min(access_time) as first_access_date FROM whole_log_table GROUP BY userID) subquery_alias WHERE first_access_date < '2015-04-28'").show(5)
#+------------+
#|first_count |
#+------------+
#|       20480|
#+------------+
```

<h4>2. Tìm kiếm sử dụng filter, select</h4>

Đối với DataFrame , tìm kiếm kèm điều kiện rất đơn giản. Giống với câu query ở trên nhưng `filter, select` dễ dàng hơn rất nhiều. Vậy `filter` và `select` khác nhau thế nào ?

Cùng là để tìm kiếm nhưng `filter` trả về những row thoả mãn điều kiện, trong đó `select` lấy dữ liệu theo column.


```python
#Ví dụ filter
print whole_log_df.filter(whole_log_df["access_time"] < "2015-04-28").count()
#41434
print whole_log_df.filter(whole_log_df["access_time"] > "2015-05-01").show(3)
#+--------------------+------+-----------+
#|         access_time|userID| campaignID|
#+--------------------+------+-----------+
#|2015-05-01 22:11:...|114157|Campaign002|
#|2015-05-01 23:36:...| 93708|Campaign055|
#|2015-05-01 22:51:...| 57798|Campaign046|
#+--------------------+------+-----------+

#Ví dụ select
print whole_log_df.select("access_time", "userID").show(3)
#+--------------------+------+
#|         access_time|userID|
#+--------------------+------+
#|2015-04-27 20:40:...|144012|
#|2015-04-27 00:27:...| 24485|
#|2015-04-27 00:28:...| 24485|
#+--------------------+------+
```


<h4>3. Sử dụng groupBy </h4>

groupBy có chức năng giống với reduceByKey của RDD, nhưng nó còn cung cấp [1 rổ method](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.GroupedData). Ở đây mình sẽ run thử code count, agg.


**groupBy→count**

Ví dụ sau sẽ lấy key là `campaignID` và tiến hành `groupBy`. Sau đó dùng `count()` để lấy số record ứng với mỗi key.

```python
print whole_log_df.groupBy("campaignID").count().sort("count", ascending=False).show(5)
#+-----------+-----+
#| campaignID|count|
#+-----------+-----+
#|Campaign116|22193|
#|Campaign027|19206|
#|Campaign047|18081|
#|Campaign107|13295|
#|Campaign131| 9068|
#+-----------+-----+

print whole_log_df.groupBy("campaignID", "userID").count().sort("count", ascending=False).show(5)
#+-----------+------+-----+
#| campaignID|userID|count|
#+-----------+------+-----+
#|Campaign047| 30292|  633|
#|Campaign086|107624|  623|
#|Campaign047|121150|  517|
#|Campaign086| 22975|  491|
#|Campaign122| 90714|  431|
#+-----------+------+-----+
```


**groupBy→pivot**


Pivot thì từ Spark v1.6 trở lên được đưa vào, có chức năng giống với pivot trong SQL. Thử áp dụng xem sao nhỉ :

Trước khi pivot (`agged_df`)

* Số hàng = số `UserID(=75,545) * campainID(=133) `

* Số cột = `3`

Sau khi pivot(`pivot_df`)

* Số hàng = số `UserID(=75,545)`
* Số cột =  `UserID + số CampainID = 1 + 133 = 134`

Tất nhiên là bạn phải groupBy(cột giữ nguyên).pivot(cột muốn chuyển sang ngang).sum()


```python
agged_df = whole_log_df.groupBy("userID", "campaignID").count()
print agged_df.show(3)

#+------+-----------+-----+
#|userID| campaignID|count|
#+------+-----------+-----+
#|155812|Campaign107|    4|
#|103339|Campaign027|    1|
#|169114|Campaign112|    1|
#+------+-----------+-----+

#Những cell không có giá trị -> null
pivot_df = agged_df.groupBy("userID").pivot("campaignID").sum("count")
print pivot_df.printSchema()

#root
# |-- userID: integer (nullable = true)
# |-- Campaign001: long (nullable = true)
# |-- Campaign002: long (nullable = true)
# ..
# |-- Campaign133: long (nullable = true)

#TH muốn add 0 vào cell NULL
pivot_df2 = agged_df.groupBy("userID").pivot("campaignID").sum("count").fillna(0)
```


<h4>4. Thêm cột sử dụng UDF</h4>


Trong Spark DataFrame có thể sử dụng UDF, với ứng dụng chính là thêm cột. Như đã nói ở trên, bản chất DataFrame là immutable nên khi thêm cột thì 1 DataFrame mới sẽ được tạo.

```python
from pyspark.sql.functions import UserDefinedFunction
from pyspark.sql.types import DoubleType

def add_day_column(access_time):
    return int(access_time.strftime("%Y%m%d"))

my_udf = UserDefinedFunction(add_day_column, IntegerType())
print whole_log_df.withColumn("access_day", my_udf("access_time")).show(5)

#+--------------------+------+-----------+----------+
#|         access_time|userID| campaignID|access_day|
#+--------------------+------+-----------+----------+
#|2015-04-27 20:40:...|144012|Campaign077|  20150427|
#|2015-04-27 00:27:...| 24485|Campaign063|  20150427|
#|2015-04-27 00:28:...| 24485|Campaign063|  20150427|
#|2015-04-27 00:33:...| 24485|Campaign038|  20150427|
#|2015-04-27 01:00:...| 24485|Campaign063|  20150427|
#+--------------------+------+-----------+----------+
```

Cũng có thể sử dụng `lambda`

```python
my_udf2 = UserDefinedFunction(lambda x: x + 5, IntegerType())
print whole_log_df.withColumn("userID_2", my_udf2("userID")).show(5)

#+--------------------+------+-----------+--------+
#|         access_time|userID| campaignID|userID_2|
#+--------------------+------+-----------+--------+
#|2015-04-27 20:40:...|144012|Campaign077|  144017|
#|2015-04-27 00:27:...| 24485|Campaign063|   24490|
#|2015-04-27 00:28:...| 24485|Campaign063|   24490|
#|2015-04-27 00:33:...| 24485|Campaign038|   24490|
#|2015-04-27 01:00:...| 24485|Campaign063|   24490|
#+--------------------+------+-----------+--------+
```


Ngược lại, muốn xóa cột thì sử dụng `df.drop()`


```python
print whole_log_df.drop("userID").show(3)

#+--------------------+-----------+
#|         access_time| campaignID|
#+--------------------+-----------+
#|2015-04-27 20:40:...|Campaign077|
#|2015-04-27 00:27:...|Campaign063|
#|2015-04-27 00:28:...|Campaign063|
#+--------------------+-----------+
```


<h4>5. Join 2 DataFrame </h4>


Tính năng rất quan trọng khi xử lý dữ liệu chính là join.
Mình sẽ sử dụng data đầu vào ban đầu để tạo ra 1 DataFrame mới chứa những userID có count >100

```python
heavy_user_df1 = whole_log_df.groupBy("userID").count()
heavy_user_df2 = heavy_user_df1.filter(heavy_user_df1 ["count"] >= 100)

print heavy_user_df2 .printSchema()
print heavy_user_df2 .show(3)
print heavy_user_df2 .count()

#root
# |-- userID: integer (nullable = true)
# |-- count: long (nullable = false)
#
#+------+-----+
#|userID|count|
#+------+-----+
#| 84231|  134|
#| 13431|  128|
#|144432|  113|
#+------+-----+
#
#177
```

Được `heavy_user_df2 ` rồi tiến hành join (mặc định là inner join).

Các kiểu join bao gồm : inner, outer, left_outer, rignt_outer


```python
joinded_df = whole_log_df.join(heavy_user_df2, whole_log_df["userID"] == heavy_user_df2["userID"], "inner").drop(heavy_user_df2["userID"]).drop("count")
print joinded_df.printSchema()
print joinded_df.show(3)
print joinded_df.count()

#root
# |-- access_time: timestamp (nullable = true)
# |-- campaignID: string (nullable = true)
# |-- userID: integer (nullable = true)

#None
#+--------------------+-----------+------+
#|         access_time| campaignID|userID|
#+--------------------+-----------+------+
#|2015-04-27 02:07:...|Campaign086| 13431|
#|2015-04-28 00:07:...|Campaign086| 13431|
#|2015-04-29 06:01:...|Campaign047| 13431|
#+--------------------+-----------+------+
#
#38729
```


<h4>6. Lấy data theo cột trong DataFrame </h4>

* Lấy lable của cột (`df.columns`) -> Trả về list tên cột (not DataFrame )
* Lấy riêng 1 cột (`df.select("userID").map(lambda x: x[0]).collect()`) -> Trả về list userID (not RDD/Dataframe)
* Lấy cột distinct, chỉ cần thêm .distinct()

```python
print whole_log_df.columns
#['access_time', 'userID', 'campaignID']

print whole_log_df.select("userID").map(lambda x: x[0]).collect()[:5]
#[144012, 24485, 24485, 24485, 24485]

print whole_log_df.select("userID").distinct().map(lambda x:x[0]).collect()[:5]
#[4831, 48631, 143031, 39631, 80831]
```


<h4>7. Từ DataFrame , tạo RDD</h4>


Có 2 cách chính là:

* Sử dụng `map()` : mỗi hàng của DataFrame được chuyển sang RDD theo dạng list
* Sử dụng `.rdd`: mỗi hàng của DataFrame được chuyển sang RDD [Row Object](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.Row) (Tức là mỗi hàng sẽ là 1 Object) Tiếp theo sử dụng `.asDict()` với Row Object để chuyển về RDD Key-Value.


```python
#convert to rdd by ".map"
print whole_log_df.groupBy("campaignID").count().map(lambda x: [x[0], x[1]]).take(5)
#[[u'Campaign033', 786], [u'Campaign034', 3867], [u'Campaign035', 963], [u'Campaign036', 1267], [u'Campaign037', 1010]]

# rdd -> normal list can be done with "collect".
print whole_log_df.groupBy("campaignID").count().map(lambda x: [x[0], x[1]]).collect()[:5]
#[[u'Campaign033', 786], [u'Campaign034', 3867], [u'Campaign035', 963], [u'Campaign036', 1267], [u'Campaign037', 1010]]

#convert to rdd by ".rdd" will return "Row" object
print whole_log_df.groupBy("campaignID").rdd.take(3)
#[Row(campaignID=u'Campaign033', count=786), Row(campaignID=u'Campaign034', count=3867), Row(campaignID=u'Campaign035', count=963)]

#`.asDict()` will convert to Key-Value RDD from Row object
print whole_log_df.groupBy("campaignID").rdd.map(lambda x:x.asDict()).take(3)
#[{'count': 786, 'campaignID': u'Campaign033'}, {'count': 3867, 'campaignID': u'Campaign034'}, {'count': 963, 'campaignID': u'Campaign035'}]
```

<h3>IV. Tổng kết</h3>

Trên đây mình đã giới thiệu với các bạn những kiến thức cơ bản về Spark DataFrame, và tất nhiên là bạn có thể làm nhiều điều hơn(ngoài phạm vi bài viết này) với Spark DataFrame. Nhưng kiến thức cơ bản không bao giờ thừa phải không.

Hẹn gặp lại các bạn ở các bài viết sau.
