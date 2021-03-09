---
title: Một vài lưu ý khi làm việc với BigData
author: thanhtt
date: 2021-03-09 10:33:00 +0700
categories: [blog, python, bigdata]
tags: [blog, python, scikit-learn]
math: true
mermaid: true
image:
  src: '/assets/img/posts/2018/02/okkk.png'
---

Dự án tôi đang tham gia là phát triển hệ thống recommender liên quan đến giao dịch tiền tệ. Và tất nhiên lượng data cần thiết là rất lớn, có những lúc lên đến hàng chục TB. Tuy thời gian tiếp cận chưa lâu nhưng cũng cóp nhặt được một số kiến thức liên quan đến PySpark (Python API của Spark)cho là hay nên cũng muốn chia sẻ với mọi người cũng như memo lại.

Spark là một framework được xây dựng để hỗ trợ việc xử lý dữ liệu một cách phân tán. Lợi thế là nhanh,  tiếp cận dễ dàng, hỗ trợ nhiều kiểu tính toán hơn Hadoop MapReduce.

# PySpark

Thực tế Pyspark được buil trên Java API của Spark. Bên trong **Python driver program**, SparkContext sẽ sử dụng Py4j để chạy **JVM** và khởi tạo **JavaSparkContext**. **JavaSparkContext** có nhiệm vụ giao tiếp với các **Spark executors** của các cluster. Python API khi gọi đến object **SparkContext**, những lời gọi, yêu cầu đó sẽ được **translate** bên trong **Java API calls** và được chuyển đến **JavaSparkContext**, dữ liệu được xử lý bằng Python sau đó được cached/shuffled trong JVM.

<div style='max-width:70%'>

![pyspark_internals](/assets/img/posts/2018/02/pyspark_internals.png)

</div>
RDD (Resilient Distributed Datasets) được định nghĩa trong Spark Core. Nó đại diện cho một collection các item đã được phân tán trên các cluster, và có thể xử lý phân tán. PySpark sử dụng PySpark RDDs và nó chỉ là 1 object của Python nên khi bạn viết code RDD transformations trên Java thực ra khi run, những transformations đó được ánh xạ lên object PythonRDD trên Java.

# Spark SQL và DataFrame

Spark hỗ trợ một số **higher-level tool** như Spark SQL cho việc xử lý data có cấu trúc. *Mlib* cho Machine Learning, *GraphX* cho xử lý dạng biểu đồ.

![spark_stack](/assets/img/posts/2018/02/spark_stack.png)

Bất cứ khi nào xử lý dữ liệu có cấu trúc điều đầu tiên mình muốn suggest là sử dụng Spark SQL, với interfaces được cung cấp bởi Spark SQL dẫn đến việc xử lý dễ dàng và hiệu quả. Một số cách để tương tác với Spark SQL là: SQL, DataFrames API, Datasets API. Và trong dự án của tôi là sử dụng DataFrames API. DataFrame trong Spark base dựa theo DataFrame trong R hay Pandas. Nó cũng khá tương tự như một bảng trong hệ quản trị CSDL quan hệ. DataFrames có thể được tạo từ file chứa dữ liệu có cấu trúc, tables trong Hive hoặc RDDs. Và nó cũng có một số điểm chung với RDD như immutable, lazy và distributed.

# Một vài best practices

**Broadcast variables:** Khi bạn có một lượng dữ liệu lớn muốn xử lý trên các nodes, sử dụng broadcast variables để giảm chi phí communication cost. Nếu không lượng dữ liệu này sẽ được gửi riêng sang các nodes mỗi khi xử lý và với cơ chế chuyển mặc định được tối ưu cho các biến mang ít dữ liệu, đối với biến chứa lượng lớn dữ liệu thì sẽ chậm hơn. Và biến broadcast lưu dữ liệu dưới dạng read-only cached và deserialized trên mỗi cluster.

Ví dụ Broadcast được tạo như sau:

```python
broadcastVar = sc.broadcast([1, 2, 3])
<pyspark.broadcast.Broadcast object at 0x102789f10>

```

Sử dụng thì cũng đơn giản không kém:

```python
broadcastVar.value
[1, 2, 3]
```

**Parquet and Spark:** Parquet lưu trữ dữ liệu dưới dạng Columnar. góp phần tăng hiệu năng truy xuất trên Spark lên rất nhiều lần. Bởi vì nó có thể tính toán và chỉ lấy ra 1 phần dữ liệu cần thiết (như 1 vài cột trên CSV), mà không cần phải đụng tới các phần khác của data row. Ngoài ra Parquet còn hỗ trợ flexible compression do đó tiết kiệm được rất nhiều không gian HDFS. Nhiều đánh giá cho thấy Parquet cho khả năng hiệu năng cải thiện đến 10x so với sử dụng data file thông thường.

Lưu trữ DF dưới dạng parquet rất đơn giản:

```python
df.write.parquet(outputDir)
# Nó sẽ tạo 1 dir outputDir và lưu trữ dữ liệu tại đó.
```

**Overwrite save mode ở cluster:**

Khi lưu trữ DataFrame xuống data source, mặc định thì Spark sẽ throws exception nếu data đã tồn tại. Spark cung cấp nhiều [SaveMode](https://spark.apache.org/docs/2.2.0/api/java/index.html?org/apache/spark/sql/SaveMode.html), trong đó có `overwrite` là 1 role rất quan trọng trong việc runing ở cluster.

**Clean code và performance:**

Khi xử lý lượng lớn dữ liệu, bạn cần chú ý đến việc clean code để tăng hiệu suất chương trình. Cần chú ý đến từng vấn để nhỏ nhất, tối ưu hóa tới mức tối đa. Ví dụ đơn giản như collect data về rồi xử lý trên pandas chắc chắn sẽ chậm hơn xử lý phân tán trực tiếp trên DataFrame.

 **Sử dụng multiple regexs**

 Điều cuối cùng mình muốn chia sẻ là khi bạn muốn match nhiều pattern regex, lời khuyên là nên sử dụng `(|)` thay vì chia làm nhiều lần, mỗi lần match với 1 pattern. Lý do là sẽ chỉ cần duyệt qua data 1 lần duy nhất.
