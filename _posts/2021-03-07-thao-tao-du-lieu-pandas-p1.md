---
title: Thao tác dữ liệu với pandas - Phần 1
author: thanhtt
date: 2021-03-07 11:33:00 +0700
categories: [blog, pandas]
tags: [blog, pandas]
math: true
mermaid: true
image:
  src: '/assets/img/posts/chibi-pandas.jpg'
---

# Tại sao phải biết Pandas

Giống như muốn chăn rau, à trồng rau thì phải có cuốc, mặc dù không có cuốc thì dùng tay vẫn trồng được, nhưng mà có phải khổ hơn không. Trong việc xử lý dữ liệu với Python cũng thế, bạn không dùng Pandas cũng được, nhưng sẽ vất vả hơn rất nhiều.
OK, xem pandas có gì hay nào.

# Pandas là gì
Về cơ bản Pandas là 1 open source, được cộng đồng đánh giá là high-performance, việc xử lý dữ liệu, tính toán sẽ dễ dàng hơn rất nhiều cách truyền thống. OK, xem pandas có gì thú vị nào.

# Tạo data


Công việc đầu tiên là import thư viện Pandas và tạo Pandas DataFrame


```python
import pandas as pd

#Tạo data cần dùng
df_sample =\
pd.DataFrame([["day1","day2","day1","day2","day1","day2"],
              ["A","B","A","B","C","C"],
              [100,150,200,150,100,50],
              [120,160,100,180,110,80]] ).T  

#Thêm tên column
df_sample.columns = ["day_no","class","score1","score2"]  
#Gắn thêm index
df_sample.index   = [11,12,13,14,15,16]
import pandas as pd

# Tạo data cần dùng
df_sample = pd.DataFrame([["day1", "day2", "day1", "day2", "day1", "day2"],
                          ["A", "B", "A", "B", "C", "C"],
                          [100, 150, 200, 150, 100, 50],
                          [120, 160, 100, 180, 110, 80]]).T

# Thêm tên column
df_sample.columns = ["day_no", "class", "score1", "score2"]

# Gắn với index
df_sample.index = [11, 12, 13, 14, 15, 16]
df_sample
#    day_no class score1 score2
# 11   day1     A    100    120
# 12   day2     B    150    160
# 13   day1     A    200    100
# 14   day2     B    150    180
# 15   day1     C    100    110
# 16   day2     C     50     80
```


## Thao tác với Column/Index

Bạn có thể truy cập thay đổi index hoặc column như sau:

```python
# Lấy tên của cột
df_sample.columns
# Index([u'day_no', u'class', u'score1', u'score2'], dtype='object')

# Lấy tên
df_sample.index
# Int64Index([11, 12, 13, 14, 15, 16], dtype='int64')

#Ghi đè tên columns
df_sample.columns = ["day_no","class","point1","point2"]

#Ghi đè index
df_sample.index = [11,12,13,14,15,16]

# Sử dụng method rename point1 →　point
df_sample.rename(columns={'point1': 'point'})
# day_no class point point2
# 11   day1     A   100    120
# 12   day2     B   150    160
# 13   day1     A   200    100
# 14   day2     B   150    180
# 15   day1     C   100    110
# 16   day2     C    50     80

```

## Kiểm tra thông tin của data

```python
# Số row
len(df_sample)

# Kích thước data
#Trả về（số hàng、số cột)
df_sample.shape


# Thông tin từng column
df_sample.info()
# <class 'pandas.core.frame.DataFrame'>
# Int64Index: 6 entries, 11 to 16
# Data columns (total 4 columns):
# day_no    6 non-null object
# class     6 non-null object
# point1    6 non-null object
# point2    6 non-null object
# dtypes: object(4)
# memory usage: 240.0+ bytes

# Trả về các giá trị như TBC, phân tán, tứ phân vị
df_sample.describe()
#        day_no class  point1  point2
# count       6     6       6       6
# unique      2     3       4       6
# top      day1     B     150     110
# freq        3     2       2       1

# Hiển thị 10 dòng đầu tiên
df_sample.head(10)
```

# Thao tác với data
Lấy 1 số cột theo ý muốn

```python
# Với cách gọi như dưới chính là sử dụng method __get_item___
df_sample["day_no"]
# 11    day1
# 12    day2
# 13    day1
# 14    day2
# 15    day1
# 16    day2
# Name: day_no, dtype: object

# Giống như trên, nhưng muốn lấy nhiều cột một lúc
df_sample[["day_no","score1"]]

# Sử dụng loc
# Phương pháp ：iloc[rows, columns]
df_sample.loc[:,"day_no"]  # Muốn hiển thị cả cột thì thêm「:」
df_sample.loc[:,["day_no","score1"]] #TH muốn lấy nhiều cột một lúc

# Sử dụng iloc
# Phương pháp ：iloc[rows số bao nhiêu , column số bao nhiêu]
df_sample.iloc[:,0]
df_sample.iloc[:,0:2] #TH lấy nhiều cột


# Sử dụng ix
# Dùng số thứ tự hoặc tên cột đều được
df_sample.ix[:,"day_no"] # Tuy nhiên nếu lấy 1 cột thì KQ trả về dạng Pandas.Series Object
df_sample.ix[:,["day_no","score1"]] # Ngược lại, lấy nhiều cột thì là Pandas.Dataframe

df_sample.ix[0:4,"score1"] # hàng dùng số thứ tự, cột dùng tên cũng ok hết

series_bool = [True,False,True,False]
df_sample.ix[:,series_bool]  #Ngoài ra、lấy theo thứ tự array Boolean cũng được
# day_no class point1 point2
# 11   day1     A    100    120
# 12   day2     B    150    160
# 13   day1     A    200    100
# 14   day2     B    150    180
# 15   day1     C    100    110
# 16   day2     C     50     80


score_select = pd.Series(df_sample.columns).str.contains("score") # Xem có cột nào contains "score"
# 0    True
# 1    False
# 2    False
# 3    False
df_sample.ix[:, np.array(score_select)]
```


## Subsetting
Lấy ra 1 phần data thoả mãn điều kiện

```python
df_sample[df_sample.day_no == "day1"]  # Lọc ra nguyên row có day_no = day1
#    day_no class point1 point2
# 11   day1     A    100    120
# 13   day1     A    200    100
# 15   day1     C    100    110

# Cách khác là dùng arr boolean để lọc row dataframe như sau
series_bool = [True,False,True,False,True,False] #Lưu ý: số row = len(series_bool)
df_sample[series_bool]


#Sử dụng method query của Pandas
df_sample.query("day_no == 'day1'")
# So với cách trên không phải viết lại tên dataframe 2 lần :D
# Chú ý là điều kiện bắt buộc là str

# TH nhiều đkien or là "|" and là "&"
df_sample.query("day_no == 'day1'|day_no == 'day2'")

# Nếu muốn sử dụng biến số
select_condition = "day1"
df_sample.query("day_no == select_condition")  # ☓ báo lỗi
# Do đkien bắt buộc là kiểu str nên muốn sử dụng biến số thì không thể viết kiểu thông thường, cần thêm @
df_sample.query("day_no == @select_condition")  # ◯ cách viết đúng


#Subsetting sử dụng index
df_sample.query("index == 11 ")  # Lấy row có index 11
df_sample.query("index  in [11,12] ") #　or là「in」
```

## Sorting
Tiến hành sắp xếp lại

```python
df_sample.sort_values("point1")  # Sort point1 theo thứ tự tăng dần
df_sample.sort_values(["point1","point2"])  # Sort 2 cột theo thứ tự tăng dần


df_sample.sort_values("point1",ascending=False)  #point1 giam dan
```

## pandas.concat
Tiến hành thêm cột, record
```python

# Thêm record
# Trước tiên phải tạo 1 DF cấu trúc giống DF hiện có
df_addition_row = pd.DataFrame([["day1","A",100,180]])
df_addition_row.columns =["day_no","class","point1","point2"]
df_addition_row.index   =[17] #Index đương nhiên phải khác nhau

pd.concat([df_sample,df_addition_row],axis=0)
# Axis=0: Kiểu kết hợp theo hướng thẳng đứng, -> thêm cột thì axis = 1


# Thêm cột
# Thêm cột point3
df_addition_col = pd.DataFrame([[120,160,100,180,110,80]]).T #Tạo 1 DF có cột giống cấu trúc df_sample
df_addition_col.columns =["point3"]
df_addition_col.index   = [11,12,13,14,15,16]

pd.concat([df_sample,df_addition_col],axis=1)
```
