---
title: Thao tác dữ liệu với pandas - Phần 2
author: thanhtt
date: 2021-03-07 13:33:00 +0700
categories: [blog, pandas]
tags: [blog, pandas]
math: true
mermaid: true
image:
  src: '/assets/img/posts/chibi-pandas.jpg'
---

Chào các bạn, hôm nay mình sẽ viết tiếp phần 2, cũng là phần cuối về việc giới thiệu, sử dụng Pandas trong Python để thao tác, xử lý dữ liệu.
Bài lần trước các bạn đã biết cách tạo Pandas DataFrame, các cách hiển thị dữ liêu, filter cũng như thêm cột vào DataFrame. Lần này, chúng ta tiếp tục thử tính toán, `Pivoting`, và cách sử dụng `Group by`.

## Tính toán thống kê
Thử tính toán theo hàng hoặc cột của DataFrame

```python
#Tính tổng theo cột
df_sample["point1"].sum(axis=0) # Tính tổng tất cả giá trị của point1
#axis=0 nghĩa là hướng tính sum theo chiều dọc, mặc định là 0 nên không cần viết lại cũng được

df_sample[["point1","point2"]].sum(axis=0)  #score1,score2 khi đồng thời muốn tính 2 cột. Trả về 2 giá trị.
#point1    750.0
#point2    750.0


#Tính theo hàng
df_sample[["point1","point2"]].sum(axis=1)  
#Tính tổng theo từng dòng, tổng = cột point1 + point2
#axis=1 thì hướng tính sum theo hàng ngang. Ở Pandas, sử dụng axis nhiều nên bạn biết rõ thì sẽ tốt hơn.
```
Để biểu thị axis = 0 và 1 nghĩa là thế nào, đơn giản như hình sau:
![axis](http://localhost:2368/content/images/2017/08/axis.jpg)
Như hình trên, ví dụ muốn tính toán theo hàng, tức là theo chiều ngang →　 axis=1

## Pivoting
Hỗ trợ rất nhiều trong việc tính toán group by theo cột, được ứng dụng rất nhiều trong thực tế công việc.

```python
df_sample.pivot_table("point1",     #Chỉ định cột cần tính
                       aggfunc="sum",  # Cách tính
                       fill_value=0,   # Trong TH không có giá trị thì fill 0
                       index="class",  # Giống như groupby, Cột nào sẽ làm hàng
                       columns="day_no")   #Cột nào sẽ làm cột
#day_no  day1  day2
#class             
#A        300     0
#B          0   300
#C        100    50

```

## Group_by
```python
df_sample_grouped = df_sample.groupby("day_no", as_index=True)  # Group_by theo day_no
df_sample_grouped[["point1","point2"]].sum()
# Sử dụng sum với object df_sample_grouped.
#        point1  point2
#day_no                
#day1       400     330
#day2       350     420
```

**Lưu ý:** `as_index=True` thì key sử dụng làm `groupBy` sẽ thành index
`as_index=False` thì sẽ đánh index lại từ 0.
Lợi ích của việc sử dụng `as_index=True` là nếu muốn lấy dữ liệu `day1` thì chỉ cần:
```python
df_sample_grouped.loc['day1']
#point1    400
#point2    330
#Name: day1, dtype: int64
```
Nếu `as_index=False` và muốn lấy dữ liệu `day1` thì index lúc này không phải là day_no nên cần phải làm như sau :

```python
df_sample_grouped.loc[df_sample_grouped.day_no == 'day1']
#  day_no    point1  point2
#0   day1     400     330
```
Về hiệu năng thì tương tự như việc đánh index trong SQL, sử dụng `df_sample_grouped.loc['day1']` sẽ nhanh hơn rất nhiều.
##Tổng kết
Phần 2 cũng là kết thúc chuỗi bài viết về Pandas, trên đây là những chức năng hay sử dụng trong quá trình thực tế. Chắc chắn sẽ giúp ích cho các bạn mới làm quen với Pandas. Hẹn gặp lại các bạn ở các bài viết sau.
