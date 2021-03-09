---
title: Tìm hiểu phương pháp Kmeans
author: thanhtt
date: 2021-03-09 02:33:00 +0700
categories: [blog, python, scikit-learn]
tags: [blog, python, scikit-learn]
math: true
mermaid: true
image:
  src: '/assets/img/posts/1_OFfN4I73rLPfhVqCcdcG5A.png'
---

Nghe đến thuật toán chắc hẳn đa số các bạn đã ngửi thấy mùi vị của cái khó, khó ở đây là khó hiểu, tài liệu thì trên internet thì nhiều, mà chất lượng thì không phải chỗ nào cũng tốt. Nhất là với các bạn mới bước chân vào con đường machine learning.

Hiểu được khó khăn của các bạn nên lần này mình sẽ giới thiệu đến các bạn phương pháp tiếp cận mới. Không chuyên sâu vào các công thức toán học rắc rối, mà chú trọng về lý giải cách thuật toán hoạt động như thế nào. Một khi bạn đã hiểu thì làm việc, ứng dụng vào thực tế sẽ dễ dàng hơn rất nhiều. Cụ thể lần này sẽ là phương pháp k-means.

# Phương pháp k-means là gì

Ngắn gọn phương pháp k-means là 1 trong số những thuật toán phân cụm (clustering). Đầu vào tập dữ liệu cần phân cụm và số cụm (cluster), đầu ra chúng ta sẽ được kết quả dữ liệu đã được phân về các cluster.

 Mục đích là phân dữ liệu thành các cụm (cluster) khác nhau sao cho dữ liệu trong cùng một cụm có tính chất giống nhau.

# Phân tích chi tiết

 Đầu tiên là chuẩn bị dữ liệu cần phân cụm. Tiếp theo quyết định số lượng cụm (cluster) cần phân chia. Ở ví dụ này mình thử chọn số cluster là 3. Ở đây data được thể hiện dưới dạng các điểm cho dễ quan sát. Cự ly của các dữ liệu được hiểu là độ dài đoạn thẳng nối giữa 2 điểm với nhau.

 ![1-1](/assets/img/posts/2017/10/1-1.png)

### Bước 2：

 Chọn ngẫu nhiên 3 điểm làm điểm trung tâm của cluster.

 ![2](/assets/img/posts/2017/10/2.png)


### Bước 3:

Với các điểm dữ liệu không được chọn là điểm trung tâm thì tính toán khoảng cách từ chính điểm đó đến các cluster và quyết định cluster nào gần với mình nhất.

![3](/assets/img/posts/2017/10/3.png)


### Bước 4:

Từ bước tính toán trên, tiến hành phân loại các điểm về các cluster đã quyết định(cluster gần nó nhất). Vậy là đã phân ra được 3 cụm.


![4](/assets/img/posts/2017/10/4.png)



Quá dễ dàng nhỉ. Tuy nhiên nhìn vào hình trên bạn có thể nhận ra ngay các cụm dữ liệu này chưa phải là chuẩn nhất. Chính xác là do điểm trung tâm cụm được chọn chưa chính xác. Nên bước 5 sẽ được tiến hành.

### Bước 5:

Bước trên chúng ta đã thu được 3 cụm, bây giờ tiến hành tính trọng tâm của các điểm dữ liệu của từng cụm. Sau đó di chuyển điểm trung tâm của cụm sang vị trí vừa tính được.


![5](/assets/img/posts/2017/10/5.png)


Vị trí mà 3 điểm trung tâm của cluster vừa di chuyển đến được hiểu ngắn gọn chính là điểm trung tâm đang di chuyển đến vị trí chính xác hơn.

### Bước 6:

Một lần nữa tiến hành bước 3, tính toán lại khoảng các các điểm đến các điểm trung tâm. Sau đó phân loại lại các điểm dữ liệu về các cụm.

![6](/assets/img/posts/2017/10/6.png)

### Bước 7:

Sau đó lặp lại quá trình di chuyển cluster trung tâm và phân loại lại các điểm về các cụm gần nhất.　

Quá trình này sẽ dừng khi sau khi dữ liệu sau khi phân cụm lại không thay đổi gì so với lần trước.

![7-1](/assets/img/posts/2017/10/7-1.png)

Với ví dụ này thì với với lần lặp thứ 4, thuật toán đã phân cụm thành công.


# Những lưu ý khi sử dụng phương pháp k-means

Trước khi sử dụng phương pháp này, chúng ta phải quyết định trước số lượng cluster, tuy nhiên trong quá trình tính toán số lượng cluster có thể khác với số lượng cluster mình dự đoán nên kết quả sẽ không chính xác.

Vì vậy để giải quyết vấn đề này, để có thể chọn ra số lượng cluster thích hợp thì cần phải phân tích dữ liệu cẩn thận, chạy thử k-means với nhiều biến số số lượng cluster.

Cùng ví dụ trên nếu thay số lượng cluster thành 2, kết quả phân loại sẽ thành ra như sau:

![8](/assets/img/posts/2017/10/8.png)

# Ví dụ

Mình sử dụng python2 và code viết trên jupyter notebook

## Tạo dữ liệu

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, HTML # dùng để hiển thị
%matplotlib inline
```

```python
from sklearn.datasets import make_blobs  #  dùng để tạo dữ liệu giả

# (x,y)
X,y=make_blobs(n_samples=150,         # tổng số điểm sample
               n_features=2,          # số lượng feature(số chiều) default:2
               centers=3,             # số lượng cluster
               cluster_std=0.5,       # độ lệch chuẩn giữa các cluster
               shuffle=True,          #  có trộn các sample với nhau không
               random_state=0)  

plt.scatter(X[:,0],X[:,1],c='black',marker='o',s=50)
plt.grid()
plt.show()
```

![----------2017-10-29-19.32.27](/assets/img/posts/2017/10/----------2017-10-29-19.32.27.png)


Bạn sẽ thu được bộ dữ liệu như trên. Lần này mình sẽ dùng bộ data này để tiếp tục sử dụng k-means

## Áp dụng k-means

Phân cụm sao cho SSE (Sai số trong mỗi cluster) là nhỏ nhất.
Sai số chính là khoảng cách từ các điểm thuộc 1 cluster với điểm trung tâm.

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3,            # số cluster
            init='random',           # vị trí center của cluster  default: 'k-means++'
            n_init=10,               # số lần chọn center của cluster default: '10'  trong số lần chọn  , sẽ chọn ra model có  SSE nhỏ nhất
            max_iter=300,            # Tiến hành chạy k-means nhiều nhất bao nhiêu lần default: '300'
            tol=1e-04,               # Khi tiến hành hội tụ các điểm, sai số cho phép là bao nhiêu, default: '1e-04'
            random_state=0)

y_km = km.fit_predict(X)

```


```python
plt.scatter(X[y_km==0,0],         # hiển thị y_km（cluster）=0
                    X[y_km==0,1],
                    s=50,
                    c='lightgreen',
                    marker='s',
                    label='cluster 1')
plt.scatter(X[y_km==1,0], # hiển thị y_km（cluster）=1
                    X[y_km==1,1],
                    s=50,
                    c='orange',
                    marker='o',
                    label='cluster 2')
plt.scatter(X[y_km==2,0],# hiển thị y_km（cluster）=2
                   X[y_km==2,1],
                    s=50,
                    c='lightblue',
                    marker='v',
                    label='cluster 3')
plt.scatter(km.cluster_centers_[:,0],   # km.cluster_centers_ điểm trung tâm
                    km.cluster_centers_[:,1],
                    s=250,
                    marker='*',
                    c='red',
                    label='centroids')
plt.legend()
plt.grid()
plt.show()


```

![----------2017-10-29-19.59.39](/assets/img/posts/2017/10/----------2017-10-29-19.59.39.png)


Và trên là kết quả. Việc sử dụng thư viện có sẵn trên sklearn, rất dễ dàng đúng không.

Có 1 lưu ý nhỏ là một model k-means được đánh giá là tốt khi giá trị SSE là nhỏ nhất. Với model trên ta có thể lấy giá trị SSE bằng cách sau:

```python
print ('Distortion: %.2f'% km.inertia_)

#Distortion: 72.48
```


# Tài liệu tham khảo

https://qiita.com/Takumi0204/items/1b2c990670b508e3426b

https://www.datascience.com/blog/k-means-clustering

https://www.coursera.org/learn/machine-learning/lecture/93VPG/k-means-algorithm
