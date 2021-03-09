---
title: Làm quen với Pipeline trong Python sklearn
author: thanhtt
date: 2021-03-09 05:33:00 +0700
categories: [blog, python, scikit-learn]
tags: [blog, python, scikit-learn]
math: true
mermaid: true
image:
  src: '/assets/img/posts/1_h5naKpgTe5qordP8xJgGKQ.png'
---

Khi chuẩn bị data cho model trong machine learning, các điểm dữ liệu đôi khi chênh lệch nhau quá lớn, một thành phần có khoảng giá trị từ 0 đến 1000, thành phần kia chỉ có khoảng giá trị từ 0 đến 1 chẳng hạn. Lúc này, chúng ta cần chuẩn hóa dữ liệu trước khi thực hiện các bước tiếp theo. Còn gọi là [Feature scaling](https://en.wikipedia.org/wiki/Feature_scaling).

Cũng có trường hợp lượng feature của data lớn nên chúng ta cần chọn ra một số lượng nhỏ hơn các feature phù hợp với bài toán. Còn gọi là [Feature selection](https://en.wikipedia.org/wiki/Feature_selection).

Sau khi sử dụng các phương pháp tiền xử lý xong, bước tiếp theo là đưa dữ liệu vào model. Và tất cả các bước trên có thể gói gọn vào Pipeline để mọi việc được đơn giản hoá. Tại sao nên dùng Pipeline và lợi ích nó đem lại như thế nào, cùng thực hành và bạn sẽ thấy được cái hay của Pipeline đem lại.

# Chuẩn bị data

Lần này mình sẽ sử dụng bộ data: Breast Cancer Wisconsin. Bộ dataset này bao gồm 569 sample chuẩn đoán ác tính, lành tính với tế bào khối u.
Nói qua 1 chút về bộ data này nhé:

2 cột đầu tiên tương ứng với sample id và kết quả chuẩn đoán. M là ác tính, B là lành tính. Từ cột thứ 3 đến 32 chứa dữ liệu được tính toán từ hình ảnh chụp tế bào khối u. Từ những số liệu này, việc cần làm là xây dựng model dự đoán khối u ác tính hay lành tính.

Với 3 bước đơn giản sau để chuẩn bị data:

```python
import pandas as pd
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data', header =None)
```


```python
from sklearn.preprocessing import LabelEncoder

# X chứa data của feature
X = df.loc[:, 2:].values

# y là target
y = df.loc[:, 1].values
le = LabelEncoder()

# Convert target dạng chữ về số
y = le.fit_transform(y)

le.transform(['M', 'B'])
# array([1, 0])
```

```python
from sklearn.model_selection import train_test_split

# Chia data thành 2 phần: training data, test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)
```

# Ứng dụng Pipeline

Để có thể sử dụng LogisticRegression chúng ta cần chuẩn hoá dữ liệu.

Ngoài ra còn có công đoạn giảm số chiều của dữ liệu. Vậy tại sao lại cần giảm số chiều của dữ liệu làm gì? Thực ra các feature vectors trong các bài toán thực tế có thể có số chiều rất lớn, tới vài nghìn. Ngoài ra, số lượng các điểm dữ liệu cũng thường rất lớn. Nếu thực hiện lưu trữ và tính toán trực tiếp trên dữ liệu có số chiều cao này thì sẽ gặp khó khăn cả về việc lưu trữ và tốc độ tính toán. Vì vậy, giảm số chiều dữ liệu là một bước quan trọng trong nhiều bài toán

Hiện tại số chiều của dữ liệu đang là 30, mình sẽ sử dụng PCA để giảm số chiều xuống còn 2. Cụ thể về PCA bạn có thể tham khảo [tại đây](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html).

Vì vậy thay vì lần lượt làm 3 công đoạn riêng, mình sẽ tạo 1 Pipeline, Pipeline sẽ có nhiệm vụ kết hợp 3 Object: StandardScaler, PCA, LogisticRegression.

```python
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('pca', PCA(n_components=2)),
                    ('clf', LogisticRegression(random_state=1))
                   ])

pipe_lr.fit(X_train, y_train)

'Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test)
#'Test Accuracy: 0.947'

```

Pipeline là list tuple. Ví dụ như `('clf', LogisticRegression(random_state=1))` thì `clf` là tên gán cho object `LogisticRegression` phía sau, chỉ là tên nên bạn muốn đặt sao cũng được, nhưng nên đặt tên có nghĩa. Và chúng ta có thể truy cập các component bên trong Pipeline qua tên định danh này.

```python
pipe_lr.named_steps['scl']
#StandardScaler(copy=True, with_mean=True, with_std=True)
```

Bước trung gian trong Pipeline là transform và cuối cùng là bước dự đoán. Ở ví dụ trên có 2 bước trung gian: StandardScaler và PCA. Bước cuối là model phân loại hồi quy LogisticRegression.

Khi tiến hành gọi method `fit` của `pipe_lr` thì Object `StandardScaler` sẽ tiến hành gọi method `fit` và `transform` với đầu vào là Training Data.

Training Data sau khi được scaling sẽ đi đến bước tiếp theo là PCA. Tại bước này, số chiều dữ liệu hiện tại là 30 giảm xuống còn `n_components` chiều. Cũng giống như bước trên, method fit và transform sẽ được gọi.

Và data sau khi được transform qua 2 bước trên sẽ được chuyển đến model `LogisticRegression`.

Lưu ý nhỏ là:

Số bước trung gian không giới hạn. Ví dụ bạn có thể thêm bước lựa chọn đặc trưng sử dụng `SelectKBest`, `RFE`, `RFECV` chẳng bạn. Hay sử dụng thêm `OneHotEncoder`.

Sơ đồ luồng xử lý minh hoạ cho những gì mình nói ở trên:

<div style='max-width:70%; align='middle'>

![pipeline-diagram](/assets/img/posts/2017/12/pipeline-diagram.png)

</div>

Ngoài ra trong trường hợp muốn set lại parameter cho model hoặc các transform object bạn có thể làm như sau:

```python
pipe_lr.set_params(pca__n_components = 15, clf__C=0.5)
pipe_lr.fit(X_train, y_train)

'Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test)
#'Test Accuracy: 0.982'
```

Như ở ví dụ tên định danh của object và parameter được nối với nhau bởi dấu `__`, rất đơn giản phải không.

# Ví dụ với SelectKBest

Như mình đã nói ở trên, khi lượng feature của data lớn nên chúng ta cần chọn ra một số lượng nhỏ hơn các feature phù hợp với bài toán. Tuy nhiên với dữ liệu Breast Cancer Wisconsin thì lượng feature cũng không nhiều. Nhưng cứ thử xem kết quả có cải thiện được không nhé.


```python
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_regression

k_filter = SelectKBest(f_regression, k=15)
pipe_lr = Pipeline([('scl', StandardScaler()),
                    ('kbest', k_filter),
                    ('clf', LogisticRegression(random_state=1))
                   ])

pipe_lr.fit(X_train, y_train)
'Test Accuracy: %.3f' % pipe_lr.score(X_test, y_test)
#'Test Accuracy: 0.956'
```

Mình thử chọn ra 15 feature tốt nhất nhưng rất tiếc là kết quả không được cải thiện. Và đây là kết quả khi chưa tunning hyper parameter.

# Tổng kết

Mình viết bài viết này muốn giới thiệu với các bạn những kiến thức cơ bản về Pipeline. Làm việc với machine learning thì chắc chắn bạn sẽ gặp và ứng dụng rất nhiều bài toán khác với pipeline. Mong rằng những kiến thức cơ bản trên sẽ giúp các bạn làm việc dễ dàng hơn.

Hẹn gặp lại các bạn ở các bài viết sau ^^

Source code bạn có thể xem [tại đây.](https://github.com/thanhttvn/machinelearning_codetudau/tree/master/pipeline_example)
