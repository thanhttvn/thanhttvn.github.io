---
title: Hồi quy tuyến tính - Hồi quy đơn giản, phức tạp
author: thanhtt
date: 2021-03-07 15:33:00 +0700
categories: [blog, python, scikit-learn]
tags: [blog, python, scikit-learn]
math: true
mermaid: true
---


Với bài viết này, mình sẽ chia sẻ về phương pháp hồi quy tuyến tính là gì, cách tạo model hồi quy tuyến tính sử dụng thư viện scikit-learn của Python.
Đầu tiên, hồi quy tuyến tính là gì nhỉ ?

# Hồi quy tuyến tình là
Model hồi quy tuyến tính (Linear Regression) được biểu diễn dưới dạng công thức toán học như ảnh dưới, là loại model dự đoán giá trị của biến mục tiêu dựa trên giá trị của biến giải thích.
![linearregression3-768x174](/assets/img/posts/linearregression3-768x174.png)
Trường hợp chỉ có 1 biến giải thích thì sẽ được gọi là phân tích hồi quy đơn giản. biến số giải thích từ 2 trở lên được gọi là phân tích hồi quy phức tạp.
# Hồi quy tuyến tính sử dụng scikit-learn
Trong scikit-learn cung cấp class `sklearn.linear_model.LinearRegression`, Class này dùng để biểu diễn công thức hồi quy tuyến tính sau đó dự đoán kết quả với data test.

## Cách sử dụng class sklearn.linear_model.LinearRegression
```python
sklearn.linear_model.LinearRegression(fit_intercept=True, normalize=False, copy_X=True, n_jobs=1)
```
### Biến số
<table>
<tbody><tr>
<th>fit_intercept</th>
<td>nếu setting False, Khong tính toán những điểm cắt với trục y. Biến mục tiêu chắc chắn phải nằm trên đường thẳng đi qua gốc toạ độ.  (Default: True)</td>
</tr>
<tr>
<th>normalize</th>
<td>Nếu set True thì chuẩn hoá biến giải thích trước. (Default: False)</td>
</tr>
<tr>
<th>copy_X</th>
<td>Có lưu lại data trong memory rồi mới chạy không (Default: True)</td>
</tr>
<tr>
<th>n_jobs</th>
<td>Số job sử dụng khi tính toán, -1 sẽ là dùng hết. (Default: 1)</td>
</tr>
</tbody></table>

### Attribute của class LinearRegression

<table>
<tbody><tr>
<th>coef_</th>
<td>Trả về hệ số hồi quy</td>
</tr>
<tr>
<th>intercept_</th>
<td>Trả về sai số</td>
</tr>
</tbody></table>

### Method

<table>
<tbody><tr>
<th>fit(X, y[, sample_weight])</th>
<td>Tiến hành tìm phương trình hồi quy tuyến tính</td>
</tr>
<tr>
<th>get_params([deep])</th>
<td>Lấy parameter đã sử dụng</td>
</tr>
<tr>
<th>predict(X)</th>
<td>Sử dụng model vừa tạo được tiến hành dự đoán</td>
</tr>
<tr>
<th>score(X, y[, sample_weight])</th>
<td>Đưa ra hệ số quyết định R<sup>2</sup>.
Không phải lúc nào giá trị dự đoán cũng giống vs giá trị thực tế. Hệ số quyết định ~ 1 thì độ chính xác càng cao, ~ 0 thì sai lệch lớn</td>
</tr>
<tr>
</tr>
</tbody></table>

## Chuẩn bị dữ liệu
Lần này chúng ta sẽ sử dụng bộ data dùng để phân tích chất lượng của rượu [winequality-red.csv](http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv) của đại học California-Berkeley.

Các column của data
<table>
<tbody><tr>
<th>fixed acidity</th>
<td>Nồng độ axit tartaric</td>
</tr>
<tr>
<th>volatile acidity</th>
<td>Tính axit</td>
</tr>
<tr>
<th>citric acid</th>
<td>Nồng độ axit Citric</td>
</tr>
<tr>
<th>residual sugar</th>
<td>Nồng độ đường dư</td>
</tr>
<tr>
<th>chlorides</th>
<td>Nồng độ clo</td>
</tr>
<tr>
<th>free sulfur dioxide</th>
<td>Nồng độ acid sulfurous tự do</td>
</tr>
<tr>
<th>total sulfur dioxide</th>
<td>Nồng độ acid sulfurus</td>
</tr>
<tr>
<th>density</th>
<td>Mật độ m/v(khối lượng/đơn vị thể tích)</td>
</tr>
<tr>
<th>pH</th>
<td>pH</td>
</tr>
<tr>
<th>sulphates</th>
<td>Nồng độ sunfat</td>
</tr>
<tr>
<th>alcohol</th>
<td>Nồng độ alc</td>
</tr>
<tr>
<th>quality</th>
<td>Điểm đánh giá chất lượng rượu từ 0-10 </td>
</tr>
</tbody></table>

## Đọc dữ liệu từ file csv

Tại đây mình sử dụng jupyter notebook để code và dùng notebook cũng thuận tiện cho việc hiển thị biểu đồ.

```python
import pandas as pd
import numpy as np

wine = pd.read_csv("winequality-red.csv", sep=";")
wine.head
```

![----------2017-09-17-14.07.56](/assets/img/posts/----------2017-09-17-14.07.56.png)

## Phân tích hồi  quy đơn giản
Như mình đã nói, hồi quy đơn giản khi chỉ có 1 biến giải thích. Lúc này phương trình sẽ có dạng `y = ax+b`.

```python
# import thư viện sklearn.linear_model.LinearRegression
from sklearn import linear_model
clf = linear_model.LinearRegression()

# Sử dụng "density (mật độ)" làm biến giải thích
X = wine.loc[:, ['density']].as_matrix()

# Sử dụng "alcohol (Số độ cồn)" làm biến mục đích
Y = wine['alcohol'].as_matrix()

# Tạo model suy đoán
clf.fit(X, Y)

# Hệ số hồi quy
print(clf.coef_)

# Sai số
print(clf.intercept_)

# Score
print(clf.score(X, Y))
```

Kết quả sẽ được như sau:
![----------2017-09-17-14.27.30](/assets/img/posts/----------2017-09-17-14.27.30.png)

Vậy công thức hồi quy đơn giản dạng `y = ax+b` sẽ là:
`[alcohol] = -280.16382307 × [density] + 289.675343383`

Biểu diễn kết quả tính toán phía trên lên toạ độ 2 chiều. Đường thẳng chính là phương trình hồi quy tìm được.

```python
# sử dụng package matplotlib
import matplotlib.pyplot as plt

# Biểu diễn sự phân bố tập dữ liệu input
# c: color
plt.scatter(X, Y, c='b')

# Đường thẳng hồi quy
plt.plot(X, clf.predict(X))
plt.show()
```
![----------2017-09-17-14.38.41](/assets/img/posts/----------2017-09-17-14.38.41.png)

## Phân tích hồi quy phức tạp
Tiếp tục với ví dụ tìm phương trình hồi quy phức tạp.
Mình sẽ chọn 'quality' làm biến mục tiêu, các biến còn lại là biến giải thích.
```python
from sklearn import linear_model
clf = linear_model.LinearRegression()

# Tạo dataframe chỉ chứa data làm biến giải thích
wine_except_quality = wine.drop("quality", axis=1)
X = wine_except_quality

# Sử dụng quality làm biến mục tiêu
Y = wine['quality']

# Tạo model
clf.fit(X, Y)

# Hệ số hồi quy
print(pd.DataFrame({"Name":wine_except_quality.columns,
                    "Coefficients":clf.coef_}).sort_values(by='Coefficients') )

# Sai số
print(clf.intercept_)
```
![----------2017-09-17-14.48.04](/assets/img/posts/----------2017-09-17-14.48.04.png)
Dễ dàng tìm được phương trình hồi quy phức tạp như sau:
```
[quality] = -17.881164 × [density] + -1.874225 × [chlorides] +
            -1.083590 × [volatile acidity] + -0.413653 × [pH] +
            -0.182564 × [citric acid] + -0.003265 × [total sulfur dioxide] +
            0.004361 × [free sulfur dioxide] + 0.016331 × [residual sugar] +
            0.024991 × [fixed acidity] + 0.276198 × [alcohol] +
            0.916334 × [sulphates] + 21.9652084495

```
## Chuẩn hoá các biến số
Mục đích để biết mức độ ảnh hưởng của các biến số giải thích đến biến mục tiêu như thế nào. Tiến hành chuẩn hoá xong, chúng ta sẽ biết được yếu tố nào ảnh hưởng lớn nhất đến chất lượng của rượu.

```python
from sklearn import linear_model
clf = linear_model.LinearRegression()

# chuẩn hoá dữ liệu các cột
wine2 = wine.apply(lambda x: (x - np.mean(x)) / (np.max(x) - np.min(x)))
wine2.head()

# Tạo dataframe không chứa quality làm biến giải thích
X = wine2.drop("quality", axis=1)

# Sử dụng quality làm biến mục tiêu
Y = wine2['quality']

clf.fit(X, Y)

print(pd.DataFrame({"Name":X.columns, "Coefficients":np.abs(clf.coef_)}).sort_values(by='Coefficients') )

print(clf.intercept_)
```
Dữ liệu sau chuẩn hoá
![----------2017-09-17-15.14.52](/assets/img/posts/----------2017-09-17-15.14.52.png)

![----------2017-09-17-15.15.06](/assets/img/posts/----------2017-09-17-15.15.06.png)
Nhìn vào kết quả, ta có thể dễ dàng thấy alcohol có giá trị lớn nhất, đồng nghĩa với việc biến số này ảnh hưởng lớn nhất đến chất lượng của rượu.
