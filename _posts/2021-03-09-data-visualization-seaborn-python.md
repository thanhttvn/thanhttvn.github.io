---
title: Phải chăng matplotlib đã hết thời, Data Visualization bằng seaborn trên Python
author: thanhtt
date: 2021-03-09 13:33:00 +0700
categories: [blog, python]
tags: [blog, python]
math: true
mermaid: true
image:
  src: '/assets/img/posts/2018/03/seaborn.png'
---

Làm việc với Machine Learning, một công việc không thể thiếu là visualization dữ liệu. Nghĩ mãi không tìm được từ tiếng việt nào cho chuẩn, nói chung data visualization là tìm cách hiển thị dữ liệu dạng đồ thị làm sao để không chỉ mình mà người khác dễ dàng hiểu được.

Nhắc đến matplotlib chắc hẳn ai cũng biết và đã từng sử dụng. Vậy bạn đã sử dụng đến seaborn chưa ?

# Seaborn là cái ... gì thế ?

"Seaborn is a Python visualization library based on matplotlib. It provides a high-level interface for drawing attractive statistical graphics." https://seaborn.pydata.org/

Đơn giản seaborn là một thư viện base trên matplotlib, nó như 1 interface, wapper của matplotlib để bạn có thể dễ dàng visualization hơn, đồng ý là matplotlib rất tuyệt vời tuy nhiên seaborn lại có điểm mạnh là quá dễ để làm quen, không như matplotlib có 1 mớ các tham số mà đôi lúc không hiểu các tham số đó có mục đích gì nên khá mất công tìm hiểu.

# Cài đặt

```
pip install seaborn
```

Hoặc với anaconda

```
conda install seaborn
```

# Chuẩn bị
Như đã nói ở trên, seaborn như 1 wapper của matplotlib nên khi sử dụng cần import cả matplotlib. Ngoài ra còn có numpy, pandas quá nổi tiếng chắc ai cũng biết rồi nên mình không nhắc lại tác dụng để làm gì nữa.

```python
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

%matplotlib inline
```

Mình sẽ sử dụng 3 bộ data mẫu sau: titanic, tips, iris

```python
titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")
```

# Biểu đồ biểu thị sự phân bố, tần suất

Đầu tiên vấn đề thực tế hay gặp nhất phải kể đến là muốn biết sự phân bố của dữ liệu. Dạng biểu đồ phổ biến nhất là histogram, về mặt khái niệm thì biểu đồ Histogram là một tập hợp các cột biểu thị tần suất xuất hiện của một đại lượng cần theo dõi theo giá trị tăng dần theo trục hoành.


### Trường hợp 1 biến số(distplot)

```python
# Khởi tạo dữ liệu
x = np.random.normal(size=100)
```

seaborn cung cấp method distplot, khi dùng distplot thì bạn sẽ được biểu đồ chứa cả 2 dạng histogram và KDE. Điều này mình rất thích, không chỉ biểu thị nguyên histogram mà còn biểu thị cả sự phân bố dữ liệu mà code quá đơn giản như sau:

```python
sns.distplot(x)
```

![----------2018-03-25-14.46.31](/assets/img/posts/2018/03/----------2018-03-25-14.46.31.png)


### Trường hợp 2 biến số(jointplot)

Trường hợp này biến số là x, y. Mình sẽ sử dụng pandas DataFrame lưu trữ dữ liệu.

```python
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])
df.head()
```

![----------2018-03-25-14.57.08](/assets/img/posts/2018/03/----------2018-03-25-14.57.08.png)


Ảnh mình trích từ notebook, bạn có thể dowload link github ở dưới nhé.

Để biểu thị sự phân bố của 2 biến này, seaborn cung cấp method jointplot().

```python
sns.jointplot(x="x", y="y", data=df)
```

![----------2018-03-25-15.00.47](/assets/img/posts/2018/03/----------2018-03-25-15.00.47.png)


Ngoài biểu diễn theo trục x và y, biểu đồ còn biểu thị cả histogram từng biến. Tuyệt !

# Biểu thị thông tin theo nhóm

Khi làm việc với data, chắc hẳn nhiều lúc bạn muốn biểu diễn thông tin theo từng nhóm. Như là khách hàng nhóm A họ thường mua gì, giá cả khoảng nào... Seaborn cũng hỗ trợ rất tốt các dạng biểu đồ này.

### Biểu thị sự phân bố data theo từng nhóm

Ví dụ với data tips, mình muốn biến doanh thu theo ngày bị ảnh hưởng như thế nào với người hút thuốc hoặc không hút thuốc, nam hoặc nữ thì sẽ làm thế nào ?

Đơn giản là sử dụng method stripplot():

```python
sns.stripplot(x="day", y="total_bill", data=tips)
```

![----------2018-03-25-16.53.54](/assets/img/posts/2018/03/----------2018-03-25-16.53.54.png)


Với cách trên các điểm bị đè lên nhau, nếu bạn muốn các điểm hiển thị tách biệt nhau thì sử dụng swarmplot như sau:

```python
sns.swarmplot(x="day", y="total_bill", data=tips)
```

![----------2018-03-25-16.56.39](/assets/img/posts/2018/03/----------2018-03-25-16.56.39.png)


OK, nhìn rất trực quan và đẹp phải không ^^. Đến đây biểu đồ chưa cho biết khách nam, nữ sẽ tiêu tiền khác nhau như thế nào, để thêm đặc trưng giới tính vào biểu đồ chỉ cần sử dụng thêm parameter hue:

```python
sns.swarmplot(x="day", y="total_bill", hue="sex", data=tips)
```

![----------2018-03-25-16.59.50](/assets/img/posts/2018/03/----------2018-03-25-16.59.50.png)


### Biểu đồ hộp (box-plot)

Nếu bạn làm việc nhiều với biểu đồ hoặc hay đọc nghiên cứu khoa hoặc, chắc hẳn bạn sẽ biết đến biểu đồ này, thật sự biểu đồ cột không mang nhiều ý nghĩa nên gần đây giới khoa học khuyến khích đưa biểu đồ này vào thực tiễn. Nó mang khá nhiều giá trị như median, bách phân vị 25%,75%, hoặc dữ liệu outliers. Ví dụ như sau:

```python
sns.boxplot(x="day", y="total_bill", data=tips)
```

![----------2018-03-25-17.28.58](/assets/img/posts/2018/03/----------2018-03-25-17.28.58.png)

Nhìn biểu đồ chúng ta có thể hiểu được ví dụ ngày chủ nhật:

* Trung bình mỗi đối tượng chi phí khoảng 20 USD.
* Khoảng 25% đối tượng tiêu bằng hoặc ít hơn 15 USD.
* Tượng tự có 75% đối tượng tiêu bằng hoặc ít hơn 26 USD.
* Và có 2 giá trị outliers, nghĩa là đột biến, khác với chu kỳ, nên tính riêng để có có cái nhìn khách quan hơn.

# Tạo model hồi quy tuyến tính

Nếu bạn chưa biết về hồi quy tuyến tính là gì thì tham khảo [bài viết này](http://codetudau.com/hoi-quy-tuyen-tinh/) của mình, mình đã nói khá rõ.

Nếu sử dụng seaborn, công việc quá đơn giản như sau:

```python
sns.regplot(x="total_bill", y="tip", data=tips)
```

![----------2018-03-25-17.45.11](/assets/img/posts/2018/03/----------2018-03-25-17.45.11.png)


Tương tự như ở trên, nếu sử dụng tham số hue, biểu đồ sẽ hiển thị theo nhiều nhóm.

```python
sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips)
```

![----------2018-03-25-17.48.36](/assets/img/posts/2018/03/----------2018-03-25-17.48.36.png)


# Hiển thị thông tin chi tiết từ lượng data lớn

Vấn đề đặt ra khi cầm trên tay một lượng lớn data nhưng mới đầu chưa hiểu gì về lượng data này, chưa biết nó chứa những thông tin gì, liệu có thông tin nào hay ho không? Bạn sẽ làm thế nào ?

Nếu là mình thì mình muốn biểu diễn toàn bộ data theo nhiều cách để có cái nhìn trực quan nhất. Với seaborn mọi việc càng đơn giản hơn, bạn không cần phải mò mẫm vẽ từng biểu đồ như sau:

```python
sns.pairplot(iris, hue="species", size=2.5)
# size: chiều cao mỗi biểu đồ theo inch
```

![----------2018-03-25-17.58.17](/assets/img/posts/2018/03/----------2018-03-25-17.58.17.png)

# Tổng kết

Đến đây bạn cảm thấy việc sử dụng seaborn thế nào? Cá nhân mình thấy cú pháp đơn giản, dễ hiểu, dễ sử dụng và làm quen hơn matplotlib rất nhiều. Cũng do seaborn thực chất base trên matplotlib nên có vẻ đã khắc phục được những yếu điểm của matplotlib là điều dễ hiểu. Ngoài những cách biểu diễn dữ liệu như mình trình bày còn rất nhiều cách khác, tùy biến nhiều hơn tùy thuộc vào mục đích của các bạn.

## Link github source code

https://github.com/thanhttvn/machinelearning_codetudau/tree/master/seaborn_example
