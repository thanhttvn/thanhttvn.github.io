---
title: Tunning Hyper Parameter với scikit-learn
author: thanhtt
date: 2021-03-09 01:33:00 +0700
categories: [blog, python, scikit-learn]
tags: [blog, python, scikit-learn]
math: true
mermaid: true
image:
  src: '/assets/img/posts/learning_resources_03-05.png'
---

# Mở đầu

Chào quý vị và các bạn, thời tiết hôm nay nắng đẹp, có những cơn gió thu mát rười rượi. Rảnh quá mang truyện Tấm Cám ra đọc. Có đoạn như thế này:

```python
Ít lâu sau nhà vua mở hội trong mấy đêm ngày. Già trẻ gái trai các làng đều nô nức đi xem, trên các nẻo đường, quần áo mớ ba mớ bẩy dập dìu tuôn về kinh như nước chảy. Hai mẹ con Cám cũng sắm sửa quần áo đẹp để đi trẩy hội. Thấy Tấm cũng muốn đi, mụ dì ghẻ nguýt dài, sau đó mụ lấy một đấu gạo trộn lẫn với một đấu thóc, bảo Tấm:

- Khi nào nhặt riêng gạo và thóc ra hai đấu thì mới được đi xem hội.

Nói đoạn, hai mẹ con quần áo xúng xính lên đường. Tấm tủi thân òa lên khóc. Bụt lại hiện lên hỏi:

- Làm sao con khóc?

Tấm chỉ vào cái thúng, thưa:

- Dì con bắt phải nhặt thóc ra thóc, gạo ra gạo, rồi mới được đi xem hội, lúc nhặt xong thì hội đã tan rồi còn gì mà xem.

Bụt bảo: - Con đừng khóc nữa. Con mang cái thúng đặt ra giữa sân, để ta sai chim sẻ xuống nhặt giúp.
```

Chợt nghĩ ra có một vấn đề hơi có tý liên quan mình muốn chia sẻ với các bạn, đó chính là việc tìm kiếm HyperParameter cho model. Và tất nhiên trong lúc làm việc không có ông bụt nào giúp bạn tìm hyper parameter cả ^^.

Ông bụt chính là bạn, và có thể coi scikit-learn như những con chim sẻ, cần cù thử các Parameter rồi tìm ra giá trị tốt nhất. Thú vị phết nhỉ. Còn cụ thể ra sao cùng tìm hiểu nhé.

# Tunning Parameter là gì ???

Với các model bạn đang sử dụng, điều không thể thiếu là các parameter, và tất nhiên là tuỳ thuộc mỗi bài toán cụ thể, số dữ liệu training bạn đang có, sẽ có các parameter thích hợp. Và việc thử nhiều parameter khác nhau là điều đương nhiên cần thiết.

Ví dụ như `RandomForest` có các parameter như: `n_estimators, max_features, max_depth, min_samples_split...` Và việc thay đổi giá trị các parameter trên sẽ ảnh hưởng đến độ chính xác của model, công việc của chúng ta là tìm cho bằng được parameter đẹp trai nhất. Chính là việc `Tunning HyperParameter`.

# Grid Search và Random Search

Hôm nay mình sẽ giới thiệu 2 phương pháp chính, giang hồ gọi tên là : `Grid Search` và `Random Search`.

Ví dụ chúng ta đang cần tunning với 2 parameter, cùng xem 2 đại hiệp kia sẻ xử lý chúng ra sao nhỉ.

![grid-search](/assets/img/posts/2017/10/grid-search.png)

Với Grid Search, giả dụ giá trị của 2 parameter lần lượt từ 0-9. Grid Search sẽ lần lượt ghép từng giá trị của param 1 với param 2 để tính toán độ chính xác của model. Đảm bảo không bỏ sót cặp parameter nào.

Ưu điểm: Diệt nhầm còn hơn bỏ sót, nên thường được ưu tiên lựa chọn.

Nhược điểm: Tuy nhiên đối với các model cần thiết lập nhiều parameter và nhiều giá trị thì việc tunning sẽ mất rất nhiều thời gian, hàng giờ, vài giờ thậm chỉ có thể tính bằng ngày.

![random-search](/assets/img/posts/2017/10/random-search.png)

Còn với Random Search, đúng như tên gọi, từ những giá trị parameter bạn setting, Random Search sẽ chọn ngẫu nhiên các cặp parameter để tiến hành độ chính xác của model.

Ưu điểm: Random nên sẽ không chạy đủ các trường hợp như Grid Search nên sẽ nhanh hơn đáng kể.

Nhược điểm: được cái lọ thì mất cái chai. Rất dễ bị trường hợp bỏ qua hyper parameter nhất.

# Ứng dụng

Cả 2 phương pháp trên đều được scikit-learn hỗ trợ.
Lần này mình sử dụng data Machine Learning Repository từ UCI, thử nghiệm với 2 cách trên. Code trên `jupyter notebook` với `python2.7`.

## Bước 1: Dowload data

Bộ data này phục vụ trong việc chuẩn đoán ác tính, lành tính với căn bệnh ung thư vú ^^.

```python
df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases'
                  '/breast-cancer-wisconsin/wdbc.data', header=None)
df.shape

# (569, 32)
```

## Bước 2: Chia data

Chúng ta cần chia data làm 2 phần feature data và target data.

```python
X_feature = df.loc[:, 2:].values
y_label = df.loc[:, 1].values

X_feature

# array([[  1.79900000e+01,   1.03800000e+01,   1.22800000e+02, ...,
#           2.65400000e-01,   4.60100000e-01,   1.18900000e-01],
#        [  2.05700000e+01,   1.77700000e+01,   1.32900000e+02, ...,
#           1.86000000e-01,   2.75000000e-01,   8.90200000e-02],
#        [  1.96900000e+01,   2.12500000e+01,   1.30000000e+02, ...,
#           2.43000000e-01,   3.61300000e-01,   8.75800000e-02],
#        ...,
#        [  1.66000000e+01,   2.80800000e+01,   1.08300000e+02, ...,
#           1.41800000e-01,   2.21800000e-01,   7.82000000e-02],
#        [  2.06000000e+01,   2.93300000e+01,   1.40100000e+02, ...,
#           2.65000000e-01,   4.08700000e-01,   1.24000000e-01],
#        [  7.76000000e+00,   2.45400000e+01,   4.79200000e+01, ...,
#           0.00000000e+00,   2.87100000e-01,   7.03900000e-02]])

y_label

# array(['M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M',
#        'M', 'M', 'M', 'M', 'M', 'M', 'B', 'B', 'B', 'M', 'M', 'M', 'M',
#        'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'M', 'B', 'M',
#        'M', 'M', 'M', 'M', 'M', 'M', 'M', 'B', 'M', 'B', 'B', 'B', 'B',
#        'B', 'M', 'M', 'B', 'M', 'M', 'B', 'B', 'B', 'B', 'M', 'B', 'M',
#        'M', 'B', 'B', 'B', 'B', 'M', 'B', 'M', 'M', 'B', 'M', 'B', 'M',
#        'M', 'B', 'B', 'B', 'M', 'M', 'B', 'M', 'M', 'M', 'B', 'B', 'B',
#        'M', 'B', 'B', 'M', 'M', 'B', 'B', 'B', 'M', 'M', 'B', 'B', 'B',
```

Tiếp theo đương nhiên cần chia thành 2 bộ data training và test.

```python
from sklearn.model_selection import train_test_split

# test size: 20%
# random_state: đảm bảo mỗi lần split đều ra output giống nhau.
X_train, X_test, y_train, y_test = train_test_split(
    X_feature, y_label, test_size=0.20, random_state=1
)

```

## Bước 3: Sử dụng default parameter thử tìm độ chính xác của model

Ở bài toán này, mình sử dụng model RandomForest, đương nhiên với các model khác cũng tương tự.

```python
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report

 def model_check(model):
     model.fit(X_train,y_train)
     y_train_pred = classification_report(y_train,model.predict(X_train))
     y_test_pred  = classification_report(y_test,model.predict(X_test))

     print("""【{model_name}】\n Train Accuracy: \n{train}
           \n Test Accuracy:  \n{test}""".format(model_name=model.__class__.__name__, train=y_train_pred, test=y_test_pred))

print(model_check(RandomForestClassifier()))

# [RandomForestClassifier]
#      Train Accuracy:
#                  precision    recall  f1-score   support
#
#               B       1.00      1.00      1.00        67
#               M       1.00      1.00      1.00        75
#
#     avg / total       1.00      1.00      1.00       142
#
#
#      Test Accuracy:
#                  precision    recall  f1-score   support
#
#               B       0.89      0.93      0.91        72
#               M       0.93      0.89      0.91        70
#
#     avg / total       0.91      0.91      0.91       142
```

Độ chính xác khi thử với tập train là 100%, còn với tập test data là 91%, bây giờ chúng ta thử tunning parameter xem có thể tìm ra được model nào tốt hơn không.

## Bước 4: Thử với Grid Search

```python
 #Grid search

 from sklearn.grid_search import GridSearchCV

 # use a full grid over all parameters
 param_grid = {"max_depth": [2,3, None],
              "n_estimators":[50,100,200,300,400,500],
              "max_features": [1, 3, 10],
              "min_samples_split": [2, 3, 10],
              "min_samples_leaf": [1, 3, 10],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

 forest_grid = GridSearchCV(estimator=RandomForestClassifier(random_state=0),
                 param_grid = param_grid,   
                 scoring="accuracy",  #metrics
                 cv = 3,              #cross-validation
                 n_jobs = 1)          #number of core

 forest_grid.fit(X_train,y_train) #fit

 forest_grid_best = forest_grid.best_estimator_ #best estimator
 print("Best Model Parameter: ",forest_grid.best_params_)

#    [RandomForestClassifier]
#      Train Accuracy:
#                  precision    recall  f1-score   support
#
#               B       0.99      0.99      0.99        67
#               M       0.99      0.99      0.99        75
#
#     avg / total       0.99      0.99      0.99       142
#
#
#      Test Accuracy:  
#                  precision    recall  f1-score   support
#
#               B       0.96      0.89      0.92        72
#               M       0.89      0.96      0.92        70
#
#     avg / total       0.92      0.92      0.92       142
```

Kết quả độ chính xác, f1-score đã được cải thiện


## Bước 5: Thử với Random Search

```python
#Random search
from sklearn.grid_search import RandomizedSearchCV
from scipy.stats import randint as sp_randint

param_dist = {"max_depth": [3, None],                  #distribution
              "n_estimators":[50,100,200,300,400,500],
              "max_features": sp_randint(1, 11),
              "min_samples_split": sp_randint(2, 11),
              "min_samples_leaf": sp_randint(1, 11),
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

forest_random = RandomizedSearchCV( estimator=RandomForestClassifier( random_state=0 ),
                                    param_distributions=param_dist,
                                    cv=3,              #CV
                                    n_iter=1944,          #interation num
                                    scoring="accuracy", #metrics
                                    n_jobs=1,           #num of core
                                    verbose=0,          
                                    random_state=1)

forest_random.fit(X,y)
forest_random_best = forest_random.best_estimator_ #best estimator
print("Best Model Parameter: ",forest_random.best_params_)

#    [RandomForestClassifier]
#      Train Accuracy:
#                  precision    recall  f1-score   support
#
#               B       1.00      1.00      1.00        67
#               M       1.00      1.00      1.00        75
#
#     avg / total       1.00      1.00      1.00       142
#
#
#      Test Accuracy:  
#                  precision    recall  f1-score   support
#
#               B       0.94      0.92      0.93        72
#               M       0.92      0.94      0.93        70
#
#     avg / total       0.93      0.93      0.93       142
```

So với default độ chính xác đã cải thiện được 2%.
