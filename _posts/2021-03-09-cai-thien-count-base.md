---
title: Cải thiện phương pháp Count Base với Pointwise Mutual Information
author: thanhtt
date: 2021-03-09 18:33:00 +0700
categories: [blog, python, NLP]
tags: [blog, python, NLP]
math: true
mermaid: true
image:
  src: '/assets/img/posts/2019/03/PPMI.png'
---

Ở [bài viết trước](https://codetudau.com/word-to-vector-voi-phuong-phap-count-base/) mình đã giới thiệu với các bạn một phương pháp vector hoá từ có tên là Count Base. Dựa vào việc tạo ra một **co-occurence matrix** của từ, chắc hẳn bạn đã tạo được vector của từ rồi nhỉ. Tuy nhiên **co-occurence matrix** đó vẫn còn những chỗ cần cải thiện để có thể áp dụng được với thực tế.

# Pointwise Mutual Information

Ở bài viết trước, các bạn có thể thấy là mỗi phần tử của **co-occurence matrix** đều là 1 số chỉ số lần xuất hiện. Tuy nhiên là con số đơn thuần như thế lại có vấn đề là đối với những từ có tần suất xuất hiện nhiều.
Ví dụ: Nếu xét đến sự xuất hiện đồng thời của từ 「the」 và 「car」trong 1 bộ corpus nào đó, thì có cụm từ 「...the car...」 thường xuyên xuất hiện thì chúng ta sẽ có giá trị số lần đồng thời xuất hiện của từ 「the」và「car」sẽ rất nhiều. Tiếp theo xét đến từ 「car」và「drive」, rõ ràng 2 từ này cũng có sự tương quan . Tuy nhiên nếu đơn giản chỉ nhìn vào tần số xuất hiện thì từ「car」chắc chắn sẽ có tính tương quan với từ「the」mạnh mẽ hơn với từ 「 drive」.
> *Sự tương quan chính là mối quan hệ giữa 2 từ, từ này thường xuất hiện cùng từ kia. Về cơ bản, phương pháp vector hoá từ sử dụng Count Base dựa trên các từ bên cạnh hay chính là dựa trên tính tương quan giữa các từ. Vì vậy cải thiện được con số biểu thị tính tương quan giữa các từ chính là việc tạo ra được vector chính xác hơn.*

Để giải quyết vấn đề trên có một phương pháp có tên là **Pointwise Mutual Information** (**PMI**) thường xuyên được sử dụng. Về mặt định nghĩa cũng không có gì phức tạp cả:

$$
PMI(x, y) = log{2}  \frac{P(x, y)}{P(x)P(y)}
$$

* P(x)là xác suất xuất hiện x
* P(y)là xác suất xuất hiện y
* P(x, y)là xác suất xuất hiện đồng thời cả x và y
* PMI càng lớn thì tính tương quan giữa x và y càng lớn.

 Áp dụng vào xử lý ngôn ngữ tự nhiên thì P(x) chính là xác suất xuất hiện từ x trong bộ corpus.
 Ví dụ 1 bộ corpus có 10000 từ, từ 「the」xuất hiện 100 lần thì ta có P("the") = 0.01. P(x, y) là xác suất xuất hiện đồng thời x và y, vậy ví dụ số lần xuất hiện xuất hiện đồng thời 「the」 và 「car」 là 10 lần thì P("the", "car") = 0.001

 Giả sử tần số đồng thời xuất hiện của từ x và y ký hiệu là C(x, y). Số lần xuất hiện từ x, y tương ứng là C(x), C(y). Số từ trong corpus là N thì công thức trên có thể viết lại như sau:
$$
PMI(x, y) = log_{2}  \frac{P(x, y)}{P(x)P(y)} = log_{2}  \frac{ \frac{C(x,y)}{N} }{ \frac{C(x)}{N}\frac{C(y)}{N} } = log_{2}  \frac{C(x, y) ・ N}{C(x)C(y)}
$$

 OK, vậy là đã có công thức để tính PMI từ **co-occurence matrix**. Xét 1 ví dụ cụ thể sau:
*  Số từ trong corpus (N):10000
*  Từ 「the」:1000 lần
*  Từ 「car」:20 lần
*  Từ 「drive」:10 lần
*  Từ 「the」và「car」cùng xuất hiện: 10 lần
*  Từ 「car」và「drive」cùng xuất hiện: 5 lần

Với ví dụ trên thì vấn đề của việc chỉ nhìn vào số lần xuất hiện đồng thời thì có thể thấy từ 「car」có tính tương quan với từ 「the」cao hơn từ 「drive」. Vậy PMI thì sao:
$$
PMI("the", "car") = log_{2}  \frac{10・10000}{1000・20}  \approx  2.32
$$

$$
PMI("drive", "car") = log_{2}  \frac{5・10000}{20・10}  \approx  7.97
$$
OK, có vẻ ổn rồi đấy. Nếu sử dụng PMI thì 「car」với 「drive」có giá trị lớn hơn tức là có mối tương quan cao hơn 「the」 với「car」. Về cơ bản do từ 「the」xuất hiện nhiều lần nên PMI bị giảm. Thực tế cũng vậy những từ thường xuyên xuất hiện thì thường không mang nhiều ý nghĩa, ví dụ ở tiếng việt có những từ như là: 「thì」「là」...

Tuy nhiên chưa xong, PMI hiện tại có một vấn đề là trong trường hợp số lần đồng thời xuất hiện của 2 từ là 0 thì $$log_{2}0 = - \infty$$. Để giải quyết vấn đề này thì thực tế hay sử dụng **Positive PMI** (PPMI). Cũng rất đơn giản chỉ là lấy minimum là 0 .

$$
PPMI(x, y) = max(0, PMI(x, y))
$$

Ngoài ra cũng để tránh $$log_{2}0 = - \infty $$ trong quá trình sử dụng thực tế thì thông thường sẽ cộng thêm 1 số cực nhỏ: 1e-8 như dưới đây:

```python

def ppmi(C, verbose=False, eps = 1e-8):
    '''PPMI
    :param C: co-occurence matrix
    :param verbose: process log
    :return:
    '''
    M = np.zeros_like(C, dtype=np.float32)
    N = np.sum(C)
    S = np.sum(C, axis=0)
    total = C.shape[0] * C.shape[1]
    cnt = 0

    for i in range(C.shape[0]):
        for j in range(C.shape[1]):
            pmi = np.log2(C[i, j] * N / (S[j]*S[i]) + eps)
            M[i, j] = max(0, pmi)

            if verbose:
                cnt += 1
                if cnt % (total//100) == 0:
                    print('%.1f%% done' % (100*cnt/total))
    return M

```

Kết hợp với [bài viết trước](https://codetudau.com/word-to-vector-voi-phuong-phap-count-base/) thì chúng ta sẽ được kết quả như sau:

```python

# coding: utf-8
import sys
sys.path.append('..')
import numpy as np
from common.util import preprocess, create_co_matrix, cos_similarity, ppmi


text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)
vocab_size = len(word_to_id)
C = create_co_matrix(corpus, vocab_size)
W = ppmi(C)

np.set_printoptions(precision=3)  # round number
print('covariance matrix')

# >>> print(C)
# [[0 1 0 0 0 0 0]
#  [1 0 1 0 1 1 0]
#  [0 1 0 1 0 0 0]
#  [0 0 1 0 1 0 0]
#  [0 1 0 1 0 0 0]
#  [0 1 0 0 0 0 1]
#  [0 0 0 0 0 1 0]]

# >>> print('PPMI')
# PPMI
# >>> print(W)
# [[0.    1.807 0.    0.    0.    0.    0.   ]
#  [1.807 0.    0.807 0.    0.807 0.807 0.   ]
#  [0.    0.807 0.    1.807 0.    0.    0.   ]
#  [0.    0.    1.807 0.    1.807 0.    0.   ]
#  [0.    0.807 0.    1.807 0.    0.    0.   ]
#  [0.    0.807 0.    0.    0.    0.    2.807]
#  [0.    0.    0.    0.    0.    2.807 0.   ]]
# >>>

```

# Tổng kết

Đây cũng là một phương pháp được sử dụng khá nhiều trong xử lý ngôn ngữ. PMI không hề phức tạp phải không các bạn. Tuy nhiên chưa dừng lại ở đây, mặc dù so với **co-occurence matrix** là một bước cải tiến nhưng PPMI vẫn còn một vấn đề lớn. Các bạn có nhận thấy điều gì không ?
Đó chính là khi lượng corpus càng lớn thì số chiều các vector của từ càng tăng, có thể đến hàng trăm nghìn chiều.Và với vector có số chiều lớn như thế thì việc xử lý rất khó khăn. Ngoài ra, nhìn vào matrix được tạo ra có thể nhận thấy ngay là có rất nhiều vị trí giá trị 0. Và các giá trị đấy lại không mang ý nghĩa, dễ bị ảnh hưởng bởi noise. Vậy có cách nào để giải quyết không ? Hẹn gặp lại các bạn ở bài viết sau. :)
