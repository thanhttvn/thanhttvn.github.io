---
title: Word To Vector với phương pháp Count Base
author: thanhtt
date: 2021-03-09 15:33:00 +0700
categories: [blog, python, deep-learning]
tags: [blog, python, deep-learning]
math: true
mermaid: true
image:
  src: '/assets/img/posts/2019/01/count-base.png'
---

Đi cùng với lịch sử phát triển của xử lý ngôn ngữ tự nhiên, đã có rất nhiều nghiên cứu về việc vector hoá từ. Nhìn vào các nghiên cứu đó có thể thấy hầu hết các phương pháp dựa trên một idea cơ bản đó chính là:
> Ý nghĩa của 1 từ được tạo thành từ các từ xung quanh.

Giới khoa học gọi ngắn gọn là **distributional hypothesis**, và dựa vào idea này, rất nhiều nghiên cứu về việc vector hoá từ được diễn ra. Thực chất bản chất của 1 từ không có ý nghĩa, mà tuỳ thuộc vào bối cảnh, đoạn văn đang nói tới mà mới mang ý nghĩa.

Ví dụ:
* I drink beer.
* We drink wine.

Có thể dễ dàng nhận thấy gần với 「**drink**」 là đồ uống. Và xét tiếp ví dụ sau:


* I guzzle beer.
* We guzzle wine.


Ồ có thể nhận ra được từ 「**drink**」 và 「**guzzle**」 được sử dụng trong cùng ngữ cảnh giống nhau, hơn nữa khả năng cao 「**drink**」 và 「**guzzle**」 có nghĩa giống nhau.
Vậy từ đây phương pháp nào ra đời ?

## Phương pháp CountBase

Phần đầu mình có nhắc đến **context(bối cảnh)**, context được nhắc đến ở đây có nghĩa là những từ lân cận với từ đang cần vector hoá.

![count-base-context](/assets/img/posts/2019/01/count-base-context.png)


Như ví dụ trên giả sử window size là 2, từ đang cần vector hoá là 「**goodbye**」 thì có 2 từ liền kề bên phải và trái được coi là context. Độ lớn của context chính là window size. Tương tự window size bằng 1 tức là phía bên phải và trái 1 từ.
<sub>(*)Một số tài liệu lại sử dụng window size với ý nghĩa là size của một phía. Tuy nhiên đa số được hiểu như trên.</sub>

Từ idea ban đầu có một phương pháp dễ dàng nghĩ tới là sẽ tiến hành count những từ xung quanh. Cụ thể là tương ứng với window size sẽ count những từ thuộc context. Chính vì vậy phương pháp này được coi là một  Count Base Method. Cũng tuỳ tài liệu mà được gọi là Statistical Method.

Tiếp theo mình sẽ vừa code vừa giới thiệu chi tiết phương pháp này.
Đầu tiên là chuẩn bị dữ liệu.

```python
# coding: utf-8
import sys
sys.path.append('..')
import os
import numpy as np


def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word
    corpus = np.array([word_to_id[w] for w in words])
    return corpus, word_to_id, id_to_word

text = 'You say goodbye and I say hello.'
corpus, word_to_id, id_to_word = preprocess(text)

print(corpus)
#[0 1 2 3 4 1 5 6]
print(id_to_word)
#{0: 'you', 1: 'say', 2: 'goodbye', 3: 'and', 4: 'i', 5: 'hello', 6: '.'}
```

Dễ dàng thấy được tổng số từ là 7. Tiếp theo set window size bằng 1 và đếm những từ thuộc context.

![count-base-2](/assets/img/posts/2019/01/count-base-2.png)


Với window size bằng 1 thì bối cảnh từ 「**you**」 chỉ có duy nhất từ 「**say**」. Vậy từ 「**say**」 mang giá trị 1, còn các từ khác  0.

![count-base-3](/assets/img/posts/2019/01/count-base-3.png)


Tương tự từ tiếp theo là từ 「**say**」. Sẽ được kết quả như sau:


![count-base-4](/assets/img/posts/2019/01/count-base-4.png)


OK, vậy là từ 「**say**」 có thể biểu diễn bằng vector [1, 0, 1, 0, 1, 1, 0]. Và làm tương tự với các từ còn lại.

![count-base-5](/assets/img/posts/2019/01/count-base-5.png)


Hình trên chính là một ma trận được gọi là **co-occurence matrix**. Vậy code như thế nào để ra kết quả như trên, thực tế lượng corpus rất lớn nên không thể dùng cơm để tạo ma trận được rồi. :D

```python
def create_co_matrix(corpus, vocab_size, window_size=1):
    '''create co-occurence matrix
    :param corpus: danh sách word id
    :param vocab_size:số từ
    :param window_size: window size
    :return: co-occurence matrix
    '''
    corpus_size = len(corpus)
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)
    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size + 1):
            left_idx = idx - i
            right_idx = idx + i
            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1
            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1
    return co_matrix

create_co_matrix(corpus, 7, window_size=1)

# array([[0, 1, 0, 0, 0, 0, 0],
       [1, 0, 1, 0, 1, 1, 0],
       [0, 1, 0, 1, 0, 0, 0],
       [0, 0, 1, 0, 1, 0, 0],
       [0, 1, 0, 1, 0, 0, 0],
       [0, 1, 0, 0, 0, 0, 1],
       [0, 0, 0, 0, 0, 1, 0]], dtype=int32)
```

# Mức độ tương đồng của 2 vector
OK, phần trước chúng ta đã có thể tạo được ma trận Co-occurence. Tức là đã có vector của từng từ rồi, việc còn lại cũng rất quan trọng trong NLP đó là tính mức độ tương đồng của các vector. Tại sao nó lại mang ý nghĩa quan trọng ? Thực tế có thể thấy 1 từ sẽ có các từ khác đồng nghĩa, chính vì vậy xác định được các từ đồng nghĩa cũng hết sức quan trọng. Và độ tương đồng của 2 vector từ càng lớn thì 2 từ đó càng mang ý nghĩa giống nhau.

Và nói đến phương pháp thì cũng có thể nghĩ đến rất nhiều phương pháp như tính tích vô hướng của 2 vector,  hay tính khoản cách eculid... Tuy nhiên liên quan đến việc tính độ tương đồng của vector từ thì **cosine similarity** được sử dụng rất nhiều.
Định nghĩa đối với 2 vector $$  x =  \big(x_{1},  x_{2},  x_{3},..., x_{n}\big) $$ và $$  y =  \big(y_{1},  y_{2},  y_{3},..., y_{n}\big) $$

$$ similarity \big(x, y \big) =  \frac{x * y}{ \parallel x \parallel \parallel y \parallel }  =  \frac{ x_{1}y_{1} + ... + x_{n}y_{n}}{ \sqrt{  x_{1}^{2} + ... + x_{n}^{2} } \sqrt{  y_{1}^{2} + ... + y_{n}^{2} } } $$

Phần tử là **tích vô hướng** 2 vector, phần mẫu là **norm L2** của từng vector. Điểm cần chú ý ở công thức trên là việc lấy tích vô hướng sau khi regularize.

> Norm là cách tính biểu thị độ lớn của vector.

Nhìn một cách trực quan thì **cosine similarity** cho biết hướng của 2 vector như thế nào. 2 vector hoàn toàn cùng hướng với nhau thì cosine similarity = 1, hoàn toàn ngược hướng thì cosine similarity = -1.

Code của công thức trên đơn giản như sau:

```python
def cos_similarity(x, y, eps=1e-8):
    '''Tính cosine similarity
    :param x: vector
    :param y: vector
    :param eps: Tránh trường hợp chia cho 0 khi vector 0
    :return:
    '''
    nx = x / (np.sqrt(np.sum(x ** 2)) + eps) # regularize vector x
    ny = y / (np.sqrt(np.sum(y ** 2)) + eps) # regularize vector y
    return np.dot(nx, ny)
```

OK, vậy là chúng ta đã tính được độ tương đồng của 2 vector, tiếp tục với ví dụ trên thử tính cosine similarity của từ 「 **i**」và「**you**」xem sao?

```python

c0 = C[word_to_id['you']]  # vector của「you」
c1 = C[word_to_id['i']]  #vector của「i」
print(cos_similarity(c0, c1))

#0.7071067691154799
```

# Tìm từ đồng nghĩa

Từ việc tính được cosine similarity như trên, không phải là quá dễ dàng để tìm được những từ đồng nghĩa hay sao. Tuy nhiên chính từ ví dụ này, cũng có thể nhận thấy phương pháp **Count Base** có khuyết điểm.

```python

def most_similar(query, word_to_id, id_to_word, word_matrix, top=5):
    '''Tìm từ đồng nghĩa
    :param query: từ cần tìm
    :param word_to_id: dict để word to id
    :param id_to_word: dict để từ id to word
    :param word_matrix: co-occurence matrix
    :param top: lấy top bao nhiêu
    '''
    if query not in word_to_id:
        print('%s is not found' % query)
        return
    print('\n[query] ' + query)
    query_id = word_to_id[query]
    query_vec = word_matrix[query_id]
    vocab_size = len(id_to_word)
    similarity = np.zeros(vocab_size)
    for i in range(vocab_size):
        similarity[i] = cos_similarity(word_matrix[i], query_vec)
    count = 0
    for i in (-1 * similarity).argsort():
        if id_to_word[i] == query:
            continue
        print(' %s: %s' % (id_to_word[i], similarity[i]))
        count += 1
        if count >= top:
            return

most_similar("you", word_to_id, id_to_word, C)

#[query] you
# goodbye: 0.7071067691154799
# i: 0.7071067691154799
# hello: 0.7071067691154799
# say: 0.0
# and: 0.0
```

Nhìn từ kết quả có thể nhận thấy giống nhất từ 「 **you**」có 3 từ:「**goodbye**」,「**i**」, 「**hello**」.
「 **you**」 và 「**i**」đúng là 2 danh từ chỉ người, cũng mang nghĩa khá tương đồng → ok đúng.
「 **you**」với 「**goodbye**」và 「**hello**」mà có giá trị cosine similarity lớn thì đúng là sai thật. Đương nhiên là với lượng corpus quá ít như này cũng là 1 nguyên nhân. Bạn hãy thử với lượng corpus nhiều hơn xem kết quả có được như mong đợi không nhé.
Ngoài ra nhìn vào **co-occurence matrix** có thể nhận thấy ma trận có số chiều rất lớn. Nếu lượng data nhiều thì xử lý sẽ rất chậm. Và đương nhiên cũng sẽ có cách cải thiện vấn đề này.

# Lời kết

Phạm vi bài viết này chỉ mới nói đến kiến thức cơ bản nhất của phương pháp **Count Base**, kiến thức cơ bản này mình đọc từ cuốn sách (link phía dưới) thấy hay nên đã viết lại chi tiết nhất để ai cũng có thể hiểu được và ứng dụng ngay. Và đương nhiên bài viết này không dừng lại tại đây, những bài viết tiếp theo mình sẽ nói đến các phương pháp cải tiến phương pháp này để hiệu quả hơn nữa.

# Tài liệu

https://www.oreilly.co.jp/books/9784873118369/
