---
title: Đi tìm cội nguồn của Deep Learning - Perceptron
author: thanhtt
date: 2021-03-09 13:33:00 +0700
categories: [blog, python]
tags: [blog, python]
math: true
mermaid: true
image:
  src: '/assets/img/posts/2018/04/perceptron-deeplearning.png'
---

Hẳn là ai cũng biết sông bắt nguồn từ suối, các hồ nước từ độ cao lớn hơn. Và chắc hẳn nhiều người đã biết đến DeepLearning hoặc đã, đang làm việc với DeepLearning tuy nhiên bạn có biết nguồn gốc của DeepLearning(Neural Network)　từ đâu mà có không?

Đó chính là Perceptron algorithm, bài viết này mình sẽ giải thích cặn kẽ với các bạn Perceptron là gì? Tại sao nó lại là cội nguồn của DeepLearning.

Đối tượng bài viết:
* Nếu bạn muốn tìm hiểu DeepLearning mà chưa biết bắt đầu từ đâu.
* Nếu bạn đang làm việc với DeepLearning nhưng lại chưa nắm được kiến thức cơ bản.

# Perceptron là gì?

Perceptron sẽ nhận input đầu vào là nhiều tín hiệu khác nhau, output là 1 tín hiệu duy nhất. "Tín hiệu" ở đây bạn có thể tưởng tượng như một vật nào đó có dòng chảy như dòng điện hay sông chẳng hạn. Dòng điện thì sẽ được truyền đi theo đường dây, việc dòng điện được truyền đi có bản chất là dòng dịch chuyển các electron dựa trên sự chênh lệch hiệu điện thế. Khác với dòng điện, tín hiệu của Perceptron mang 2 giá trị "1/0", 1 mang ý nghĩa là truyền tín hiệu, 0 là không truyền tín hiệu.

Như ví dụ dưới đây biểu diễn 1 Perceptron nhận 2 tín hiệu đầu vào.

![Perceptron-1](/assets/img/posts/2018/04/Perceptron-1.png)


* x1, x2 là tín hiệu input
* y là tín hiệu output
* w1, w2 là weight
* ○ có thể gọi là node hoặc neuron

Bạn có đang thắc mắc weight là cái khỉ gì không ?
Hiểu đơn giản giống như dòng điện, weight chính là trở kháng. Trở kháng càng cao thì dòng điện càng khó được truyền đi phải không. Tuy nhiên weight của Perceptron thì ngược lại, giá trị càng cao thì tín hiệu càng dễ được truyền đi.

Khi tín hiệu x được gửi đến neuron thì tại mỗi neuron sẽ nhân với weight $ ( x_1 \omega _1,  x_2 \omega _2) $. Output sẽ được tính đơn giản như sau:

![----------2018-04-15-11.02.43](/assets/img/posts/2018/04/----------2018-04-15-11.02.43.png)


$  \theta  $ có tên gọi là ngưỡng (threshold). Chỉ khi nào giá trị vượt ngưỡng thì y mới return về 1. Vì vậy chức năng của Perceptron chính là điều khiển các tín hiệu. Chốt lại là với weight càng lớn thì độ quan trọng của tín hiệu càng cao.

# Mạch logic đơn giản

## Mạch AND

Phần trên mình đã trình bày nguyên lý hoạt động của Perceptron, tiếp theo là ứng dụng của nó ra sao. Chắc hẳn bạn cũng biết mạch AND là gì. Đơn giản là mạch AND có 2 input và 1 output, chỉ khi input $ x_1, x_2 $ bằng 1 thì output mới bằng 1.

![-and-gate](/assets/img/posts/2018/04/-and-gate.png)


Vậy thì nếu muốn biểu diễn mạch AND bằng Perceptron thì sẽ làm thế nào ? Thực chất công việc cần làm là lựa chọn $ ( \omega _1, \omega _2, \theta ) $ sao cho thỏa mãn như hình. Ví dụ $ ( \omega _1, \omega _2, \theta ) $ = (0.5, 0.5, 0.7). Với parameter này, chỉ khi cả $ x_1, x_2 $ bằng 1 thì tổng mới $ > \theta $ và output y = 1.

![AND-perceptron](/assets/img/posts/2018/04/AND-perceptron.png)

## Mạch NAND và OR

NAND có ý nghĩa là Not AND nên output sẽ ngược lại so với AND. Tương tự như trên để biểu diễn mạch NAND bằng Perceptron thì cần lựa chọn parameter $ ( \omega _1, \omega _2, \theta ) $. Ví dụ parameter là $ ( \omega _1, \omega _2, \theta ) = (-0.5, -0.5, -0.7)$


![NAND-perceptron-1](/assets/img/posts/2018/04/NAND-perceptron-1.png)


Cũng không có khó khăn gì đúng không các bạn ^^. Với mạch OR thì như sau:

![OR-gate](/assets/img/posts/2018/04/OR-gate.png)


Vậy bạn thử suy nghĩ xem nên chọn parameter như thế nào cho phù hợp và để lại comment phía ↓ nhé. ^^

*Ở đây, việc chọn parameter cho Perceptron hoàn toàn bằng tay. Công việc của Machinelearning sẽ thay chúng ta lựa chọn parameter cho phù hợp và công việc của chúng ta là code model Perceptron*

Ở trên mình đã nói đến 3 mạch cơ bản là AND, NAND, OR. Về bản chất chúng hoàn toàn giống nhau, sự khác nhau chỉ là ở parameter Perceptron $ ( \omega _1, \omega _2, \theta ) $  mà thôi. Chính vì vậy với 1 model duy nhất, bằng việc thay đổi parameter thích hợp thì sẽ  transform được mạch AND, NAND hay OR.

## Implement Perceptron

### Weight và Bias

Biểu diễn lại công thức đầu tiên của Perceptron như sau:


![----------2018-04-15-12.33.02](/assets/img/posts/2018/04/----------2018-04-15-12.33.02.png)


b chính là threshold ở công thức trên, và ở đây thay vì gọi như vậy sẽ được gọi là bias, vì threshold bây giờ sẽ là 0.  $ \omega _1, \omega _2 $ sẽ là weight. Chức năng của bias và weight đương nhiên là khác nhau. Với chức năng của weight như đã nói ở trên là quyết định độ quan trọng của mỗi input, còn bias có chức năng tương tự như ngưỡng (threshold).

Chú ý:
* Tùy trường hợp, tài liệu mà $ \omega _1, \omega _2, b $ cùng được gọi là weight.

### Implement AND với weight và bias.


```python
# coding: utf-8
import numpy as np

def AND(x1, x2):
    x = np.array([x1, x2])　 #input
    w = np.array([0.5, 0.5]) #weight
    b = -0.7 #bias
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

if __name__ == '__main__':
    for xs in [(0, 0), (1, 0), (0, 1), (1, 1)]:
        y = AND(xs[0], xs[1])
        print(str(xs) + " -> " + str(y))
```

### Implement NAND và OR

```python

# coding: utf-8
import numpy as np


def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

```

## Perceptron nhiều tầng

Perceptron có thể biểu diễn được mạch AND, NAND, OR nhưng lại không thể biểu diễn được mạch XOR. Không tin bạn thử làm xem :)). Thực ra sự bá đạo của Perceptron là có thể chồng được nhiều tầng lên nhau. Với việc xếp chồng nhiều tầng, Perceptron sẽ biểu diễn được mạch XOR.

### Kết hợp các mạch có sẵn

Mạch XOR:

![-xor-gate](/assets/img/posts/2018/04/-xor-gate.png)


Tạo mạch XOR có nhiều cách nhưng cách đơn giản nhất là kết hợp từ các mạch AND, NAND, OR có sẵn.

```python
def XOR(x1,x2):
    gate1 = NAND(x1,x2)
    gate2 = OR(x1,x2)
    y = AND(gate1, gate2)
    return y
```

![XOR-Per](/assets/img/posts/2018/04/XOR-Per.png)


Lúc này sơ đồ luồng di chuyển sẽ như sau:


 ![xor](/content/images/2018/04/xor.png)

 # Tổng kết
 Trên đây mình đã giới thiệu với các bạn Perceptron là gì. Ứng dụng nó ra sao. Đây là một thuật toán cực kỳ đơn giản nên chắc hẳn các bạn đã có thể hiểu được ngay. Bài viết sau mình sẽ giới thiệu về nền tảng của neural network. Và tất nhiên newral network sẽ được xây dựng dựa trên Perceptron nên hiểu nguyên lý hoạt động của Perceptron rất quan trọng.

 Hẹn gặp lại các bạn ở bài viết sau.
