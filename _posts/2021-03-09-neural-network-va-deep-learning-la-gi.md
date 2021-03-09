---
title: Neural Network và Deep Learning là gì ?
author: thanhtt
date: 2021-03-09 15:33:00 +0700
categories: [blog, python, deep-learning]
tags: [blog, python, deep-learning]
math: true
mermaid: true
image:
  src: '/assets/img/posts/2018/07/neural-network.png'
---

Chào các bạn, hôm nay đẹp trời lại có thời gian rảnh mình sẽ viết tiếp chuỗi bài về Deep Learning. Như bài trước mình đã giới thiệu với các bạn về Perceptron, nếu bạn chưa biết thì bạn có thể xem lại [tại đây](https://codetudau.com/deeplearning-perceptron-la-gi/).
Và bài này mình sẽ giới thiệu về Neural Network(NN) và NN có mối liên hệ như thế nào với Deep Learning.

Mục tiêu chính của bài viết:
* Hiểu được Neuron Network là gì
* Deep Learning là gì

# Neural Network là gì ?

## Khởi đầu với nơron
Một cách ngắn gọn nhất thì Neural là mô hình toán học mô phỏng nơron trong hệ thống thần kinh con người. Model đó biểu hiện cho một số chức năng của nơron(neuron) thần kinh con người.

![noron](/assets/img/posts/2018/06/noron.png)

Đầu tiên, nhìn hình thôi các bạn đã thấy có sự tương đồng rồi đúng không ^^. Tuy nhiên mục đích bài viết là giúp các bạn hiểu gốc rễ vấn đề, biết ý nghĩa từng tham số, chắc chắn sẽ giúp bạn hiểu rõ hơn về cái mà mình đang làm, đang tìm hiểu. Với phần gốc rễ đã chắc thì chắc chắn sẽ dễ dàng hơn khi tiếp cận những bài báo mới hiện nay.

Đầu tiên là tính chất truyền đi của thông tin trên neuron, khi neuron nhận tín hiệu đầu vào từ các dendrite, khi tín hiệu vượt qua một ngưỡng(threshold) thì tín hiệu sẽ được truyền đi sang neuron khác (Neurons Fire) theo sợi trục(axon). Neural của model toán học ở đây cũng được mô phỏng tương tự như vậy. Công thức tính output y sẽ như sau:

$$ y= a( w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} -  \theta ) (1) $$

*Ái chà, cái gì vậy trời, từ đâu lòi ra công thức loằng ngoằng vậy.*

Không phải vội từ từ mình sẽ giải thích.

Việc **Neurons Fire** khi nhận tín hiệu từ các neuron khác được tính phép cộng thông thường ( $$ x_{1} + x_{2} $$ ). Tuy nhiên lại không chỉ đơn giản như thế. Tại sao ?

*Ví dụ như này:*

*Bạn đang tham gia 1 trận đấu tenis, não của bạn sẽ nhận các tín hiệu từ các giác quan như hình ảnh từ mắt, âm thanh từ tai, cảm giác từ các tế bào ở tứ chi, thậm chí là cả mùi vị từ mũi ... Và bạn đang thi đấu, bạn sẽ tập trung vào điều gì, bạn có dễ bị phân tâm từ mùi hôi hôi từ chính đôi tất 2 bữa nay chưa giặt không, hay bạn đang chỉ chú tâm tới từng động tác của đối thủ ?
Tại sao lại thế nhỉ, rõ ràng thông tin não bộ nhận được là đầy đủ... Đó, bạn đã mường tượng ra vấn đề gì chưa. Đó chính là nhờ cấu trúc phức tạp của từng neuron của hệ thần kinh.
Cụ thể là từ input nhận được, việc xử lý từng thông tin đó được gắn với 1 trọng số(weight), mấy thông tin không quan trọng sẽ có weight thấp hơn, cái ta cần là các thông tin có ích cho trận đấu*

Trong neural, weight(ký hiệu: w) cũng mang ý nghĩa như vậy.

Và  $$\theta$$ chính là ngưỡng(threshold) như mình đề cập ở trên.

**a** là một function mà người tạo ra model ký hiệu, có tên trên giấy khai sinh là **activation function**, có nhiệm vụ là chuẩn hoá output. Bạn thử tưởng tượng công thức trên bỏ đi activation function thì output y sẽ là 1 giá trị không có giới hạn (-inf -> inf), vậy làm sao biết khi nào fired hoặc không. Ở điểm này, đã chứng minh được bộ não con người quá siêu việt, mặc dù không có activation function nhưng cũng đã quản lý được trạng thái fired hoặc not fire. Activation Function có các đại diện tiêu biểu như:
* Step function
* Linear function
* Sigmoid function
* Tanh function
* ReLu

Trong phạm vi bài viết này, bạn chỉ cần hiểu Activation Function có nhiệm vụ là chuẩn hoá output của neural là được. Có lẽ mình sẽ viết 1 bài riêng nói về chi tiết về các Activation Function nêu trên (nếu nhận được sự ủng hộ nhiệt tình từ các bạn ^^).

Từ công thức (1), thực tế threshold trong phạm vi toán học có thể mang cả dấu (-) và (+) nên các bác đầu to hơn bình thường 1 chút đã đưa vào thuật ngữ **bias**: $$ bias = b = - \theta$$ . Đơn giản ta sẽ có công thức sau:
$$ y= a( w_{1}x_{1} + w_{2}x_{2} + w_{3}x_{3} + b ) (2) $$

Đến đây chắc hẳn bạn đã thấy sự liên quan với Perceptron mà mình đã giới thiệu ở bài trước. Nếu chưa, chắc hẳn bạn chưa đọc hoặc đã quên mất rồi, xin mời bạn đọc lại bài đó [tại đây](https://codetudau.com/deeplearning-perceptron-la-gi/) nhé.

## Đi đến Neural Network
Ở trên là công thức tính output của 1 unit trong một mạng lưới neural(Neural Network). Vậy rốt cục Neural Network có mặt mũi ra sao ?

![network-neurale](/assets/img/posts/2018/06/network-neurale.png)

Ồ, thì ra là sự kết hợp từ nhiều unit. Và tín hiệu sẽ được xử lý theo từng tầng(layer), như trên hình, tầng ở giữa được gọi là tầng ẩn(hidden layer), còn lại là tầng input và output.
Tầng sau sẽ nhận giá trị output của tầng trước để tiến hành xử lý. Còn xử lý ra sao là một chuyện khác, phụ thuộc vào từng bài toán mà công việc xử lý sẽ khác nhau. Và số lượng Hidden layer là không giới hạn, việc lựa chọn số tầng ẩn và cách xử lý ở mỗi tầng là chuyện không hề đơn giản. Nhưng đều có công thức cơ bản như (2) chỉ khác là thay đổi **Activation function**, **Input** Nói chung công thức ở trên là công thức tổng quát.


# Deep Learning

Cái tên **Deep Learning** ra đời với mục đích nhấn mạnh các **Hidden layers** của Neural Network. Có thể hiểu Deep Learning chính là Neural Network với nhiều **Hidden layers**.

![TB010-Deep-Neural-Network](/assets/img/posts/2018/06/TB010-Deep-Neural-Network.jpg)

À thế sao lại cần nhiều **Hidden layers** làm gì ?

Ví dụ như quá trình trưởng thành của "bướm". Mà bạn đang nghĩ đi đâu đấy... nhưng mình thích cách suy nghĩ của bạn. ^^ Thôi nghiêm túc, quá trình trưởng thành gồm các bước:
* Đầu tiên là trạng thái trong trứng
* Sau đó nở thành sâu
* Sâu ăn rất nhiều và đóng kén thành nhộng
* Và tada.. nở thành bướm

Ở Deep Learning cũng vậy, không có cách nào đi tắt đón đầu, mỗi Hidden layers sẽ có một nhiệm vụ, output của tầng này sẽ là input của tầng sau. Ở các bài viết sau (về CNN chẳng hạn) mình sẽ cho các bạn thấy rõ điều này.


# Ứng dụng của Deep Learning

Không thể phủ nhận được những thành công ngoài mong đợi của Deep Learning ở khắp các lĩnh vực phổ biến.

* Self-driving cars
* Voice Search & Voice-Activated Assistants
* Automatic Machine Translation
* Automatic Text Generation
* Nhận dạng ảnh (Image Recognition)
* ...

Bạn có thể tham khảo trend hiện nay [tại đây](https://medium.com/@vratulmittal/top-15-deep-learning-applications-that-will-rule-the-world-in-2018-and-beyond-7c6130c43b01).

# Tổng kết

Tiếp nối bài trước về Perceptron, ở bài này mình đã giới thiệu với các bạn về Neural Network từ ví dụ đầy sinh động về chính bộ não con người. ^^ Và đưa đến khái niệm Deep Learning mong những kiến thức trên sẽ có ích cho bạn. Và hẹn gặp lại các bạn ở các bài viết sau tiếp tục đi sâu và rộng hơn về Deep Learning.
