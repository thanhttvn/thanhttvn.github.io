---
title: Đánh giá model của machine learning(Precision, Recall, Bias & Variance, Cross Validation)
author: thanhtt
date: 2021-03-09 04:33:00 +0700
categories: [blog, python]
tags: [blog, python]
math: true
mermaid: true
image:
  src: 'https://drive.google.com/uc?id=0B05rqFCwNCjkQ3pjMlhaeTJEdjg&export=download'
---


### Dạo đầu
Làm gì cũng vậy, đều có công đoạn mang tên là đánh giá. Đơn giản, gần gũi như việc lấy vợ, có một công đoạn mang tên là đưa người yêu về ra mắt, mục đích chính là để bố mẹ, anh em họ hàng oánh giá. Tất nhiên giá cao bao giờ cũng được ưu tiên :D

Machine Learning cũng không có ngoại lệ, khi chúng ta xây dựng mô hình(model) có hàng tá model ta có thể sử dụng. Ví dụ bạn sử dụng RandomForest model, tương tự còn có ExtraTrees, AdaBoost...

Câu hỏi đặt ra là model này có tốt không. Một model tốt sẽ cho kết quả chính xác khi dự đoán kết quả với dữ liệu mới. Nên việc đánh giá model là một bước rất quan trọng để có thể xác định model có thể sử dụng được không, từ đó có thể tiếp tục tiến hành tuning parameter, chọn lựa lại feature hay sử dụng model khác, cũng có thể là bỏ cuộc ^^.

Tất nhiên không có model nào là tốt nhất với tất cả các hoàn cảnh, nó phụ thuộc vào đặc trưng của model, đặc trưng của dữ liệu, nên việc thử data của mình trên nhiều loại model là việc hết sức bình thường và nên làm.

Trước tiên đến với các phương pháp đánh giá, có 2 khái niệm rất quan trọng chính là Overfitting và Underfit mình muốn giới thiệu đến với các bạn.

### Overfitting, Underfitting là gì

![](https://drive.google.com/uc?id=0B05rqFCwNCjkZnlpT2FIamY4SG8&export=download)

Như trên hình(bài toán đang phân loại ○ và X), bạn có thể dùng mô hình Logistic Regression để giải quyết. Nếu bạn chưa biết có thể tham khảo thêm [tại đây](https://machinelearningcoban.com/2017/01/27/logisticregression/).

Theo thứ tự từ trái sang phải lần lượt là ví dụ về Underfitting, bình thường và Overfitting.

* Trong trường hợp Underfitting, model quá đơn giản nên rất nhiều X không được phân loại nên độ chính xác ngay cả trên tập Training Data rất tệ.

* Ngược lại với trường hợp Overfitting thì khi nhìn vào hình, bạn có thể thấy model lại quá phức tạp, mô tả cả noise data(2 dấu X nằm trong phần ○) nên độ chính xác trên tập Training là 100% nhưng thực tế với data mới(không có trong tập Training Data) thì độ chính xác rất tồi tệ.

Do vậy 1 model lý tưởng là model không quá đơn giản, không quá phức tạp và không dễ bị ảnh hưởng do noise.


### Cross Validation

Đầu tiên phải kể đến phương pháp `cross validation`, được đánh giá là phương pháp tai tiếng nhất, à nổi tiếng nhất.

Thông thường chúng ta hay làm như sau:

Chia data thành 2 phần, Training Data và Test Data. Tiến hành dùng Training Data để tạo model, dùng Test Data để dự đoán rồi xác định tỷ lệ đoán xịt, đoán trúng. Thông thường tỷ lệ khi chia data `Training:Test = 70:30`

![](https://drive.google.com/uc?id=0B05rqFCwNCjkbEJDWnpmYTlkMGM&export=download)

Tuy nhiên, có trường hợp một model cho cross validation tốt nhưng áp dụng với data mới thì kết quả lại không được như ý muốn.

Giả dụ trường hợp `Overfitting`, là hiện tượng mô hình tìm được quá khớp với dữ liệu training. Khớp quá nên mô hình có xu hướng mô tả cả nhiễu , thành ra khi cho test data vào toạch vô số kể. Thường xảy ra khi lượng data quá nhỏ so với độ phức tạp của model. Độ phức tạp của mô hình có thể được coi là bậc của đa thức cần tìm.

Những lúc như thế "Vậy thì thay đổi model để với test data cũng có kết quả tốt là xong", và rồi chúng ta cố gắng thay đổi model cho tỷ lệ dự đoán đúng trên Test Data cao, "Ngon, hoàn thành", nhưng chưa chắc dễ dàng vậy đâu, rất có thể model lại overfitting với Test Data.

Tóm lại, việc chia data làm 2 phần Training Data và Test Data thì vẫn chưa thể đưa ra kết luận chính xác cho model được.
Vậy nên mình sẽ làm như sau:

![](https://drive.google.com/uc?id=0B05rqFCwNCjkeHBDVVVvUXVodVE&export=download)

Ở bước chia dữ liệu, không chỉ chia làm 2 phần Training, Test mà chia thêm 1 phần là cross validation. Tỷ lệ thông thường: `Training:CV:Test = 60:20:20`

Tiếp theo:

<ol>
<li>Sử dụng Training Data để tìm parameter và tạo model.</li>
<li>Sử dụng Cross validation data để đánh giá độ chính xác của model.</li>
<li>Nếu độ chính xác thấp, tunning parameter để nâng cao độ chính xác của model.</li>
<li>Sau khi thu được model cuối cùng thì tiến hành đánh giá độ chính xác với Test data.</li>
</ol>

**Chú ý:**

* **Việc tìm high parameter chỉ tiến hành trên Training Data.** Sự khác nhau giữa Cross Validation Data Set và Test Data Set là việc thay đổi model, high parameter sao cho nâng cao được độ chính xác trên Cross Validation Data Set. Còn lại Test Set chỉ phục vụ cho phát đánh giá cuối cùng thôi nhé.

* Phương pháp tính Cross Validation ở trên có tên là Holdout method, ngoài ra còn có các cách khác như: k-fold cross-validation, Repeated random sub-sampling validation... các bạn có thể tìm hiểu thêm [tại đây](https://goo.gl/HGwdpL).
Tuy nhiên phương pháp Holdout method là phương pháp đơn giản và thường được sử dụng nhất.


### Bias & Variance

Nhắc lại một chút, 1 model tuyệt vời ông mặt trời là cả Training Set, Cross Validation Set, Test Set có độ lỗi thấp.

Trong trường hợp cả **Training Set, Cross Validation Set, Test Set có độ lỗi thấp** thì model được gắn với cái tên rất feed: **Underfit** hoặc **high bias** (*).

Trường hợp **Training Set** lỗi thấp nhưng trên Cross Validation, Test Set lỗi lớn thì được gọi là **Overfit** hoặc **High variance** (*)

À mà "Lỗi" là gì thế ???

Lỗi của một model đã dùng parameter θ để training được tính bằng cách lấy bình phương của giá trị dự đoán hθ(x) - giá trị thực tế y tại các data point.

![](https://drive.google.com/uc?id=0B05rqFCwNCjkaEJqTmhabFNFUmc&export=download)

m là số data samples.

Chúng ta cố gắng tìm kiếm parameter θ sao cho <span class="mrow" id="MathJax-Span-157"><span class="mi" id="MathJax-Span-158" style="font-family: MathJax_Math-italic;">J<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.056em;"></span></span><span class="mo" id="MathJax-Span-159" style="font-family: MathJax_Main;">(</span><span class="mi" id="MathJax-Span-160" style="font-family: MathJax_Math-italic;">θ</span><span class="mo" id="MathJax-Span-161" style="font-family: MathJax_Main;">)</span></span> nhỏ nhất nhưng để không xảy ra tình trạng Overfitting, chúng ta sẽ sử dụng thêm parameter chuẩn hóa λ (regularization parameter).

![](https://drive.google.com/uc?id=0B05rqFCwNCjkbS1qTU5RQTNXeEE&export=download)

Bằng cách này sẽ tránh được trường hợp giá trị θ lớn sẽ khó tìm được J(θ) nhỏ nhất, sẽ tránh được overfitting (high variance).

**Vậy lựa chọn λ như thế nào là hợp lý ??**

Với câu hỏi trên, giả sử trục tung là độ lỗi, trục hoành là λ, biểu diễn trên đồ thị ta sẽ được câu trả lời.

![](https://drive.google.com/uc?id=0B05rqFCwNCjkQ3pjMlhaeTJEdjg&export=download)

Do λ nhỏ quá thì sẽ bị overfitting (high variance), <span class="mrow" id="MathJax-Span-251"><span class="msubsup" id="MathJax-Span-252"><span style="display: inline-block; position: relative; width: 2.252em; height: 0px;"><span style="position: absolute; clip: rect(3.162em 1000.64em 4.179em -999.997em); top: -4.013em; left: 0.003em;"><span class="mi" id="MathJax-Span-253" style="font-family: MathJax_Math-italic;">J<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.056em;"></span></span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span><span style="position: absolute; top: -3.853em; left: 0.538em;"><span class="texatom" id="MathJax-Span-254"><span class="mrow" id="MathJax-Span-255"><span class="mi" id="MathJax-Span-256" style="font-size: 70.7%; font-family: MathJax_Math-italic;">t</span><span class="mi" id="MathJax-Span-257" style="font-size: 70.7%; font-family: MathJax_Math-italic;">r</span><span class="mi" id="MathJax-Span-258" style="font-size: 70.7%; font-family: MathJax_Math-italic;">a</span><span class="mi" id="MathJax-Span-259" style="font-size: 70.7%; font-family: MathJax_Math-italic;">i</span><span class="mi" id="MathJax-Span-260" style="font-size: 70.7%; font-family: MathJax_Math-italic;">n</span></span></span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span></span></span><span class="mo" id="MathJax-Span-261" style="font-family: MathJax_Main;">(</span><span class="mi" id="MathJax-Span-262" style="font-family: MathJax_Math-italic;">θ</span><span class="mo" id="MathJax-Span-263" style="font-family: MathJax_Main;">)</span></span> của Training Data sẽ nhỏ , độ lỗi Jcv của Cross Validation Set trở nên lớn.

Ngược lại λ lớn quá model sẽ bị Underfit hoặc high bias. Cả 2 độ lỗi  của Training Data, Cross Validation Set sẽ cùng trở nên lớn.

Do vậy chọn λ tại điểm khoanh đỏ sẽ cho <strong><span class="MathJax_Preview" style="color: inherit;"></span><span class="MathJax" id="MathJax-Element-25-Frame" tabindex="0" data-mathml="<math xmlns=&quot;http://www.w3.org/1998/Math/MathML&quot;><msub><mi>J</mi><mrow class=&quot;MJX-TeXAtom-ORD&quot;><mi>C</mi><mi>V</mi></mrow></msub><mo stretchy=&quot;false&quot;>(</mo><mi>&amp;#x03B8;</mi><mo stretchy=&quot;false&quot;>)</mo></math>" role="presentation" style="position: relative;"><nobr aria-hidden="true"><span class="math" id="MathJax-Span-303" role="math" style="width: 3.308em; display: inline-block;"><span style="display: inline-block; position: relative; width: 2.657em; height: 0px; font-size: 124%;"><span style="position: absolute; clip: rect(1.655em 1002.61em 2.858em -999.997em); top: -2.502em; left: 0.003em;"><span class="mrow" id="MathJax-Span-304"><span class="msubsup" id="MathJax-Span-305"><span style="display: inline-block; position: relative; width: 1.505em; height: 0px;"><span style="position: absolute; clip: rect(3.208em 1000.5em 4.16em -999.997em); top: -4.005em; left: 0.003em;"><span class="mi" id="MathJax-Span-306" style="font-family: STIXGeneral-Italic;">J<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.053em;"></span></span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span><span style="position: absolute; top: -3.854em; left: 0.453em;"><span class="texatom" id="MathJax-Span-307"><span class="mrow" id="MathJax-Span-308"><span class="mi" id="MathJax-Span-309" style="font-size: 70.7%; font-family: STIXGeneral-Italic;">C<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span class="mi" id="MathJax-Span-310" style="font-size: 70.7%; font-family: STIXGeneral-Italic;">V<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.053em;"></span></span></span></span><span style="display: inline-block; width: 0px; height: 4.01em;"></span></span></span></span><span class="mo" id="MathJax-Span-311" style="font-family: STIXGeneral-Regular;">(</span><span class="mi" id="MathJax-Span-312" style="font-family: STIXGeneral-Italic;">θ<span style="display: inline-block; overflow: hidden; height: 1px; width: 0.003em;"></span></span><span class="mo" id="MathJax-Span-313" style="font-family: STIXGeneral-Regular;">)</span></span><span style="display: inline-block; width: 0px; height: 2.507em;"></span></span></span><span style="display: inline-block; overflow: hidden; vertical-align: -0.307em; border-left: 0px solid; width: 0px; height: 1.183em;"></span></span></nobr></span></strong> nhỏ nhất nhỉ.

Bài này mình cũng không đi sâu và việc phòng tránh Underfitting và Overfitting nhưng có cách thường được sử dụng như sau:

**Chữa bệnh Underfitting(High bias)**
<ol>
<li>Tìm kiếm biến giải thích(feature) khác.</li>
<li>Thêm vào các feature dạng (<span class="mrow" id="MathJax-Span-318"><span class="msubsup" id="MathJax-Span-319"><span style="display: inline-block; position: relative; width: 1.02em; height: 0px;"><span style="position: absolute; clip: rect(3.43em 1000.54em 4.179em -999.997em); top: -4.013em; left: 0.003em;"><span class="mi" id="MathJax-Span-320" style="font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span><span style="position: absolute; clip: rect(3.376em 1000.43em 4.179em -999.997em); top: -4.334em; left: 0.592em;"><span class="mn" id="MathJax-Span-321" style="font-size: 70.7%; font-family: MathJax_Main;">2</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span><span style="position: absolute; clip: rect(3.376em 1000.43em 4.179em -999.997em); top: -3.692em; left: 0.592em;"><span class="mn" id="MathJax-Span-322" style="font-size: 70.7%; font-family: MathJax_Main;">1</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span></span></span><span class="mo" id="MathJax-Span-323" style="font-family: MathJax_Main;">,</span><span class="msubsup" id="MathJax-Span-324" style="padding-left: 0.163em;"><span style="display: inline-block; position: relative; width: 1.02em; height: 0px;"><span style="position: absolute; clip: rect(3.43em 1000.54em 4.179em -999.997em); top: -4.013em; left: 0.003em;"><span class="mi" id="MathJax-Span-325" style="font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span><span style="position: absolute; clip: rect(3.376em 1000.43em 4.179em -999.997em); top: -4.334em; left: 0.592em;"><span class="mn" id="MathJax-Span-326" style="font-size: 70.7%; font-family: MathJax_Main;">2</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span><span style="position: absolute; clip: rect(3.376em 1000.43em 4.179em -999.997em); top: -3.692em; left: 0.592em;"><span class="mn" id="MathJax-Span-327" style="font-size: 70.7%; font-family: MathJax_Main;">2</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span></span></span><span class="mo" id="MathJax-Span-328" style="font-family: MathJax_Main;">,</span><span class="msubsup" id="MathJax-Span-329" style="padding-left: 0.163em;"><span style="display: inline-block; position: relative; width: 1.02em; height: 0px;"><span style="position: absolute; clip: rect(3.43em 1000.54em 4.179em -999.997em); top: -4.013em; left: 0.003em;"><span class="mi" id="MathJax-Span-330" style="font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span><span style="position: absolute; top: -3.853em; left: 0.592em;"><span class="mn" id="MathJax-Span-331" style="font-size: 70.7%; font-family: MathJax_Main;">1</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span></span></span><span class="msubsup" id="MathJax-Span-332"><span style="display: inline-block; position: relative; width: 1.02em; height: 0px;"><span style="position: absolute; clip: rect(3.43em 1000.54em 4.179em -999.997em); top: -4.013em; left: 0.003em;"><span class="mi" id="MathJax-Span-333" style="font-family: MathJax_Math-italic;">x</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span><span style="position: absolute; top: -3.853em; left: 0.592em;"><span class="mn" id="MathJax-Span-334" style="font-size: 70.7%; font-family: MathJax_Main;">2</span><span style="display: inline-block; width: 0px; height: 4.019em;"></span></span></span></span></span>)</li>
<li>Giảm parameter λ xuống</li>
</ol>

**Chữa bệnh Overfitting (High variance)**

<ol>
<li>Tăng số lượng Training Data</li>
<li>Giảm số lượng biến giải thích(feature)</li>
<li>Tăng độ lớn của parameter chuẩn hóa λ</li>
</ol>

**Chú thích (*):**

Như biểu đồ trên, khi High Variance thì độ lỗi trên tập train sẽ thấp nhưng khi đó trên Test Data độ lỗi lớn chính là hiện tượng Overfitting.

Ngược lại khi High Bias thì độ lỗi trên Training Data lớn và đương nhiên độ lỗi trên Test Data cũng sẽ lớn. Cũng chính là hiện tượng Underfitting.

### Precision & Recall

Cách đánh giá này thường được áp dụng cho các bài toán phân lớp có hai lớp dữ liệu. Cụ thể hơn, trong hai lớp dữ liệu này có một lớp nghiêm trọng hơn lớp kia và cần được dự đoán chính xác.

![](https://drive.google.com/uc?id=0B05rqFCwNCjkYjlvcV9YZ2dlSjg&export=download)

p: viết tắt của positive

n: viết tắt của negative


Ví dụ như việc xác định mail spam, việc nhầm mail quan trọng thành mail spam nguy hiểm hơn là bỏ sót mail spam. Hay việc dự đoán động đất nhầm với tỷ lệ thấp tốt hơn là bỏ sót.

Trong những bài toán này, người ta thường định nghĩa lớp dữ liệu quan trọng cần được xác định đúng là lớp Positive (P-dương tính), lớp còn lại được gọi là Negative (N-âm tính).
Ta định nghĩa True Positive (TP), False Positive (FP), True Negative (TN), False Negative (FN) dựa trên confusion matrix như trên.

Tỷ lệ chính xác (Precision) và tỷ lệ tái hiện tính theo công thức:

![](https://drive.google.com/uc?id=0B05rqFCwNCjkTXJLWnUwUFVYdXc&export=download)

Tỷ lệ chính xác (Precision) là việc lấy tỷ lệ của **số dự đoán y = 1 đúng với thực tế y cũng = 1** với **tổng số tất cả số lần dự đoán y = 1** (vùng khoanh đỏ ở trên hình.) Giá trị càng cao, càng tốt.

Tỷ lệ tái hiện (Recall) là tỷ lệ của **số dự đoán y = 1 đúng với thực tế y cũng = 1** chia cho **tổng số trường hợp thực tế y = 1**. Giá trị càng cao, các tốt.

**Ví dụ bài toán lọc mail rác:**

![](https://drive.google.com/uc?id=0B05rqFCwNCjkc201VkoxS0JGU2s&export=download)

Prec = 8/(8+32) = 20%

Rec = 8/10 = 80%

Từ kết quả ta có thể kết luận như sau:

Tỷ lệ xác suất bộ lọc chính xác khi xác định 1 mail là thư rác là 20%.

Tỷ lệ xác suất một thư rác bị bộ lọc phát hiện là 80%.

Ta có thể thấy tỷ lệ phát hiện ra thư rác khá cao, nhưng tỷ lệ chính xác lại thấp, việc xác định thư quan trọng thành thư rác là việc cực kỳ nguy hiểm nên với kết quả trên cần cải tiến sao cho cả giá trị Precision = 100% là tốt nhất.


### Kết luận

Trên đây mình đã giới thiệu đến các bạn các cách đánh giá model của machine learning, cùng với đó là khái niệm Overfitting (High variance) hay Underfitting(High bias).

Ngoài ra là một số cách khắc phục tình trạng underfit và overfit. Hi vọng sẽ giúp đỡ được các bạn trong quá trình tìm hiểu và làm việc với machine learning.
