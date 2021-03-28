---
title: Cài đặt hệ thống gợi ý bài viết Made in Vietnam và FREE
author: thanhtt
date: 2021-03-28 15:33:00 +0700
categories: [blog, recommender system]
tags: [blog, recommender system]
math: true
mermaid: true
image:
  src: 'https://blog.vietnamlab.vn/content/images/17hiALjvKH0XanUqZh0GAcZrdhF0gnQ8C.png'
---

Lâu lắm mới có dịp viết blog, dạo này AE khoẻ không :)

Hiện tại mình làm việc tại một công ty nho nhỏ. Ở công ty này có phong trào viết blog để anh chị em trong công ty chia sẻ được những gì mình đang tìm hiểu, nghiên cứu. Hơn nữa bài viết nào nhiều pageview nhất sẽ được thưởng 500k. Lúc đầu anh chị em hào hứng lắm.

Tuy nhiên gần đây AE không hứng thú viết blog nữa, dẫn đến phong trào đi xuống. AE vẫn viết nhưng viết theo kiểu chống chế, viết cho có, dẫn đến chất lượng không còn được như trước.

Thấy vậy sếp giao cho mình nhiệm vụ điều tra nguyên nhân và tìm cách cải thiện phong trào viết blog của công ty.

![13LK1CDBbNVsf4cnGKTzcgkg_dBhyQxU1](https://blog.vietnamlab.vn/content/images/13LK1CDBbNVsf4cnGKTzcgkg_dBhyQxU1.jpg)

Sau một thời gian lê la trà đá vỉa hè, thanh niên A đã biết nguyên nhân tại sao AE không muốn viết blog là:

- Để viết được một bài blog tốn rất nhiều tâm huyết nhưng đến khi publish lại lẹt đẹt vài view nên dần dần AE không còn mặn mà với việc viết blog nữa.
- Blog có số lượng bài tương đối nhiều dẫn đến các bài viết cũ không hiện lên hompage nên lượt view lại càng lẹt đẹt đi.

Mình nảy sinh ra ý tưởng làm tính năng gợi ý các bài viết liên quan đến bài viết người đọc đang đọc để có thể cải thiện lượng pageview của blog công ty.

Tuy nhiên nếu chỉ làm riêng cho blog công ty thì lại thấy lãng phí công sức nên quyết định hợp tác với anh B(Trưởng nhóm System) để xây dựng một hệ thống recommender system có thể tích hợp trên tất cả các website khác. Và đương nhiên phải cài đặt dễ dàng nhất có thể.

![1h6SQIMT6DuuUP4xFC9_QwtptA1cHKGgS](https://blog.vietnamlab.vn/content/images/1h6SQIMT6DuuUP4xFC9_QwtptA1cHKGgS.jpg)

Sau một thời gian dự án đã hoàn thành và sẵn sàng cho mọi người trải nghiệm. Và dịch vụ đó có tên là REEM.

# Tính năng của REEM
## Đề xuất các bài viết phù hợp

- Kết hợp nhiều model để đề xuất các bài viết phù hợp nhất cho người đọc. Sẽ giúp người đọc đọc nhiều hơn. :))
- Bài viết đề xuất chính là các bài cùng chủ đề hoặc các bài mà có thể người dùng sẽ quan tâm trên chính website của bạn.
- Tự động phát hiện các bài viết mới và tự động đưa ra bài viết gợi ý phù hợp.
- Cài đặt cực dễ. Các bạn chỉ cần chèn 1 đoạn javascript không dài quá một dòng là xong.
- **FREE**, đúng vậy. Mình đã xin sếp rồi, tính năng này hoàn toàn free trọn đời nhé. Riêng về chuyện server thì không phải lo, của nhà làm ra mà.

![1sJ9lLs5oriTo0oW8jI-AInCqD0GQ-0KL](https://blog.vietnamlab.vn/content/images/1sJ9lLs5oriTo0oW8jI-AInCqD0GQ-0KL.png)

## Quản lý thông tin lượng truy cập Blog

Sau khi đăng ký hoàn tất AE có thể truy cập vào màn hình quản lý. Bao gồm các chức năng:
- Thống kê lượng truy cập
- Điều chỉnh lại khung gợi ý theo ý mình. (kích cỡ, số lượng bài viết gợi ý ...)

![1o0WyV91DbcKA2k3n39NqqS56YABQAUGJ](https://blog.vietnamlab.vn/content/images/1o0WyV91DbcKA2k3n39NqqS56YABQAUGJ.png)

## AB Test

Tính năng này dành riêng cho Admin, với mục đích là tuning việc cài đặt các Model để cho hiệu quả cao nhất.


## Tính năng quảng cáo(Đang tìm mối :))

- Nhiều người bảo làm FREE thì cạp *** mà ăn à. Đúng vậy, tính năng tiếp theo sẽ là hiển thị quảng cáo xen kẽ trong khung đề xuất.
- Tuy nhiên chưa tìm được công ty quảng cáo nào để liên kết. Tiện thể AE nào có mối là các công ty có quảng cáo thì giới thiệu để hợp tác nhé.

## Yêu cầu phi tính năng

Về cơ bản AE chúng tớ đã làm giải quyết được các vấn đề sau:

- Hệ thống yêu cầu bảo mật cao
- Dễ dàng scale up khi nhiều người truy cập
- Xử lý được dữ liệu lớn khi khối lượng log nhiều
- Cài đặt dịch vụ dễ dàng
- Làm vì đam mê, không cần lương :))
-
# Cách đăng ký
![1b-JBa9Ih78iQGUudOzSyuU5sdwJwl_Xy](https://blog.vietnamlab.vn/content/images/1b-JBa9Ih78iQGUudOzSyuU5sdwJwl_Xy.png)

- Bước 1: Truy cập https://reem.vn
- Bước 2: Click vào đăng ký và điền các thông tin cần thiết.
- Bước 3: Tổng đài REEM sẽ gọi điện confirm các thông tin AE điền chính xác chưa và tiến hành kích hoạt tài khoản cho AE.
- Bước 4: Chèn link javascript vào trang web của AE và tận hưởng. :)

# Lời kết

Chắc chắn rồi, dự án vẫn đang phát triển với dự định phát triển nhiều model dự đoán thú vị hơn. AE mà có blog thì hãy cài đặt đi nhé, nhất là mấy anh em làm ở các media lớn như baomoi, dantri, kenh14, thanhnien... mà hợp tác với mình thì vui hơn nữa. :)
