# Đề xuất nghiên cứu cho VMS60

## Tên đề tài

**Học khả năng chống chịu của mạng giao thông dưới các kịch bản gián đoạn bằng lý thuyết đồ thị phổ và mạng nơ-ron đồ thị**

**English:** *Learning Road-Network Resilience under Disruptions with Spectral Graph Theory and Graph Neural Networks*

## Tóm tắt đề xuất

Mạng giao thông có thể được biểu diễn bằng đồ thị có trọng số, trong đó các
nút là giao lộ và các cạnh là đoạn đường. Khi nhiều đoạn đường đồng thời mất
khả năng phục vụ, việc tính lại các chỉ số kết nối cho số lượng lớn kịch bản có
thể tốn kém. Nghiên cứu này đề xuất kết hợp lý thuyết đồ thị phổ với mạng
nơ-ron đồ thị để dự báo nhanh mức suy giảm khả năng kết nối. Algebraic
connectivity, tức trị riêng nhỏ thứ hai của ma trận Laplacian, được dùng làm
đại lượng nền. Một xấp xỉ bậc nhất dựa trên Fiedler vector được xây dựng làm
baseline có thể diễn giải; ScenarioGCN sau đó học các tương tác phi tuyến khi
nhiều cạnh bị gián đoạn. Thí nghiệm được tiến hành trước hết trên các đồ thị
hình học ngẫu nhiên, với phép chia train/test theo đồ thị gốc để kiểm tra khả
năng tổng quát hóa. Giai đoạn tiếp theo áp dụng cùng protocol cho một mạng
đường thu từ OpenStreetMap. Chất lượng được đánh giá bằng MAE, RMSE, hệ số
xác định, tương quan thứ hạng và thời gian suy luận. Nghiên cứu hướng đến một
quy trình tái lập để sàng lọc nhanh các kịch bản rủi ro; không xem kết quả mô
phỏng là dự báo ngập thực tế nếu chưa tích hợp dữ liệu nguy cơ ngập đã kiểm
chứng.

## Bối cảnh và khoảng trống

Algebraic connectivity phản ánh mức kết nối của đồ thị và bằng không khi đồ
thị không liên thông. Đạo hàm bậc nhất theo trọng số một cạnh có thể được biểu
diễn qua hiệu hai thành phần của Fiedler vector. Xấp xỉ này dễ giải thích nhưng
có thể mất chính xác khi nhiều cạnh bị xóa đồng thời, khi trị riêng gần nhau,
hoặc khi đồ thị tiến gần trạng thái mất liên thông. Đây là khoảng trống mà mô
hình học máy có thể hỗ trợ: học phần tương tác bậc cao trong khi vẫn dùng các
đặc trưng phổ làm prior.

## Mục tiêu

1. Xây dựng chỉ số suy giảm kết nối chuẩn hóa giữa các đồ thị khác kích thước.
2. Đánh giá miền hiệu lực của xấp xỉ nhiễu loạn phổ bậc nhất.
3. Xây dựng ScenarioGCN dự báo tác động của gián đoạn nhiều cạnh.
4. Kiểm tra khả năng tổng quát hóa sang đồ thị chưa xuất hiện khi huấn luyện.
5. Tạo pipeline mã nguồn mở, tái lập được, sẵn sàng mở rộng sang mạng đường thực.

## Câu hỏi và giả thuyết

**RQ1.** Sai số của xấp xỉ phổ tăng như thế nào theo số cạnh bị gián đoạn?

**H1.** Xấp xỉ phổ cạnh tranh khi chỉ mất ít cạnh nhưng suy giảm khi tương tác
giữa nhiều cạnh trở nên đáng kể.

**RQ2.** ScenarioGCN có giảm MAE trên các đồ thị test chưa từng gặp không?

**H2.** Mô hình học được quan hệ phi tuyến và có MAE thấp hơn baseline phổ ở
nhóm kịch bản nhiều cạnh, nhưng không nhất thiết vượt baseline ở trường hợp
một cạnh.

**RQ3.** Mô hình có bảo toàn thứ hạng mức nguy hiểm của các kịch bản không?

**H3.** Spearman correlation của mô hình lớn hơn baseline hằng và đủ cao để
dùng cho sàng lọc kịch bản.

## Mô hình toán học

Với đồ thị liên thông có trọng số `G`, đặt Laplacian `L=D-A` và
`lambda_2(G)` là trị riêng nhỏ thứ hai. Với đồ thị sau gián đoạn `G-S`, biến
mục tiêu là

```text
y(G,S) = clip(1 - lambda_2(G-S) / lambda_2(G), 0, 1).
```

Giá trị 0 nghĩa là không suy giảm; giá trị 1 bao gồm trường hợp mất liên
thông. Nếu `u_2` là Fiedler vector chuẩn hóa thì độ nhạy theo trọng số cạnh
`e=(i,j)` là

```text
d lambda_2 / d w_e = (u_2[i] - u_2[j])^2.
```

Baseline cho tập cạnh bị xóa `S`:

```text
y_spectral = clip(sum_{e in S} w_e (u_2[i]-u_2[j])^2 / lambda_2(G), 0, 1).
```

Đây là xấp xỉ bậc nhất, không phải đẳng thức cho gián đoạn hữu hạn.

## Đóng góp dự kiến

- Protocol chống rò rỉ dữ liệu: chia tập theo đồ thị gốc thay vì theo kịch bản.
- Baseline phổ có diễn giải và phân tích sai số theo cường độ gián đoạn.
- Kiến trúc GCN nhỏ, không phụ thuộc PyTorch Geometric, dễ tái lập.
- Đánh giá đồng thời sai số giá trị, thứ hạng và chi phí tính toán.
- Trường hợp nghiên cứu trên mạng đường Việt Nam ở giai đoạn mở rộng.

## Giới hạn và tuyên bố khoa học

- Đồ thị hình học ngẫu nhiên chỉ là benchmark, không thay thế mạng giao thông thật.
- Xóa cạnh ngẫu nhiên không đồng nghĩa với ngập lụt.
- Algebraic connectivity đo kết nối cấu trúc, không đo trực tiếp lưu lượng hoặc thời gian đi lại.
- Không khẳng định GCN tốt hơn trước khi có nhiều seed và kiểm định thống kê.
- Nếu dùng dữ liệu OpenStreetMap, phải ghi ngày tải, phạm vi và quy tắc tiền xử lý.

## Kết quả tối thiểu để nộp poster

1. Thí nghiệm tối thiểu 5 random seeds.
2. Báo cáo mean ± standard deviation cho MAE và Spearman.
3. Phân tích riêng theo số cạnh bị xóa.
4. Một ablation bỏ đặc trưng Fiedler khỏi GCN.
5. Một ví dụ bản đồ hoặc đồ thị trực quan về kịch bản quan trọng.
6. Mã nguồn, cấu hình và phiên bản dữ liệu.

## Tài liệu nền chọn lọc

- M. Fiedler, “Algebraic connectivity of graphs,” *Czechoslovak Mathematical
  Journal*, 1973, DOI: 10.21136/CMJ.1973.101168.
- T. N. Kipf và M. Welling, “Semi-Supervised Classification with Graph
  Convolutional Networks,” ICLR 2017, arXiv:1609.02907.
- M. Zaheer et al., “Deep Sets,” NeurIPS 2017, arXiv:1703.06114.
- G. Boeing, “OSMnx: New Methods for Acquiring, Constructing, Analyzing, and
  Visualizing Complex Street Networks,” *Computers, Environment and Urban
  Systems*, 2017.
- P. Y. R. Sohouenou et al., “Using a random road graph model to understand
  road networks robustness to link failures,” *International Journal of
  Critical Infrastructure Protection*, 2020.
