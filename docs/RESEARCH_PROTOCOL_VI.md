# Protocol nghiên cứu

## 1. Đơn vị phân tích

Một quan sát là cặp `(G,S)`, trong đó `G` là đồ thị gốc và `S` là tập cạnh bị
xóa. Nhiều quan sát có thể dùng chung `G`. Vì vậy tuyệt đối không chia ngẫu
nhiên từng quan sát: làm vậy sẽ đưa cùng cấu trúc đồ thị vào cả train và test.
Code hiện tại chia theo `graph_id`.

## 2. Dữ liệu giai đoạn 1

Sinh random geometric graph vì loại đồ thị này có tính địa phương không gian
gần với mạng đường hơn Erdős–Rényi. Mỗi cạnh có trọng số nghịch đảo chiều dài.
Với từng đồ thị:

- lấy 20 kịch bản;
- xóa từ 1 đến 8 cạnh tùy kích thước mạng;
- tính lại `lambda_2` để tạo ground truth;
- lưu adjacency sau gián đoạn và đặc trưng node.

Đặc trưng hiện tại gồm normalized degree, tỷ lệ cạnh kề bị mất, trị tuyệt đối
Fiedler coordinate và tọa độ không gian chuẩn hóa.

## 3. Baseline và mô hình

**Constant baseline:** dự báo mọi test scenario bằng trung bình target của
train set. Baseline này kiểm tra mô hình có thực sự học được hơn một hằng số.

**Spectral baseline:** tổng độ nhạy bậc nhất của các cạnh bị xóa.

**ScenarioGCN:** ba lớp truyền thông điệp trên adjacency bị hỏng, mean/max
pooling và regression head sigmoid. Mô hình trả giá trị trong `[0,1]`.

## 4. Chỉ số đánh giá

- MAE: dễ diễn giải trên thang suy giảm `[0,1]`.
- RMSE: phạt mạnh sai số lớn.
- R²: so sánh mức biến thiên được giải thích.
- Spearman: đánh giá năng lực xếp hạng kịch bản.
- Thời gian suy luận: đo riêng sau khi cố định hardware và batch protocol.

Metric chính phải được đăng ký trước là **test MAE**; Spearman là metric phụ.

## 5. Thí nghiệm bắt buộc

### E1 — Benchmark chính

Chạy 5 seeds: `11, 22, 33, 44, 55`, mỗi seed ít nhất 2.000 quan sát và 100
epochs. Tổng hợp mean ± SD.

### E2 — Theo cường độ gián đoạn

Nhóm test set theo `failed_count`: `1`, `2–3`, `4–5`, `6+`. So sánh GCN và
xấp xỉ phổ. Thí nghiệm này trực tiếp kiểm tra H1/H2.

### E3 — Ablation

- bỏ Fiedler coordinate;
- bỏ tọa độ;
- chỉ dùng degree và tỷ lệ cạnh hỏng.

### E4 — Generalization

Train trên 35–65 node, test trên 70–100 node. Đây là kiểm tra ngoại suy kích
thước, cần báo riêng và không trộn với test chính.

### E5 — Mạng đường thực

Tải một khu vực vừa phải bằng OSMnx, chuyển multigraph có hướng thành đồ thị
vô hướng có trọng số, lấy largest connected component, rồi sinh cùng protocol.
Không dùng toàn Hà Nội ở lần chạy đầu.

## 6. Phân tích thống kê

Đơn vị lặp độc lập là **seed**, không phải từng scenario. Báo hiệu số MAE theo
seed. Với 5 seed, ưu tiên bootstrap confidence interval hoặc Wilcoxon signed-rank;
không xem hàng nghìn scenario phụ thuộc trong cùng seed là hàng nghìn lần lặp.

## 7. Tiêu chí go/no-go

- Nếu GCN hơn spectral baseline ở ít nhất 4/5 seed và hiệu số có ý nghĩa thực
  tiễn, dùng GCN làm kết quả chính.
- Nếu GCN chỉ hơn ở nhóm nhiều cạnh, đó vẫn là kết quả đúng với giả thuyết và
  có giá trị.
- Nếu GCN không hơn, poster chuyển trọng tâm sang miền hiệu lực của xấp xỉ phổ
  và phân tích vì sao mô hình học máy không giúp ích.
- Không chọn seed đẹp hoặc loại kịch bản sau khi xem kết quả.

## 8. Threats to validity

- Distribution shift giữa graph tổng hợp và mạng thật.
- Trọng số nghịch đảo chiều dài chưa phản ánh capacity.
- Fiedler eigenvalue có thể không đơn, làm đạo hàm không ổn định.
- Mất cạnh không độc lập trong thiên tai không gian.
- GCN thông thường có thể khó nhận biết các thuộc tính phổ toàn cục.
