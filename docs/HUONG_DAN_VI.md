# Hướng dẫn thực hiện từng bước

## Bước 1 — Chạy và xác nhận pipeline

```powershell
python run_experiment.py --samples 80 --epochs 3 --output outputs/smoke
```

Mở `outputs/smoke/metrics.json`. Không dùng kết quả smoke test trong poster.

## Bước 2 — Chạy thí nghiệm pilot

```powershell
python run_experiment.py --samples 600 --epochs 80 --seed 42 --output outputs/pilot42
```

Kiểm tra learning curve. Nếu validation loss tăng trong khi train loss giảm,
mô hình đang overfit; giảm hidden dimension, tăng dropout hoặc tăng dữ liệu.

## Bước 3 — Đọc kết quả đúng cách

- MAE thấp hơn là tốt hơn.
- R² âm nghĩa là mô hình tệ hơn dự báo hằng theo metric bình phương.
- Spearman cao nghĩa là xếp hạng tốt, kể cả khi calibration chưa tốt.
- So sánh GCN với `spectral_first_order`, không chỉ với constant baseline.

Không kết luận từ một seed. Pilot chỉ có tác dụng tìm lỗi và ước lượng thời gian.

## Bước 4 — Chạy benchmark nhiều seed

Chạy lần lượt 5 lệnh, thay `SEED` bằng 11, 22, 33, 44 và 55:

```powershell
python run_experiment.py --samples 2000 --epochs 100 --seed SEED --output outputs/seed_SEED
```

Lưu nguyên các thư mục output. Sau đó mới viết script tổng hợp mean ± SD.

## Bước 5 — Thêm mạng đường thật

Cài OSMnx trong một môi trường riêng nếu cần:

```powershell
pip install osmnx
```

Tải một quận/khu vực nhỏ trước. Ghi chính xác địa danh truy vấn, ngày tải,
`network_type`, phép đơn giản hóa và cách gộp cạnh song song. Tuân thủ chính
sách sử dụng của OpenStreetMap/Nominatim/Overpass; cache dữ liệu thay vì tải
lặp lại.

## Bước 6 — Tạo kịch bản gần với ngập

Thứ tự tăng độ tin cậy:

1. xóa cạnh hoàn toàn ngẫu nhiên;
2. xóa cạnh theo vùng tròn/ô lưới để mô phỏng tương quan không gian;
3. xác suất gián đoạn phụ thuộc độ cao;
4. dùng bản đồ nguy cơ/ngập có nguồn và giấy phép rõ ràng.

Ở mức 1–2, gọi kết quả là “spatial disruption scenarios”, chưa gọi là dự báo ngập.

## Bước 7 — Viết abstract

Chỉ điền số sau khi benchmark hoàn tất:

> We study the rapid estimation of structural connectivity loss in road-like
> graphs under multi-edge disruptions. We define resilience loss through the
> relative decrease of algebraic connectivity and derive a first-order
> Fiedler-vector baseline. To capture nonlinear interactions among simultaneous
> failures, we develop a scenario-aware graph convolutional regressor. Using
> graph-level held-out evaluation over [N] random geometric graphs and [M]
> disruption scenarios, the proposed method achieves [MAE] MAE and [rho]
> Spearman correlation, compared with [baseline values]. Error stratification
> shows [main finding]. The resulting reproducible pipeline provides a basis
> for screening disruption scenarios before extension to real street networks.

## Bước 8 — Cấu trúc poster

- 15%: vấn đề, câu hỏi nghiên cứu và đóng góp.
- 20%: định nghĩa target và xấp xỉ Fiedler.
- 20%: sơ đồ ScenarioGCN và protocol chia dữ liệu.
- 30%: bảng metric, scatter plot, kết quả theo số cạnh hỏng.
- 15%: giới hạn, mã QR tới code và hướng phát triển.

Tiêu đề hình phải tự giải thích được. Một hình chính nên trả lời một câu hỏi,
không chỉ trang trí.

## Bước 9 — Hồ sơ xin hỗ trợ

Trong phần đăng ký, nhấn mạnh:

- bạn là nhà nghiên cứu trẻ và ở ngoài Hà Nội nếu đúng thực tế;
- kết quả đã có, không chỉ là ý tưởng;
- mã nguồn và protocol tái lập;
- giao thoa giữa toán hiện đại, AI và bài toán hạ tầng;
- nhu cầu hỗ trợ đi lại/chỗ ở trình bày ngắn gọn, chính xác.

