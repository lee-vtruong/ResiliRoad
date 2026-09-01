# VMS60 — Spectral resilience + AI

Research starter kit cho đề tài poster:

> **Learning Road-Network Resilience under Disruptions with Spectral Graph Theory and Graph Neural Networks**

Tác giả: **Le Van Truong** — University of Science, VNU-HCM.

Mục tiêu là dự báo mức suy giảm *algebraic connectivity* khi một tập cạnh của
mạng giao thông bị gián đoạn. Repository có ba lớp kết quả:

1. ground truth tính trực tiếp từ phổ Laplacian;
2. baseline giải tích bậc nhất dùng Fiedler vector;
3. mô hình ScenarioGCN học từ đồ thị sau gián đoạn.

## Chạy nhanh

Yêu cầu Python 3.11+ và các gói: `numpy scipy networkx pandas scikit-learn
torch matplotlib`.

```powershell
python run_experiment.py --samples 600 --epochs 80 --output outputs/quick
```

Chạy smoke test nhanh:

```powershell
python run_experiment.py --samples 80 --epochs 3 --output outputs/smoke
```

Chạy benchmark 5 seed và tổng hợp:

```powershell
python run_experiment.py --samples 2000 --epochs 100 --seed 11 --output outputs/benchmark/seed_11
# Lặp lại với seed 22, 33, 44, 55
python aggregate_results.py --input outputs/benchmark --output outputs/summary
```

Kết quả gồm:

- `metrics.json`: MAE, RMSE, R² và Spearman correlation;
- `predictions.csv`: nhãn thật và dự báo trên test set;
- `prediction_scatter.png`: hình dùng để kiểm tra chất lượng mô hình;
- `learning_curve.png`: loss train/validation;
- `dataset_summary.json`: cấu hình sinh dữ liệu.

## Tài liệu

- [Proposal tiếng Việt](docs/PROPOSAL_VI.md)
- [Quy trình nghiên cứu](docs/RESEARCH_PROTOCOL_VI.md)
- [Hướng dẫn từng bước](docs/HUONG_DAN_VI.md)
- [Báo cáo sơ bộ](report/output/pdf/ResiliRoad_Preliminary_Report.pdf)

## Kết quả mở rộng

Tổng cộng 10.000 kịch bản trên 5 seed; 1.500 kịch bản test thuộc các đồ thị
không xuất hiện trong train/validation.

| Phương pháp | MAE | RMSE | R² | Spearman |
|---|---:|---:|---:|---:|
| Direct GCN | 0.0656 ± 0.0132 | 0.1492 ± 0.0273 | 0.5464 ± 0.0952 | 0.6279 ± 0.0571 |
| GCN không có Fiedler feature | 0.0694 ± 0.0133 | 0.1517 ± 0.0351 | 0.5372 ± 0.0910 | 0.6308 ± 0.0542 |
| Residual spectral-GCN | **0.0390 ± 0.0099** | **0.1083 ± 0.0320** | **0.7633 ± 0.0759** | 0.8764 ± 0.0375 |
| Spectral first order | 0.0652 ± 0.0243 | 0.2090 ± 0.0508 | 0.1331 ± 0.1222 | **0.9507 ± 0.0172** |

Trên 1.500 kịch bản zero-shot của mạng OSM 163 nút/221 cạnh, residual đạt
MAE `0.1018 ± 0.0155` và Spearman `0.9269 ± 0.0119`, tốt hơn direct GCN.
Các số là mean ± sample SD qua 5 seed. Đây là nghiên cứu gián đoạn cấu trúc,
không phải mô hình dự báo ngập thực tế.

Chạy nghiên cứu mở rộng:

```powershell
python download_osm.py
python run_extended_benchmark.py --seed 11 --samples 2000 --osm-samples 300 --epochs 100 --output outputs/extended/seed_11
# Lặp lại seed 22, 33, 44, 55
python aggregate_extended.py
```

## Nguyên tắc diễn giải

Đây là dữ liệu mô phỏng, không phải dự báo ngập thực tế. Chỉ được dùng từ
“flood” sau khi bổ sung lớp nguy cơ ngập có nguồn dữ liệu đáng tin cậy. Trong
giai đoạn đầu, tên chính xác là *network disruptions*.
