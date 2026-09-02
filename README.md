# ResiliRoad

Mã nguồn và báo cáo cho đề tài **Residual Spectral Graph Learning for
Connectivity-Loss Estimation under Multi-Edge Disruptions**.

Mô hình ước lượng suy giảm algebraic connectivity khi nhiều cạnh của mạng đường
bị gián đoạn. Dự án so sánh spectral first-order approximation, direct/residual
GCN, Deep Sets, summary MLP, ablation Fiedler feature và tọa độ.

## Kết quả chính

Thí nghiệm dùng 5 seed, graph-disjoint synthetic split, hai cơ chế gián đoạn
(độc lập và cụm không gian), cùng zero-shot transfer trên 5 mạng OSM Việt Nam.
Bootstrap lấy mẫu theo seed, không xem các scenario cùng seed là mẫu độc lập.

| OSM failure mode | Direct GCN MAE | Residual GCN MAE | Paired improvement [95% CI] |
|---|---:|---:|---:|
| Independent | 0.106 | 0.097 | 0.0085 [0.0039, 0.0131] |
| Spatial cluster | 0.102 | 0.090 | 0.0121 [0.0026, 0.0247] |

Bỏ Fiedler node feature hoặc tọa độ chưa tạo khác biệt MAE có ý nghĩa ở mức 5
seed; kết quả hỗ trợ residual prior, nhưng chưa hỗ trợ tuyên bố rằng từng feature
riêng lẻ là cần thiết. Đây là nghiên cứu gián đoạn cấu trúc, không phải dự báo
ngập hay mô hình giao thông.

## Chạy toàn bộ

Yêu cầu Python 3.11+ và các gói trong `requirements.txt`. Không cần GPU; benchmark
đã báo cáo được chạy hoàn toàn bằng CPU.

```powershell
pip install -r requirements.txt
.\reproduce_all.ps1
```

Hoặc chạy từng phần:

```powershell
python download_osm.py
python run_paper_benchmark.py --seed 11 --output outputs/paper/seed_11
python analyze_paper_results.py --input outputs/paper --output outputs/paper_summary
python collect_environment.py
```

Lặp benchmark với seed `11, 22, 33, 44, 55`. Script `reproduce_all.ps1` thực
hiện vòng lặp này và biên dịch báo cáo.

## Tài liệu và artefact

- [Báo cáo PDF](report/output/pdf/ResiliRoad_Preliminary_Report.pdf)
- [Proposal tiếng Việt](docs/PROPOSAL_VI.md)
- [Protocol nghiên cứu](docs/RESEARCH_PROTOCOL_VI.md)
- [Ma trận thí nghiệm](docs/EXPERIMENT_MATRIX.md)
- [Metadata OSM](data/osm/manifest.json)
- `outputs/paper_summary/`: metrics, bootstrap CI, paired differences, error
  analysis, runtime và thông tin môi trường.
