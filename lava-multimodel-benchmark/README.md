# LAVA Multi-model Audio Deepfake Benchmark

Hệ thống train, đánh giá và so sánh nhiều kiến trúc phát hiện giọng nói deepfake trên cùng một giao thức dữ liệu. Project giữ nguyên model `MobileNetV3Small-LSTM` đã train và bổ sung pipeline cho năm đối chứng.

## 1. Các model

| Key dùng trong lệnh | Tên hiển thị | Đầu vào | Ghi chú |
|---|---|---|---|
| `mobilenetv3small_lstm` | MobileNetV3Small-LSTM | 6 Mel-spectrogram | Keras ImageNet + LSTM(128), baseline đã train |
| `efficientnetb0_lstm` | EfficientNet-B0-LSTM | 6 Mel-spectrogram | Keras ImageNet + LSTM(128) |
| `shufflenetv2_lstm` | ShuffleNetV2-LSTM | 6 Mel-spectrogram | LAVA TensorFlow ShuffleNetV2 1.0x |
| `mnasnet_lstm` | MnasNet-LSTM | 6 Mel-spectrogram | LAVA TensorFlow MnasNet-A1-style |
| `rawnet2_lava` | RawNet2-LAVA | Waveform 3 giây | Bản benchmark TensorFlow, không phải checkpoint chính thức |
| `aasist_lite` | AASIST-Lite | Waveform 3 giây | Bản lightweight TensorFlow, không phải checkpoint chính thức |

Không được mô tả `RawNet2-LAVA` và `AASIST-Lite` là tái lập chính thức của bài báo. Muốn so sánh với bản chính thức phải tích hợp repository/checkpoint của tác giả và giữ nguyên manifest test của project này.

## 2. Cấu trúc dự án

```text
lava-multimodel-benchmark/
├── app.py                         # App baseline cũ
├── benchmark_app.py               # App chọn model và xem bảng so sánh
├── config.py                      # Cấu hình âm thanh, dữ liệu, train
├── train.py                       # Baseline MobileNetV3 cũ
├── evaluate.py                    # Baseline evaluator cũ
├── requirements.txt
├── data/
│   ├── REAL/                      # File giọng thật
│   └── FAKE/                      # File giọng giả
├── models/                        # Model production baseline cũ
├── src/                           # Tiền xử lý và tiện ích baseline
└── benchmark/
    ├── protocol.py                # Đóng băng train/validation/test manifest
    ├── data.py                    # Dataset Mel/waveform dùng chung
    ├── model_registry.py          # Registry sáu model
    ├── import_baseline.py         # Nhập model MobileNet đã train
    ├── train_model.py             # Train một model
    ├── evaluate_model.py          # Test + latency + size + RTF
    ├── aggregate_results.py       # CSV/Markdown/biểu đồ cuối
    └── run_all.py                 # Chạy tuần tự nhiều model
```

Artifact được lưu riêng, không ghi đè lẫn nhau:

```text
outputs/experiments/<model_key>/seed_42/
├── model.keras
├── threshold.txt
├── metadata.json
├── history.json
├── training_history.png
├── metrics.json
└── test_scores.csv
```

## 3. Cài đặt trên Windows

Khuyến nghị Python 3.11 và TensorFlow 2.15.

```bat
cd D:\antt_new02\lava-multimodel-benchmark
py -3.11 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Kiểm tra:

```bat
python -c "import tensorflow as tf; print(tf.__version__); print(tf.config.list_physical_devices())"
```

## 4. Chuẩn bị dữ liệu

```text
data/REAL/*.wav|flac|mp3|ogg|m4a
data/FAKE/*.wav|flac|mp3|ogg|m4a
```

Tạo manifest một lần:

```bat
python -m benchmark.protocol
```

Kết quả:

```text
outputs/protocol/splits_seed42.csv
```

Không dùng `--force` sau khi đã bắt đầu benchmark. Nếu tạo lại manifest, mọi model cũ phải train/evaluate lại vì test set đã thay đổi.

## 5. Nhập MobileNetV3Small-LSTM đã train

Đặt model baseline tại:

```text
models/lava_mobilenetv3_lstm.keras
```

Nếu có `models/best_threshold.txt`, script sẽ dùng threshold đó. Nếu chưa có,
script tự hiệu chỉnh threshold trên validation split cố định.

Sau đó chạy:

```bat
python -m benchmark.import_baseline
python -m benchmark.evaluate_model --model mobilenetv3small_lstm --seed 42
```

Không cần train lại baseline.

Nếu model/threshold nằm chỗ khác:

```bat
python -m benchmark.import_baseline ^
  --model-source "D:\duong-dan\lava_mobilenetv3_lstm.keras"
```

## 6. Train từng model

Nên chạy từng model, đợi model trước hoàn tất rồi mới chạy model tiếp theo.

### EfficientNet-B0-LSTM

```bat
python -m benchmark.train_model --model efficientnetb0_lstm --seed 42
python -m benchmark.evaluate_model --model efficientnetb0_lstm --seed 42
```

### ShuffleNetV2-LSTM

```bat
python -m benchmark.train_model --model shufflenetv2_lstm --seed 42
python -m benchmark.evaluate_model --model shufflenetv2_lstm --seed 42
```

### MnasNet-LSTM

```bat
python -m benchmark.train_model --model mnasnet_lstm --seed 42
python -m benchmark.evaluate_model --model mnasnet_lstm --seed 42
```

### RawNet2-LAVA

```bat
python -m benchmark.train_model --model rawnet2_lava --seed 42 --warmup-epochs 0 --finetune-epochs 50
python -m benchmark.evaluate_model --model rawnet2_lava --seed 42
```

### AASIST-Lite

```bat
python -m benchmark.train_model --model aasist_lite --seed 42 --warmup-epochs 0 --finetune-epochs 50
python -m benchmark.evaluate_model --model aasist_lite --seed 42
```

Mặc định model ảnh chạy 50 epoch warm-up + tối đa 50 epoch fine-tuning với Early Stopping. Để smoke test trước:

```bat
python -m benchmark.train_model --model efficientnetb0_lstm --warmup-epochs 1 --finetune-epochs 1
```

Smoke test chỉ kiểm tra code, không dùng số liệu đó trong báo cáo.

## 7. Chạy nhiều model tuần tự

Ví dụ chạy ba model ảnh còn lại:

```bat
python -m benchmark.run_all --models efficientnetb0_lstm shufflenetv2_lstm mnasnet_lstm
```

Không nên chạy nhiều terminal train song song trên laptop vì RAM, CPU/GPU và nhiệt độ sẽ cạnh tranh, làm sai phép đo thời gian.

## 8. Hoàn thiện bảng

Sau mỗi lần evaluate, chạy:

```bat
python -m benchmark.aggregate_results --seed 42
```

Kết quả:

```text
outputs/comparison/model_comparison_seed42.csv
outputs/comparison/model_comparison_seed42.md
outputs/comparison/model_comparison_seed42.png
```

Dòng chưa train vẫn hiện `…`; dòng có `metrics.json` sẽ tự điền Accuracy, F1 lớp FAKE, Macro-F1, ROC-AUC, EER, số tham số, dung lượng model, latency và Real-Time Factor.

Chỉ so sánh các dòng có cùng `manifest_sha256`. Script sẽ từ chối tổng hợp nếu phát hiện model dùng test manifest khác nhau.

## 9. Chạy giao diện

```bat
streamlit run benchmark_app.py
```

App cho phép chọn model đã train, tải file âm thanh, xem `P(FAKE)` và bảng benchmark.

## 10. Quy tắc báo cáo khoa học

1. Threshold được chọn trên validation, tuyệt đối không tối ưu trên test.
2. Tất cả model dùng cùng manifest file.
3. Accuracy không đủ khi dữ liệu mất cân bằng; phải báo cáo Macro-F1 và confusion matrix.
4. EER càng thấp càng tốt; Accuracy/F1/AUC càng cao càng tốt.
5. RTF dưới 1 nghĩa là xử lý nhanh hơn thời lượng âm thanh trên đúng máy đã đo.
6. Không trộn số liệu công bố từ bài báo với số liệu tự đo trong cùng bảng mà không ghi rõ nguồn và protocol.
7. Nếu có speaker/source ID, nên nâng cấp manifest thành group-disjoint split trước khi đưa ra tuyên bố tổng quát hóa.

## 11. Xử lý lỗi thường gặp

`Missing model/threshold`: hãy train model hoặc chạy `benchmark.import_baseline`.
Script import sẽ tự hiệu chỉnh threshold nếu baseline chưa có `best_threshold.txt`;
không tự tạo `0.5` cho kết quả báo cáo.

Hết RAM: giảm batch size:

```bat
python -m benchmark.train_model --model efficientnetb0_lstm --batch-size 4
```

Không tải được ImageNet weights: kiểm tra Internet lần đầu. ShuffleNetV2, MnasNet, RawNet2-LAVA và AASIST-Lite không cần tải pretrained weights.
