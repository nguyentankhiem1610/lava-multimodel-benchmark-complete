# Full Audit Report (LEGACY BASELINE SNAPSHOT)

> Tài liệu này mô tả trạng thái baseline trước khi pipeline multi-model được
> bổ sung. Trạng thái và lệnh hiện hành nằm trong `README.md`; kết quả mới nằm
> trong `outputs/experiments` và `outputs/comparison`.

Ngày audit: 2026-08-28  
Implementation root: `D:\audio-deepfake-mobilenet-lstm`  
Specification source: `prompt.docx`

## Kết luận điều hành

Implementation source code sau sửa đã đúng pipeline TensorFlow/Keras `MobileNetV3Small + LSTM`, giữ chronology của 6 Mel segments và thống nhất `REAL=0`, `FAKE=1`, output=`P(FAKE)`. Ba repo reference không bị chỉnh sửa và không tham gia runtime.

Project source hiện dùng contract “một detector, một production model”: `models/lava_mobilenetv3_lstm.keras`. Warm-up và fine-tuning là hai stage nội bộ của một lệnh `python train.py`; một checkpoint validation xuyên suốt lifecycle chọn global-best weights. Checkpoint `.keras` tốt nhất của lần train đã audit được save/load-verify sang contract mới và threshold được calibrate lại trên validation (`0.38`). Các model/plot stage cũ đã được chuyển vào `outputs/legacy_models/` và `outputs/legacy_training/`, không còn là runtime dependency.

## Workspace map

| Nhóm | Thành phần | Quyền xử lý |
|---|---|---|
| IMPLEMENTATION | `app.py`, `config.py`, `train.py`, `evaluate.py`, `predict.py`, `src/`, `requirements.txt` | Đã audit và sửa |
| DATA / ARTIFACT | `data/`, `models/`, `outputs/` | Dùng cho verification; không coi data/model trong reference là production |
| READ-ONLY REFERENCE | `deepfake-audio-detection/` | Chỉ đọc/đối chiếu |
| READ-ONLY REFERENCE | `enhancing-deepfake-detection-using-mobilenet-lstm-hybrid-model-main/` | Chỉ đọc/đối chiếu |
| READ-ONLY REFERENCE | `mobilenetv3.pytorch/` | Chỉ đọc/đối chiếu |

## Requirement matrix

| Requirement | Status | Evidence | Implementation file | Reference used |
|---|---|---|---|---|
| Chỉ audit/sửa implementation root | PASS | Recursive scan được tách theo ba nhóm; không có edit trong reference | Toàn root | Cả ba repo |
| TensorFlow/Keras, không có PyTorch runtime | PASS | Không có `torch`, `torchvision`, `.pth` hoặc CUDA hard-code trong implementation | `requirements.txt`, root scripts, `src/` | `mobilenetv3.pytorch` chỉ để đối chiếu |
| MobileNetV3Small pretrained ImageNet | PASS | `MobileNetV3Small(include_top=False, weights="imagenet", pooling="avg")` | `src/model.py` | MobileNetV3 PyTorch + Keras Applications |
| TimeDistributed bọc backbone | PASS | Layer `time_distributed_mobilenetv3small`; tensor `(B,6,576)` | `src/model.py` | Hybrid reference, đã sửa semantics |
| LSTM nhận sequence embedding thật | PASS | LSTM input giữ 6 timestep, output `(B,128)` | `src/model.py` | Hybrid notebook chỉ có sequence length 1 nên không reuse trực tiếp |
| Sigmoid output `(B,1)` | PASS | Layer `probability_fake`, activation sigmoid | `src/model.py` | Cả hai reference ML |
| REAL=0, FAKE=1, output=P(FAKE) | FIXED | Một contract dùng chung cho dataset, metrics, CLI và UI | `config.py`, `src/inference.py` | Audio reference |
| Load/resample/normalize duration | FIXED | Đọc giới hạn 3 giây, mono, resample 22,050 Hz, pad/truncate | `src/preprocessing.py` | `deepfake-audio-detection/train_model_v2.py` |
| Temporal segmentation chronology | PASS | Reshape tuần tự thành 6 segment, không shuffle nội bộ | `src/preprocessing.py` | Hybrid temporal idea |
| Mel -> dB -> resize -> RGB | FIXED | STFT + Mel filter bank + dB top-80 + 224x224 RGB | `src/preprocessing.py` | Audio preprocessing reference |
| MobileNetV3 input scale | FIXED | Tensor trả `float32` trong `0..255`, phù hợp internal Rescaling | `src/preprocessing.py`, `src/model.py` | Keras MobileNetV3 behavior |
| REAL và FAKE preprocessing | PASS | Cả hai smoke test trả `(6,224,224,3)`, finite float32 | `src/preprocessing.py` | Root data |
| Chỉ scan root dataset | PASS | Scanner chỉ nhận `config.REAL_DIR` và `config.FAKE_DIR` | `src/dataset.py` | Không dùng reference data |
| Split files trước augmentation | PASS | Original path split trước; augmentation nằm trong training dataset | `src/dataset.py` | Audio reference được sửa leakage order |
| Train/val/test độc lập | FIXED | Stratified 70/15/15, seed 42, disjoint; hiện là 44/10/10 | `src/dataset.py`, `config.py` | sklearn pattern từ reference |
| tf.data shuffle/batch/prefetch | FIXED | `_PrefetchDataset`, shuffle train-only, batch, AUTOTUNE prefetch | `src/dataset.py` | Không reuse generator cũ |
| Validation/test không augmentation | PASS | `training=False` cho cả val/test | `train.py`, `evaluate.py` | Specification |
| Class weight từ train-only | PASS | `get_class_weights(train_data[1])` | `train.py`, `src/dataset.py` | Audio reference idea |
| Warm-up backbone freeze | PASS | Backbone trainable=False; 369,281 trainable params trong smoke test | `src/model.py`, `train.py` | Transfer-learning pattern |
| Internal fine-tuning transition | PASS | Last 20 layers considered, compile lại với LR `1e-5` | `src/model.py`, `train.py` | Hybrid notebook fine-tune idea |
| BatchNormalization handling | PASS | 0 BN layer trainable trong fine-tuning | `src/model.py` | Keras fine-tuning practice |
| Checkpoint contract | PASS | Một global `val_loss` checkpoint xuyên suốt lifecycle; chỉ xuất một final `.keras` | `src/metrics.py`, `train.py` | Callback ideas từ references |
| Threshold tune trên validation | PASS | Search F1 từ validation raw probabilities và persist atomically | `train.py`, `src/metrics.py` | Root pre-audit implementation |
| Test không tham gia tuning | PASS | Evaluate chỉ load persisted threshold | `evaluate.py` | Specification |
| Metrics đầy đủ và ROC-AUC raw | PASS | Accuracy/Precision/Recall/F1/Macro-F1/AUC/EER/confusion/report | `evaluate.py` | Audio evaluation reference, mở rộng |
| Predict CLI contract | FIXED | Có prediction, confidence, raw P(FAKE), threshold | `predict.py`, `src/inference.py` | Audio reference CLI |
| Streamlit contract | FIXED | Upload, playback, waveform, Mel, result, confidence, probability, threshold | `app.py` | Audio reference Streamlit |
| Không bịa per-segment prediction | PASS | UI chỉ mô tả segment chronology, không đưa class từng segment | `app.py` | Specification |
| Không dùng Physics Layer | PASS | Không có physics/ensemble path trong implementation | Root implementation | Audio reference chỉ đọc |
| Requirements production | FIXED | Pin TensorFlow/NumPy/librosa/OpenCV headless/matplotlib/Streamlit/sklearn/soundfile/setuptools<81 | `requirements.txt` | Reference requirements chỉ đối chiếu |
| Matplotlib non-interactive | PASS | `Agg` được set trước `pyplot` | `src/utils.py`, `app.py` | Specification |
| CPU/GPU behavior | PASS | Không CUDA hard-code; TensorFlow tự phát hiện GPU, CPU fallback | `train.py` | Specification |
| Config centralization | FIXED | Paths, audio, image, split, seed, lifecycle stages, threshold đều tập trung | `config.py` | Specification |
| Reproducibility | FIXED | Python, NumPy, TensorFlow và sklearn cùng seed 42 | `train.py`, `src/dataset.py`, `config.py` | Specification |
| Production artifact contract | PASS | Một final model load-verified + validation threshold; không legacy fallback | `models/`, `src/artifacts.py` | Single-detector contract |

## Verification đã chạy

| Test | Kết quả |
|---|---|
| Compile/import toàn bộ root modules | PASS |
| Dataset scan | 8 REAL, 56 FAKE; không scan reference |
| Deterministic stratified split | 44 train, 10 validation, 10 test; disjoint |
| REAL preprocessing | `(6,224,224,3)`, float32, finite, trong `0..255` |
| FAKE preprocessing | `(6,224,224,3)`, float32, finite, trong `0..255` |
| tf.data batch | `(2,6,224,224,3)` + `(2,)`; `_PrefetchDataset` |
| ImageNet model build | PASS; backbone output 576 |
| Tensor trace | `(B,6,224,224,3) -> (B,6,576) -> (B,128) -> (B,1)` |
| Lifecycle transition | Trainable params `369,281 -> 646,481`; BN trainable `0` |
| Forward pass | PASS trên CPU |
| Tiny training step | PASS |
| `.keras` save/load/forward | PASS; temporary checkpoint đã được xóa |
| Threshold loader | PASS; đọc validation-calibrated threshold `0.38` |
| Predict REAL | PASS với production model; output đủ Prediction/Confidence/P(FAKE)/Threshold |
| Predict FAKE | PASS với production model; output đủ Prediction/Confidence/P(FAKE)/Threshold |
| Evaluate smoke | Chạy thành công; legacy AUC `0.5000`, TN=0, FP=2, FN=0, TP=8 |
| Streamlit import/model load | PASS |
| Streamlit server startup | PASS tại port 8501; tiến trình test đã dừng |

Không chạy full training theo đúng scope smoke test. Tiny training chỉ dùng temporary directory và không thay production weights; production artifact hiện tại được migrate từ checkpoint `.keras` đã train, sau khi architecture/save/load và validation calibration đều thành công.

## Source provenance report

| Final component | Origin / inspiration | Reference file | Root implementation file | Adaptation performed |
|---|---|---|---|---|
| Audio loading và duration normalization | `deepfake-audio-detection` | `train_model_v2.py`, `test_model.py` | `src/preprocessing.py` | Giữ 22,050 Hz/3 giây; đổi sang bounded SoundFile read + polyphase resample để không đọc toàn bộ file 10 phút |
| Mel voice-band features | `deepfake-audio-detection` | `train_model_v2.py` | `src/preprocessing.py` | Giữ 128 Mel, 20-8,000 Hz, FFT 2048, hop 512; tách thành 6 temporal images và scale đúng MobileNetV3 |
| Training-only augmentation | `deepfake-audio-detection` | `train_model_v2.py` | `src/augmentation.py`, `src/dataset.py` | Shift/pitch/noise/volume, chỉ sau file split và chỉ trên train |
| REAL/FAKE labels | `deepfake-audio-detection` | `train_model_v2.py` | `config.py`, `src/dataset.py`, `src/inference.py` | Chuẩn hóa thành contract duy nhất REAL=0/FAKE=1/P(FAKE) |
| Callbacks và plots | `deepfake-audio-detection` + hybrid reference | `train_model_v2.py`, notebook cells 6-9 | `src/metrics.py`, `src/utils.py` | Một global-best checkpoint và một lifecycle history plot |
| Streamlit audio UX | `deepfake-audio-detection` | `app.py` | `app.py` | Reuse ý tưởng upload/playback/plots; bắt buộc dùng shared preprocessing/inference và threshold |
| Spatial-temporal concept | Hybrid MobileNet-LSTM reference | `deepfake-det.ipynb` cell 4 | `src/model.py` | Thay sequence length 1 của reference bằng 6 Mel segments thật và TimeDistributed backbone |
| Warm-up-then-fine-tune lifecycle | Hybrid MobileNet-LSTM reference | `deepfake-det.ipynb` cell 13 | `src/model.py`, `train.py` | Một run; partial unfreeze, compile lại, LR thấp, freeze toàn bộ BN |
| MobileNetV3 architecture | `mobilenetv3.pytorch` + Keras Applications | `mobilenetv3.py` | `src/model.py` | Đối chiếu MobileNetV3Small; thay toàn bộ PyTorch/.pth bằng `tf.keras.applications.MobileNetV3Small(weights="imagenet")` |

Không có source code PyTorch nào được copy vào runtime. CNN cũ, MFCC, rolloff, ZCR, image/video pipeline và Physics Layer chỉ được tham khảo hoặc chủ động không reuse vì không thuộc baseline cuối.

## Sai/thiếu ban đầu và file đã sửa

Các lỗi chính ban đầu:

- Mel image bị chia về `0..1` trong khi MobileNetV3Small mặc định đã có internal preprocessing cho input `0..255`.
- Dùng `keras.utils.Sequence`, không phải `tf.data.Dataset`; validation/test generator vẫn shuffle index.
- Split ratio và seed hard-code; config thiếu split/model paths/default threshold/fine-tune settings.
- Fine-tune làm 2 BatchNormalization layer trainable.
- Checkpoint `.h5` legacy và model-path logic bị duplicate ở bốn entry point.
- Predict thiếu threshold trong output; UI thiếu threshold và có mojibake icon.
- Audio WAV rất dài bị loader cũ xử lý chậm bất thường; bounded read chưa được đảm bảo.
- Requirements chưa pin Streamlit/OpenCV/setuptools và không bảo vệ NumPy/TensorFlow compatibility đầy đủ.
- Có `fix.py` với absolute path và mutation logic stale; file này đã được loại bỏ.

File implementation đã sửa/tạo:

```text
app.py
config.py
train.py
evaluate.py
predict.py
requirements.txt
src\augmentation.py
src\dataset.py
src\inference.py
src\metrics.py
src\model.py
src\preprocessing.py
src\utils.py
README.md
AUDIT_REPORT.md
```

## Technical debt còn lại

1. Dataset rất nhỏ và lệch lớp: 8 REAL, 56 FAKE. Metric hiện tại không đủ để kết luận chất lượng production.
2. Tên file cho thấy cùng speaker identity có thể xuất hiện ở nhiều split. Specification yêu cầu file-level split và implementation đáp ứng, nhưng nghiên cứu nghiêm ngặt nên bổ sung speaker/group split để đo generalization.
3. `tf.data` dùng `tf.py_function` để gọi decoder/STFT Python. Semantics đúng nhưng graph portability và throughput chưa tối ưu; có thể chuyển sang TensorFlow I/O/native ops sau.
4. Augmentation deterministic theo file giúp reproducibility nhưng cùng một file nhận cùng transform giữa các epoch. Có thể thêm stateless epoch-aware seed nếu cần độ đa dạng cao hơn.
5. Môi trường Python toàn cục tại thời điểm audit bị trộn TensorFlow meta-package 2.15 với `tensorflow-intel` 2.13 và có xung đột `typing-extensions`. Không sửa global environment; bắt buộc dùng venv mới theo README.
6. Chưa có model `.keras` mới vì full retraining không nằm trong dry-run. Đây là blocker duy nhất đối với chất lượng inference thực tế.

## Readiness cuối

| Capability | Readiness | Ghi chú |
|---|---|---|
| TRAIN | YES | Source/pipeline sẵn sàng; dùng venv sạch |
| EVALUATE | CONDITIONAL | Command chạy được; metric chỉ có ý nghĩa sau retrain |
| PREDICT | CONDITIONAL | CLI chạy được; legacy artifact chỉ để smoke-test |
| STREAMLIT | CONDITIONAL | Server/UI chạy được; cần `.keras` mới để kết quả đáng tin |

Commands chính xác nằm trong `README.md`.
