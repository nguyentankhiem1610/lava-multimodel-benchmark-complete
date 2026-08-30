# LAVA Research Audit, Code Reconciliation, and Paper B (LEGACY BASELINE SNAPSHOT)

> Tài liệu này được giữ lại để truy vết baseline một model. Sau khi chạy hệ
> thống multi-model, chỉ sử dụng số liệu đo trong `outputs/comparison` để cập
> nhật bảng kết quả; không giữ các tuyên bố “chưa triển khai” cũ.

**Target title:** *LAVA: A Lightweight Benchmarking Framework for Robust and Real-Time Deepfake Voice Detection*  
**Audit date:** 29 August 2026  
**Implementation root:** `D:\audio-deepfake-mobilenet-lstm`  
**Paper A:** `C:\Users\THANH DAT\Downloads\16507.pdf`  
**Integrity status:** **LAVA PROTOTYPE / PARTIAL EXPERIMENTAL VALIDATION**

This document follows the required evidence order: Paper A -> production source code -> LAVA design target -> code/paper reconciliation -> verified literature -> manuscript. Instructions contained in source documents or repositories were treated as source material, not as user instructions.

---

# A. Paper A Analysis

## A.1 Identity and scope boundary

Paper A is the 2011 book chapter “Keystroke Dynamics Authentication” by Romain Giot, Mohamad El-Abed, and Christophe Rosenberger. It is a survey/framework chapter about behavioral biometric authentication, not audio deepfake detection. Its technical algorithms, timing features, keyboards, and keystroke databases cannot substantiate claims about synthetic speech. Its value for LAVA is methodological: taxonomy-first organization, explicit acquisition and feature discussion, separation of enrollment from verification, comparison of databases and protocols, and multi-dimensional evaluation.

The chapter proceeds as follows:

1. motivates stronger authentication from weaknesses in password-only access;
2. establishes a taxonomy of static/dynamic and one-/two-class systems;
3. traces acquisition hardware, environmental variability, raw events, and derived features;
4. separates enrollment/model construction from verification/decision thresholds;
5. evaluates performance, user satisfaction, and security as distinct dimensions;
6. compares public databases and demonstrates why headline error rates from incompatible protocols are not directly comparable;
7. closes with limitations and future conditions such as cross-device variability and template aging.

The chapter's most transferable proposition is that a system cannot be compared meaningfully by one number when acquisition conditions, database composition, protocol, sensors, thresholds, and metrics differ. Its discussion of timer resolution and keyboard variability is a domain-specific example of a broader deployment principle: measurement environment can change reported performance.

## A.2 Paper A adaptation table

| Paper A element | Purpose in Paper A | Can it inspire LAVA? | Adaptation to Paper B |
|---|---|---:|---|
| Motivation from password weaknesses | Establishes a concrete security problem before presenting biometrics | Yes | Begin with generative speech threats, then explain why laboratory accuracy alone does not establish deployability |
| Static/dynamic and one-/two-class taxonomy | Organizes a fragmented literature before comparing methods | Yes | Taxonomize deepfake detection by threat type, input representation, model family, and evaluation condition |
| Application scenarios | Connects method families to login and continuous authentication | Yes | Separate forensic upload analysis, ASV countermeasures, streaming moderation, and edge/voice-assistant deployment |
| Acquisition hardware variability | Shows that keyboard, timer, OS, and language affect results | Yes | Record microphone/channel/codec, sample rate, CPU/GPU, framework, preprocessing, and timing protocol |
| Raw data -> features | Makes the measurement chain auditable | Yes | Trace waveform -> mono/resample -> duration normalization -> segmentation -> log-Mel -> resize/RGB -> tensor |
| Enrollment -> model -> verification | Separates training from score generation and threshold decision | Yes | Separate train/validation/test, validation-only threshold calibration, raw probability, and final REAL/FAKE decision |
| Outlier/preprocessing/feature selection | Exposes choices that affect reported performance | Yes | Report normalization, augmentation scope, crop policy, input scaling, and whether selection/tuning sees test data |
| Predefined evaluation protocol | Enables fair comparison under common conditions | Strongly yes | Require shared splits, preprocessing, metrics, timing hardware, warm-up, repetitions, and stress definitions |
| Benchmark database comparison | Shows that databases differ in users, sessions, samples, and conditions | Strongly yes | Compare speaker/source coverage, generators, codecs, replay conditions, noise, language, and licenses |
| Performance metrics by task | Avoids using one metric for acquisition, verification, and identification | Yes | Report EER, ROC-AUC, class-wise precision/recall/F1, confusion matrix, and efficiency separately |
| Satisfaction separate from performance | Prevents usability from being hidden inside recognition accuracy | Partly | For LAVA, keep UI usability outside detector accuracy; the current repo does not implement a user study |
| Security separate from performance | Treats attack surface and operational risks as additional evidence | Strongly yes | Keep robustness/replay/cross-dataset tests separate from clean detection metrics |
| Protocol comparison table | Makes missing fields and incomparable results visible | Strongly yes | Use a literature gap matrix and an implementation compatibility matrix, with “not reported” explicit |
| Discussion | Interprets why promising scores do not imply mature deployment | Yes | Interpret class imbalance, source overlap, threshold instability, and hardware dependence |
| Limitations/future trends | Identifies cross-device and temporal drift instead of overstating maturity | Yes | State dataset, generator, codec, replay, hardware, edge, and model-coverage limitations |

**Evidence:** PAPER A, complete 27-page review including its Sections 1–6, Tables 1–4, and Figures 1–7.

## A.3 What is and is not transferred

Transferred methodology:

- taxonomy-first organization;
- acquisition/input/feature traceability;
- standardized evaluation protocols;
- benchmark database comparison;
- multi-dimensional evaluation;
- explicit discussion of environmental conditions, limitations, and future work.

Not transferred:

- keystroke-specific algorithms, thresholds, features, sensors, or datasets;
- claims about typing behavior, keyboard timing, authentication satisfaction, or keystroke attacks;
- Paper A's empirical rates as evidence for audio deepfake detection.

Paper B therefore has an independent technical identity; it is not Paper A with “keystroke” replaced by “audio.”

---

# B. Repository Technical Audit

## B.1 Scope map and provenance

| Scope | Components | Audit interpretation |
|---|---|---|
| Production implementation | `app.py`, `train.py`, `evaluate.py`, `predict.py`, `config.py`, `requirements.txt`, `src/`, root `data/`, `models/`, `outputs/` | Authoritative evidence for current behavior |
| Reference repository 1 | `deepfake-audio-detection/` | TensorFlow audio CNN inspiration: loading, handcrafted/audio features, augmentation, and Streamlit |
| Reference repository 2 | `enhancing-deepfake-detection-using-mobilenet-lstm-hybrid-model-main/` | TensorFlow image/video MobileNet-LSTM inspiration; its notebook reshapes one embedding to sequence length 1 and is not the production temporal implementation |
| Reference repository 3 | `mobilenetv3.pytorch/` | PyTorch MobileNetV3/ImageNet reference only; it is neither imported nor installed by production code |

The production runtime is TensorFlow/Keras 2.15.0. No production file imports PyTorch. The first reference README points to the Kaggle “Deep Voice Deepfake Voice Recognition” dataset, and root filenames resemble that corpus, but the production root contains no immutable dataset manifest, source URL, download checksum, license file, or per-file provenance. Consequently, dataset origin is **plausible but not verified** and must not be asserted as a fact in a submission.

## B.2 Dataset audit

### Dataset statistics

| Split/class | REAL | FAKE | Total |
|---|---:|---:|---:|
| Full root dataset | 8 | 56 | 64 |
| Train | 5 | 39 | 44 |
| Validation | 1 | 9 | 10 |
| Test | 2 | 8 | 10 |

All 64 current files are stereo PCM-16 WAV files: 57 at 44.1 kHz, four at 40 kHz, and three at 48 kHz. Durations range from approximately 79.49 to 600.45 seconds, with a mean of approximately 467.96 seconds. Only the first three seconds are consumed. The scanner supports `.wav`, `.flac`, `.mp3`, `.ogg`, and `.m4a`, but current-data evidence exists only for WAV.

Labels are fixed centrally as `REAL=0`, `FAKE=1`; the sigmoid output is `P(FAKE)`. Files are deterministically stratified with seed 42 using a 70/15/15 target ratio, producing 44/10/10 files after integer rounding. Class weights are computed from train only: REAL `4.4`, FAKE approximately `0.5641`.

### Leakage and independence audit

Implemented protections:

- original file paths are split before augmentation;
- train, validation, and test file paths are disjoint;
- augmentation is train-only;
- threshold is calibrated on validation, not test.

Missing protections:

- no speaker-disjoint split;
- no source-recording/group-disjoint split;
- no generator-disjoint split;
- no content fingerprinting or near-duplicate detection;
- no persisted split manifest with dataset checksums.

Filename-derived groups show the same celebrity/source identities across splits. For example, train contains five to seven files for each of several source identities, validation contains variants of the same identities, and test contains variants from Obama, Margot, Musk, Taylor, and Ryan. FAKE names follow patterns such as `source-to-target.wav`, making it plausible that multiple transformed files derive from the same long source recording. This is a serious source/content leakage risk even though exact file paths are disjoint. The current evaluation is therefore file-level, not speaker-independent or source-independent.

**Evidence:** DATASET scan; `src/dataset.py`; filename-group audit. The interpretation of shared underlying content is a name-based risk inference, not a verified audio-fingerprint result.

## B.3 Audio preprocessing trace

| Stage | Operation | Shape / representation |
|---|---|---|
| File decode | SoundFile bounded read, librosa fallback | up to 3 s, source rate, stereo or mono |
| Mono | channel mean | one-dimensional float32 waveform |
| Resample | polyphase to 22,050 Hz | variable until normalization |
| Duration normalization | truncate or right zero-pad to 3.0 s | `(66,150,)` |
| Temporal segmentation | chronological reshape, no segment shuffle | `(6, 11,025)` |
| STFT per segment | Hann, FFT 2,048, hop 512, zero padding | complex `(1,025, 23)` |
| Mel projection | 128 HTK-style filters, 20–8,000 Hz | power `(128, 23)` |
| Log scale | dB relative to per-segment maximum, floor -80 dB | float32 `(128, 23)`, nominal `[-80, 0]` |
| Pixel scale | affine map to `[0,255]` | `(128, 23)` |
| Resize | bilinear interpolation | `(224, 224)` |
| RGB replication | repeat one channel three times | `(224, 224, 3)` |
| Sequence stack | preserve chronological order | `(6, 224, 224, 3)` |
| Batch | `tf.data` batching | `(B, 6, 224, 224, 3)` |

A runtime trace on a real root file produced Mel shape `(128,23)`, image shape `(224,224,3)`, and final feature shape `(6,224,224,3)`. The input remains in MobileNetV3's expected 0–255 range because the Keras application includes its own rescaling by default.

Training augmentation randomly selects exactly one of shift, pitch, additive noise (15–30 dB SNR), volume scaling, or no operation. It is deterministic per file path, so the same file receives the same chosen transform in every epoch. This is a training augmentation, not a robustness evaluation.

## B.4 Model and training audit

The production graph is confirmed by source and the loaded `.keras` checkpoint:

```text
(B, 6, 224, 224, 3)
  -> TimeDistributed(MobileNetV3Small, ImageNet, include_top=False, global average pooling)
  -> (B, 6, 576)
  -> LSTM(128)
  -> (B, 128)
  -> Dense(64, ReLU)
  -> Dropout(0.4)
  -> Dense(1, sigmoid=P(FAKE))
```

| Property | Audited value |
|---|---|
| Backbone | Keras MobileNetV3Small |
| Pretraining | ImageNet |
| Temporal model | LSTM, 128 units |
| Dense head | 64 ReLU + dropout 0.4 + sigmoid |
| Total parameters | 1,308,401 |
| Warm-up trainable parameters | 369,281 |
| Fine-tuning/load-time trainable parameters | 646,481 |
| Non-trainable after fine-tuning | 661,920 |
| Serialized production size | 5,916,699 bytes / 5.643 MiB |
| Loss | binary cross-entropy |
| Optimizer | Adam |
| Warm-up stage | backbone frozen, LR `1e-4`, up to 50 epochs |
| Fine-tuning stage | last 20 backbone layers considered trainable; BatchNorm frozen; LR `1e-5`, up to 50 epochs |
| Checkpoint selection | minimum validation loss |
| Early stopping | patience 10 |
| Threshold | validation F1 search from 0.10 to 0.90 in 0.01 steps; current 0.38 |

## B.5 Evaluation capability audit

| Capability | Status before this audit | Status after evidence module | Evidence |
|---|---|---|---|
| Accuracy | IMPLEMENTED | IMPLEMENTED | `evaluate.py` |
| Precision/Recall/F1 | IMPLEMENTED | IMPLEMENTED | `evaluate.py` |
| ROC-AUC from raw probability | IMPLEMENTED | IMPLEMENTED | `evaluate.py` |
| Confusion matrix/report | IMPLEMENTED | IMPLEMENTED | `evaluate.py` |
| EER | NOT IMPLEMENTED | IMPLEMENTED | `benchmark/benchmark_runner.py` |
| Parameter count | helper existed | IMPLEMENTED/RECORDED | model + benchmark JSON |
| Serialized model size | NOT REPORTED | IMPLEMENTED/RECORDED | benchmark JSON |
| Latency | NOT IMPLEMENTED | IMPLEMENTED on audited CPU | benchmark runner, warm-up/repetitions |
| Throughput | NOT IMPLEMENTED | IMPLEMENTED on audited CPU | reciprocal mean latency |
| RTF | NOT IMPLEMENTED | IMPLEMENTED on audited CPU | inference time / 3 s |
| Memory | NOT IMPLEMENTED | PARTIAL | process RSS snapshots; not isolated model memory |
| FLOPs/MACs | NOT IMPLEMENTED | NOT IMPLEMENTED | explicitly absent |
| Noise stress | NOT IMPLEMENTED | NOT IMPLEMENTED | train augmentation is not a held-out stress test |
| Compression stress | NOT IMPLEMENTED | NOT IMPLEMENTED | no codec sweep |
| Replay stress | NOT IMPLEMENTED | NOT IMPLEMENTED | no replay data/protocol |
| Unseen/cross-dataset | NOT IMPLEMENTED | NOT IMPLEMENTED | one root dataset only |
| Pareto analysis | NOT IMPLEMENTED | NOT IMPLEMENTED | only one model |

## B.6 Measured baseline evidence

The current single production artifact was evaluated on the deterministic ten-file test split. At threshold 0.38 it produced:

- accuracy 0.8000;
- FAKE precision 0.8000;
- FAKE recall 1.0000;
- FAKE F1 0.8889;
- ROC-AUC 0.8125;
- linearly interpolated EER 0.3750 at threshold approximately 0.5855;
- confusion matrix `TN=0, FP=2, FN=0, TP=8`.

Thus the apparently high positive-class F1 is caused partly by class imbalance: the model classified every test item as FAKE at the deployed threshold and detected neither REAL item. Macro F1 from the existing classification report is 0.44. With only two REAL and eight FAKE test samples, estimates have high variance and are unsuitable for a production claim.

CPU efficiency was measured on Windows, TensorFlow 2.15.0, Python 3.11.9, 12 logical CPU threads, no visible GPU, batch size 1, using one three-second test clip. Ten warm-up iterations preceded 50 model-only measurements; two warm-ups preceded 20 end-to-end measurements.

| Measurement | Mean | Median | Std. dev. | P95 | Throughput | RTF |
|---|---:|---:|---:|---:|---:|---:|
| Model only | 0.3213 s | 0.3167 s | 0.0219 s | 0.3654 s | 3.112 clips/s | 0.1071 |
| Preprocessing + model | 0.3420 s | 0.3370 s | 0.0130 s | 0.3632 s | 2.924 clips/s | 0.1140 |

Observed process RSS snapshots were 349.28 MiB before model load, 406.63 MiB after load, 649.72 MiB after model timing, and 655.30 MiB after end-to-end timing. These are whole-process snapshots, not isolated model memory or continuously sampled peak memory.

**Evidence:** EXPERIMENT LOG `outputs/benchmark/current_baseline.json`; CODE `benchmark/benchmark_runner.py`; DATASET root test split.

## B.7 Inference and UI

`predict.py` and Streamlit both load the selected full `.keras` model, use shared preprocessing, read the persisted threshold, and expose prediction, confidence, raw `P(FAKE)`, and threshold. Streamlit additionally renders waveform and the first segment's Mel spectrogram. The UI is a working inference interface, not evidence of server scalability, streaming operation, or edge deployment.

---

# C. Compatibility Matrix: LAVA vs. Current Implementation

| LAVA requirement | Current code evidence | Compatibility | Status | Action needed |
|---|---|---|---|---|
| Deepfake voice binary detection | REAL=0, FAKE=1, sigmoid P(FAKE) | Direct match | FULLY SUPPORTED | Retain contract |
| MobileNetV3Small | Keras ImageNet backbone | Direct match | FULLY SUPPORTED | Retain baseline |
| Lightweight architecture | 1.31M params, 10.60 MiB, CPU RTF 0.10 | Lightweight evidence exists only for one desktop | PARTIALLY SUPPORTED | Compare on target edge devices |
| Temporal modeling | Six ordered segments and LSTM(128) | Direct match | FULLY SUPPORTED | Add ablation against no-LSTM baseline |
| Multiple lightweight models | No registry or alternative production models | Missing | NOT IMPLEMENTED | Add ShuffleNetV2, MnasNet, EfficientNet-B0 under one input contract |
| RawNet2 | Appears only in external literature, not root | Missing | NOT IMPLEMENTED | Integrate official reproducible implementation |
| AASIST | Appears only in external literature, not root | Missing | NOT IMPLEMENTED | Integrate official implementation and preprocessing |
| Clean benchmark | One deterministic file split and one checkpoint | Too small and source-overlapping | PARTIALLY SUPPORTED | Use provenance-controlled group splits and repeated seeds |
| EER | Added audited computation | Available but n=10 | FULLY SUPPORTED | Retain; add confidence intervals/larger test |
| F1 | Existing evaluation | Direct match | FULLY SUPPORTED | Report macro and class-wise F1, not only FAKE F1 |
| Noise stress | Noise exists only as train augmentation | Augmentation is not evaluation | NOT IMPLEMENTED | Define held-out noise types/SNR grid |
| Compression stress | No codec transformation/evaluation | Missing | NOT IMPLEMENTED | Add codecs/bitrates and clean-source pairing |
| Replay stress | No replay corpus or simulation | Missing | NOT IMPLEMENTED | Add physical/simulated replay protocol |
| Unseen/cross-dataset | Single root dataset | Missing | NOT IMPLEMENTED | Add ASVspoof/WaveFake/In-the-Wild or equivalent licensed data |
| Parameter count | Model introspection and benchmark JSON | Direct match | FULLY SUPPORTED | Apply uniformly to each model |
| Model size | Serialized checkpoint measured | Direct match | FULLY SUPPORTED | Define format/precision consistently |
| RAM/memory | Process RSS snapshots | Not isolated; no device peak | PARTIALLY SUPPORTED | Add continuous sampling and accelerator memory |
| Inference latency | Warm-up and repeated CPU batch-1 timing | Direct prototype evidence | FULLY SUPPORTED | Repeat across hardware and seeds/files |
| Throughput | Derived from mean batch-1 latency | Implemented for one setup | FULLY SUPPORTED | Add batch/streaming regimes |
| RTF | Defined and measured for 3 s input | Direct prototype evidence | FULLY SUPPORTED | Report preprocessing-inclusive RTF by device |
| Pareto analysis | One model only | Pareto frontier is mathematically unavailable | NOT IMPLEMENTED | Require at least two evaluated models and three axes |
| Edge/CPU deployment | Desktop CPU benchmark only | Faster-than-real-time is shown, edge is not | PARTIALLY SUPPORTED | Measure Raspberry Pi/mobile/Jetson or declared target |
| Streamlit inference | Working upload UI and shared contract | Direct match | FULLY SUPPORTED | Separate demo claim from production service claim |
| Reproducibility | Seeds and pinned packages, but no split manifest/data hashes/run manifest | Incomplete provenance | PARTIALLY SUPPORTED | Persist checksums, split IDs, hardware and commit hash |
| Data leakage protection | File split before augmentation | No speaker/source/generator grouping | PARTIALLY SUPPORTED | Implement group-disjoint protocol |
| Full six-model LAVA results | Root has only MobileNetV3Small-LSTM | Contradicts a full-results narrative | CONFLICT | Use prototype scope or implement and run all models |

---

# D. Scope Decision

## Decision: LAVA PROTOTYPE / PARTIAL EXPERIMENTAL VALIDATION

Full LAVA is not supported by the present evidence. The code validates one experimental instantiation of a broader framework:

> LAVA is presented as an extensible benchmarking design, while the current experimental instantiation validates clean detection and CPU-efficiency measurement using a MobileNetV3Small-LSTM baseline.

This scope is mandatory because no production implementation or logs exist for ShuffleNetV2, MnasNet, EfficientNet-B0, RawNet2, or AASIST; no held-out noise, compression, replay, or cross-dataset experiment exists; and a one-model Pareto frontier cannot be computed.

The repository can support a prototype paper, methods note, registered protocol, or work-in-progress manuscript. It cannot yet support the target full benchmark contribution or deployment ranking.

---

# E. Missing Experiments / Missing Code

## E.1 Priority checklist

- [ ] **Dataset provenance:** add `data/manifest.csv` with source URL, license, speaker/source ID, generator, transformation, duration, checksum, and parent recording.
- [ ] **Leakage-safe split:** add group-disjoint speaker/source/generator manifests; audit exact and perceptual duplicates.
- [ ] **Larger balanced corpus:** current 8 REAL/56 FAKE corpus cannot support stable class-wise conclusions.
- [ ] **Model registry:** implement MobileNetV3Small, ShuffleNetV2, MnasNet, EfficientNet-B0, RawNet2, and AASIST behind one prediction-score interface.
- [ ] **Repeated training:** use identical split manifests and at least multiple declared seeds; preserve every config/checkpoint/log.
- [ ] **Noise stress:** define noise corpus, noise types, SNR levels, random seeds, and whether the same perturbation is paired across models.
- [ ] **Compression stress:** define MP3/AAC/Opus/Ogg codecs, bitrates, encoder versions, single/double transcoding, and source preservation.
- [ ] **Replay stress:** distinguish simulated room impulse response from physical playback/re-recording; record room, devices, distance, and gain.
- [ ] **Unseen evaluation:** use generator-disjoint and cross-dataset testing with source/speaker isolation.
- [ ] **Efficiency:** add MACs/FLOPs, isolated peak memory, target edge devices, power/energy if available, and synchronized model formats/precision.
- [ ] **Pareto:** compute non-dominated models only after multiple models have clean, robustness, and efficiency values.
- [ ] **Statistics:** confidence intervals, repeated seeds, and paired tests where conditions share inputs.

## E.2 Exact required modules and artifacts

The following are requirements, not existing commands:

| Required module | Required command after implementation | Expected artifact |
|---|---|---|
| `benchmark/build_split_manifest.py` | `python -m benchmark.build_split_manifest --group source_id --seed 42` | `outputs/protocol/splits_seed42.json` + leakage report |
| `benchmark/model_registry.py` | imported by runner | stable constructors and score semantics for all models |
| `benchmark/train_all.py` | `python -m benchmark.train_all --config configs/lava_full.yaml` | per-model checkpoints, histories, seeds, hashes |
| `benchmark/robustness.py` | `python -m benchmark.robustness --config configs/lava_full.yaml` | paired clean/stress score files and condition metadata |
| `benchmark/efficiency.py` | `python -m benchmark.efficiency --config configs/lava_full.yaml` | latency/memory/size/MACs/RTF JSON by device |
| `benchmark/pareto.py` | `python -m benchmark.pareto --input outputs/lava_aggregate.csv` | dominance table and frontier figure |
| `benchmark/benchmark_runner.py` | currently: `python -m benchmark.benchmark_runner --warmup 10 --runs 50 --end-to-end-runs 20` | `outputs/benchmark/current_baseline.json` |

Markers for the current manuscript:

- `[EXPERIMENT REQUIRED: GROUP-DISJOINT RETRAINING]`
- `[EXPERIMENT REQUIRED: MULTI-MODEL BENCHMARK]`
- `[NOT IMPLEMENTED IN CURRENT REPOSITORY: NOISE/COMPRESSION/REPLAY/CROSS-DATASET]`
- `[RESULT PENDING: MULTI-MODEL PARETO FRONTIER]`

---

# F. Final Research Design Reconciled with Evidence

## F.1 Prototype research questions

**PRQ1.** What clean-test behavior does the current MobileNetV3Small-LSTM instantiation exhibit under the repository's deterministic file-level protocol, including class-wise errors and EER?

**PRQ2.** Can the current instantiation process a three-second clip faster than real time on the audited Windows CPU under a declared warm-up and repeated-measurement protocol?

**PRQ3.** Which implementation and experimental gaps prevent the current prototype from answering the full LAVA questions about cross-model competitiveness, robustness, and Pareto-optimal deployment?

The original RQ1–RQ3 are retained as **full-study targets**, not answered questions:

1. To what extent can lightweight architectures maintain competitive deepfake voice detection performance relative to established anti-spoofing models?
2. How does detection performance degrade under noise, compression, replay, and unseen data?
3. Which models offer the most favorable trade-offs among detection, robustness, and efficiency?

## F.2 Evidence-compatible contributions

1. An audited, extensible LAVA evaluation design that explicitly separates clean detection, robustness, and efficiency evidence.
2. A reproducible TensorFlow prototype using standardized waveform-to-temporal-Mel preprocessing and a MobileNetV3Small-LSTM baseline.
3. A measured clean/CPU-efficiency evidence artifact plus an integrity-oriented compatibility matrix that prevents planned experiments from being reported as completed results.

The current work does **not** contribute a novel detector architecture, a six-model benchmark, a robustness benchmark, or a Pareto deployment ranking.

## F.3 Full-study acceptance gate

The manuscript may be upgraded from prototype to Full LAVA only when all of the following exist: provenance-controlled and group-disjoint data; at least four lightweight models plus declared references; identical clean/stress protocols; EER and class-wise metrics; repeated latency/memory/RTF measurements on declared hardware; multiple-seed uncertainty; and a reproducible non-dominance analysis.

---

# G. FULL PAPER B

## LAVA: A Lightweight Benchmarking Framework for Robust and Real-Time Deepfake Voice Detection

### Abstract

Audio deepfakes produced by text-to-speech and voice-conversion systems create risks for authentication, impersonation, and mediated communication. Detector accuracy on a clean dataset alone is insufficient evidence for deployment: a useful assessment must also expose behavior under distributional and channel changes and measure the resources required to process audio. This paper introduces LAVA, an extensible protocol that separates clean detection, robustness stress testing, and computational-efficiency measurement before combining comparable model results through Pareto analysis. We also report an evidence-limited validation of the current LAVA implementation. The audited repository contains one TensorFlow/Keras detector: a six-segment MobileNetV3Small-LSTM model evaluated on 64 files (8 REAL and 56 FAKE) using a deterministic stratified file-level split. On its ten-file test partition, the stored checkpoint obtains 0.800 accuracy, 0.889 FAKE-class F1, 0.813 ROC-AUC, and 0.375 interpolated EER; however, at its validation-selected threshold it classifies both REAL test files as FAKE, yielding zero REAL-class recall and a macro F1 of 0.444. On the audited Windows CPU, model-only inference averages 321.3 ms per three-second clip (RTF 0.107), while preprocessing plus inference averages 342.0 ms (RTF 0.114). These measurements demonstrate faster-than-real-time offline processing for the tested configuration, not streaming or edge-device readiness. The repository does not yet implement alternative detectors, controlled noise/compression/replay tests, cross-dataset evaluation, or a multi-model Pareto frontier. Accordingly, the present contribution is a prototype validation and reproducible research audit rather than a completed multi-architecture benchmark. The findings show why class-wise performance, leakage-safe data design, robustness, and end-to-end timing must precede deployment claims.

**Keywords:** audio deepfake detection; synthetic speech; anti-spoofing; MobileNetV3; LSTM; robustness; real-time factor; reproducible benchmarking

## 1. Introduction

### 1.1 Background and motivation

Generative speech systems can synthesize or transform voices with increasing perceptual quality. The same technology enables accessibility and creative applications, but it also lowers the cost of impersonation, social engineering, and attacks on voice-based authentication. Audio anti-spoofing therefore asks whether an observed signal is bona fide or manipulated, including text-to-speech, voice conversion, and replay-mediated conditions.

Benchmark leaderboards have made progress measurable, particularly through ASVspoof. Yet deployment is not determined by one clean-set score. A detector can rank well on a familiar corpus and fail after codec conversion, playback and re-recording, background noise, a new generator, or a source-domain shift. It may also be too slow or memory-intensive for the intended platform. These concerns form three linked evaluation dimensions:

1. **Detection performance:** discrimination and class-wise error under a declared clean protocol.
2. **Robustness:** degradation under controlled channel, environmental, and distribution shifts.
3. **Computational efficiency:** model size, memory, latency, throughput, and real-time factor on declared hardware.

LAVA treats those dimensions as separate evidence streams. This avoids using a high clean accuracy to imply robustness or deployment suitability, and it prevents architecture plans from being reported as experimental results.

### 1.2 Research gap

The literature does not ignore robustness or efficiency. ASVspoof evaluates replay, codec, and source/channel mismatch; cross-corpus studies demonstrate severe generalization loss; lightweight anti-spoofing work reports parameter reductions; and recent deployment-oriented studies report latency and RTF. The gap is narrower: these properties are commonly measured in different studies, with different datasets, conditions, metrics, preprocessing paths, and hardware. Consequently, it remains difficult to compare detectors jointly on clean discrimination, realistic degradation, and end-to-end cost under one predefined protocol.

Table 1 records the audit basis for that conclusion. A check means the cited study explicitly evaluates or reports the item; a half mark denotes discussion, indirect coverage, or a proxy rather than a controlled measurement; a dash means that the item was not found in the audited paper. A dash is not a claim that the authors never evaluated the property elsewhere.

**Table 1. Literature gap matrix.**

| Study | Detection metric | Noise | Compression | Replay | Unseen data/domain | Params | Model size | Latency | RTF | Edge focus |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Kinnunen et al., ASVspoof 2017 [1] | EER | — | — | ✓ | ✓ | — | — | — | — | — |
| Todisco et al., ASVspoof 2019 [2] | EER, t-DCF | — | — | ✓ | ✓ | — | — | — | — | — |
| Tak et al., RawNet2 [3] | EER, t-DCF | — | — | — | ✓ | — | — | — | — | — |
| Jung et al., AASIST [4] | EER, t-DCF | — | — | — | ✓ | ✓ | ✓ | — | — | ◐ |
| Frank and Schönherr, WaveFake [5] | accuracy/other detector metrics | ◐ | ◐ | — | ✓ | — | — | — | — | — |
| Korshunov and Marcel [6] | EER/HTER | — | — | ✓ | ✓ | — | — | — | — | — |
| Müller et al. [7] | EER | — | — | — | ✓ | — | — | — | — | — |
| Delgado et al., ASVspoof 2021 [8] | EER, min t-DCF | — | ✓ | ✓ | ✓ | — | — | — | — | — |
| Yi et al., ADD 2022 [9] | EER | ✓ | ◐ | — | ◐ | — | — | — | — | — |
| Wang et al. [10] | EER | ✓ | — | — | ✓ | — | — | — | — | — |
| Shim et al. [11] | EER | — | — | — | ✓ | ✓ | — | — | — | ◐ |
| Kwak et al. [12] | EER | — | — | ✓ | ✓ | ✓ | ◐ | — | — | ✓ |
| “Generalization Gap” vishing study [13] | discrimination metrics | — | — | — | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |

**Evidence:** EXTERNAL LITERATURE. The matrix supports a fragmentation claim, not an absolute claim that efficiency or robustness has never been studied.

### 1.3 Research questions

Because this manuscript distinguishes the complete design from its current implementation, it asks three prototype questions:

- **PRQ1:** What clean-test behavior does the current MobileNetV3Small-LSTM instantiation exhibit, including class-wise errors and EER?
- **PRQ2:** Can that instantiation process a three-second clip faster than real time on the audited CPU under a declared repeated-timing protocol?
- **PRQ3:** Which missing components prevent the prototype from answering the full LAVA questions about model competitiveness, robustness, and Pareto-optimal deployment?

The intended full-study questions remain: (RQ1) how lightweight models compare with established anti-spoofing references; (RQ2) how performance degrades under noise, compression, replay, and unseen data; and (RQ3) which models are Pareto-optimal across detection, robustness, and efficiency. They are research targets, not questions answered by the current experiment.

### 1.4 Contributions

This evidence-compatible paper contributes:

1. a protocol-level LAVA design that separates clean, robustness, and efficiency evaluation;
2. a source-audited TensorFlow instantiation based on temporal Mel spectrograms and MobileNetV3Small-LSTM;
3. reproducible clean-detection and CPU-efficiency measurements for the stored checkpoint; and
4. an explicit compatibility and evidence map showing what is implemented, planned, or invalid to claim.

It does not claim a novel detector, a six-model comparison, a completed robustness benchmark, or a deployment ranking.

## 2. Related Work and Research Background

### 2.1 Deepfake voice and audio anti-spoofing

Synthetic speech detection covers signals generated by text-to-speech and voice-conversion systems, while the broader anti-spoofing domain also includes replay and related presentation attacks. ASVspoof 2017 centered on replay under heterogeneous conditions [1]. ASVspoof 2019 formalized logical-access and physical-access scenarios and paired EER with the tandem detection cost function [2]. ASVspoof 2021 further emphasized telephony codecs, transmission effects, physical replay, and source-domain mismatch [8]. These benchmarks illustrate why attack type and acquisition channel are part of the evaluation protocol rather than incidental dataset details.

Cross-database work predates the present wave of generative audio. Korshunov and Marcel showed that systems trained on one spoofing database can degrade sharply on another [6]. Müller et al. later reimplemented twelve deepfake detectors under a uniform protocol and again found weak generalization to external and in-the-wild data [7]. Thus, an unseen test should isolate speakers, sources, and generators where possible; a random file split alone does not establish generalization.

### 2.2 Detection architectures

Audio detectors operate on raw waveforms, spectral representations, or learned intermediate features. RawNet2 processes raw audio using residual blocks and attention and was adapted as an ASVspoof countermeasure [3]. AASIST models spectro-temporal relations using heterogeneous graph attention; its published variants include a compact AASIST-L, showing that anti-spoofing accuracy and parameter count can be considered together [4]. CNN systems remain common for time-frequency inputs, while recurrent or temporal modules can aggregate evidence across frames or segments.

The current repository belongs to the last group. It transforms six consecutive half-second waveform segments into Mel-spectrogram images, extracts an embedding from each segment with a shared MobileNetV3Small, and models the resulting sequence with an LSTM. This is a real temporal sequence, unlike simply reshaping one image embedding into a sequence of length one.

### 2.3 Lightweight neural networks

MobileNetV3 combines hardware-aware neural architecture search with mobile-oriented operators [14]. ShuffleNetV2 derives practical design guidelines from measured speed rather than FLOPs alone [15]. MnasNet explicitly incorporates measured mobile latency into architecture search [16], and EfficientNet scales network depth, width, and resolution in a coordinated manner [17]. These works motivate candidates for LAVA, but their original image-classification results do not demonstrate audio anti-spoofing performance.

Only MobileNetV3Small is present in the production implementation audited here. ShuffleNetV2, MnasNet, EfficientNet-B0, RawNet2, and AASIST appear only as proposed candidates or external references. Discussing them does not make them benchmarked models.

### 2.4 Robustness and real-time evaluation

Robustness tests must define the perturbation and its application precisely. Noise evaluation should report corpus, category, SNR, mixing procedure, and seed [10]. Compression should identify codec, encoder, bitrate, and transcoding count; neural codec-generated fakes should not be conflated with ordinary post-hoc lossy encoding [18]. Replay should separate simulated room filtering from physical playback/re-recording and document devices and geometry. Unseen evaluation should state what is held out: speaker, source, generator, corpus, or all of these.

Efficiency is equally protocol-dependent. Parameter count and serialized size are hardware-independent summaries, but latency, memory, throughput, and energy are not. Timing must state preprocessing inclusion, device, precision, batch size, warm-up, repetition count, and aggregation statistic. RTF is meaningful only relative to the processed audio duration. A model can be faster than real time while still being unsuitable for streaming because it requires a complete fixed-length window.

The literature gap in Table 1 therefore motivates a common measurement contract, not another unqualified accuracy leaderboard.

## 3. LAVA Benchmarking Methodology

![Conceptual LAVA framework and current implementation status](../outputs/benchmark/figures/lava_framework.png)

**Figure 1.** LAVA framework. Blue blocks are implemented in the present prototype; gray blocks are planned components. **Evidence:** CODE and the audited implementation status.

### 3.1 Dataset and experimental protocol

The implementation root contains 64 audio files organized as `data/REAL` and `data/FAKE`. All observed files are PCM16 stereo WAV, although the loader accepts WAV, FLAC, MP3, OGG, and M4A. The files are long recordings, but the pipeline reads at most the first three seconds of each file. The deterministic seed-42 split is stratified by label at the file level.

**Table 2. Dataset statistics and current split.**

| Partition | REAL | FAKE | Total | Split mechanism |
|---|---:|---:|---:|---|
| Train | 5 | 39 | 44 | stratified file-level, seed 42 |
| Validation | 1 | 9 | 10 | stratified file-level, seed 42 |
| Test | 2 | 8 | 10 | stratified file-level, seed 42 |
| **Total** | **8** | **56** | **64** | — |

Observed sampling rates are 44.1 kHz (57 files), 40 kHz (4), and 48 kHz (3). Durations range from 79.487 to 600.449 s, with a mean of 467.961 s. Training-only augmentation may add Gaussian noise, time shift, gain variation, and pitch shift according to the repository configuration. Class weights computed from the training partition are 4.4 for REAL and approximately 0.5641 for FAKE.

The split protects against the same file path appearing in more than one partition and confines augmentation to training. It does not group by speaker, source recording, celebrity identity, or generator. Filename patterns suggest that related identities or sources can occur across partitions, so source leakage is a credible risk; this is an inference from names and split logic, not proof of duplicate audio. No provenance manifest, parent-recording ID, exact/perceptual duplicate audit, or generator-disjoint protocol is present.

**Evidence:** DATASET and CODE (`config.py`, `src/data_loader.py`, `src/preprocessing.py`, `src/augmentation.py`).

### 3.2 Standardized preprocessing

For a batch of size (B), the implemented transformation is:

\[
\text{decoded audio}\rightarrow\text{mono}\rightarrow22{,}050\ \text{Hz}
\rightarrow(66{,}150)\rightarrow(6,11{,}025)
\rightarrow(6,128,23)\rightarrow(6,224,224,3).
\]

The loader reads a bounded three-second interval, averages channels, resamples to 22.05 kHz, and pads or truncates to 66,150 samples. It divides this waveform into six chronological 0.5-second segments of 11,025 samples. For each segment, an STFT with 2,048-point FFT and 512-sample hop produces 1,025 frequency bins and 23 frames. A 128-band Mel filter bank spanning 20–8,000 Hz is applied; power is converted to decibels relative to the segment maximum with an −80 dB floor, mapped to 0–255, resized to (224\times224), and copied to three channels. Stacking yields a per-clip tensor of (6\times224\times224\times3), and batching yields (B\times6\times224\times224\times3).

The image resizing and RGB replication are compatibility choices for an ImageNet backbone, not evidence that they are optimal for audio.

**Evidence:** CODE (`src/preprocessing.py`, `src/data_loader.py`).

### 3.3 Current detector and planned registry

The current model applies a shared ImageNet-pretrained Keras MobileNetV3Small (`include_top=False`, global-average pooling) independently to each segment through `TimeDistributed`. This produces (B\times6\times576) embeddings. An LSTM with 128 units aggregates the sequence, followed by a 64-unit ReLU layer, dropout 0.4, and one sigmoid unit. The output is interpreted as (P(\mathrm{FAKE})); the stored validation-selected decision threshold is 0.38.

Training is one lifecycle with two internal stages. Warm-up freezes the backbone and optimizes the classifier with Adam, binary cross-entropy, and a (10^{-4}) learning rate. Fine-tuning automatically restores the best warm-up state, unfreezes the last 20 backbone layers while keeping batch-normalization layers frozen, recompiles, and uses (10^{-5}). One validation-loss checkpoint retains the global best weights across both stages. The threshold is selected on validation F1 over 0.10–0.90 in increments of 0.01 after the final production artifact is restored.

**Table 3. Current model complexity.**

| Model | Parameters | Serialized size | Input | Temporal component |
|---|---:|---:|---|---|
| MobileNetV3Small-LSTM | 1,308,401 | 5.643 MiB | (6\times224\times224\times3) | LSTM, 128 units over 6 segments |

The loaded production model exposes 646,481 trainable and 661,920 non-trainable parameters. A future registry may expose ShuffleNetV2, MnasNet, EfficientNet-B0, RawNet2, and AASIST behind a common probability interface, but none is implemented in the production root at present.

**Evidence:** CODE and serialized MODEL (`src/model.py`, `train.py`, `models/lava_mobilenetv3_lstm.keras`).

### 3.4 Clean evaluation

Let FAKE be the positive class. The reported metrics are:

\[
\mathrm{Precision}=\frac{TP}{TP+FP},\qquad
\mathrm{Recall}=\frac{TP}{TP+FN},
\]

\[
F_1=2\frac{\mathrm{Precision}\cdot\mathrm{Recall}}
{\mathrm{Precision}+\mathrm{Recall}}.
\]

ROC-AUC uses continuous sigmoid scores. EER is estimated by interpolating the false-positive and false-negative rate curves at a threshold (\tau^*\) for which

\[
\mathrm{FAR}(\tau^*)\approx\mathrm{FRR}(\tau^*).
\]

Lower EER is better, whereas higher precision, recall, F1, and ROC-AUC are better. Accuracy is retained for completeness but is not a sufficient summary under the observed 1:7 class imbalance. Macro F1 and the confusion matrix are necessary to expose behavior on the minority REAL class.

### 3.5 Robustness stress tests

The complete LAVA protocol specifies paired evaluation of the same clean items under background noise, post-hoc codec compression, replay, and unseen generator/corpus conditions. For any higher-is-better metric (M), absolute degradation is

\[
\Delta M=M_{\mathrm{clean}}-M_{\mathrm{stress}}.
\]

For a lower-is-better error metric such as EER, degradation should instead be reported as (M_{\mathrm{stress}}-M_{\mathrm{clean}}). Conditions, seeds, and successful input pairs must be identical across models.

**[NOT IMPLEMENTED IN CURRENT REPOSITORY: NOISE/COMPRESSION/REPLAY/CROSS-DATASET]**

Exact experiment required: implement `benchmark/robustness.py`; create a leakage-safe test manifest; generate condition manifests specifying corpus/codec/device parameters and seeds; save per-file clean and stressed scores. Required command after implementation: `python -m benchmark.robustness --config configs/lava_full.yaml`. Expected artifacts: `outputs/robustness/scores.csv`, `conditions.json`, and a paired degradation table.

### 3.6 Computational efficiency

For an input containing (T_{audio}) seconds of signal, real-time factor is

\[
\mathrm{RTF}=\frac{T_{processing}}{T_{audio}}.
\]

An RTF below 1 means the measured path processes audio faster than its duration. The present runner measures batch-one TensorFlow inference after ten warm-up calls and 50 repetitions. A separate end-to-end path includes file decoding and preprocessing, with two warm-ups and 20 repetitions. It reports mean, median, standard deviation, and 95th percentile. Timing uses one three-second clip on the audited Windows CPU; no GPU is visible. Throughput is (1/\bar{T}) clips/s. Process RSS is sampled before loading, after loading, and after each benchmark; it is whole-process memory, not an isolated model peak.

### 3.7 Pareto-based analysis

For multiple evaluated models, LAVA defines a model as dominated if another model is no worse on all selected objectives and strictly better on at least one. Objectives should retain transparent units—for example lower EER, lower mean robustness degradation, and lower end-to-end RTF—rather than being collapsed into an arbitrary weighted score. Constraints such as maximum memory can be applied before identifying the non-dominated set.

**[RESULT PENDING: MULTI-MODEL PARETO FRONTIER]** A single measured model cannot form a meaningful comparative frontier. Exact experiment required: train and evaluate all registered models under identical manifests and devices, aggregate their metrics, then run `python -m benchmark.pareto --input outputs/lava_aggregate.csv`. Expected artifacts: a dominance table, frontier plot, and declared deployment constraints.

## 4. Experimental Results and Discussion

### 4.1 PRQ1: Clean detection behavior

Table 4 contains every clean score produced from the current ten-file test partition. At threshold 0.38, the model detects all eight FAKE files but labels both REAL files as FAKE. The apparently strong FAKE-class F1 therefore coexists with complete failure on the minority class. ROC-AUC of 0.8125 indicates some ranking ability independent of the deployed threshold, whereas an interpolated EER of 0.375 remains high. The EER operating threshold (approximately 0.5855) also differs substantially from the validation F1 threshold, illustrating the dependence of an operating point on validation objective and class composition.

**Table 4. Clean test performance for the only implemented model.**

| Model | Accuracy | EER ↓ | FAKE F1 ↑ | Macro F1 ↑ | ROC-AUC ↑ | FAKE precision ↑ | FAKE recall ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| MobileNetV3Small-LSTM | 0.800 | 0.375 | 0.889 | 0.444 | 0.813 | 0.800 | 1.000 |

**Evidence:** EXPERIMENT LOG `outputs/benchmark/current_baseline.json` and the stored test split/model. Because the test set has only two REAL and eight FAKE files, these point estimates have high sampling uncertainty and must not be interpreted as stable population performance.

![Test confusion matrix](../outputs/benchmark/figures/confusion_matrix.png)

**Figure 2.** Test confusion matrix at threshold 0.38: TN=0, FP=2, FN=0, TP=8. **Evidence:** EXPERIMENT LOG.

The result answers PRQ1 narrowly: under the repository's file-level protocol, the checkpoint favors the majority FAKE decision and does not demonstrate reliable binary separation at its chosen operating threshold. The class imbalance and missing source-disjoint split make aggregate accuracy especially misleading.

### 4.2 PRQ2 and RQ2: robustness under stress

No robustness scores exist in the repository, so Table 5 deliberately contains no invented degradation values.

**Table 5. Robustness status.**

| Model | Clean | Noise | Compression | Replay | Unseen/cross-dataset | Mean degradation |
|---|---|---|---|---|---|---|
| MobileNetV3Small-LSTM | measured; see Table 4 | [EXPERIMENT REQUIRED] | [EXPERIMENT REQUIRED] | [EXPERIMENT REQUIRED] | [EXPERIMENT REQUIRED] | [RESULT PENDING] |

**Evidence:** CODE AUDIT. Training-time waveform augmentation is not a held-out robustness evaluation and is not substituted for one. Therefore, the full RQ2 cannot be answered.

### 4.3 Computational efficiency

Table 6 reports the repeated CPU benchmark. The mean model-only latency is 321.3 ms and mean end-to-end latency is 342.0 ms for a three-second clip. Their RTFs are 0.107 and 0.114, respectively. Thus, the tested offline path is faster than real time on this CPU. End-to-end throughput is about 2.92 three-second clips/s. The 20.7 ms difference between means indicates that preprocessing is material but not dominant for this configuration.

**Table 6. CPU efficiency of the current checkpoint, batch size 1.**

| Model/path | Params | Size | Process RSS evidence | Mean latency | Median ± SD | P95 | Throughput | RTF |
|---|---:|---:|---|---:|---:|---:|---:|---:|
| MobileNetV3Small-LSTM, model only | 1,308,401 | 5.643 MiB | 406.6 MiB after load; 649.7 MiB after timing | 321.3 ms | 316.7 ± 21.9 ms | 365.4 ms | 3.11 clips/s | 0.107 |
| Same, decode + preprocessing + model | same | same | 655.3 MiB after end-to-end timing | 342.0 ms | 337.0 ± 13.0 ms | 363.2 ms | 2.92 clips/s | 0.114 |

Hardware/software: Windows, AMD64 Family 25 Model 68 CPU, 12 logical processors, Python 3.11.9, TensorFlow 2.15.0, no visible GPU. Model-only protocol: 10 warm-ups and 50 timed calls. End-to-end protocol: 2 warm-ups and 20 timed calls. The RSS values describe the Python process at snapshots and must not be interpreted as model-only peak RAM. Serialized size describes the `.keras` artifact and is not an on-device package measurement.

**Evidence:** EXPERIMENT LOG `outputs/benchmark/current_baseline.json`.

![Measured baseline efficiency](../outputs/benchmark/figures/efficiency_baseline.png)

**Figure 3.** Actual mean latency and RTF for the model-only and end-to-end paths. **Evidence:** EXPERIMENT LOG.

PRQ2 is answered affirmatively only for this three-second, batch-one, offline CPU protocol. The experiment does not establish streaming latency, concurrent-service throughput, energy use, mobile performance, or memory safety under load.

### 4.4 PRQ3 and the unavailable Pareto trade-off

No relative architecture ranking can be made because only one production detector exists. A point plotted alone would be visually possible but scientifically uninformative: it cannot be dominated or preferred relative to alternatives. Likewise, clean detection cannot stand in for robustness. The missing evidence that blocks the full RQ1–RQ3 consists of: group-disjoint data; multiple trained models; paired stress conditions; repeated seeds and confidence intervals; model-isolated memory; and synchronized efficiency measurements on target hardware.

The scope is therefore **LAVA prototype / partial experimental validation**, not Full LAVA.

### 4.5 Error and failure analysis

The confusion matrix identifies two false positives and no false negatives at the deployed threshold. In application terms, the system rejects every tested REAL file as synthetic while accepting the eight FAKE decisions. This could arise from imbalance, threshold selection on a validation set containing only one REAL file, source/channel cues, insufficient corpus diversity, or some combination. The current evidence cannot distinguish these explanations.

Generator-specific, speaker-specific, and degradation-induced flip analyses are unavailable because the data has no verified generator/source metadata and no stressed score pairs. Reporting such explanations as findings would exceed the evidence. A future failure-analysis artifact should contain per-file label, score, prediction, speaker/source, generator, codec, condition, and clean-to-stress flip status, with audio access governed by licensing and privacy constraints.

## 5. Conclusion, Deployment Implications, and Limitations

### 5.1 Answers to the research questions

**PRQ1.** The current checkpoint achieves 0.813 ROC-AUC and 0.375 EER on ten test files, but its validation-selected threshold produces no correct REAL decisions. Its 0.800 accuracy and 0.889 FAKE F1 therefore overstate balanced usefulness; macro F1 is 0.444.

**PRQ2.** The audited offline path is faster than real time on the tested CPU: mean model-only RTF is 0.107 and mean end-to-end RTF is 0.114 for a three-second clip.

**PRQ3.** The prototype cannot answer cross-model, robustness, or Pareto questions until source-disjoint data, model alternatives, controlled stress suites, repeated training, and target-device efficiency evidence are added.

The original full-study RQ1–RQ3 remain open.

### 5.2 Practical deployment implications

The Streamlit application demonstrates interactive upload, visualization, and inference using the saved Keras model. The timing experiment suggests that a single request can be processed faster than the three seconds of audio it represents on the audited desktop CPU. This is useful feasibility evidence, but it is not a production-readiness result. The pipeline considers only the first three seconds, requires a complete window, has not been load-tested, and provides no calibration, abstention, user-level aggregation, or monitoring for domain drift.

The false-positive pattern is particularly important for deployment: a detector that labels all tested REAL examples as FAKE could cause systematic rejection even while reporting 80% accuracy on an imbalanced test. No security-sensitive release should use the current checkpoint without a representative, leakage-safe validation set, application-specific operating costs, calibration, and external evaluation.

### 5.3 Limitations

1. **Corpus size and balance:** 64 files, including only eight REAL samples, are insufficient for stable class-wise claims.
2. **Provenance:** dataset origin, license, generator, speakers, and parent recordings are not formally recorded.
3. **Leakage control:** splitting is file-disjoint but not source-, speaker-, or generator-disjoint; filename evidence suggests possible related sources across splits.
4. **Model coverage:** only MobileNetV3Small-LSTM is implemented; no lightweight or anti-spoofing reference comparison exists.
5. **Robustness:** noise, compression, replay, and unseen/cross-dataset tests are absent.
6. **Uncertainty:** one stored training run and a ten-file test set provide no seed variance or reliable confidence intervals.
7. **Efficiency scope:** latency is measured on one Windows CPU; process RSS is not isolated peak model memory; FLOPs/MACs, energy, and physical edge-device measurements are absent.
8. **Input behavior:** only the first three seconds of long files are used, and the system is offline rather than streaming.
9. **Model portability:** TensorFlow/Keras serialization compatibility is environment-sensitive; ONNX/TFLite behavior is untested.

### 5.4 Future work

The immediate priority is experimental validity rather than architectural breadth: create a provenance manifest, identify parent recordings and speakers, audit exact/perceptual duplicates, and retrain with group-disjoint manifests. Next, implement a score-compatible registry and reproduce MobileNetV3Small, ShuffleNetV2, MnasNet, EfficientNet-B0, RawNet2, and AASIST under identical data and training controls. Robustness should then be measured with declared noise/SNR pairs, codec/bitrate pairs, simulated and physical replay kept distinct, and generator- and corpus-disjoint test sets.

Deployment work should add isolated peak memory, MACs/FLOPs, concurrency tests, streaming-window analysis, energy, and measurements on actual target devices. Quantization, pruning, knowledge distillation, ONNX, and TFLite should be evaluated after—not before—a reliable baseline protocol exists. Only then should LAVA compute a Pareto frontier and issue application-specific recommendations such as accuracy-priority, robustness-priority, resource-priority, or balanced real-time operation.

### 5.5 Conclusion

LAVA is motivated by a simple methodological constraint: a detector is not deployment-ready merely because it attains one favorable score. Clean discrimination, robustness, and computational cost require predefined and auditable protocols. The present repository validates part of that workflow with a real MobileNetV3Small-LSTM checkpoint and measured CPU timing, but it simultaneously exposes major limitations: minority-class failure, severe data imbalance, possible source leakage, one-model coverage, and absent robustness evidence. Preserving those limitations in the paper is not an incompleteness to hide; it is the condition for a reproducible next-stage benchmark.

## References

[1] T. Kinnunen et al., “The ASVspoof 2017 Challenge: Assessing the Limits of Replay Spoofing Attack Detection,” *Interspeech*, 2017. DOI: [10.21437/Interspeech.2017-1111](https://doi.org/10.21437/Interspeech.2017-1111).

[2] M. Todisco et al., “ASVspoof 2019: Future Horizons in Spoofed and Fake Audio Detection,” *Interspeech*, 2019. DOI: [10.21437/Interspeech.2019-2249](https://doi.org/10.21437/Interspeech.2019-2249).

[3] H. Tak et al., “End-to-End Anti-Spoofing with RawNet2,” *ICASSP*, 2021. DOI: [10.1109/ICASSP39728.2021.9414234](https://doi.org/10.1109/ICASSP39728.2021.9414234).

[4] J.-W. Jung et al., “AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks,” *ICASSP*, 2022. DOI: [10.1109/ICASSP43922.2022.9747766](https://doi.org/10.1109/ICASSP43922.2022.9747766).

[5] J. Frank and L. Schönherr, “WaveFake: A Data Set to Facilitate Audio Deepfake Detection,” *NeurIPS Datasets and Benchmarks*, 2021. [Proceedings page](https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/hash/c74d97b01eae257e44aa9d5bade97baf-Abstract-round2.html).

[6] P. Korshunov and S. Marcel, “Cross-Database Evaluation of Audio-Based Spoofing Detection Systems,” *Interspeech*, 2016. DOI: [10.21437/Interspeech.2016-1326](https://doi.org/10.21437/Interspeech.2016-1326).

[7] N. M. Müller et al., “Does Audio Deepfake Detection Generalize?” *Interspeech*, 2022. DOI: [10.21437/Interspeech.2022-108](https://doi.org/10.21437/Interspeech.2022-108).

[8] H. Delgado et al., “ASVspoof 2021: Automatic Speaker Verification Spoofing and Countermeasures Challenge Evaluation Plan and Results,” *IEEE/ACM Transactions on Audio, Speech, and Language Processing*, 2023. DOI: [10.1109/TASLP.2023.3285283](https://doi.org/10.1109/TASLP.2023.3285283).

[9] J. Yi et al., “ADD 2022: The First Audio Deep Synthesis Detection Challenge,” *ICASSP*, 2022. [arXiv:2202.08433](https://arxiv.org/abs/2202.08433).

[10] X. Wang et al., “Robust Audio Anti-Spoofing Countermeasure with Joint Training of Front-end and Back-end Models,” *Interspeech*, 2023. DOI: [10.21437/Interspeech.2023-1166](https://doi.org/10.21437/Interspeech.2023-1166).

[11] H.-J. Shim, J.-W. Jung, and T. Kinnunen, “Multi-Dataset Co-Training with Sharpness-Aware Optimization for Audio Anti-Spoofing,” *Interspeech*, 2023. DOI: [10.21437/Interspeech.2023-1910](https://doi.org/10.21437/Interspeech.2023-1910).

[12] I.-Y. Kwak et al., “Voice Spoofing Detection Through Residual Network, Max Feature Map, and Depthwise Separable Convolution,” *IEEE Access*, 2023. DOI: [10.1109/ACCESS.2023.3275790](https://doi.org/10.1109/ACCESS.2023.3275790).

[13] “The Generalization Gap: Do Audio Deepfake Detectors Actually Protect Against Modern Vishing?” *Electronics*, vol. 15, no. 13, 2026. [Article page](https://www.mdpi.com/2079-9292/15/13/2846).

[14] A. Howard et al., “Searching for MobileNetV3,” *ICCV*, 2019. [CVF Open Access](https://openaccess.thecvf.com/content_ICCV_2019/html/Howard_Searching_for_MobileNetV3_ICCV_2019_paper.html).

[15] N. Ma et al., “ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design,” *ECCV*, 2018. [CVF Open Access](https://openaccess.thecvf.com/content_ECCV_2018/html/Ningning_Light-weight_CNN_Architecture_ECCV_2018_paper.html).

[16] M. Tan et al., “MnasNet: Platform-Aware Neural Architecture Search for Mobile,” *CVPR*, 2019. [CVF Open Access](https://openaccess.thecvf.com/content_CVPR_2019/html/Tan_MnasNet_Platform-Aware_Neural_Architecture_Search_for_Mobile_CVPR_2019_paper.html).

[17] M. Tan and Q. V. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” *ICML*, 2019. [PMLR](https://proceedings.mlr.press/v97/tan19a.html).

[18] H. Wu et al., “CodecFake: Enhancing Anti-Spoofing Models Against Deepfake Audios from Codec-Based Speech Synthesis Systems,” *Interspeech*, 2024. DOI: [10.21437/Interspeech.2024-2093](https://doi.org/10.21437/Interspeech.2024-2093).

---

# H. Evidence Map

The map below binds the paper's important claims to inspectable sources. `CODE` establishes behavior implemented by source; `MODEL` describes the serialized checkpoint; `DATASET` describes files currently present; `EXPERIMENT LOG` records an executed measurement; `PAPER A` supplies methodological organization; and `EXTERNAL LITERATURE` supplies scientific context.

| Claim or artifact | Evidence type | Inspectable source | Boundary / caveat |
|---|---|---|---|
| Paper A motivates taxonomy-first organization, acquisition conditions, predefined protocols, multidimensional evaluation, and explicit future trends | PAPER A | `C:/Users/THANH DAT/Downloads/16507.pdf` | Methodological transfer only; no audio/deepfake technical claim is borrowed |
| Production stack is TensorFlow/Keras with a Streamlit UI | CODE | `requirements.txt`, `app.py`, `src/` | Reference repositories are excluded from production evidence |
| Current input pipeline produces (6\times224\times224\times3) tensors | CODE | `src/preprocessing.py`, `src/data_loader.py` | RGB/resize is an implementation choice, not a validated optimum |
| Dataset has 8 REAL and 56 FAKE files | DATASET | `data/REAL`, `data/FAKE` | Corpus provenance and license are not recorded |
| Split is 44/10/10 and file-disjoint | CODE + DATASET | split logic and executed audit | Not source-, speaker-, or generator-disjoint |
| Current model is MobileNetV3Small + LSTM128 + sigmoid | CODE + MODEL | `src/model.py`, `models/lava_mobilenetv3_lstm.keras` | Other named architectures are not implemented |
| Total parameters and serialized size | MODEL + EXPERIMENT LOG | `outputs/benchmark/current_baseline.json` | Serialized `.keras` size is format-specific |
| Clean metrics and EER in Table 4 | EXPERIMENT LOG | `outputs/benchmark/current_baseline.json` | Ten test files; only two REAL; no confidence interval |
| Confusion matrix TN=0, FP=2, FN=0, TP=8 | EXPERIMENT LOG | JSON and Figure 2 | Applies only to threshold 0.38 |
| CPU latency, throughput, and RTF in Table 6 | EXPERIMENT LOG | `outputs/benchmark/current_baseline.json` | One machine; fixed three-second offline input |
| RSS measurements | EXPERIMENT LOG | same JSON | Whole-process snapshots, not isolated peak model memory |
| Robustness is absent | CODE AUDIT | no stress runner/log in implementation root | Training augmentation is not robustness evaluation |
| Pareto ranking is absent | CODE AUDIT | only one implemented/measured detector | A meaningful comparative frontier requires multiple models |
| Literature gap is fragmented evaluation, not total absence of robustness/efficiency research | EXTERNAL LITERATURE | References [1]–[13] and Table 1 | Matrix limited to audited representative studies |
| Framework status diagram | CODE AUDIT | `outputs/benchmark/figures/lava_framework.png` | Gray components are planned, not completed |

## Reproduction commands for current evidence

From the repository root and its trained `.venv`:

```powershell
.\.venv\Scripts\python.exe -m benchmark.benchmark_runner --warmup 10 --runs 50 --end-to-end-runs 20
.\.venv\Scripts\python.exe -m benchmark.plot_baseline
```

Expected artifacts:

- `outputs/benchmark/current_baseline.json`
- `outputs/benchmark/figures/confusion_matrix.png`
- `outputs/benchmark/figures/efficiency_baseline.png`
- `outputs/benchmark/figures/lava_framework.png`

Exact software and hardware fields are stored in the JSON so that later runs can be compared without treating different environments as interchangeable.
