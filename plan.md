# KẾ HOẠCH THỰC HIỆN ĐỒ ÁN

**Đề tài:** Thiết kế và xây dựng ứng dụng AI nhận diện phương tiện giao thông sử dụng mô hình CNN và thuật toán Heuristic

**Mục tiêu:** Từ video giao thông (MP4/AVI, ≥ 720p, camera cố định), hệ thống tự động **phát hiện → phân loại → theo dõi → đếm** phương tiện thuộc 4 lớp: xe máy, ô tô, xe buýt, xe tải.

> **Trạng thái:** toàn bộ phần code đã xong và đã kiểm thử (31 unit test + chạy trọn quy trình trên video giả lập).
> Phần còn lại là **thu thập dữ liệu thật, chạy thực nghiệm, viết báo cáo**. Cách dùng từng lệnh: xem [README.md](README.md).

---

## 1. Kiến trúc tổng quát

```
Video ──► [MOG2] tìm vùng chuyển động ──► cắt ảnh, resize 64×64
      ──► [Custom CNN] phân loại (motorbike / car / bus / truck / other)
      ──► [Centroid Tracker] gán ID theo tâm xe qua các khung hình
      ──► [Vạch ảo] mỗi ID đi qua vạch thì đếm đúng 1 lần
      ──► Video kết quả (khung bao, nhãn, ID, bộ đếm) + file CSV thống kê
```

| Khối | Vai trò | File mã nguồn |
|---|---|---|
| Phát hiện | MOG2 + morphology + gộp mảnh vỡ + lọc khung bao + ROI | `src/detector.py` |
| Phân loại | Custom CNN 64×64×3 (PyTorch) | `train/model.py`, `src/classifier.py` |
| Theo dõi | Centroid tracker: Hungarian theo khoảng cách tâm, dự đoán vị trí theo vận tốc | `src/tracker.py` |
| Đếm | Vượt vạch ảo, bỏ phiếu mềm loại xe theo ID, lọc rung / `other` | `src/counter.py` |
| Ứng dụng | Ghép pipeline, vẽ kết quả, xuất video / CSV / JSON, đo FPS từng bước | `src/main.py` |

## 2. Chỉ tiêu cần đạt

| Chỉ tiêu | Mục tiêu | Cách đo |
|---|---|---|
| Accuracy phân loại (tập test) | > 90% | `py -m train.evaluate` |
| F1-Score trung bình (macro) | > 0,90 | `py -m train.evaluate`, `py -m train.kfold` |
| Độ chính xác đếm (tổng) | ≥ 90% | `py -m tools.eval_counting`: `(1 - \|máy - tay\| / tay) × 100%` |
| Sai số đếm theo từng lớp + bỏ sót / đếm thừa / nhầm loại | báo cáo riêng | `py -m tools.eval_counting` (cần đếm tay từng xe) |
| Tốc độ xử lý | 15–20 FPS trên CPU, không dùng GPU rời | `py -m src.main --no-display` → `processing_fps` trong summary |

> Mọi số liệu đưa vào báo cáo phải đo thật trên dữ liệu của nhóm, không lấy số từ tài liệu tham khảo.
> Kết quả trên video giả lập (đếm 98,7%, ~34–68 FPS) chỉ chứng minh phần mềm chạy đúng, **không phải kết quả đồ án**.

---

## 3. Các giai đoạn thực hiện

### Giai đoạn 0 – Chuẩn bị môi trường ✅
- [x] Python 3.12.7 (gọi bằng `py`); Python hệ thống đã có đủ gói: OpenCV, NumPy, PyTorch (CPU), scikit-learn, SciPy, Matplotlib, PyYAML, pytest
- [x] Cấu trúc thư mục, `requirements.txt`, `.gitignore`, `README.md`
- [x] Chọn **PyTorch** thay TensorFlow: đã có sẵn trên máy, không phải tải gì; kiến trúc mạng giữ nguyên như báo cáo
- [ ] *(Để sau)* Khởi tạo Git
- [ ] *(Tuỳ chọn)* Xoá thư mục `venv/` rỗng, hoặc chọn Python hệ thống làm interpreter trong VS Code

### Giai đoạn 1 – Thu thập dữ liệu – ĐANG LÀM
**Công cụ ✅**
- [x] `tools/extract_crops.py`: cắt ảnh vùng chuyển động, theo dõi từng xe, tối đa 5 ảnh / xe ở các vị trí khác nhau, bỏ xe chạm mép khung; `--preview`, `--select-roi`
- [x] `tools/label_crops.py`: gán nhãn bằng phím 1–5 / x / z
- [x] `tools/dataset_stats.py`: bảng số ảnh theo nguồn × lớp, cảnh báo lớp < 500 ảnh
- [x] `tools/import_miotcd.py`: nhập MIO-TCD (hoặc dataset phân loại dạng thư mục lớp) vào `data/crops/pub_*`
- [x] `tools/crops_from_yolo.py`: cắt xe từ dataset YOLO (Roboflow) vào `data/crops/pub_*`

**Dữ liệu (người dùng thực hiện)**
- [ ] Thu thập ≥ 6 video giao thông, camera cố định, ≥ 720p, nhiều góc quay và thời điểm, đặt vào `data/raw_videos/` (tên file không dấu)
- [ ] **Để riêng 2–3 video kiểm thử** (~5 phút mỗi video) cho đánh giá đếm; không cắt ảnh train từ các video này
- [ ] Với mỗi video huấn luyện: `extract_crops --preview` để chỉnh tham số, sau đó chạy thật
- [ ] Gán nhãn bằng `label_crops`
- [ ] Bổ sung dataset công khai nếu thiếu xe buýt / xe tải (tự tải MIO-TCD hoặc Roboflow, rồi chạy tool nhập)
- [ ] Đạt ≥ 500 ảnh mỗi lớp (lý tưởng ≥ 1000), kiểm tra bằng `dataset_stats`
- [ ] Chốt dùng 4 lớp hay 5 lớp (có `other`); khuyến nghị 5 lớp vì `other` loại được nhiễu MOG2 (người đi bộ, bóng, khối nhiều xe)

### Giai đoạn 2 – Tiền xử lý và chia dữ liệu
- [x] `train/data.py`: đọc ảnh, BGR → RGB, resize 64×64, chuẩn hoá [0,1]; dùng chung cho train và suy luận
- [x] `tools/split_dataset.py`: chia 70/15/15 **theo video** (tự cân tỉ lệ từng lớp) hoặc **theo thời gian** trong từng video có khoảng đệm; tự loại video có file đếm tay; `pub_*` chỉ vào train; ghi `split.json`
- [x] Tăng cường dữ liệu chỉ cho tập train: lật ngang, xoay ±10°, zoom ±10%, dịch ±5%, độ sáng / tương phản ±20%
- [x] Trọng số lớp (`class_weight`) bù mất cân bằng
- [ ] Chạy `split_dataset` trên dữ liệu thật, kiểm tra mỗi tập đủ các lớp

### Giai đoạn 3 – Huấn luyện Custom CNN
- [x] `train/model.py`, kiến trúc khớp báo cáo (683.716 tham số với 4 lớp):

  | Lớp | Cấu hình | Output |
  |---|---|---|
  | Input | 64×64×3 | 64×64×3 |
  | Conv + ReLU | 32 filter, 3×3, valid | 62×62×32 |
  | MaxPooling | 2×2 | 31×31×32 |
  | Conv + ReLU | 64 filter, 3×3 | 29×29×64 |
  | MaxPooling | 2×2 | 14×14×64 |
  | Conv + ReLU | 128 filter, 3×3 | 12×12×128 |
  | MaxPooling | 2×2 | 6×6×128 |
  | Flatten | | 4608 |
  | Dense + ReLU | 128 | 128 |
  | Dropout | 0,5 | 128 |
  | Dense + Softmax | số lớp (4 hoặc 5) | 4/5 |

- [x] `train/train_cnn.py`: Adam (lr 1e-3), cross-entropy, batch 32, tối đa 50 epoch, dừng sớm (patience 8, lấy lại trọng số tốt nhất), giảm lr khi chững; lưu `models/vehicle_cnn.pt`, `train_log.csv`, biểu đồ, `model_summary.txt`
- [x] `train/kfold.py`: StratifiedGroupKFold 5 fold (nhóm = video × đoạn 30 s), accuracy / P / R / F1 từng fold, trung bình ± độ lệch chuẩn
- [x] `train/evaluate.py`: confusion matrix (ảnh), P / R / F1 từng lớp, ảnh dự đoán sai, thời gian suy luận
- [ ] Train trên dữ liệu thật, đạt accuracy test > 90%, macro F1 > 0,90
- [ ] Chạy K-Fold, phân tích các cặp lớp hay nhầm (dự kiến xe buýt ↔ xe tải) bằng ảnh trong `outputs/misclassified/`
- [ ] *(Nên làm)* So sánh với MobileNetV2 (transfer learning). **Cần tải torchvision và trọng số pretrained**, hỏi trước khi làm

### Giai đoạn 4 – Khối phát hiện MOG2
- [x] `src/detector.py`: MOG2 (history 500, varThreshold 16, detectShadows) → bỏ bóng → open / close / dilate → gộp mảnh vỡ cùng xe → lọc diện tích và tỉ lệ khung → ROI
- [x] Tham số từng video trong `configs/<video>.yaml`; tạo bằng chuột với `tools/make_config.py`
- [ ] Tinh chỉnh tham số trên video thật (xem bảng "Chỉnh khi chạy trên video thật" trong README)
- [ ] Ghi lại các lỗi điển hình cho báo cáo: nhiều xe máy dính thành một khối, xe dừng đèn đỏ bị nhập vào nền, bóng đổ dài

### Giai đoạn 5 – Theo dõi và đếm ✅
- [x] `src/tracker.py`: Hungarian theo khoảng cách tâm có ngưỡng, dự đoán vị trí theo vận tốc, xoá ID sau `max_disappeared` khung hình
- [x] `src/counter.py`: vạch ngang hoặc dọc bất kỳ, đếm 2 chiều hoặc 1 chiều, mỗi ID tối đa 1 lượt; chốt lượt khi đủ phiếu và xe đã đi qua vạch ≥ 10 px (lọc rung); bỏ phiếu mềm; không đếm `other` / độ tin cậy thấp; ghi lý do mọi lượt bị loại
- [x] Unit test cho detector, tracker, counter, model, chia dữ liệu, đánh giá đếm (`tests/`, 31 test)

### Giai đoạn 6 – Ghép ứng dụng ✅
- [x] `src/classifier.py`: phân loại theo batch
- [x] `src/main.py`: chạy pipeline, chỉ phân loại xe đã nằm trọn trong khung (tối đa 5 lần / xe), vẽ khung / nhãn / ID / vạch / bảng đếm / FPS, xuất video, `<video>_counts.csv`, `<video>_rejected.csv`, `<video>_summary.json` (kèm thời gian từng bước); `process_width` để tăng FPS
- [ ] *(Tuỳ chọn)* Giao diện đồ hoạ: chọn video, xem kết quả. Streamlit cần tải thêm; Tkinter có sẵn

### Giai đoạn 7 – Đánh giá toàn hệ thống
- [x] `tools/manual_count.py`: đếm tay bằng bàn phím khi xem video (đúng vạch đếm của config), lưu từng xe
- [x] `tools/eval_counting.py`: bảng Máy / Tay / Sai lệch / Chính xác từng lớp + tổng; ghép từng xe → đúng loại / nhầm loại / bỏ sót / đếm thừa; tự loại khoảng thời gian hệ thống không đếm (warmup)
- [ ] Tạo config cho từng video kiểm thử bằng `make_config`
- [ ] Đếm tay các video kiểm thử (2 người đếm độc lập, so sánh bằng `eval_counting`)
- [ ] Chạy `src.main` và `eval_counting`, lưu bảng kết quả
- [ ] Đo FPS trên CPU bằng `--no-display` (ghi cấu hình máy); thử `process_width: 960 / 640`
- [ ] Thử nghiệm theo điều kiện: đông / vắng, ngày / tối, các góc quay khác nhau
- [ ] Chụp ảnh minh họa các ca lỗi (dùng `<video>_rejected.csv`, `outputs/misclassified/`) và phân tích nguyên nhân

### Giai đoạn 8 – Báo cáo và bảo vệ
- [ ] Viết báo cáo 5 chương:
  1. Mở đầu: lý do chọn đề tài, mục tiêu, phạm vi
  2. Cơ sở lý thuyết: Object Detection và Tracking, CNN, MOG2, Heuristic, các khó khăn (che khuất, kích thước, ánh sáng)
  3. Phân tích và thiết kế: đầu vào / đầu ra, kiến trúc hệ thống, kiến trúc CNN, quy tắc theo dõi / đếm
  4. Triển khai và thực nghiệm: dữ liệu, huấn luyện, K-Fold, confusion matrix, kết quả đếm, FPS
  5. Kết luận: kết quả đạt được, hạn chế, hướng phát triển (YOLO, SORT / DeepSORT / ByteTrack, dữ liệu ban đêm)
- [ ] Slide thuyết trình khoảng 15–20 trang
- [ ] Quay video demo dự phòng (`src.main --output`)
- [ ] Chuẩn bị câu hỏi phản biện:
  - Vì sao dùng MOG2 + CNN mà không dùng YOLO?
  - Con số độ chính xác đếm đo cái gì, có hạn chế gì (lỗi thừa / thiếu bù trừ nhau)?
  - Hệ thống xử lý che khuất và xe đi sát nhau thế nào?
  - Vì sao Flatten ra 4608?
  - Làm sao tránh đếm trùng? Vì sao chỉ phân loại khi xe đã nằm trọn trong khung?

---

## 4. Tiến độ tổng hợp

| Tuần | Công việc chính | Kết quả bàn giao | Trạng thái |
|---|---|---|---|
| 1 | Môi trường, toàn bộ code, kiểm thử trên video giả lập | Mã nguồn + test | ✅ |
| 1–2 | Thu video, cắt ảnh, gán nhãn | Dataset đã gán nhãn | ⏳ |
| 2–3 | Chia dữ liệu, train CNN, K-Fold, đánh giá | Model + confusion matrix | |
| 3–4 | Config từng video, chỉnh MOG2 / tracker | `configs/*.yaml` | |
| 4–5 | Đếm tay, đánh giá đếm, đo FPS, phân tích lỗi | Bảng kết quả | |
| 6 | Báo cáo, slide, demo | Hồ sơ bảo vệ | |

## 5. Rủi ro và cách xử lý

| Rủi ro | Ảnh hưởng | Cách xử lý |
|---|---|---|
| Thiếu ảnh xe buýt và xe tải | CNN phân loại lệch | Dataset công khai (`pub_*`), augmentation, `class_weight` |
| MOG2 gộp nhiều xe máy thành 1 khối | Đếm thiếu xe máy | Đặt vạch ở vùng xe thưa, chỉnh morphology / `merge_ratio`, lớp `other`, đặt camera cao hơn |
| Bóng đổ và thay đổi ánh sáng | Phát hiện giả | `detectShadows`, ROI, lọc diện tích, lớp `other` |
| Xe dừng lâu bị nhập vào nền | Mất track | Đặt vạch xa khu vực dừng đèn, tăng `history` |
| ID bị đổi khi xe đi sát nhau | Đếm trùng hoặc thiếu | Hungarian + dự đoán vận tốc, chỉnh `max_distance`, mỗi ID chỉ đếm 1 lần |
| Xe vừa vào khung bị phân loại sai | Bỏ sót ô tô / xe tải | Chỉ phân loại khi xe nằm trọn trong khung (đã áp dụng) |
| Không đạt 15 FPS trên CPU | Không đạt yêu cầu phi chức năng | `process_width`, phân loại theo batch, tối đa 5 lần / xe, `--no-display` |
| Rò rỉ dữ liệu train / test | Accuracy ảo | Chia theo video / thời gian, K-Fold theo nhóm, tối đa 5 ảnh / xe |

## 6. Danh sách sản phẩm cuối cùng
- [x] Mã nguồn hoàn chỉnh kèm `README.md` hướng dẫn chạy và bộ test
- [ ] Mã nguồn trên Git
- [ ] Dataset đã gán nhãn và bảng thống kê
- [ ] Model `vehicle_cnn.pt` và `labels.json`
- [ ] Kết quả: biểu đồ huấn luyện, bảng K-Fold, confusion matrix, bảng đếm xe, số đo FPS
- [ ] Video demo đầu ra
- [ ] Báo cáo đồ án và slide thuyết trình
