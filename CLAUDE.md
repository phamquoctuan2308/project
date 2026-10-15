# CLAUDE.md

Hướng dẫn cho Claude Code khi làm việc trong repo này.

## Dự án
Đồ án: **Ứng dụng AI nhận diện và đếm phương tiện giao thông bằng Custom CNN + Heuristic.**
Pipeline: `MOG2 (phát hiện vùng chuyển động) → Custom CNN 64×64 (phân loại) → Centroid Tracker (gán ID) → Vạch ảo (đếm)`.
Các lớp: `motorbike`, `car`, `bus`, `truck`, cùng lớp lọc nhiễu `other` (tuỳ chọn khi train).
Kế hoạch và checklist tiến độ nằm trong [plan.md](plan.md); hướng dẫn sử dụng đầy đủ nằm trong [README.md](README.md). Khi hoàn thành một đầu việc, đánh dấu `[x]` trong plan.md.

## Ngôn ngữ
- Trao đổi với người dùng bằng **tiếng Việt**.
- Tên biến, hàm, class, file viết bằng tiếng Anh. Docstring và comment có thể viết tiếng Việt, giữ ngắn gọn.
- Chữ vẽ lên khung hình bằng `cv2.putText` chỉ dùng ASCII (OpenCV không hiển thị dấu tiếng Việt).

## Môi trường
- Windows 11, Python 3.12.7 gọi bằng `py` (lệnh `python` không có trong PATH). Python **hệ thống** đã có đủ gói: OpenCV 5.0, NumPy, **PyTorch 2.13 bản CPU**, scikit-learn, SciPy, Matplotlib, PyYAML, pytest.
- Thư mục `venv/` rỗng (tạo nhầm, chưa cài gói). Chạy mọi thứ bằng Python hệ thống.
- **Không tự cài hay tải gói / dữ liệu** (pip, dataset, model). Người dùng muốn tự quyết khi nào tải; cần gói mới thì hỏi trước.
- Framework CNN: **PyTorch** (đã có sẵn, không cần tải). torch không có CUDA nên train và suy luận đều chạy CPU (20 nhân).
- Import torch mất ~5 s: các tool chỉ xử lý ảnh / file (`tools/*` trừ những tool cần mô hình) không được import torch ở mức module; `train/data.py` chỉ nạp torch bên trong `to_tensor`.
- Mỗi script có `for stream in (sys.stdout, sys.stderr): stream.reconfigure(encoding="utf-8")` đầu `main()` vì console Windows mặc định cp1252, in tiếng Việt sẽ lỗi.
- Yêu cầu phi chức năng: **15–20 FPS trên CPU**. Đo trên video giả lập 720p: ~34 FPS (ghi video kết quả), ~68 FPS với `process_width: 960`.

## Cấu trúc thư mục
```
prj/
├── data/                       # KHÔNG commit (trừ ground_truth/)
│   ├── raw_videos/             # video gốc
│   ├── crops/<nguồn>/          # ảnh cắt + crops.csv; sau gán nhãn: <nguồn>/<lớp>/, ảnh bỏ: _discard/
│   │                           # nguồn "pub_*" = dataset công khai -> chỉ vào tập train
│   ├── dataset/{train,val,test}/<lớp>/ + split.json   # do tools.split_dataset tạo lại hoàn toàn
│   └── ground_truth/<video>.csv    # đếm tay: time_s,class,direction
├── src/                        # pipeline suy luận
│   ├── config.py               # DEFAULT_CONFIG + load_config (khớp configs/example.yaml)
│   ├── detector.py             # MotionDetector (MOG2), crop_box, touches_border, merge_boxes
│   ├── classifier.py           # VehicleClassifier.predict_proba(crops)
│   ├── tracker.py              # Track, CentroidTracker (Hungarian + vận tốc)
│   ├── counter.py              # LineCounter (vượt vạch, chốt lượt, lý do loại)
│   └── main.py                 # CLI chạy toàn bộ pipeline trên video
├── train/
│   ├── data.py                 # ALL_CLASSES, prepare/to_tensor (dùng chung train + suy luận), load_split
│   ├── model.py                # VehicleCNN, summary, save/load_checkpoint, predict_proba
│   ├── train_cnn.py            # fit(), augment(), lưu models/vehicle_cnn.pt
│   ├── kfold.py                # StratifiedGroupKFold
│   └── evaluate.py             # confusion matrix, P/R/F1, ảnh sai, thời gian suy luận
├── tools/                      # extract_crops, label_crops, dataset_stats, split_dataset, import_miotcd,
│                               # crops_from_yolo, make_config, manual_count, eval_counting
├── configs/example.yaml        # mẫu cấu hình có chú thích; configs/<video>.yaml cho từng video
├── tests/                      # pytest (31 test)
├── models/                     # vehicle_cnn.pt + labels.json (KHÔNG commit file .pt)
├── outputs/                    # log, biểu đồ, CSV / JSON kết quả, video kết quả (KHÔNG commit)
└── report/                     # báo cáo, slide
```

## Lệnh thường dùng
Chạy từ thư mục gốc `prj/`, luôn dùng dạng `py -m <module>`:
```bash
py -m pytest                                                     # 31 test, ~5 s
py -m tools.extract_crops --video data/raw_videos/v1.mp4 --preview
py -m tools.label_crops --dir data/crops/v1
py -m tools.split_dataset [--mode time] [--overwrite]
py -m train.train_cnn [--classes motorbike car bus truck]
py -m train.evaluate
py -m train.kfold --folds 5
py -m tools.make_config --video data/raw_videos/test1.mp4
py -m src.main --video data/raw_videos/test1.mp4 --config configs/test1.yaml --output outputs/test1_result.mp4
py -m tools.manual_count --video data/raw_videos/test1.mp4 --config configs/test1.yaml
py -m tools.eval_counting --pred outputs/test1_counts.csv --gt data/ground_truth/test1.csv
```
Mọi tool đều nhận đường dẫn qua tham số (`--dataset`, `--crops`, `--out-dir`, `--outputs`, `--model`...), nên có thể chạy thử trong thư mục tạm mà không đụng vào `data/`.

## Quy ước kỹ thuật
- **Kiến trúc CNN cố định theo báo cáo:** 3 khối Conv 3×3 (32/64/128, không padding, ReLU) + MaxPool 2×2 → Flatten 4608 → Dense 128 ReLU → Dropout 0,5 → Dense + Softmax. Tổng 683.716 tham số (4 lớp). Không đổi kiến trúc nếu chưa hỏi người dùng.
- Tiền xử lý dùng chung: `train.data.prepare` (BGR → RGB, resize 64×64, uint8) + `to_tensor` (/255, NCHW). Cắt ảnh dùng chung: `src.detector.crop_box` với `CROP_PAD = 0.1` (cả lúc tạo dataset lẫn lúc suy luận).
- **Điều kiện ảnh phải khớp giữa train và suy luận:** `extract_crops` bỏ xe chạm mép khung / ROI, nên `main` cũng chỉ phân loại xe đã nằm trọn trong vùng xét (`touches_border`). Ngoại lệ: xe đã vượt vạch mà chưa đủ phiếu. Bỏ quy tắc này làm sót nhiều ô tô / xe tải (đã đo: 65% → 99%).
- Thứ tự lớp: `train.data.ALL_CLASSES` là thứ tự chuẩn; thứ tự của mô hình nằm trong checkpoint `.pt` (bản sao ở `models/labels.json`).
- Tham số theo từng video (ROI, vạch đếm, chiều đếm, ngưỡng MOG2 / tracker / counter) nằm trong `configs/*.yaml`, toạ độ theo pixel video gốc. Thêm khoá mới thì sửa đồng thời `src/config.py` và `configs/example.yaml` (test kiểm tra hai nơi khớp nhau).
- Bounding box thống nhất dạng `(x, y, w, h)`, tâm là `(x + w//2, y + h//2)`.
- Quy tắc đếm: vạch ngang thì `y_prev < Y ≤ y_curr` là `down`, ngược lại là `up` (vạch dọc: `left` / `right`); mỗi ID tối đa một lượt. Lượt được chốt khi đủ `min_votes` lần phân loại và đã đi qua vạch ≥ `min_travel` px (hoặc track mất / chờ quá `max_wait`). Loại xe = bỏ phiếu mềm (tổng xác suất); không đếm `other` và độ tin cậy < `min_confidence`. Lượt bị loại ghi vào `<video>_rejected.csv` kèm lý do.
- Seed cố định (`random`, `numpy`, `torch`) trong train và K-Fold.

## Nguyên tắc về dữ liệu và kết quả
- **Chia train / val / test theo video nguồn** (`--mode video`), hoặc theo thời gian trong từng video có khoảng đệm (`--mode time`). Không chia ngẫu nhiên từng ảnh.
- Video dùng để đánh giá đếm xe **không được** dùng để lấy ảnh huấn luyện. `split_dataset` tự loại nguồn trùng tên với file trong `data/ground_truth/`.
- Dataset công khai (`pub_*`) chỉ vào tập train; val / test là dữ liệu của nhóm.
- **Không bịa hoặc chép số liệu.** Mọi con số trong báo cáo (accuracy, F1, bảng đếm, FPS) phải do script trong repo tạo ra và lưu trong `outputs/`. Kết quả trên video giả lập chỉ là kiểm thử phần mềm, không phải kết quả đồ án. Con số 96,8% hay 212/219 trong tài liệu tham khảo không phải kết quả của dự án này.
- Khi báo cáo độ chính xác đếm, luôn kèm sai số **từng lớp** và kết quả ghép từng xe (bỏ sót / đếm thừa / nhầm loại), không chỉ con số tổng.

## Khi làm việc
- Sau khi sửa code, chạy `py -m pytest`.
- Thay đổi ảnh hưởng tới FPS thì đo lại bằng `src.main --no-display` (xem `stage_ms_per_frame` trong summary) và báo cho người dùng.
- Không commit `data/` (trừ `data/ground_truth/`), `outputs/`, `venv/`, file `.pt`.
