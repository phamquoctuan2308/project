# Nhận diện và đếm phương tiện giao thông (Custom CNN + Heuristic)

Từ video camera giao thông cố định, hệ thống tự động **phát hiện → phân loại → theo dõi → đếm** xe máy, ô tô, xe buýt, xe tải.

```
Video ─► MOG2 (vùng chuyển động) ─► cắt ảnh 64×64 ─► Custom CNN (loại xe)
      ─► Centroid Tracker (gán ID) ─► Vạch ảo (mỗi ID đếm 1 lần) ─► video kết quả + CSV thống kê
```

| Khối | File |
|---|---|
| Phát hiện MOG2, cắt ảnh dùng chung | `src/detector.py` |
| Custom CNN, tiền xử lý dùng chung | `train/model.py`, `train/data.py` |
| Phân loại khi chạy | `src/classifier.py` |
| Theo dõi (Hungarian + vận tốc) | `src/tracker.py` |
| Đếm qua vạch ảo, bỏ phiếu loại xe | `src/counter.py` |
| Chạy toàn bộ trên một video | `src/main.py` |
| Cấu hình từng video | `src/config.py`, `configs/*.yaml` |

## Cài đặt

Python 3.12 (Windows gọi bằng `py`). Các gói cần có nằm trong `requirements.txt` (OpenCV, NumPy, PyTorch, scikit-learn, SciPy, Matplotlib, PyYAML, pytest).
Máy phát triển đã có đủ các gói này nên **không cần cài thêm**. Máy khác thì chạy `py -m pip install -r requirements.txt`.

Mọi lệnh chạy từ thư mục gốc `prj/` và phải dùng dạng `py -m <module>`.

Kiểm tra nhanh: `py -m pytest` (31 test, khoảng 5 giây).

## Quy trình từ đầu đến cuối

### 1. Thu thập và gán nhãn dữ liệu
Đặt video vào `data/raw_videos/` (camera cố định, ≥ 720p, tên file không dấu). **Để riêng 2–3 video ~5 phút chỉ dùng để đánh giá đếm.** Không cắt ảnh huấn luyện từ các video này.

```bash
py -m tools.extract_crops --video data/raw_videos/v1.mp4 --preview      # xem mask MOG2, chỉnh --min-area / ROI
py -m tools.extract_crops --video data/raw_videos/v1.mp4 --select-roi   # cắt ảnh -> data/crops/v1/
py -m tools.label_crops --dir data/crops/v1                             # gán nhãn bằng phím 1-5, x = bỏ
py -m tools.dataset_stats                                               # bảng số ảnh theo nguồn x lớp
```
`extract_crops` theo dõi từng xe và lưu tối đa 5 ảnh mỗi xe, ở các vị trí khác nhau. Xe đang chạm mép khung hình (bị cắt dở) không được lưu.

Phím trong `label_crops`: `1` motorbike · `2` car · `3` bus · `4` truck · `5` other (người đi bộ, bóng đổ, nhiều xe dính nhau) · `x` bỏ ảnh hỏng · `z` hoàn tác.

**Bổ sung dataset công khai** (tự tải về trước), kết quả nằm trong `data/crops/pub_<tên>/` và **chỉ được đưa vào tập train**:
```bash
py -m tools.import_miotcd --src <MIO-TCD-Classification>/train --max-per-class 1500   # thêm --with-other nếu cần
py -m tools.crops_from_yolo --src <thư_mục_export_YOLO_Roboflow> --name vnvehicle
```

### 2. Chia train / val / test
```bash
py -m tools.split_dataset                  # mỗi video nằm trọn một tập, tự cân tỉ lệ 70/15/15 theo từng lớp
py -m tools.split_dataset --mode time      # ít video: chia mỗi video theo thời gian, bỏ ảnh sát điểm cắt
```
Video đã có file đếm tay trong `data/ground_truth/` tự động bị loại. `data/dataset/` được tạo lại hoàn toàn mỗi lần chạy (dùng `--overwrite` khi đã có ảnh). Cách chia được ghi vào `data/dataset/split.json`.

### 3. Huấn luyện và đánh giá CNN
```bash
py -m train.train_cnn                                   # -> models/vehicle_cnn.pt, outputs/train_log.csv, biểu đồ
py -m train.train_cnn --classes motorbike car bus truck # train 4 lớp (bỏ other)
py -m train.evaluate                                    # tập test: accuracy, P/R/F1, confusion matrix, ảnh sai
py -m train.kfold --folds 5                             # 5-Fold theo nhóm, trung bình ± độ lệch chuẩn
```
Mặc định dùng mọi lớp có ảnh trong `data/dataset/train/` (5 lớp nếu có `other`). Train trên CPU: Adam, lr 1e-3, batch 32, dừng sớm theo val_loss, tăng cường dữ liệu (lật, xoay ±10°, zoom ±10%, độ sáng / tương phản ±20%), trọng số lớp.

### 4. Cấu hình video và đếm xe
```bash
py -m tools.make_config --video data/raw_videos/test1.mp4         # chọn ROI, click 2 điểm vạch đếm -> configs/test1.yaml
py -m src.main --video data/raw_videos/test1.mp4 --config configs/test1.yaml --output outputs/test1_result.mp4
py -m src.main --video data/raw_videos/test1.mp4 --config configs/test1.yaml --no-display     # đo FPS
```
Kết quả trong `outputs/`:
- `test1_counts.csv`: từng xe đã đếm.
- `test1_rejected.csv`: các lượt vượt vạch bị loại, kèm lý do.
- `test1_summary.json`: số đếm theo lớp và chiều, FPS, thời gian từng bước.

Ý nghĩa từng tham số xem trong `configs/example.yaml`. Mọi toạ độ tính theo pixel của video gốc.

### 5. Đếm tay và đánh giá đếm
```bash
py -m tools.manual_count --video data/raw_videos/test1.mp4 --config configs/test1.yaml          # -> data/ground_truth/test1.csv
py -m tools.eval_counting --pred outputs/test1_counts.csv --gt data/ground_truth/test1.csv
```
`manual_count`: bấm `1`–`4` khi tâm xe chạm vạch, `d` đổi chiều, `[` `]` chỉnh tốc độ phát, `space` dừng, `z` xoá lượt cuối.

`eval_counting` cho ra:
- bảng **Máy / Tay / Sai lệch / Chính xác** cho từng lớp và tổng, theo công thức `(1 − |máy − tay| / tay) × 100%`;
- kết quả ghép từng xe: số xe đúng loại, nhầm loại, bỏ sót, đếm thừa.

Để so sánh 2 người đếm tay: người thứ hai lưu vào `--out data/ground_truth/test1_nguoi2.csv`, rồi chạy `eval_counting` với file đó làm `--pred`.

## Quy tắc chính
- **Đếm:** mỗi ID chỉ được đếm một lần, khi tâm xe đi từ phía này sang phía kia của vạch. Với vạch `y = Y`: khung trước `y < Y` và khung hiện tại `y ≥ Y` thì tính là chiều `down`. Lượt đếm được chốt khi xe đã được phân loại ít nhất 2 lần và đã đi qua vạch ≥ 10 px; cách này lọc được rung lắc và vật đứng yên trên vạch.
- **Phân loại:** chỉ phân loại khi xe đã nằm trọn trong khung hình hoặc ROI, giống điều kiện của ảnh train. Mỗi xe được phân loại tối đa 5 lần, cách nhau 3 khung hình. Loại xe cuối cùng là lớp có tổng xác suất lớn nhất qua các lần (bỏ phiếu mềm). Xe có kết quả `other` hoặc độ tin cậy < 0,3 không được đếm.
- **Chống rò rỉ dữ liệu:** chia dataset theo video. K-Fold nhóm các ảnh cùng video trong cùng đoạn ~30 giây. Video dùng để đánh giá đếm không bao giờ được dùng để train.

## Kiến trúc CNN (khớp báo cáo)
```
Input 64x64x3
Conv 3x3, 32  + ReLU -> 62x62x32 -> MaxPool 2x2 -> 31x31x32
Conv 3x3, 64  + ReLU -> 29x29x64 -> MaxPool 2x2 -> 14x14x64
Conv 3x3, 128 + ReLU -> 12x12x128 -> MaxPool 2x2 -> 6x6x128
Flatten 4608 -> Dense 128 + ReLU -> Dropout 0.5 -> Dense + Softmax (4 hoặc 5 lớp)
Tổng tham số: 683.716 (4 lớp) / 683.845 (5 lớp)
```
Mô hình viết bằng PyTorch. Softmax nằm trong hàm loss khi train và được áp dụng tường minh khi suy luận. `outputs/model_summary.txt` in bảng từng lớp để đưa vào báo cáo.

## Chỉnh khi chạy trên video thật
| Hiện tượng | Thử chỉnh |
|---|---|
| Nhiều nhiễu nhỏ (lá cây, bóng) | tăng `detector.min_area`, `var_threshold`; khoanh `roi` sát mặt đường |
| Nhiều xe máy dính thành một khối | đặt vạch đếm ở chỗ xe thưa hơn; giảm `merge_ratio`; camera đặt cao hơn |
| Một xe bị tách thành 2 khung | tăng `merge_ratio` (ví dụ 0.3 → gộp dễ hơn) |
| ID bị nhảy, đếm trùng | giảm `tracker.max_distance`; tăng `max_disappeared` nếu xe hay bị mất dấu vài khung hình |
| FPS thấp | `process_width: 960` (hoặc 640), không dùng `--output`, chạy `--no-display` |
| Lượt đếm bị loại nhiều | xem cột `reason` trong `<video>_rejected.csv` |
