"""Đọc ảnh, tiền xử lý dùng chung cho huấn luyện và suy luận, nạp dataset theo tập."""
import re
from pathlib import Path

import cv2
import numpy as np

IMG_SIZE = 64
ALL_CLASSES = ["motorbike", "car", "bus", "truck", "other"]  # thứ tự chuẩn của các lớp
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = ("train", "val", "test")
_CROP_NAME = re.compile(r"^(?P<video>.+)_f(?P<frame>\d+)_\d+$")  # tên ảnh do extract_crops đặt


def imread(path):
    """cv2.imread an toàn với đường dẫn có dấu tiếng Việt trên Windows."""
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path, img):
    ok, buf = cv2.imencode(Path(path).suffix or ".jpg", img)
    if ok:
        buf.tofile(str(path))
    return ok


def list_images(folder):
    folder = Path(folder)
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def prepare(img_bgr):
    """Ảnh BGR (OpenCV) kích thước bất kỳ -> RGB uint8 64x64x3."""
    h, w = img_bgr.shape[:2]
    interp = cv2.INTER_AREA if h > IMG_SIZE and w > IMG_SIZE else cv2.INTER_LINEAR
    img = cv2.resize(img_bgr, (IMG_SIZE, IMG_SIZE), interpolation=interp)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def to_tensor(batch):
    """uint8 (N, 64, 64, 3) RGB -> tensor float32 (N, 3, 64, 64) chuẩn hoá về [0, 1]."""
    import torch  # nạp muộn: các tool chỉ xử lý ảnh không phải chờ ~5 s nạp torch

    x = torch.from_numpy(np.ascontiguousarray(batch)).permute(0, 3, 1, 2).contiguous()
    return x.float().div_(255.0)


def video_and_frame(path):
    """Tách (tên video, số khung hình) từ tên '<video>_f000123_0.jpg'; ảnh nguồn khác -> (None, None)."""
    m = _CROP_NAME.match(Path(path).stem)
    return (m["video"], int(m["frame"])) if m else (None, None)


def group_of(path, group_frames=900):
    """Nhóm chống rò rỉ cho K-Fold: ảnh cùng video, cùng đoạn group_frames khung hình (~30 s) chung nhóm."""
    video, frame = video_and_frame(path)
    return f"{video}#{frame // group_frames}" if video else Path(path).stem


def class_counts(split_dir, classes=ALL_CLASSES):
    return {c: len(list_images(Path(split_dir) / c)) for c in classes}


def resolve_classes(dataset_dir, requested=None):
    """Các lớp dùng để train: theo --classes, hoặc mọi lớp chuẩn có ảnh trong tập train."""
    counts = class_counts(Path(dataset_dir) / "train", requested or ALL_CLASSES)
    if requested:
        empty = [c for c in requested if counts[c] == 0]
        if empty:
            raise SystemExit(f"Lớp không có ảnh trong tập train: {', '.join(empty)}")
        return list(requested)
    classes = [c for c in ALL_CLASSES if counts[c] > 0]
    if len(classes) < 2:
        raise SystemExit(f"Cần ít nhất 2 lớp có ảnh trong {Path(dataset_dir) / 'train'} "
                         "(chạy py -m tools.split_dataset trước)")
    return classes


def load_split(dataset_dir, split, classes):
    """Nạp một tập -> (X uint8 (N, 64, 64, 3), y int64 (N,), danh sách đường dẫn)."""
    xs, ys, paths = [], [], []
    for label, cls in enumerate(classes):
        for p in list_images(Path(dataset_dir) / split / cls):
            img = imread(p)
            if img is None:
                print(f"  bỏ qua ảnh hỏng: {p}")
                continue
            xs.append(prepare(img))
            ys.append(label)
            paths.append(p)
    X = np.stack(xs) if xs else np.zeros((0, IMG_SIZE, IMG_SIZE, 3), np.uint8)
    return X, np.array(ys, dtype=np.int64), paths
