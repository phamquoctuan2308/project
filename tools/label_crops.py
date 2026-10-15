"""Gán nhãn ảnh cắt bằng bàn phím: ảnh được chuyển vào thư mục con theo lớp.

Ví dụ:
    py -m tools.label_crops --dir data/crops/v1

Phím: 1 motorbike | 2 car | 3 bus | 4 truck | 5 other (nhiễu, nhiều xe dính nhau)
      x bỏ (ảnh hỏng) | s/space bỏ qua | z hoàn tác | q/Esc thoát
Kết quả: data/crops/v1/<lớp>/*.jpg, ảnh bị bỏ vào data/crops/v1/_discard/.
Giữ ảnh theo từng video để Giai đoạn 2 chia train/val/test theo video.
"""
import argparse
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

from train.data import ALL_CLASSES, imread, list_images

KEYS = {**{ord(str(i + 1)): c for i, c in enumerate(ALL_CLASSES)}, ord("x"): "_discard"}
HELP = "1 moto  2 car  3 bus  4 truck  5 other  x bo  s skip  z undo  q quit"


def render(img, idx, total, counts):
    h, w = img.shape[:2]
    scale = 320 / max(h, w)
    img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    canvas = np.full((420, 640, 3), 40, np.uint8)
    y0, x0 = (340 - img.shape[0]) // 2 + 10, (640 - img.shape[1]) // 2
    canvas[y0:y0 + img.shape[0], x0:x0 + img.shape[1]] = img
    cv2.putText(canvas, f"{idx + 1}/{total}  {w}x{h}px", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    stats = "  ".join(f"{k}:{v}" for k, v in counts.items())
    cv2.putText(canvas, stats, (10, 375), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    cv2.putText(canvas, HELP, (10, 405), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return canvas


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", required=True, help="thư mục ảnh cắt của một video")
    root = Path(p.parse_args().dir)
    files = list_images(root)
    if not files:
        raise SystemExit(f"Không còn ảnh chưa gán nhãn trong {root}")

    counts = {c: len(list_images(root / c)) for c in KEYS.values()}
    history = []  # (đường dẫn gốc, đường dẫn mới, lớp) để hoàn tác
    i = 0
    while i < len(files):
        img = imread(files[i])
        if img is None:
            i += 1
            continue
        cv2.imshow("label_crops", render(img, i, len(files), counts))
        key = cv2.waitKey(0) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in KEYS:
            cls = KEYS[key]
            (root / cls).mkdir(exist_ok=True)
            dst = root / cls / files[i].name
            shutil.move(str(files[i]), str(dst))
            history.append((files[i], dst, cls))
            counts[cls] += 1
            i += 1
        elif key in (ord("s"), ord(" ")):
            i += 1
        elif key == ord("z") and history:
            src, dst, cls = history.pop()
            shutil.move(str(dst), str(src))
            counts[cls] -= 1
            i = files.index(src)
    cv2.destroyAllWindows()
    print("Đã gán nhãn:", ", ".join(f"{k}={v}" for k, v in counts.items()))
    print(f"Còn lại chưa gán: {len(list_images(root))} ảnh")


if __name__ == "__main__":
    main()
