"""Đếm tay (ground truth) bằng bàn phím khi xem video, để đánh giá hệ thống đếm.

Ví dụ:
    py -m tools.manual_count --video data/raw_videos/test1.mp4 --config configs/test1.yaml
    py -m tools.manual_count --video data/raw_videos/test1.mp4 --config configs/test1.yaml --out data/ground_truth/test1_nguoi2.csv
Bấm phím ngay khi tâm xe chạm vạch vàng (dùng đúng config của video để trùng vạch với hệ thống):
    1 motorbike | 2 car | 3 bus | 4 truck   ghi 1 xe theo chiều đang chọn
    d đổi chiều | space dừng / chạy | n khung hình kế (khi dừng) | [ ] chậm / nhanh | z xoá lượt cuối | q/Esc thoát
Mỗi lần bấm đều lưu ngay vào data/ground_truth/<video>.csv (time_s, class, direction).
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

from src.config import default_line, load_config
from src.counter import LineCounter

KEYS = {ord("1"): "motorbike", ord("2"): "car", ord("3"): "bus", ord("4"): "truck"}
SPEEDS = [0.1, 0.25, 0.5, 1.0, 2.0]
WINDOW = "Dem tay: 1-4 = ghi xe | d = doi chieu | space = dung | [ ] = toc do | z = xoa | q = thoat"


def save(path, events):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["time_s", "class", "direction"])
        writer.writeheader()
        writer.writerows(sorted(events, key=lambda e: e["time_s"]))


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--config", help="config của video (để vẽ đúng vạch đếm)")
    p.add_argument("--out", help="mặc định data/ground_truth/<tên_video>.csv")
    p.add_argument("--force", action="store_true", help="ghi đè file đã có")
    args = p.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else Path("data/ground_truth") / f"{video.stem}.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} đã tồn tại (dùng --out tên khác hoặc --force để ghi đè)")
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = load_config(args.config)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {video}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25

    line = cfg["line"] or default_line(W, H)
    names = LineCounter(line, []).names
    direction = cfg["direction"] if cfg["direction"] in names else names[0]
    p1, p2 = tuple(int(v) for v in line[0]), tuple(int(v) for v in line[1])
    scale = min(1.0, 1280 / W)
    events, speed_i = [], SPEEDS.index(1.0)
    frame, frame_idx, paused, step = None, 0, False, False
    while True:
        if frame is None or not paused or step:
            ok, nxt = cap.read()
            if not ok:
                break
            frame, frame_idx, step = nxt, frame_idx + 1, False
        t = frame_idx / fps
        vis = frame.copy()
        cv2.line(vis, p1, p2, (0, 255, 255), 2)
        counts = {c: sum(e["class"] == c for e in events) for c in KEYS.values()}
        lines = [f"t = {t:7.2f}s   speed x{SPEEDS[speed_i]}   {'PAUSED' if paused else ''}",
                 f"direction: {direction}  (d to change)",
                 "  ".join(f"{c}:{n}" for c, n in counts.items()) + f"   total:{len(events)}"]
        vis[6:6 + 30 * len(lines) + 10, 6:760] = (vis[6:6 + 30 * len(lines) + 10, 6:760] * 0.35).astype(np.uint8)
        for i, s in enumerate(lines):
            cv2.putText(vis, s, (16, 32 + 30 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.imshow(WINDOW, cv2.resize(vis, None, fx=scale, fy=scale) if scale < 1 else vis)

        key = cv2.waitKey(0 if paused else max(1, int(1000 / (fps * SPEEDS[speed_i])))) & 0xFF
        if key in (ord("q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key in KEYS:
            events.append({"time_s": round(t, 2), "class": KEYS[key], "direction": direction})
            save(out, events)
        elif key == ord("z") and events:
            events.pop()
            save(out, events)
        elif key == ord("d"):
            direction = names[1] if direction == names[0] else names[0]
        elif key == ord(" "):
            paused = not paused
        elif key == ord("n"):
            step = True
        elif key == ord("["):
            speed_i = max(0, speed_i - 1)
        elif key == ord("]"):
            speed_i = min(len(SPEEDS) - 1, speed_i + 1)
    cap.release()
    cv2.destroyAllWindows()
    save(out, events)
    print(f"Đã lưu {len(events)} lượt vào {out} (dừng ở giây {frame_idx / fps:.1f})")
    for c in KEYS.values():
        print(f"  {c:<10} {sum(e['class'] == c for e in events)}")


if __name__ == "__main__":
    main()
