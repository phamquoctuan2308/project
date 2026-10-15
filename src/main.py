"""Chạy toàn bộ pipeline trên video: phát hiện (MOG2) -> phân loại (CNN) -> theo dõi -> đếm qua vạch ảo.

Ví dụ:
    py -m src.main --video data/raw_videos/test1.mp4 --config configs/test1.yaml --output outputs/test1_result.mp4
    py -m src.main --video data/raw_videos/test1.mp4 --config configs/test1.yaml --no-display   # đo FPS
Kết quả trong outputs/:
    <video>_counts.csv    từng xe đã đếm (time_s, frame, track_id, class, confidence, direction)
    <video>_rejected.csv  các lượt vượt vạch bị loại và lý do (để phân tích lỗi)
    <video>_summary.json  số đếm theo lớp / chiều, FPS, thời gian từng bước, cấu hình đã dùng
Phím khi hiển thị: space = dừng / chạy, q = thoát.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from src.classifier import VehicleClassifier
from src.config import default_line, load_config
from src.counter import LineCounter
from src.detector import MotionDetector, crop_box, touches_border
from src.tracker import CentroidTracker

WINDOW = "Dem phuong tien (space = dung, q = thoat)"
FONT = cv2.FONT_HERSHEY_SIMPLEX
COLORS = {"motorbike": (0, 200, 255), "car": (0, 220, 0), "bus": (255, 128, 0),
          "truck": (60, 60, 255), "other": (160, 160, 160)}  # BGR


class StageTimer:
    """Cộng dồn thời gian từng bước để biết bước nào làm chậm FPS."""

    def __init__(self):
        self.totals = {}
        self._t = time.perf_counter()

    def lap(self, name):
        now = time.perf_counter()
        self.totals[name] = self.totals.get(name, 0.0) + now - self._t
        self._t = now


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--config", help="file YAML (xem configs/example.yaml); bỏ trống = cấu hình mặc định")
    p.add_argument("--model", default="models/vehicle_cnn.pt")
    p.add_argument("--output", help="lưu video kết quả (.mp4); bỏ trống = không lưu")
    p.add_argument("--out-dir", default="outputs", help="thư mục ghi CSV / JSON kết quả")
    p.add_argument("--no-display", action="store_true", help="không mở cửa sổ (chạy nhanh hơn, dùng để đo FPS)")
    p.add_argument("--max-frames", type=int, default=0, help="dừng sau N khung hình (0 = hết video)")
    p.add_argument("--threads", type=int, default=0, help="số luồng CPU cho CNN (0 = mặc định của torch)")
    return p.parse_args()


def draw(vis, tracks, counter, classes, roi, fps, warming):
    if roi:
        x, y, w, h = roi
        cv2.rectangle(vis, (x, y), (x + w, y + h), (255, 160, 0), 1)
    cv2.line(vis, tuple(int(v) for v in counter.p1), tuple(int(v) for v in counter.p2), (0, 255, 255), 2)
    for t in tracks:
        label, conf = t.label(classes)
        color = COLORS.get(label, (255, 255, 255))
        x, y, w, h = t.box
        cv2.rectangle(vis, (x, y), (x + w, y + h), color, 3 if t.id in counter.seen else 2)
        cv2.circle(vis, t.centroid, 4, color, -1)
        text = f"#{t.id} {label} {conf:.2f}" if label else f"#{t.id}"
        cv2.putText(vis, text, (x, max(15, y - 6)), FONT, 0.5, color, 2)

    totals = counter.totals()
    lines = [f"{c:<10}{totals[c]:>5}" for c in counter.counted_labels]
    lines += [f"{'TOTAL':<10}{sum(totals.values()):>5}", f"FPS {fps:5.1f}"]
    if warming:
        lines.append("learning background...")
    h = 30 * len(lines) + 14
    vis[6:6 + h, 6:290] = (vis[6:6 + h, 6:290] * 0.35).astype(np.uint8)
    for i, s in enumerate(lines):
        cv2.putText(vis, s, (18, 34 + 30 * i), FONT, 0.75, COLORS.get(s.split()[0], (255, 255, 255)), 2)
    return vis


def write_csv(path, rows, fields):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    args = parse_args()
    video = Path(args.video)
    if not Path(args.model).exists():
        raise SystemExit(f"Chưa có mô hình {args.model} (chạy py -m train.train_cnn trước)")
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_config(args.config)

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {video}")
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video.name}: {W}x{H}, {fps:.1f} FPS, {total} khung hình")

    classifier = VehicleClassifier(args.model)
    classes = classifier.classes
    print("Lớp của mô hình:", ", ".join(classes))

    # Phát hiện trên ảnh thu nhỏ (nếu process_width < W) rồi quy đổi box về toạ độ gốc
    scale = cfg["process_width"] / W if 0 < cfg["process_width"] < W else 1.0
    det, roi = cfg["detector"], cfg["roi"]
    detector = MotionDetector(
        history=det["history"], var_threshold=det["var_threshold"],
        min_area=det["min_area"] * scale ** 2, max_area=det["max_area"] * scale ** 2 or None,
        min_aspect=det["min_aspect"], max_aspect=det["max_aspect"], merge_ratio=det["merge_ratio"],
        roi=[int(v * scale) for v in roi] if roi else None)
    tracker = CentroidTracker(**cfg["tracker"])
    counter = LineCounter(cfg["line"] or default_line(W, H), classes, direction=cfg["direction"], **cfg["counter"])
    classify_every, max_votes = cfg["classifier"]["classify_every"], cfg["classifier"]["max_votes"]
    warmup = cfg["warmup_frames"]
    region = roi or (0, 0, W, H)

    def needs_classify(t):
        if t.votes >= max_votes or (t.last_classified is not None and frame_idx - t.last_classified < classify_every):
            return False
        # Chỉ phân loại khi xe nằm trọn trong vùng xét (giống ảnh train: xe chạm mép bị loại khỏi dataset).
        # Ngoại lệ: xe đã vượt vạch mà chưa đủ phiếu (vd xe buýt dài hơn ROI) vẫn được phân loại để không bị sót.
        return not touches_border(t.box, region) or (t.id in counter.seen and t.votes < counter.min_votes)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))

    timer = StageTimer()
    frame_idx, fps_live, paused = 0, 0.0, False
    t_start = t_prev = time.perf_counter()
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break
        frame_idx += 1
        timer.lap("read")

        small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA) if scale < 1 else frame
        boxes, _ = detector.detect(small)
        if scale < 1:
            boxes = [tuple(int(round(v / scale)) for v in b) for b in boxes]
        timer.lap("detect")

        warming = frame_idx <= warmup  # MOG2 đang học nền: chưa theo dõi / đếm
        tracks = []
        if not warming:
            tracks = tracker.update(boxes, frame_idx)
            need = [t for t in tracks if needs_classify(t)]
            timer.lap("track")
            if need:
                probs = classifier.predict_proba([crop_box(frame, t.box) for t in need])
                for t, p in zip(need, probs):
                    t.add_probs(p, frame_idx)
            timer.lap("classify")
            counter.update(tracks, frame_idx, frame_idx / fps)
            timer.lap("count")

        now = time.perf_counter()
        fps_live = 0.9 * fps_live + 0.1 / max(now - t_prev, 1e-6) if fps_live else 1 / max(now - t_prev, 1e-6)
        t_prev = now
        if writer or not args.no_display:
            vis = draw(frame, tracks, counter, classes, roi, fps_live, warming)
            if writer:
                writer.write(vis)
            if not args.no_display:
                view = cv2.resize(vis, None, fx=1280 / W, fy=1280 / W) if W > 1280 else vis
                cv2.imshow(WINDOW, view)
                key = cv2.waitKey(0 if paused else 1) & 0xFF
                if key == ord("q") or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(" "):
                    paused = not paused
        timer.lap("draw_io")
        if args.no_display and frame_idx % 500 == 0:
            print(f"  khung hình {frame_idx}/{total}  ({frame_idx / (now - t_start):.1f} FPS)")

    elapsed = time.perf_counter() - t_start
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    counter.finish()

    stem = video.stem
    events = sorted(counter.events, key=lambda e: e["frame"])
    write_csv(out_dir / f"{stem}_counts.csv", events,
              ["time_s", "frame", "track_id", "class", "confidence", "direction"])
    write_csv(out_dir / f"{stem}_rejected.csv", sorted(counter.rejected, key=lambda e: e["frame"]),
              ["time_s", "frame", "track_id", "class", "confidence", "direction", "reason"])
    totals = counter.totals()
    summary = {
        "video": str(video), "resolution": [W, H], "video_fps": fps, "frames_processed": frame_idx,
        "count_start_s": round(min(warmup, frame_idx) / fps, 2), "count_end_s": round(frame_idx / fps, 2),
        "processing_fps": round(frame_idx / elapsed, 2), "display": not args.no_display,
        "write_video": bool(writer), "cpu_threads": torch.get_num_threads(),
        "stage_ms_per_frame": {k: round(v / max(frame_idx, 1) * 1000, 2) for k, v in timer.totals.items()},
        "counts": counter.counts, "totals": totals, "total": sum(totals.values()),
        "rejected": len(counter.rejected), "model": str(args.model), "classes": classes, "config": cfg,
    }
    (out_dir / f"{stem}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")

    print(f"\nĐã xử lý {frame_idx} khung hình trong {elapsed:.1f}s -> {summary['processing_fps']} FPS"
          + ("" if args.no_display else " (có hiển thị; đo FPS chuẩn bằng --no-display)"))
    print("Thời gian từng bước (ms/khung hình):",
          ", ".join(f"{k} {v}" for k, v in summary["stage_ms_per_frame"].items()))
    print(f"\n{'Lớp':<12}" + "".join(f"{d:>8}" for d in counter.names) + f"{'tổng':>8}")
    for c in counter.counted_labels:
        print(f"{c:<12}" + "".join(f"{counter.counts[d][c]:>8}" for d in counter.names) + f"{totals[c]:>8}")
    print(f"{'TỔNG':<12}" + " " * 8 * len(counter.names) + f"{summary['total']:>8}")
    print(f"Lượt vượt vạch bị loại: {len(counter.rejected)} (xem {stem}_rejected.csv)")
    print(f"Kết quả: {out_dir / (stem + '_counts.csv')}, {out_dir / (stem + '_summary.json')}"
          + (f", {args.output}" if writer else ""))


if __name__ == "__main__":
    main()
