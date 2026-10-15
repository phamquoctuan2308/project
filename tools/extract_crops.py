"""Cắt ảnh các vùng chuyển động (MOG2) từ video để tạo dữ liệu gán nhãn.

Mỗi xe được theo dõi qua các khung hình và chỉ lưu tối đa --per-track ảnh, ở các vị trí cách nhau
ít nhất --min-move chiều cao khung hình, để dataset không đầy ảnh gần trùng của cùng một xe.
Xe chạm mép khung hình / mép ROI (đang bị cắt dở) không được lưu.

Ví dụ:
    py -m tools.extract_crops --video data/raw_videos/v1.mp4 --preview
    py -m tools.extract_crops --video data/raw_videos/v1.mp4 --select-roi
Ảnh lưu vào data/crops/<tên_video>/, kèm crops.csv ghi khung hình, toạ độ và ID xe của từng ảnh.
Không cắt ảnh từ các video dùng để đánh giá đếm xe.
"""
import argparse
import csv
import sys
from pathlib import Path

import cv2

from src.detector import CROP_PAD, MotionDetector, crop_box, touches_border
from src.tracker import CentroidTracker


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True, help="đường dẫn video (MP4/AVI)")
    p.add_argument("--out", help="thư mục lưu ảnh (mặc định data/crops/<tên_video>)")
    p.add_argument("--warmup", type=int, default=120, help="số khung hình đầu để MOG2 học nền, không lưu ảnh")
    p.add_argument("--max-frames", type=int, default=0, help="dừng sau N khung hình (0 = hết video)")
    p.add_argument("--history", type=int, default=500)
    p.add_argument("--var-threshold", type=float, default=16)
    p.add_argument("--min-area", type=int, default=800, help="diện tích khung bao tối thiểu (pixel^2)")
    p.add_argument("--max-area", type=int, default=0, help="diện tích khung bao tối đa (0 = không giới hạn)")
    p.add_argument("--merge-ratio", type=float, default=0.5, help="gộp khung chồng nhau (0 = tắt)")
    p.add_argument("--min-size", type=int, default=24, help="bỏ ảnh cắt có cạnh nhỏ hơn N pixel")
    p.add_argument("--pad", type=float, default=CROP_PAD, help="nới khung cắt thêm tỉ lệ này mỗi phía")
    p.add_argument("--roi", help="vùng quan tâm x,y,w,h")
    p.add_argument("--select-roi", action="store_true", help="tự kéo chọn ROI trên khung hình đầu")
    p.add_argument("--per-track", type=int, default=5, help="số ảnh tối đa mỗi xe")
    p.add_argument("--min-move", type=float, default=0.12,
                   help="xe phải đi thêm ít nhất tỉ lệ này của chiều cao khung hình mới lưu ảnh tiếp")
    p.add_argument("--max-distance", type=int, default=100, help="ngưỡng ghép tâm xe giữa 2 khung hình (pixel)")
    p.add_argument("--keep-border", action="store_true", help="vẫn lưu xe chạm mép khung hình / ROI")
    p.add_argument("--preview", action="store_true", help="hiện khung hình + mask để chỉnh tham số")
    return p.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    args = parse_args()
    video = Path(args.video)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise SystemExit(f"Không mở được video: {video}")

    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"{video.name}: {W}x{H}, {fps:.1f} FPS, {total} khung hình ({total / fps:.0f}s)")
    if H < 720:
        print("CẢNH BÁO: độ phân giải < 720p, thấp hơn yêu cầu đầu vào của đề tài")

    roi = tuple(int(v) for v in args.roi.split(",")) if args.roi else None
    if args.select_roi:
        ok, first = cap.read()
        if not ok:
            raise SystemExit("Không đọc được khung hình đầu")
        view_scale = min(1.0, 1280 / W)
        view = cv2.resize(first, None, fx=view_scale, fy=view_scale) if view_scale < 1 else first
        r = cv2.selectROI("Chon ROI (Enter = xac nhan, c = huy)", view, showCrosshair=False)
        cv2.destroyAllWindows()
        if r[2] and r[3]:
            roi = tuple(int(v / view_scale) for v in r)
            print(f"ROI = {','.join(map(str, roi))}  (dùng lại bằng --roi)")
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    detector = MotionDetector(history=args.history, var_threshold=args.var_threshold, min_area=args.min_area,
                              max_area=args.max_area or None, merge_ratio=args.merge_ratio, roi=roi)
    tracker = CentroidTracker(max_distance=args.max_distance, max_disappeared=5)
    region = roi or (0, 0, W, H)
    min_move = args.min_move * H

    out_dir = Path(args.out) if args.out else Path("data/crops") / video.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "crops.csv"
    new_csv = not csv_path.exists()
    csv_file = open(csv_path, "a", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    if new_csv:
        writer.writerow(["file", "video", "frame", "time_s", "track_id", "x", "y", "w", "h"])

    last_saved = {}  # track id -> (số ảnh đã lưu, tâm lúc lưu gần nhất)
    saved = frame_idx = 0
    paused = False
    while True:
        ok, frame = cap.read()
        if not ok or (args.max_frames and frame_idx >= args.max_frames):
            break
        frame_idx += 1
        # Luôn đưa khung hình vào MOG2 để mô hình nền liên tục, chỉ bỏ qua khâu cắt ảnh khi đang warmup
        boxes, mask = detector.detect(frame)
        if frame_idx > args.warmup:
            for t in tracker.update(boxes, frame_idx):
                n, (cx, cy) = last_saved.get(t.id, (0, (None, None)))
                x, y, w, h = t.box
                if n >= args.per_track or min(w, h) < args.min_size:
                    continue
                if cx is not None and abs(t.centroid[0] - cx) + abs(t.centroid[1] - cy) < min_move:
                    continue
                if not args.keep_border and touches_border(t.box, region):
                    continue
                name = f"{video.stem}_f{frame_idx:06d}_{t.id}.jpg"
                if not cv2.imwrite(str(out_dir / name), crop_box(frame, t.box, args.pad)):
                    raise SystemExit(f"Không ghi được {out_dir / name} (đường dẫn nên chỉ dùng ký tự ASCII)")
                writer.writerow([name, video.name, frame_idx, round(frame_idx / fps, 2), t.id, x, y, w, h])
                last_saved[t.id] = (n + 1, t.centroid)
                saved += 1

        if args.preview:
            vis = frame.copy()
            if roi:
                cv2.rectangle(vis, roi[:2], (roi[0] + roi[2], roi[1] + roi[3]), (255, 0, 0), 2)
            for x, y, w, h in boxes:
                cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 2)
            status = "warmup" if frame_idx <= args.warmup else f"saved {saved}"
            cv2.putText(vis, f"frame {frame_idx}  {status}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            scale = 640 / W
            both = cv2.hconcat([cv2.resize(vis, None, fx=scale, fy=scale),
                                cv2.cvtColor(cv2.resize(mask, None, fx=scale, fy=scale), cv2.COLOR_GRAY2BGR)])
            cv2.imshow("frame | mask  (space = dung, q = thoat)", both)
            key = cv2.waitKey(0 if paused else 1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                paused = not paused

        if frame_idx % 500 == 0:
            print(f"  khung hình {frame_idx}/{total}, đã lưu {saved} ảnh")

    cap.release()
    csv_file.close()
    cv2.destroyAllWindows()
    print(f"Xong: lưu {saved} ảnh của {len(last_saved)} xe vào {out_dir}")


if __name__ == "__main__":
    main()
