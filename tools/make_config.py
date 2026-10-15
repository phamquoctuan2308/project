"""Tạo file cấu hình cho một video bằng chuột: chọn ROI và đặt vạch đếm.

Ví dụ:
    py -m tools.make_config --video data/raw_videos/test1.mp4
    py -m tools.make_config --video data/raw_videos/test1.mp4 --time 20 --direction down
Bước 1: kéo chuột chọn ROI rồi Enter (bấm c để dùng cả khung hình).
Bước 2: click 2 điểm đặt vạch đếm rồi Enter (r = vẽ lại, q = thoát không lưu).
File được sinh từ configs/example.yaml (giữ chú thích), lưu ở configs/<tên_video>.yaml,
kèm ảnh xem trước outputs/<tên_video>_config.jpg để đưa vào báo cáo.
"""
import argparse
import re
import sys
from pathlib import Path

import cv2

from src.config import load_config

TEMPLATE = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
MAX_VIEW = 1280  # bề rộng tối đa của cửa sổ


def pick_line(view):
    points = []
    win = "Vach dem: click 2 diem | Enter = luu | r = ve lai | q = thoat"

    def on_mouse(event, x, y, *_):
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))

    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        vis = view.copy()
        for pt in points:
            cv2.circle(vis, pt, 5, (0, 255, 255), -1)
        if len(points) == 2:
            cv2.line(vis, points[0], points[1], (0, 255, 255), 2)
        cv2.imshow(win, vis)
        key = cv2.waitKey(30) & 0xFF
        if key == 13 and len(points) == 2:
            break
        if key == ord("r"):
            points.clear()
        if key in (ord("q"), 27):
            points = None
            break
    cv2.destroyWindow(win)
    return points


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--video", required=True)
    p.add_argument("--time", type=float, default=5.0, help="lấy khung hình tại giây thứ N để vẽ")
    p.add_argument("--direction", default="both", help="both | down | up | left | right")
    p.add_argument("--out", help="mặc định configs/<tên_video>.yaml")
    p.add_argument("--force", action="store_true", help="ghi đè file đã có")
    args = p.parse_args()

    video = Path(args.video)
    out = Path(args.out) if args.out else Path("configs") / f"{video.stem}.yaml"
    if out.exists() and not args.force:
        raise SystemExit(f"{out} đã tồn tại (thêm --force để ghi đè)")
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(args.time * fps))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise SystemExit(f"Không đọc được khung hình ở giây {args.time} của {video}")

    H, W = frame.shape[:2]
    scale = min(1.0, MAX_VIEW / W)
    view = cv2.resize(frame, None, fx=scale, fy=scale) if scale < 1 else frame.copy()
    r = cv2.selectROI("ROI: keo chuot roi Enter | c = ca khung hinh", view, showCrosshair=False)
    cv2.destroyAllWindows()
    roi = [int(v / scale) for v in r] if r[2] and r[3] else None
    points = pick_line(view)
    if points is None:
        raise SystemExit("Đã huỷ, không lưu cấu hình")
    line = [[int(x / scale), int(y / scale)] for x, y in points]

    text = TEMPLATE.read_text(encoding="utf-8")
    text = re.sub(r"^roi: null", f"roi: {roi}" if roi else "roi: null", text, count=1, flags=re.M)
    text = re.sub(r"^line: null", f"line: {line}", text, count=1, flags=re.M)
    text = re.sub(r"^direction: both", f"direction: {args.direction}", text, count=1, flags=re.M)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(text, encoding="utf-8")
    load_config(out)  # kiểm tra file vừa tạo đọc lại được

    preview = frame.copy()
    if roi:
        x, y, w, h = roi
        cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 160, 0), 2)
    cv2.line(preview, tuple(line[0]), tuple(line[1]), (0, 255, 255), 3)
    Path("outputs").mkdir(exist_ok=True)
    cv2.imwrite(str(Path("outputs") / f"{video.stem}_config.jpg"), preview)
    print(f"Đã lưu {out}: roi = {roi}, line = {line}, direction = {args.direction}")
    print(f"Ảnh xem trước: outputs/{video.stem}_config.jpg")


if __name__ == "__main__":
    main()
