"""Cắt từng xe từ dataset phát hiện định dạng YOLO (vd Roboflow export "YOLOv8") vào data/crops/pub_<tên>/<lớp>/.

Cấu trúc nguồn: <src>/data.yaml (danh sách names) và các cặp thư mục .../images/*.jpg + .../labels/*.txt
(mỗi dòng nhãn: "lớp cx cy w h" chuẩn hoá 0..1, hoặc đa giác "lớp x1 y1 x2 y2 ...").
Tên lớp nguồn được ánh xạ tự động (motor/motorcycle -> motorbike, ...); bổ sung bằng --map.

Ví dụ:
    py -m tools.crops_from_yolo --src D:/datasets/vietnamese-vehicle --name pub_vnvehicle
    py -m tools.crops_from_yolo --src D:/datasets/abc --name pub_abc --map "xe_khach=bus,container=truck"
Ảnh cắt dùng cùng mức nới khung (CROP_PAD) với pipeline. Nguồn "pub_*" chỉ vào tập train.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from src.detector import CROP_PAD, crop_box
from train.data import ALL_CLASSES, IMAGE_EXTS, imread, imwrite

ALIASES = {
    "motorbike": ["motorbike", "motorbikes", "motorcycle", "motorcycles", "motor", "moto", "scooter",
                  "xemay", "xe_may"],
    "car": ["car", "cars", "oto", "o_to", "xeoto", "xe_oto", "xehoi", "xe_hoi", "sedan", "suv", "taxi"],
    "bus": ["bus", "buses", "coach", "minibus", "xebuyt", "xe_buyt", "xekhach", "xe_khach"],
    "truck": ["truck", "trucks", "lorry", "container", "xetai", "xe_tai"],
}
DEFAULT_MAP = {alias: cls for cls, names in ALIASES.items() for alias in names}


def norm(name):
    return str(name).strip().lower().replace("-", "_").replace(" ", "_")


def read_names(src):
    for fname in ("data.yaml", "data.yml"):
        if (src / fname).exists():
            names = yaml.safe_load((src / fname).read_text(encoding="utf-8")).get("names", [])
            return [names[k] for k in sorted(names)] if isinstance(names, dict) else list(names)
    raise SystemExit(f"Không thấy data.yaml trong {src}; truyền tên lớp bằng --names")


def find_pairs(src):
    """Các cặp (ảnh, file nhãn): .../images/x.jpg <-> .../labels/x.txt."""
    pairs = []
    for img in sorted(src.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS or "images" not in img.parts:
            continue
        parts = list(img.parts)
        i = len(parts) - 1 - parts[::-1].index("images")
        parts[i] = "labels"
        label = Path(*parts).with_suffix(".txt")
        if label.exists():
            pairs.append((img, label))
    return pairs


def read_boxes(label_path):
    """-> [(id lớp, x1, y1, x2, y2) chuẩn hoá]; hỗ trợ cả box và đa giác."""
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        vals = line.split()
        if len(vals) < 5:
            continue
        cls, nums = int(float(vals[0])), [float(v) for v in vals[1:]]
        if len(nums) == 4:
            cx, cy, w, h = nums
            boxes.append((cls, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
        else:
            xs, ys = nums[0::2], nums[1::2]
            boxes.append((cls, min(xs), min(ys), max(xs), max(ys)))
    return boxes


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True)
    p.add_argument("--name", required=True, help="tên nguồn trong data/crops (tự thêm tiền tố pub_)")
    p.add_argument("--names", help="danh sách tên lớp theo id, cách nhau dấu phẩy (thay cho data.yaml)")
    p.add_argument("--map", help='ánh xạ bổ sung "tên_nguồn=lớp_đích,..."')
    p.add_argument("--max-per-class", type=int, default=1500)
    p.add_argument("--min-size", type=int, default=24, help="bỏ xe có cạnh nhỏ hơn N pixel")
    p.add_argument("--crops", default="data/crops")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src)
    names = [n.strip() for n in args.names.split(",")] if args.names else read_names(src)
    mapping = dict(DEFAULT_MAP)
    if args.map:
        mapping.update({norm(a): b.strip() for a, b in (item.split("=") for item in args.map.split(",") if item)})
    targets = [mapping.get(norm(n)) for n in names]
    for n, t in zip(names, targets):
        print(f"  lớp nguồn {n!r:<20} -> {t or '(bỏ qua)'}")
    if any(t and t not in ALL_CLASSES for t in targets):
        raise SystemExit(f"Lớp đích phải thuộc {', '.join(ALL_CLASSES)}")
    if not any(targets):
        raise SystemExit('Không lớp nguồn nào được ánh xạ; bổ sung bằng --map "tên_nguồn=lớp_đích"')

    pairs = find_pairs(src)
    if not pairs:
        raise SystemExit(f"Không thấy cặp images/ + labels/ nào trong {src}")
    name = args.name if args.name.startswith("pub_") else "pub_" + args.name
    out_root = Path(args.crops) / name
    counts = {t: 0 for t in set(targets) if t}
    rng = np.random.default_rng(args.seed)
    for i in rng.permutation(len(pairs)):  # duyệt ảnh ngẫu nhiên để mẫu lấy được trải đều dataset
        if all(n >= args.max_per_class for n in counts.values()):
            break
        img_path, label_path = pairs[i]
        img = imread(img_path)
        if img is None:
            continue
        H, W = img.shape[:2]
        for k, (cls, x1, y1, x2, y2) in enumerate(read_boxes(label_path)):
            target = targets[cls] if cls < len(targets) else None
            if not target or counts[target] >= args.max_per_class:
                continue
            box = (int(x1 * W), int(y1 * H), int((x2 - x1) * W), int((y2 - y1) * H))
            if min(box[2], box[3]) < args.min_size:
                continue
            out = out_root / target
            out.mkdir(parents=True, exist_ok=True)
            if imwrite(out / f"{name[4:]}_{img_path.stem}_{k}.jpg", crop_box(img, box, CROP_PAD)):
                counts[target] += 1

    info = {"source": str(src.resolve()), "class_names": names, "mapping": dict(zip(names, targets)),
            "max_per_class": args.max_per_class, "counts": counts, "imported_at": time.strftime("%Y-%m-%d %H:%M")}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "source.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print("\n" + "\n".join(f"{t:<10} {n} ảnh" for t, n in sorted(counts.items())))
    print(f"Đã ghi vào {out_root} (ghi nguồn trong source.json)")


if __name__ == "__main__":
    main()
