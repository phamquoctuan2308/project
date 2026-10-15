"""Nhập ảnh từ dataset phân loại có sẵn (mặc định MIO-TCD Classification) vào data/crops/pub_<tên>/<lớp>/.

Dataset nguồn phải có dạng <src>/<tên_lớp>/*.jpg (MIO-TCD: thư mục train/ sau khi giải nén).
Lớp được đổi tên theo bảng ánh xạ; lớp không có trong bảng bị bỏ qua.

Ví dụ:
    py -m tools.import_miotcd --src D:/datasets/MIO-TCD-Classification/train
    py -m tools.import_miotcd --src D:/datasets/MIO-TCD-Classification/train --with-other --max-per-class 800
    py -m tools.import_miotcd --src D:/datasets/kaggle_xe --name pub_kaggle --map "xe_may=motorbike,o_to=car"
Nguồn "pub_*" chỉ được đưa vào tập train khi chạy tools.split_dataset.
Khi dùng MIO-TCD phải trích dẫn: Luo et al., "MIO-TCD: A new benchmark dataset for vehicle
classification and localization", IEEE Transactions on Image Processing, 2018.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from train.data import ALL_CLASSES, imread, list_images

# Tên thư mục MIO-TCD (đã chuẩn hoá) -> lớp của dự án. pickup_truck, work_van bị bỏ để tránh nhầm car / truck.
MIOTCD_MAP = {"motorcycle": "motorbike", "car": "car", "bus": "bus",
              "articulated_truck": "truck", "single_unit_truck": "truck"}
MIOTCD_OTHER = {"pedestrian": "other", "bicycle": "other", "non_motorized_vehicle": "other", "background": "other"}


def norm(name):
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def parse_map(text):
    pairs = [item.split("=") for item in text.split(",") if item.strip()]
    mapping = {norm(src): dst.strip() for src, dst in pairs}
    bad = {d for d in mapping.values() if d not in ALL_CLASSES}
    if bad:
        raise SystemExit(f"Lớp đích không hợp lệ: {', '.join(bad)} (chỉ dùng {', '.join(ALL_CLASSES)})")
    return mapping


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--src", required=True, help="thư mục chứa các thư mục lớp")
    p.add_argument("--name", default="pub_miotcd", help="tên nguồn trong data/crops (tự thêm tiền tố pub_)")
    p.add_argument("--map", help='ánh xạ "lớp_nguồn=lớp_đích,..." thay cho bảng MIO-TCD')
    p.add_argument("--with-other", action="store_true", help="(MIO-TCD) lấy cả pedestrian/bicycle/background làm other")
    p.add_argument("--max-per-class", type=int, default=1500, help="số ảnh tối đa mỗi lớp đích")
    p.add_argument("--min-size", type=int, default=24, help="bỏ ảnh có cạnh nhỏ hơn N pixel")
    p.add_argument("--crops", default="data/crops")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src)
    folders = [d for d in sorted(src.iterdir()) if d.is_dir()] if src.is_dir() else []
    if not folders:
        raise SystemExit(f"Không thấy thư mục lớp nào trong {src}")
    mapping = parse_map(args.map) if args.map else {**MIOTCD_MAP, **(MIOTCD_OTHER if args.with_other else {})}
    name = args.name if args.name.startswith("pub_") else "pub_" + args.name
    out_root = Path(args.crops) / name

    candidates = {}
    for d in folders:
        target = mapping.get(norm(d.name))
        print(f"  {d.name:<24} -> {target or '(bỏ qua)'}")
        if target:
            candidates.setdefault(target, []).extend((d.name, f) for f in list_images(d))

    rng = np.random.default_rng(args.seed)
    counts = {}
    for target, items in candidates.items():
        out = out_root / target
        out.mkdir(parents=True, exist_ok=True)
        taken = 0
        for i in rng.permutation(len(items)):
            if taken >= args.max_per_class:
                break
            folder, f = items[i]
            img = imread(f)
            if img is None or min(img.shape[:2]) < args.min_size:
                continue
            shutil.copy2(f, out / f"{name[4:]}_{norm(folder)}_{f.name}")
            taken += 1
        counts[target] = taken
        print(f"{target:<10} lấy {taken} / {len(items)} ảnh")

    info = {"source": str(src.resolve()), "mapping": mapping, "max_per_class": args.max_per_class,
            "min_size": args.min_size, "counts": counts, "imported_at": time.strftime("%Y-%m-%d %H:%M")}
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "source.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nĐã nhập vào {out_root} (ghi nguồn trong source.json)")


if __name__ == "__main__":
    main()
