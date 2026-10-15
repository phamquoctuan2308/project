"""Thống kê số ảnh mỗi lớp theo từng nguồn (data/crops) và theo từng tập (data/dataset).

Ví dụ:
    py -m tools.dataset_stats
Ghi bảng ra outputs/dataset_stats.csv.
"""
import argparse
import csv
import sys
from pathlib import Path

from train.data import ALL_CLASSES, SPLITS, class_counts, list_images

TARGET = 500  # số ảnh tối thiểu mong muốn mỗi lớp (xem plan.md)


def print_table(title, rows):
    print(f"\n{title}")
    print(f"{'':<24}" + "".join(f"{c:>11}" for c in ALL_CLASSES) + f"{'tổng':>9}")
    for name, cnt in rows:
        print(f"{name:<24}" + "".join(f"{cnt[c]:>11}" for c in ALL_CLASSES) + f"{sum(cnt.values()):>9}")


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", default="data/crops")
    p.add_argument("--dataset", default="data/dataset")
    p.add_argument("--out", default="outputs/dataset_stats.csv")
    args = p.parse_args()

    crops = Path(args.crops)
    out_rows = []
    crop_rows = [(d.name, class_counts(d)) for d in sorted(crops.iterdir()) if d.is_dir()] if crops.exists() else []
    if crop_rows:
        total = {c: sum(r[1][c] for r in crop_rows) for c in ALL_CLASSES}
        print_table("ẢNH ĐÃ GÁN NHÃN THEO NGUỒN (data/crops; pub_* = dataset công khai)",
                    crop_rows + [("TỔNG", total)])
        for name, _ in crop_rows:
            unlabeled = len(list_images(crops / name))
            if unlabeled:
                print(f"  {name}: còn {unlabeled} ảnh chưa gán nhãn")
        short = [c for c in ALL_CLASSES if c != "other" and total[c] < TARGET]
        if short:
            print(f"  Lớp chưa đủ {TARGET} ảnh: {', '.join(f'{c} ({total[c]})' for c in short)}")
        out_rows += [("crops", n, cnt) for n, cnt in crop_rows]

    split_rows = [(s, class_counts(Path(args.dataset) / s)) for s in SPLITS if (Path(args.dataset) / s).exists()]
    if any(sum(c.values()) for _, c in split_rows):
        print_table("DATASET ĐÃ CHIA (data/dataset)", split_rows)
        out_rows += [("dataset", n, cnt) for n, cnt in split_rows]

    if not out_rows:
        raise SystemExit("Chưa có ảnh nào đã gán nhãn")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["source", "name", *ALL_CLASSES, "total"])
        for src, name, cnt in out_rows:
            w.writerow([src, name, *(cnt[c] for c in ALL_CLASSES), sum(cnt.values())])
    print(f"\nĐã ghi {args.out}")


if __name__ == "__main__":
    main()
