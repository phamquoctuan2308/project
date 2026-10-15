"""Chia ảnh đã gán nhãn (data/crops) thành data/dataset/{train,val,test}, không để rò rỉ giữa các tập.

Hai cách chia:
  --mode video (mặc định): mỗi video nằm trọn trong một tập; tự tìm cách gán video sao cho tỉ lệ
                           từng lớp gần 70/15/15 nhất (hoặc chỉ định bằng --val / --test). Cần >= 3 video.
  --mode time            : mỗi video chia theo thời gian (70% đầu -> train, 15% -> val, 15% cuối -> test),
                           bỏ ảnh trong khoảng --gap khung hình quanh điểm cắt. Dùng khi có ít video.
Nguồn có tên bắt đầu bằng "pub_" (dataset công khai) chỉ đưa vào tập train.
Video đã có file đếm tay trong data/ground_truth/ (video đánh giá đếm) tự động bị loại.
data/dataset được tạo lại hoàn toàn mỗi lần chạy: không bỏ ảnh trực tiếp vào đó.

Ví dụ:
    py -m tools.split_dataset
    py -m tools.split_dataset --mode time --overwrite
    py -m tools.split_dataset --val nga_tu_chieu --test cau_vuot_sang --overwrite
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from train.data import ALL_CLASSES, SPLITS, list_images, video_and_frame


def best_assignment(counts, ratios=(0.7, 0.15, 0.15), seed=42, max_exhaustive=10, samples=100_000):
    """counts (G nguồn x C lớp) -> mảng (G,) gán 0=train, 1=val, 2=test, tỉ lệ từng lớp gần ratios nhất."""
    counts = np.asarray(counts, float)
    G = len(counts)
    if G <= max_exhaustive:  # thử mọi cách gán (3^G)
        assign = np.stack(np.meshgrid(*[np.arange(3)] * G, indexing="ij"), -1).reshape(-1, G)
    else:
        assign = np.random.default_rng(seed).integers(0, 3, size=(samples, G))
    onehot = np.eye(3)[assign]                                 # (M, G, 3)
    per_split = np.einsum("mgk,gc->mkc", onehot, counts)       # (M, 3, C)
    total = counts.sum(0)
    frac = per_split / np.maximum(total, 1)
    cost = ((frac - np.asarray(ratios)[None, :, None]) ** 2).sum((1, 2))
    cost += 10 * ((per_split == 0) & (total > 0)).sum((1, 2))  # tập thiếu một lớp đang có
    cost += 100 * (onehot.sum(1) == 0).sum(1)                  # tập không có video nào
    return assign[np.argmin(cost)]


def time_split(frames, ratios=(0.7, 0.15, 0.15), gap=150):
    """Số khung hình của từng ảnh -> 0/1/2 theo đoạn thời gian, None nếu sát điểm cắt (bỏ đi)."""
    lo, hi = min(frames), max(frames)
    cut1 = lo + (hi - lo) * ratios[0]
    cut2 = lo + (hi - lo) * (ratios[0] + ratios[1])
    out = []
    for f in frames:
        if abs(f - cut1) < gap or abs(f - cut2) < gap:
            out.append(None)
        else:
            out.append(0 if f < cut1 else 1 if f < cut2 else 2)
    return out


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--crops", default="data/crops")
    p.add_argument("--dataset", default="data/dataset")
    p.add_argument("--gt-dir", default="data/ground_truth", help="video có file đếm tay ở đây bị loại")
    p.add_argument("--mode", choices=["video", "time"], default="video")
    p.add_argument("--ratios", type=float, nargs=3, default=[0.7, 0.15, 0.15], metavar=("TRAIN", "VAL", "TEST"))
    p.add_argument("--val", nargs="*", default=[], help="(mode video) chỉ định video cho tập val")
    p.add_argument("--test", nargs="*", default=[], help="(mode video) chỉ định video cho tập test")
    p.add_argument("--exclude", nargs="*", default=[], help="nguồn không dùng")
    p.add_argument("--gap", type=int, default=150, help="(mode time) số khung hình bỏ quanh điểm cắt")
    p.add_argument("--cap", type=int, default=0, help="giới hạn số ảnh mỗi lớp trong tập train (0 = không)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--overwrite", action="store_true", help="xoá ảnh cũ trong data/dataset rồi tạo lại")
    return p.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    args = parse_args()
    crops, dataset = Path(args.crops), Path(args.dataset)
    if not crops.is_dir():
        raise SystemExit(f"Không có thư mục {crops}")

    names = [d.name for d in sorted(crops.iterdir()) if d.is_dir() and not d.name.startswith(("_", "."))]
    gt_names = [p.stem for p in Path(args.gt_dir).glob("*.csv")] if Path(args.gt_dir).is_dir() else []
    auto_excl = {n for n in names if any(g == n or g.startswith(n + "_") for g in gt_names)}
    for n in sorted(auto_excl - set(args.exclude)):
        print(f"Loại {n}: đã có file đếm tay trong {args.gt_dir} (video đánh giá đếm không dùng để train)")
    exclude = set(args.exclude) | auto_excl

    sources = {}
    for n in names:
        if n in exclude:
            continue
        files = {c: list_images(crops / n / c) for c in ALL_CLASSES}
        if any(files.values()):
            sources[n] = files
    public = {n: f for n, f in sources.items() if n.startswith("pub_")}
    own = {n: f for n, f in sources.items() if not n.startswith("pub_")}
    if not own:
        raise SystemExit(f"Chưa có ảnh tự gán nhãn trong {crops}/<video>/<lớp>/")

    result = {s: {c: [] for c in ALL_CLASSES} for s in SPLITS}
    plan = {}
    if args.val or args.test or args.mode == "video":
        order = sorted(own)
        if args.val or args.test:
            unknown = set(args.val + args.test) - set(order)
            if unknown:
                raise SystemExit(f"Không có nguồn: {', '.join(sorted(unknown))}")
            assign = [1 if n in args.val else 2 if n in args.test else 0 for n in order]
        else:
            if len(order) < 3:
                raise SystemExit(f"Chỉ có {len(order)} video, cần >= 3 để chia theo video. "
                                 "Dùng --mode time để chia theo thời gian trong từng video.")
            counts = np.array([[len(own[n][c]) for c in ALL_CLASSES] for n in order])
            assign = best_assignment(counts, args.ratios, args.seed)
        for n, a in zip(order, assign):
            for c in ALL_CLASSES:
                result[SPLITS[a]][c] += own[n][c]
            plan[n] = SPLITS[a]
    else:
        for n, files in own.items():
            items = [(c, p) for c in ALL_CLASSES for p in files[c]]
            frames = [video_and_frame(p)[1] for _, p in items]
            if None in frames:
                raise SystemExit(f"{n}: có ảnh không đặt tên dạng <video>_f<khung>_<i>.jpg, không chia theo thời gian được")
            parts = time_split(frames, args.ratios, args.gap)
            for (c, p), k in zip(items, parts):
                if k is not None:
                    result[SPLITS[k]][c].append(p)
            plan[n] = f"chia theo thời gian (bỏ {parts.count(None)} ảnh sát điểm cắt)"
            empty = [SPLITS[k] for k in (1, 2) if k not in parts]
            if empty:
                print(f"CẢNH BÁO: {n} quá ngắn so với --gap {args.gap} khung hình, tập {', '.join(empty)} "
                      f"không nhận được ảnh nào từ video này (giảm --gap)")
    for n, files in public.items():
        for c in ALL_CLASSES:
            result["train"][c] += files[c]
        plan[n] = "train (dataset công khai)"

    if args.cap:
        rng = np.random.default_rng(args.seed)
        for c, files in result["train"].items():
            if len(files) > args.cap:
                keep = sorted(rng.choice(len(files), args.cap, replace=False))
                result["train"][c] = [files[i] for i in keep]

    old = [p for s in SPLITS for c in ALL_CLASSES for p in list_images(dataset / s / c)]
    if old and not args.overwrite:
        raise SystemExit(f"{dataset} đã có {len(old)} ảnh; thêm --overwrite để xoá và tạo lại")
    for p in old:
        p.unlink()
    for s in SPLITS:
        for c, files in result[s].items():
            if not files:
                continue
            out = dataset / s / c
            out.mkdir(parents=True, exist_ok=True)
            for p in files:
                dst = out / p.name
                if dst.exists():  # trùng tên giữa hai nguồn
                    dst = out / f"{p.parent.parent.name}__{p.name}"
                shutil.copy2(p, dst)

    counts = {s: {c: len(result[s][c]) for c in ALL_CLASSES} for s in SPLITS}
    info = {"mode": args.mode if not (args.val or args.test) else "video (chỉ định)", "ratios": args.ratios,
            "seed": args.seed, "cap": args.cap, "sources": plan, "excluded": sorted(exclude),
            "counts": counts, "created_at": time.strftime("%Y-%m-%d %H:%M")}
    (dataset / "split.json").write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")

    for n, where in plan.items():
        print(f"  {n:<28} -> {where}")
    print(f"\n{'':<8}" + "".join(f"{c:>11}" for c in ALL_CLASSES) + f"{'tổng':>9}")
    for s in SPLITS:
        print(f"{s:<8}" + "".join(f"{counts[s][c]:>11}" for c in ALL_CLASSES) + f"{sum(counts[s].values()):>9}")
    for c in ALL_CLASSES:
        empty = [s for s in SPLITS if counts["train"][c] and not counts[s][c]]
        if empty:
            print(f"CẢNH BÁO: lớp {c} không có ảnh trong tập {', '.join(empty)}")
    print(f"\nĐã ghi {dataset} (chi tiết trong {dataset / 'split.json'})")


if __name__ == "__main__":
    main()
