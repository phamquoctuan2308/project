"""Đánh giá chéo K-Fold (mặc định 5) cho Custom CNN trên tập train + val; tập test giữ nguyên.

Chia fold theo nhóm (StratifiedGroupKFold): ảnh cùng video trong cùng đoạn ~30 s luôn nằm chung
một fold, tránh việc một chiếc xe xuất hiện ở cả phần huấn luyện lẫn phần kiểm tra của fold.
Trong mỗi fold, 15% nhóm của phần huấn luyện được tách ra làm tập dừng sớm.

Ví dụ:
    py -m train.kfold
    py -m train.kfold --folds 5 --classes motorbike car bus truck
Ghi từng fold và trung bình ± độ lệch chuẩn vào outputs/kfold_results.csv.
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold

from train.data import group_of, load_split, resolve_classes
from train.model import VehicleCNN, predict_proba
from train.train_cnn import fit, seed_everything


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/dataset")
    p.add_argument("--classes", nargs="+")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--group-frames", type=int, default=900, help="độ dài một nhóm (khung hình)")
    p.add_argument("--inner-val", type=float, default=0.15, help="tỉ lệ nhóm tách ra để dừng sớm")
    p.add_argument("--verbose", action="store_true", help="in log từng epoch")
    p.add_argument("--outputs", default="outputs")
    return p.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    args = parse_args()
    classes = resolve_classes(args.dataset, args.classes)
    parts = [load_split(args.dataset, s, classes) for s in ("train", "val")]
    X = np.concatenate([p[0] for p in parts])
    y = np.concatenate([p[1] for p in parts])
    groups = np.array([group_of(path, args.group_frames) for p in parts for path in p[2]])
    n_groups = len(set(groups))
    print(f"{len(X)} ảnh (train + val), {n_groups} nhóm, lớp: {', '.join(classes)}")
    if n_groups < args.folds:
        raise SystemExit(f"Chỉ có {n_groups} nhóm, không chia được {args.folds} fold "
                         "(video quá ngắn: giảm --group-frames hoặc --folds)")

    log = print if args.verbose else (lambda _msg: None)
    splitter = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=args.seed)
    rows = []
    for k, (tr, te) in enumerate(splitter.split(X, y, groups), 1):
        seed_everything(args.seed + k)
        inner = GroupShuffleSplit(n_splits=1, test_size=args.inner_val, random_state=args.seed)
        fit_i, stop_i = next(inner.split(tr, groups=groups[tr]))
        fit_idx, stop_idx = tr[fit_i], tr[stop_i]
        t0 = time.perf_counter()
        model = VehicleCNN(len(classes), args.dropout)
        _, best_epoch = fit(model, X[fit_idx], y[fit_idx], X[stop_idx], y[stop_idx],
                            args.epochs, args.batch, args.lr, args.patience, log=log)
        pred = predict_proba(model, X[te]).argmax(1)
        prec, rec, f1, _ = precision_recall_fscore_support(y[te], pred, average="macro", zero_division=0)
        f1_each = precision_recall_fscore_support(y[te], pred, labels=range(len(classes)),
                                                  average=None, zero_division=0)[2]
        row = {"fold": k, "n_train": len(fit_idx), "n_test": len(te), "best_epoch": best_epoch,
               "accuracy": accuracy_score(y[te], pred), "precision": prec, "recall": rec, "f1": f1}
        for i, c in enumerate(classes):  # lớp vắng mặt trong fold -> NaN, không kéo trung bình xuống 0
            row[f"f1_{c}"] = f1_each[i] if (y[te] == i).any() else np.nan
        rows.append(row)
        missing = [c for i, c in enumerate(classes) if not (y[te] == i).any()]
        note = f"  (fold thiếu lớp: {', '.join(missing)})" if missing else ""
        print(f"Fold {k}: train {len(fit_idx)}, kiểm tra {len(te)}  ->  accuracy {row['accuracy']:.4f}  "
              f"F1 {f1:.4f}  (epoch {best_epoch}, {time.perf_counter() - t0:.0f}s){note}")

    keys = [key for key in rows[0] if key != "fold"]
    mean = {"fold": "mean", **{key: np.nanmean([r[key] for r in rows]) for key in keys}}
    std = {"fold": "std", **{key: np.nanstd([r[key] for r in rows], ddof=1) for key in keys}}
    out = Path(args.outputs)
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "kfold_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        for r in [*rows, mean, std]:
            writer.writerow({key: (f"{v:.4f}" if isinstance(v, float) else v) for key, v in r.items()})

    print(f"\n{args.folds}-Fold:  accuracy {mean['accuracy']:.4f} ± {std['accuracy']:.4f}   "
          f"F1 {mean['f1']:.4f} ± {std['f1']:.4f}")
    print(f"Đã ghi {out / 'kfold_results.csv'}")


if __name__ == "__main__":
    main()
