"""Đánh giá mô hình trên tập test: accuracy, confusion matrix, Precision / Recall / F1 từng lớp.

Ví dụ:
    py -m train.evaluate
    py -m train.evaluate --split val
Ghi outputs/test_report.txt, outputs/test_metrics.json, outputs/figures/confusion_matrix_test.png
và chép ảnh dự đoán sai vào outputs/misclassified/<lớp_thật>_thanh_<lớp_dự_đoán>/.
"""
import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

from train.data import list_images, load_split, to_tensor
from train.model import load_checkpoint, predict_proba


@torch.inference_mode()
def time_inference(model, X, repeat=100):
    """Thời gian chạy mạng (ms/ảnh) với batch 1 và batch 32, chưa gồm tiền xử lý."""
    x1, xb = to_tensor(X[:1]), to_tensor(X[:32])
    model(x1)  # chạy nháp
    model(xb)
    t0 = time.perf_counter()
    for _ in range(repeat):
        model(x1)
    single = (time.perf_counter() - t0) / repeat * 1000
    t0 = time.perf_counter()
    for _ in range(repeat // 4):
        model(xb)
    batched = (time.perf_counter() - t0) / (repeat // 4) / len(xb) * 1000
    return single, batched


def plot_confusion(cm, classes, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    norm = cm / np.maximum(cm.sum(1, keepdims=True), 1)
    n = len(classes)
    fig, axes = plt.subplots(1, 2, figsize=(5.5 * 2, 4.8))
    for ax, data, title, fmt in ((axes[0], cm, "Số lượng", "d"), (axes[1], norm, "Tỉ lệ theo hàng (recall)", ".2f")):
        ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(range(n), classes, rotation=30, ha="right")
        ax.set_yticks(range(n), classes)
        ax.set_xlabel("Dự đoán")
        ax.set_ylabel("Thực tế")
        ax.set_title(title)
        for i in range(n):
            for j in range(n):
                ax.text(j, i, format(data[i, j], fmt), ha="center", va="center",
                        color="white" if norm[i, j] > 0.5 else "black")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="models/vehicle_cnn.pt")
    p.add_argument("--dataset", default="data/dataset")
    p.add_argument("--split", default="test")
    p.add_argument("--outputs", default="outputs")
    args = p.parse_args()

    if not Path(args.model).exists():
        raise SystemExit(f"Chưa có mô hình {args.model} (chạy py -m train.train_cnn trước)")
    model, classes = load_checkpoint(args.model)
    X, y, paths = load_split(args.dataset, args.split, classes)
    if not len(X):
        raise SystemExit(f"Tập {args.split} trống")
    split_dir = Path(args.dataset) / args.split
    extra = [d.name for d in split_dir.iterdir() if d.is_dir() and d.name not in classes and list_images(d)]
    if extra:
        print(f"Bỏ qua thư mục không thuộc mô hình: {', '.join(extra)}")

    probs = predict_proba(model, X)
    pred = probs.argmax(1)
    labels = list(range(len(classes)))
    acc = accuracy_score(y, pred)
    report = classification_report(y, pred, labels=labels, target_names=classes, digits=4, zero_division=0)
    prec, rec, f1, support = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    macro = precision_recall_fscore_support(y, pred, labels=labels, average="macro", zero_division=0)[:3]
    cm = confusion_matrix(y, pred, labels=labels)
    ms_single, ms_batch = time_inference(model, X)

    text = "\n".join([
        f"Mô hình: {args.model}", f"Tập: {split_dir} ({len(X)} ảnh)", "",
        f"Accuracy: {acc:.4f}",
        f"Macro  Precision {macro[0]:.4f}  Recall {macro[1]:.4f}  F1 {macro[2]:.4f}", "", report,
        "Confusion matrix (hàng = thực tế, cột = dự đoán):",
        "            " + "".join(f"{c:>11}" for c in classes),
        *(f"{c:<12}" + "".join(f"{v:>11}" for v in row) for c, row in zip(classes, cm)), "",
        f"Thời gian chạy mạng trên CPU ({torch.get_num_threads()} luồng): "
        f"{ms_single:.2f} ms/ảnh (batch 1), {ms_batch:.3f} ms/ảnh (batch 32)",
    ])
    print(text)

    out = Path(args.outputs)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    (out / f"{args.split}_report.txt").write_text(text + "\n", encoding="utf-8")
    metrics = {
        "model": str(args.model), "split": args.split, "n_images": len(X), "classes": classes,
        "accuracy": acc, "precision_macro": macro[0], "recall_macro": macro[1], "f1_macro": macro[2],
        "per_class": {c: {"precision": prec[i], "recall": rec[i], "f1": f1[i], "support": int(support[i])}
                      for i, c in enumerate(classes)},
        "confusion_matrix": cm.tolist(), "ms_per_image_batch1": ms_single, "ms_per_image_batch32": ms_batch,
    }
    (out / f"{args.split}_metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2,
                                                               default=float), encoding="utf-8")
    plot_confusion(cm, classes, out / "figures" / f"confusion_matrix_{args.split}.png")

    wrong_dir = out / "misclassified"
    shutil.rmtree(wrong_dir, ignore_errors=True)
    wrong = np.flatnonzero(pred != y)
    for i in wrong:
        dst = wrong_dir / f"{classes[y[i]]}_thanh_{classes[pred[i]]}"
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copy2(paths[i], dst / f"{probs[i, pred[i]]:.2f}_{paths[i].name}")
    print(f"\nĐã ghi {out / f'{args.split}_report.txt'}, biểu đồ confusion matrix và {len(wrong)} ảnh sai "
          f"vào {wrong_dir}")


if __name__ == "__main__":
    main()
