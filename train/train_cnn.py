"""Huấn luyện Custom CNN phân loại phương tiện, lưu models/vehicle_cnn.pt.

Ví dụ:
    py -m train.train_cnn
    py -m train.train_cnn --classes motorbike car bus truck --epochs 60
Đọc ảnh từ data/dataset/{train,val}. Mặc định dùng mọi lớp có ảnh trong tập train
(thêm lớp "other" nếu thư mục train/other có ảnh); chỉ định lại bằng --classes.
Ghi outputs/train_log.csv, outputs/figures/training_curves.png, outputs/model_summary.txt.
"""
import argparse
import copy
import csv
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from train.data import load_split, resolve_classes, to_tensor
from train.model import VehicleCNN, save_checkpoint, summary


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def augment(x, rot_deg=10.0, zoom=0.1, shift=0.05, brightness=0.2, contrast=0.2):
    """Tăng cường dữ liệu theo batch (chỉ khi train): lật ngang, xoay, zoom, dịch, độ sáng, tương phản."""
    n = x.shape[0]

    def rnd(*shape):  # phân bố đều trong [-1, 1]
        return torch.rand(*shape) * 2 - 1

    angle = rnd(n) * math.radians(rot_deg)
    scale = 1 + rnd(n) * zoom
    flip = torch.where(torch.rand(n) < 0.5, -1.0, 1.0)
    cos, sin = torch.cos(angle) / scale, torch.sin(angle) / scale
    theta = torch.stack([
        torch.stack([cos * flip, -sin, rnd(n) * shift * 2], 1),
        torch.stack([sin * flip, cos, rnd(n) * shift * 2], 1),
    ], 1)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=False)
    mean = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - mean) * (1 + rnd(n, 1, 1, 1) * contrast) + mean
    x = x * (1 + rnd(n, 1, 1, 1) * brightness)
    return x.clamp_(0.0, 1.0)


def class_weights(y, num_classes):
    """Trọng số lớp tỉ lệ nghịch với số ảnh, bù cho lớp ít ảnh (xe buýt, xe tải)."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float64)
    return torch.tensor(len(y) / (num_classes * np.maximum(counts, 1)), dtype=torch.float32)


def run_epoch(model, X, y, loss_fn, optimizer=None, batch_size=32, use_augment=False):
    """Một lượt qua dữ liệu; có optimizer thì train, không thì chỉ đánh giá. Trả về (loss, accuracy)."""
    training = optimizer is not None
    model.train(training)
    order = torch.randperm(len(X)).numpy() if training else np.arange(len(X))
    total_loss = correct = 0.0
    with torch.set_grad_enabled(training):
        for i in range(0, len(X), batch_size):
            idx = order[i:i + batch_size]
            xb, yb = to_tensor(X[idx]), torch.from_numpy(y[idx])
            if use_augment:
                xb = augment(xb)
            logits = model(xb)
            loss = loss_fn(logits, yb)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(idx)
            correct += (logits.argmax(1) == yb).sum().item()
    return total_loss / len(X), correct / len(X)


def fit(model, Xtr, ytr, Xva, yva, epochs=50, batch_size=32, lr=1e-3, patience=8,
        use_augment=True, use_class_weight=True, log=print):
    """Train với Adam, giảm lr khi val_loss chững lại, dừng sớm theo val_loss.

    Trả về (lịch sử từng epoch, epoch tốt nhất); mô hình được nạp lại trọng số của epoch tốt nhất.
    """
    num_classes = model.classifier[-1].out_features
    train_loss_fn = nn.CrossEntropyLoss(weight=class_weights(ytr, num_classes) if use_class_weight else None)
    val_loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=3, min_lr=1e-5)
    history, best_loss, best_state, best_epoch, wait = [], math.inf, None, 0, 0
    for epoch in range(1, epochs + 1):
        t0 = time.perf_counter()
        loss, acc = run_epoch(model, Xtr, ytr, train_loss_fn, optimizer, batch_size, use_augment)
        val_loss, val_acc = run_epoch(model, Xva, yva, val_loss_fn, batch_size=256)
        scheduler.step(val_loss)
        history.append({"epoch": epoch, "loss": loss, "accuracy": acc, "val_loss": val_loss,
                        "val_accuracy": val_acc, "lr": optimizer.param_groups[0]["lr"]})
        log(f"epoch {epoch:3d}  loss {loss:.4f}  acc {acc:.4f}  val_loss {val_loss:.4f}  "
            f"val_acc {val_acc:.4f}  ({time.perf_counter() - t0:.1f}s)")
        if val_loss < best_loss - 1e-4:
            best_loss, best_state, best_epoch, wait = val_loss, copy.deepcopy(model.state_dict()), epoch, 0
        else:
            wait += 1
            if wait >= patience:
                log(f"Dừng sớm: val_loss không giảm sau {patience} epoch")
                break
    model.load_state_dict(best_state)
    log(f"Lấy trọng số epoch {best_epoch} (val_loss {best_loss:.4f})")
    return history, best_epoch


def plot_history(history, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, key, title in ((axes[0], "loss", "Loss"), (axes[1], "accuracy", "Accuracy")):
        ax.plot(epochs, [h[key] for h in history], label="train")
        ax.plot(epochs, [h["val_" + key] for h in history], label="val")
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="data/dataset")
    p.add_argument("--classes", nargs="+", help="danh sách lớp theo thứ tự (mặc định: tự phát hiện)")
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=8, help="dừng sớm sau N epoch val_loss không giảm")
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-augment", action="store_true", help="tắt tăng cường dữ liệu")
    p.add_argument("--no-class-weight", action="store_true", help="tắt trọng số lớp")
    p.add_argument("--out", default="models/vehicle_cnn.pt", help="nơi lưu mô hình")
    p.add_argument("--outputs", default="outputs", help="thư mục ghi log và biểu đồ")
    return p.parse_args()


def main():
    for stream in (sys.stdout, sys.stderr):  # console Windows mặc định cp1252, lỗi khi in tiếng Việt
        stream.reconfigure(encoding="utf-8")
    args = parse_args()
    seed_everything(args.seed)
    classes = resolve_classes(args.dataset, args.classes)
    print("Lớp:", ", ".join(f"{i}={c}" for i, c in enumerate(classes)))

    Xtr, ytr, _ = load_split(args.dataset, "train", classes)
    Xva, yva, _ = load_split(args.dataset, "val", classes)
    if not len(Xva):
        raise SystemExit("Tập val trống (chạy py -m tools.split_dataset trước)")
    counts = {c: (int((ytr == i).sum()), int((yva == i).sum())) for i, c in enumerate(classes)}
    for c, (n_tr, n_va) in counts.items():
        print(f"  {c:<10} train {n_tr:>6}   val {n_va:>6}")

    model = VehicleCNN(len(classes), args.dropout)
    table = summary(model)
    print(table)

    t0 = time.perf_counter()
    history, best_epoch = fit(model, Xtr, ytr, Xva, yva, args.epochs, args.batch, args.lr, args.patience,
                              not args.no_augment, not args.no_class_weight)
    minutes = (time.perf_counter() - t0) / 60
    best = history[best_epoch - 1]
    save_checkpoint(model, classes, args.out, best_epoch=best_epoch, val_accuracy=best["val_accuracy"],
                    val_loss=best["val_loss"], seed=args.seed, trained_at=time.strftime("%Y-%m-%d %H:%M"))

    out = Path(args.outputs)
    (out / "figures").mkdir(parents=True, exist_ok=True)
    with open(out / "train_log.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    plot_history(history, out / "figures" / "training_curves.png")
    lines = [table, "", "Lớp: " + ", ".join(classes),
             *(f"  {c:<10} train {a:>6}   val {b:>6}" for c, (a, b) in counts.items()),
             f"Epoch tốt nhất: {best_epoch}/{len(history)}  val_loss {best['val_loss']:.4f}  "
             f"val_accuracy {best['val_accuracy']:.4f}",
             f"Thời gian train: {minutes:.1f} phút (CPU, {torch.get_num_threads()} luồng)"]
    (out / "model_summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nĐã lưu mô hình: {args.out}  (val_accuracy {best['val_accuracy']:.4f}, epoch {best_epoch})")
    print(f"Log và biểu đồ: {out / 'train_log.csv'}, {out / 'figures' / 'training_curves.png'}")


if __name__ == "__main__":
    main()
