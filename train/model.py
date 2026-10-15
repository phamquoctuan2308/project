"""Kiến trúc Custom CNN phân loại phương tiện, lưu / nạp checkpoint, suy luận theo batch."""
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from train.data import IMG_SIZE, to_tensor


class VehicleCNN(nn.Module):
    """3 khối Conv 3x3 + ReLU + MaxPool 2x2 -> Flatten 4608 -> Dense 128 + ReLU -> Dropout -> Softmax.

    Lớp cuối trả về logits: Softmax nằm trong CrossEntropyLoss khi train và được áp dụng
    tường minh khi suy luận (predict_proba), tương đương lớp Dense + Softmax trong báo cáo.
    """

    def __init__(self, num_classes, dropout=0.5):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2),     # 64 -> 62 -> 31
            nn.Conv2d(32, 64, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2),    # 31 -> 29 -> 14
            nn.Conv2d(64, 128, 3), nn.ReLU(inplace=True), nn.MaxPool2d(2),   # 14 -> 12 -> 6
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),                                                     # 6 x 6 x 128 = 4608
            nn.Linear(6 * 6 * 128, 128), nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def summary(model):
    """Bảng lớp / kích thước đầu ra (H x W x C) / số tham số, tương tự model.summary() của Keras."""
    rows = [f"{'Lớp':<12}{'Đầu ra':>16}{'Tham số':>12}", "-" * 40,
            f"{'Input':<12}{f'{IMG_SIZE}x{IMG_SIZE}x3':>16}{0:>12,}"]
    x = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE)
    with torch.no_grad():
        for layer in [*model.features, *model.classifier]:
            x = layer(x)
            shape = f"{x.shape[2]}x{x.shape[3]}x{x.shape[1]}" if x.dim() == 4 else str(x.shape[1])
            params = sum(p.numel() for p in layer.parameters())
            rows.append(f"{type(layer).__name__:<12}{shape:>16}{params:>12,}")
    rows += ["-" * 40, f"Tổng số tham số: {sum(p.numel() for p in model.parameters()):,}"]
    return "\n".join(rows)


def save_checkpoint(model, classes, path, **meta):
    """Lưu trọng số + thứ tự lớp vào .pt, kèm bản sao labels.json cùng thư mục cho dễ đọc."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "classes": list(classes), "img_size": IMG_SIZE,
                "dropout": model.classifier[3].p, **meta}, path)
    labels = {"classes": list(classes), "img_size": IMG_SIZE}
    (path.parent / "labels.json").write_text(json.dumps(labels, ensure_ascii=False, indent=2), encoding="utf-8")


def load_checkpoint(path):
    """Nạp checkpoint -> (mô hình ở chế độ eval, danh sách lớp)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    model = VehicleCNN(len(ckpt["classes"]), ckpt.get("dropout", 0.5))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, list(ckpt["classes"])


@torch.inference_mode()
def predict_proba(model, X, batch_size=256):
    """X uint8 (N, 64, 64, 3) RGB -> xác suất (N, số lớp)."""
    model.eval()
    out = [torch.softmax(model(to_tensor(X[i:i + batch_size])), 1) for i in range(0, len(X), batch_size)]
    return torch.cat(out).numpy() if out else np.zeros((0, model.classifier[-1].out_features), np.float32)
