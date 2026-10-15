"""Nạp Custom CNN và phân loại ảnh cắt phương tiện (VehicleClassifier)."""
import numpy as np
import torch

from train.data import prepare, to_tensor
from train.model import load_checkpoint


class VehicleClassifier:
    """Bọc mô hình đã train: nhận ảnh cắt BGR từ khung hình, trả xác suất từng lớp."""

    def __init__(self, model_path="models/vehicle_cnn.pt"):
        self.model, self.classes = load_checkpoint(model_path)

    @torch.inference_mode()
    def predict_proba(self, crops):
        """Danh sách ảnh cắt BGR -> mảng xác suất (N, số lớp), chạy một batch."""
        if not crops:
            return np.zeros((0, len(self.classes)), np.float32)
        x = to_tensor(np.stack([prepare(c) for c in crops]))
        return torch.softmax(self.model(x), 1).numpy()
