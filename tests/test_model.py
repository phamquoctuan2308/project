import numpy as np
import torch

from train.data import prepare, to_tensor
from train.model import VehicleCNN, load_checkpoint, predict_proba, save_checkpoint, summary
from train.train_cnn import augment


def test_architecture_matches_report():
    model = VehicleCNN(4)
    x = torch.zeros(2, 3, 64, 64)
    assert model.features(x).shape == (2, 128, 6, 6)  # Flatten -> 4608
    assert model(x).shape == (2, 4)
    assert sum(p.numel() for p in model.parameters()) == 683_716
    assert "4608" in summary(model)


def test_prepare_and_to_tensor():
    img = np.zeros((30, 90, 3), np.uint8)
    img[..., 0] = 255  # kênh B của ảnh BGR
    x = prepare(img)
    assert x.shape == (64, 64, 3) and x[0, 0, 2] == 255  # sau khi đổi sang RGB, B nằm ở kênh cuối
    t = to_tensor(x[None])
    assert t.shape == (1, 3, 64, 64) and t.dtype == torch.float32 and float(t.max()) == 1.0


def test_checkpoint_roundtrip(tmp_path):
    model = VehicleCNN(3)
    save_checkpoint(model, ["a", "b", "c"], tmp_path / "m.pt")
    loaded, classes = load_checkpoint(tmp_path / "m.pt")
    X = np.random.default_rng(0).integers(0, 255, (5, 64, 64, 3), dtype=np.uint8)
    assert classes == ["a", "b", "c"] and (tmp_path / "labels.json").exists()
    assert np.allclose(predict_proba(model, X), predict_proba(loaded, X), atol=1e-6)
    assert np.allclose(predict_proba(loaded, X).sum(1), 1.0, atol=1e-5)


def test_augment_keeps_shape_and_range():
    x = torch.rand(8, 3, 64, 64)
    y = augment(x)
    assert y.shape == x.shape and float(y.min()) >= 0 and float(y.max()) <= 1
