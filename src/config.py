"""Cấu hình mặc định và đọc file cấu hình YAML của từng video (configs/*.yaml)."""
import copy
from pathlib import Path

import yaml

# Giữ khớp với configs/example.yaml (tests/test_config.py kiểm tra điều này)
DEFAULT_CONFIG = {
    "roi": None,
    "line": None,
    "direction": "both",
    "process_width": 0,
    "warmup_frames": 60,
    "detector": {"history": 500, "var_threshold": 16, "min_area": 800, "max_area": 0,
                 "min_aspect": 0.2, "max_aspect": 5.0, "merge_ratio": 0.5},
    "tracker": {"max_distance": 100, "max_disappeared": 10},
    "classifier": {"classify_every": 3, "max_votes": 5},
    "counter": {"min_votes": 2, "max_wait": 15, "min_confidence": 0.3, "min_travel": 10},
}


def load_config(path=None):
    """Trộn file cấu hình (nếu có) vào cấu hình mặc định; báo lỗi khi có khoá gõ sai."""
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    if not path:
        return cfg
    user = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    unknown = set(user) - set(cfg)
    if unknown:
        raise SystemExit(f"Khoá không hợp lệ trong {path}: {', '.join(sorted(unknown))}")
    for key, value in user.items():
        if isinstance(cfg[key], dict):
            bad = set(value or {}) - set(cfg[key])
            if bad:
                raise SystemExit(f"Khoá không hợp lệ trong {path} -> {key}: {', '.join(sorted(bad))}")
            cfg[key].update(value or {})
        else:
            cfg[key] = value
    return cfg


def default_line(width, height):
    """Vạch ngang toàn khung ở 60% chiều cao, dùng khi config không khai báo line."""
    y = int(height * 0.6)
    return [[0, y], [width - 1, y]]
