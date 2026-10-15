from pathlib import Path

import numpy as np
import pytest
import yaml

from src.config import DEFAULT_CONFIG, load_config
from tools.eval_counting import count_table, match_events
from tools.split_dataset import best_assignment, time_split
from train.data import group_of, video_and_frame


def test_example_config_matches_defaults():
    example = Path(__file__).resolve().parents[1] / "configs" / "example.yaml"
    assert yaml.safe_load(example.read_text(encoding="utf-8")) == DEFAULT_CONFIG


def test_config_rejects_typos(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("detector:\n  min_aera: 5\n", encoding="utf-8")
    with pytest.raises(SystemExit):
        load_config(path)


def test_video_assignment_covers_all_classes():
    counts = np.array([[100, 50, 5, 8], [120, 40, 3, 0], [80, 60, 4, 6],
                       [90, 30, 0, 5], [110, 45, 6, 7], [100, 50, 2, 4]])
    assign = best_assignment(counts)
    for k in range(3):
        assert counts[assign == k].sum(0).min() > 0  # mỗi tập có đủ lớp
    assert (assign == 0).sum() > (assign == 1).sum()


def test_time_split_leaves_gap():
    frames = list(range(0, 1001, 5))
    parts = time_split(frames, gap=50)
    assert {0, 1, 2} <= set(parts)
    for f, k in zip(frames, parts):
        if k is not None:
            assert abs(f - 700) >= 50 and abs(f - 850) >= 50


def test_crop_name_parsing():
    assert video_and_frame("data/crops/v1/car/nga_tu_1_f000123_2.jpg") == ("nga_tu_1", 123)
    assert video_and_frame("miotcd_car_00001.jpg") == (None, None)
    assert group_of("v1_f000100_0.jpg", 900) == group_of("v1_f000800_3.jpg", 900) != group_of("v1_f000950_0.jpg", 900)


def ev(t, c, d="down"):
    return {"time_s": t, "class": c, "direction": d}


def test_matching_prefers_same_class_and_finds_errors():
    gt = [ev(10, "car"), ev(10.5, "motorbike"), ev(20, "bus")]
    pred = [ev(10.4, "motorbike"), ev(10.9, "car"), ev(30, "truck")]
    matches, missed, extra = match_events(gt, pred, tol=1.5)
    assert sorted(matches) == [(0, 1), (1, 0)] and missed == [2] and extra == [2]


def test_count_table_uses_report_formula():
    rows, total, _ = count_table({"motorbike": 158, "car": 45, "bus": 5, "truck": 11},
                                 {"motorbike": 150, "car": 45, "bus": 5, "truck": 12})
    assert (total["machine"], total["manual"], total["diff"]) == (212, 219, -7)
    assert total["accuracy"] == pytest.approx(96.80, abs=0.01)
    assert {r["class"]: r["diff"] for r in rows} == {"motorbike": -8, "car": 0, "bus": 0, "truck": 1}
