import numpy as np
import pytest

from src.counter import LineCounter
from src.tracker import CentroidTracker

CLASSES = ["motorbike", "car", "bus", "truck", "other"]
CAR = [0.05, 0.85, 0.05, 0.05, 0.0]


def make(**kw):
    return LineCounter([[0, 500], [1000, 500]], CLASSES, **kw)


def drive(counter, ys, probs=CAR, x=100):
    """Cho một xe đi qua các toạ độ tâm y, mỗi khung hình phân loại một lần."""
    tracker = CentroidTracker(max_distance=200)
    for f, y in enumerate(ys, 1):
        tracks = tracker.update([(x - 10, y - 10, 20, 20)], f)
        for t in tracks:
            t.add_probs(np.array(probs, float), f)
        counter.update(tracks, f, f / 25)
    counter.finish()
    return counter


def total(counter):
    return sum(counter.totals().values())


def test_counts_once_when_crossing_down():
    c = drive(make(), [440, 460, 480, 500, 520, 540])
    assert c.counts["down"]["car"] == 1 and total(c) == 1
    assert c.events[0]["direction"] == "down" and c.events[0]["class"] == "car"


def test_jitter_around_line_counted_once():
    c = drive(make(), [470, 485, 495, 505, 495, 505, 495, 520, 560])
    assert total(c) == 1


def test_up_direction():
    c = drive(make(), [560, 540, 520, 500, 480, 460])
    assert c.counts["up"]["car"] == 1 and c.counts["down"]["car"] == 0


def test_direction_filter():
    c = drive(make(direction="down"), [560, 540, 520, 500, 480, 460])
    assert total(c) == 0 and c.rejected[0]["reason"] == "ngược chiều đếm"


def test_other_class_not_counted():
    c = drive(make(), [460, 480, 500, 520, 540], probs=[0, 0, 0, 0, 1])
    assert total(c) == 0 and "other" in c.rejected[0]["reason"]


def test_low_confidence_rejected():
    c = drive(make(min_confidence=0.5), [460, 480, 500, 520, 540], probs=[0.3, 0.25, 0.2, 0.25, 0])
    assert total(c) == 0


def test_object_jittering_on_line_not_counted():
    c = drive(make(min_travel=10), [497, 503, 498, 502, 499, 503])
    assert total(c) == 0


def test_vehicle_first_seen_near_line_still_counted():
    # xuất hiện ngay trên vạch: chờ xe đi thêm cho đủ min_travel rồi mới chốt
    c = drive(make(min_travel=10), [497, 503, 509, 515])
    assert total(c) == 1


def test_soft_vote_uses_summed_probabilities():
    tracker = CentroidTracker()
    t = tracker.update([(0, 0, 10, 10)], 1)[0]
    for probs in ([0.55, 0.45, 0, 0, 0], [0.55, 0.45, 0, 0, 0], [0.05, 0.95, 0, 0, 0]):
        t.add_probs(np.array(probs), 1)
    label, conf = t.label(CLASSES)
    assert label == "car" and conf == pytest.approx(1.85 / 3)


def test_vertical_line_directions():
    c = LineCounter([[500, 0], [500, 700]], CLASSES)
    assert c.names == ("left", "right")
    tracker = CentroidTracker(max_distance=200)
    for f, x in enumerate([560, 540, 520, 500, 480, 460], 1):
        tracks = tracker.update([(x - 10, 290, 20, 20)], f)
        for t in tracks:
            t.add_probs(np.array(CAR), f)
        c.update(tracks, f, f / 25)
    c.finish()
    assert c.counts["left"]["car"] == 1
