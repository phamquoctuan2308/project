import numpy as np

from src.detector import MotionDetector, crop_box, merge_boxes


def frame_with(rect=None):
    frame = np.full((240, 320, 3), 100, np.uint8)
    if rect:
        x, y, w, h = rect
        frame[y:y + h, x:x + w] = (0, 0, 220)
    return frame


def test_detects_moving_rectangle():
    det = MotionDetector(min_area=200)
    for _ in range(30):
        det.detect(frame_with())
    for i in range(5):
        boxes, _ = det.detect(frame_with((100, 50 + 5 * i, 40, 60)))
    assert len(boxes) == 1
    x, y, w, h = boxes[0]
    assert abs(x - 100) <= 6 and abs(w - 40) <= 12 and abs(h - 60) <= 12


def test_roi_ignores_motion_outside():
    det = MotionDetector(min_area=200, roi=(0, 0, 80, 240))
    for _ in range(30):
        det.detect(frame_with())
    boxes, _ = det.detect(frame_with((200, 50, 40, 60)))
    assert boxes == []


def test_merge_boxes():
    assert merge_boxes([(0, 0, 10, 10), (5, 5, 10, 10)], 0.2) == [(0, 0, 15, 15)]
    assert len(merge_boxes([(0, 0, 10, 10), (5, 5, 10, 10)], 0.5)) == 2
    assert len(merge_boxes([(0, 0, 10, 10), (20, 20, 5, 5)], 0.1)) == 2


def test_crop_box_pads_and_clips():
    frame = np.zeros((100, 100, 3), np.uint8)
    assert crop_box(frame, (0, 0, 50, 50), 0.1).shape == (55, 55, 3)
    assert crop_box(frame, (40, 40, 50, 50), 0.1).shape == (60, 60, 3)
