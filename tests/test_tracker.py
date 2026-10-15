from src.tracker import CentroidTracker


def test_keeps_id_while_moving():
    tracker = CentroidTracker(max_distance=50, max_disappeared=3)
    ids = {tracker.update([(100, 10 * f, 20, 20)], f)[0].id for f in range(1, 11)}
    assert ids == {1}


def test_two_objects_keep_separate_ids():
    tracker = CentroidTracker(max_distance=50)
    for f in range(1, 8):
        tracks = tracker.update([(50, 10 * f, 20, 20), (300, 10 * f, 20, 20)], f)
    assert {t.centroid[0]: t.id for t in tracks} == {60: 1, 310: 2}


def test_close_objects_matched_by_predicted_position():
    # hai xe cách nhau 30 px cùng đi xuống 12 px / khung hình: vẫn giữ đúng ID
    tracker = CentroidTracker(max_distance=40)
    for f in range(1, 10):
        tracks = tracker.update([(100, 12 * f, 20, 20), (130, 12 * f, 20, 20)], f)
    assert {t.centroid[0]: t.id for t in tracks} == {110: 1, 140: 2}


def test_track_removed_after_disappearing():
    tracker = CentroidTracker(max_disappeared=2)
    track = tracker.update([(0, 0, 10, 10)], 1)[0]
    for f in range(2, 5):
        tracker.update([], f)
    assert not tracker.tracks and not track.active


def test_reappears_with_same_id():
    tracker = CentroidTracker(max_distance=60, max_disappeared=5)
    tracker.update([(100, 100, 20, 20)], 1)
    tracker.update([(100, 110, 20, 20)], 2)
    tracker.update([], 3)
    tracker.update([], 4)
    track = tracker.update([(100, 140, 20, 20)], 5)[0]
    assert track.id == 1 and track.prev_centroid == (110, 120)


def test_far_detection_creates_new_track():
    tracker = CentroidTracker(max_distance=30)
    tracker.update([(0, 0, 10, 10)], 1)
    tracks = tracker.update([(200, 200, 10, 10)], 2)
    assert [t.id for t in tracks] == [2]
