"""Theo dõi phương tiện qua các khung hình bằng tâm xe (CentroidTracker)."""
import numpy as np
from scipy.optimize import linear_sum_assignment

_FAR = 1e9  # chi phí cho cặp vượt ngưỡng khoảng cách, không bao giờ được ghép


def centroid(box):
    x, y, w, h = box
    return (x + w // 2, y + h // 2)


class Track:
    """Một phương tiện đang được theo dõi: vị trí, vận tốc và kết quả phân loại tích luỹ."""

    def __init__(self, track_id, box, frame_idx):
        self.id = track_id
        self.box = box
        self.centroid = centroid(box)
        self.first_centroid = self.centroid
        self.prev_centroid = None     # tâm ở lần ghép trước, dùng để xét vượt vạch
        self.velocity = np.zeros(2)   # pixel / khung hình
        self.last_frame = frame_idx
        self.hits = 1
        self.missed = 0               # số khung hình liên tiếp không ghép được
        self.updated = True           # được ghép / tạo ở khung hình hiện tại
        self.active = True            # False khi đã bị xoá khỏi tracker
        self.prob_sum = None          # tổng xác suất CNN qua các lần phân loại (bỏ phiếu mềm)
        self.votes = 0
        self.last_classified = None

    def predict(self, frame_idx):
        """Vị trí tâm dự đoán theo vận tốc gần nhất (giúp ghép đúng khi xe đi sát nhau)."""
        return np.array(self.centroid, float) + self.velocity * (frame_idx - self.last_frame)

    def update(self, box, frame_idx):
        new = centroid(box)
        step = (np.array(new, float) - self.centroid) / max(1, frame_idx - self.last_frame)
        self.velocity = 0.5 * self.velocity + 0.5 * step
        self.prev_centroid, self.centroid, self.box = self.centroid, new, box
        self.last_frame, self.hits, self.missed, self.updated = frame_idx, self.hits + 1, 0, True

    def add_probs(self, probs, frame_idx):
        self.prob_sum = np.array(probs, float) if self.prob_sum is None else self.prob_sum + probs
        self.votes += 1
        self.last_classified = frame_idx

    def label(self, classes):
        """(lớp, độ tin cậy trung bình) theo bỏ phiếu mềm; (None, 0.0) nếu chưa phân loại lần nào."""
        if self.prob_sum is None:
            return None, 0.0
        i = int(np.argmax(self.prob_sum))
        return classes[i], float(self.prob_sum[i] / self.votes)


class CentroidTracker:
    """Ghép box giữa các khung hình theo khoảng cách tâm (thuật toán Hungarian, có ngưỡng)."""

    def __init__(self, max_distance=100, max_disappeared=10):
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared
        self.tracks = {}
        self.next_id = 1

    def update(self, boxes, frame_idx):
        """Nhận các box của khung hình mới; trả về các track được ghép hoặc tạo mới ở khung hình này."""
        ids = list(self.tracks)
        for t in self.tracks.values():
            t.updated = False
        unmatched = set(range(len(boxes)))
        if ids and boxes:
            pred = np.array([self.tracks[i].predict(frame_idx) for i in ids])
            cent = np.array([centroid(b) for b in boxes], float)
            dist = np.linalg.norm(pred[:, None] - cent[None], axis=2)
            cost = np.where(dist <= self.max_distance, dist, _FAR)
            for r, c in zip(*linear_sum_assignment(cost)):
                if cost[r, c] < _FAR:
                    self.tracks[ids[r]].update(boxes[c], frame_idx)
                    unmatched.discard(c)
        for i in ids:
            t = self.tracks[i]
            if not t.updated:
                t.missed += 1
                if t.missed > self.max_disappeared:
                    t.active = False
                    del self.tracks[i]
        for c in sorted(unmatched):
            self.tracks[self.next_id] = Track(self.next_id, boxes[c], frame_idx)
            self.next_id += 1
        return [t for t in self.tracks.values() if t.updated]
