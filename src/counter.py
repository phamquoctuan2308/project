"""Đếm phương tiện khi tâm xe vượt qua vạch ảo (LineCounter)."""
import numpy as np


class LineCounter:
    """Đếm mỗi track tối đa một lần khi tâm đi từ một phía của vạch sang phía kia.

    Vạch nằm ngang (|dx| >= |dy|): "down" = đi từ trên xuống qua vạch, "up" = từ dưới lên.
    Vạch nằm dọc: "left" = đi từ phải sang trái, "right" = từ trái sang phải.
    Với vạch y = Y: khung trước y < Y và khung này y >= Y -> "down" (đúng quy tắc trong báo cáo).
    Lượt vượt vạch được chốt khi track đã có đủ min_votes lần phân loại và đã dịch chuyển
    ít nhất min_travel pixel theo hướng qua vạch (hoặc khi track biến mất / chờ quá max_wait khung hình).
    """

    def __init__(self, line, classes, direction="both", min_votes=2, max_wait=15,
                 min_confidence=0.3, min_travel=10, ignore=("other",)):
        (x1, y1), (x2, y2) = line
        if abs(x2 - x1) >= abs(y2 - y1):
            if x1 > x2:
                (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
            self.names = ("down", "up")
        else:
            if y1 > y2:
                (x1, y1), (x2, y2) = (x2, y2), (x1, y1)
            self.names = ("left", "right")
        if direction != "both" and direction not in self.names:
            raise ValueError(f"direction phải là both hoặc {' / '.join(self.names)} với vạch này")
        self.p1, self.p2 = np.array([x1, y1], float), np.array([x2, y2], float)
        self.d = self.p2 - self.p1
        self.length = float(np.linalg.norm(self.d))
        self.classes = list(classes)
        self.allowed = set(self.names) if direction == "both" else {direction}
        self.min_votes, self.max_wait = min_votes, max_wait
        self.min_confidence, self.min_travel = min_confidence, min_travel
        self.counted_labels = [c for c in self.classes if c not in ignore]
        self.counts = {d: {c: 0 for c in self.counted_labels} for d in self.names}
        self.events, self.rejected, self.pending = [], [], []
        self.seen = set()  # id các track đã vượt vạch (đã đếm hoặc đã loại), không xét lại

    def signed_distance(self, p):
        """Khoảng cách có dấu (pixel) tới vạch; dương = phía mà chiều names[0] đi tới."""
        return float(self.d[0] * (p[1] - self.p1[1]) - self.d[1] * (p[0] - self.p1[0])) / self.length

    def _on_segment(self, p):
        t = float(np.dot(np.asarray(p, float) - self.p1, self.d)) / self.length ** 2
        return -0.05 <= t <= 1.05

    def _travel(self, track):
        return self.signed_distance(track.centroid) - self.signed_distance(track.first_centroid)

    def update(self, tracks, frame_idx, time_s):
        """Xét các track được cập nhật ở khung hình này; chốt các lượt vượt vạch đã đủ điều kiện."""
        for t in tracks:
            if t.id in self.seen or t.prev_centroid is None:
                continue
            s0, s1 = self.signed_distance(t.prev_centroid), self.signed_distance(t.centroid)
            if s0 < 0 <= s1:
                direction = self.names[0]
            elif s0 >= 0 > s1:
                direction = self.names[1]
            else:
                continue
            if not self._on_segment(np.add(t.prev_centroid, t.centroid) / 2):
                continue
            self.seen.add(t.id)
            event = {"track": t, "frame": frame_idx, "time_s": time_s, "direction": direction}
            if direction in self.allowed:
                self.pending.append(event)
            else:
                self._reject(event, None, 0.0, "ngược chiều đếm")
        self._flush(frame_idx)

    def _flush(self, frame_idx, force=False):
        keep = []
        for ev in self.pending:
            t = ev["track"]
            ready = t.votes >= self.min_votes and abs(self._travel(t)) >= self.min_travel
            if force or ready or not t.active or frame_idx - ev["frame"] >= self.max_wait:
                self._finalize(ev)
            else:
                keep.append(ev)
        self.pending = keep

    def _finalize(self, ev):
        t = ev["track"]
        label, conf = t.label(self.classes)
        travel = self._travel(t)
        sign = 1 if ev["direction"] == self.names[0] else -1
        if abs(travel) < self.min_travel or np.sign(travel) != sign:
            return self._reject(ev, label, conf, "không di chuyển rõ qua vạch")
        if label is None:
            return self._reject(ev, label, conf, "chưa được phân loại")
        if label not in self.counted_labels:
            return self._reject(ev, label, conf, f"lớp {label}")
        if conf < self.min_confidence:
            return self._reject(ev, label, conf, "độ tin cậy thấp")
        self.counts[ev["direction"]][label] += 1
        self.events.append(self._record(ev, label, conf))

    def _record(self, ev, label, conf):
        return {"time_s": round(ev["time_s"], 2), "frame": ev["frame"], "track_id": ev["track"].id,
                "class": label or "", "confidence": round(conf, 3), "direction": ev["direction"]}

    def _reject(self, ev, label, conf, reason):
        self.rejected.append({**self._record(ev, label, conf), "reason": reason})

    def finish(self):
        """Chốt mọi lượt còn chờ (gọi khi hết video)."""
        self._flush(None, force=True)

    def totals(self):
        return {c: sum(self.counts[d][c] for d in self.names) for c in self.counted_labels}
