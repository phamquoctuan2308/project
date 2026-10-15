"""Phát hiện vùng chuyển động bằng MOG2 (MotionDetector)."""
import cv2
import numpy as np

CROP_PAD = 0.1  # nới khung cắt 10% mỗi phía; dùng chung khi cắt ảnh train và khi suy luận


def crop_box(frame, box, pad=CROP_PAD):
    """Cắt vùng (x, y, w, h) có nới thêm pad, không vượt ra ngoài khung hình."""
    x, y, w, h = box
    dx, dy = int(w * pad), int(h * pad)
    H, W = frame.shape[:2]
    return frame[max(0, y - dy):min(H, y + h + dy), max(0, x - dx):min(W, x + w + dx)]


def touches_border(box, region, margin=2):
    """Box chạm mép vùng xét (x, y, w, h): xe đang đi vào / ra, chỉ thấy một phần thân."""
    x, y, w, h = box
    rx, ry, rw, rh = region
    return x <= rx + margin or y <= ry + margin or x + w >= rx + rw - margin or y + h >= ry + rh - margin


def _overlap(a, b):
    """Diện tích giao / diện tích box nhỏ hơn."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    iw = min(ax + aw, bx + bw) - max(ax, bx)
    ih = min(ay + ah, by + bh) - max(ay, by)
    return iw * ih / min(aw * ah, bw * bh) if iw > 0 and ih > 0 else 0.0


def merge_boxes(boxes, ratio):
    """Gộp hai box khi phần giao chiếm > ratio diện tích box nhỏ hơn (mảnh vỡ của cùng một xe)."""
    boxes = list(boxes)
    merged = True
    while merged:
        merged = False
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if _overlap(boxes[i], boxes[j]) > ratio:
                    (ax, ay, aw, ah), (bx, by, bw, bh) = boxes[i], boxes[j]
                    x1, y1 = min(ax, bx), min(ay, by)
                    boxes[i] = (x1, y1, max(ax + aw, bx + bw) - x1, max(ay + ah, by + bh) - y1)
                    del boxes[j]
                    merged = True
                    break
            if merged:
                break
    return boxes


class MotionDetector:
    """Trừ nền MOG2 + morphology + lọc khung bao, trả về bounding box (x, y, w, h)."""

    def __init__(self, history=500, var_threshold=16, min_area=800, max_area=None,
                 min_aspect=0.2, max_aspect=5.0, merge_ratio=0.5, roi=None, close_size=7):
        self.subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history, varThreshold=var_threshold, detectShadows=True)
        self.min_area = min_area      # diện tích khung bao tối thiểu (pixel^2)
        self.max_area = max_area
        self.min_aspect = min_aspect  # giới hạn tỉ lệ rộng / cao
        self.max_aspect = max_aspect
        self.merge_ratio = merge_ratio
        self.roi = roi  # (x, y, w, h) hoặc None = cả khung hình
        self.open_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        self.close_kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_size, close_size))
        self._roi_mask = None

    def foreground_mask(self, frame):
        mask = self.subtractor.apply(frame)
        # MOG2 đánh dấu bóng = 127, tiền cảnh = 255 -> chỉ giữ tiền cảnh
        _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.open_kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.close_kernel)
        mask = cv2.dilate(mask, self.open_kernel, iterations=2)
        if self.roi is not None:
            if self._roi_mask is None or self._roi_mask.shape != mask.shape:
                self._roi_mask = np.zeros_like(mask)
                x, y, w, h = self.roi
                self._roi_mask[y:y + h, x:x + w] = 255
            mask = cv2.bitwise_and(mask, self._roi_mask)
        return mask

    def detect(self, frame):
        """Trả về (danh sách box (x, y, w, h), mask tiền cảnh)."""
        mask = self.foreground_mask(frame)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # Giữ cả mảnh nhỏ (>= 1/4 ngưỡng) để còn gộp được các mảnh của cùng một xe trước khi lọc
        boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= self.min_area / 4]
        if self.merge_ratio:
            boxes = merge_boxes(boxes, self.merge_ratio)
        keep = []
        for x, y, w, h in boxes:
            area = w * h
            if area < self.min_area or (self.max_area and area > self.max_area):
                continue
            if not self.min_aspect <= w / h <= self.max_aspect:
                continue
            keep.append((x, y, w, h))
        return keep, mask
