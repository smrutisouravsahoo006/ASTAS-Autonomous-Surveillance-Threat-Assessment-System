import cv2
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple
import time


# ============================================================
# Detection Data Structure
# ============================================================

@dataclass
class Detection:
    class_id: int
    class_name: str
    confidence: float
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    area: int
    timestamp: float


# ============================================================
# Object Detector Class
# ============================================================

class ObjectDetector:

    def __init__(self,
                 model_path: str = "yolov8n.pt",
                 confidence_threshold: float = 0.5,
                 iou_threshold: float = 0.45,
                 device: str = "cpu"):

        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = device

        # COCO class names
        self.class_names = [
            'person','bicycle','car','motorcycle','airplane','bus','train','truck',
            'boat','traffic light','fire hydrant','stop sign','parking meter','bench',
            'bird','cat','dog','horse','sheep','cow','elephant','bear','zebra',
            'giraffe','backpack','umbrella','handbag','tie','suitcase','frisbee',
            'skis','snowboard','sports ball','kite','baseball bat','baseball glove',
            'skateboard','surfboard','tennis racket','bottle','wine glass','cup',
            'fork','knife','spoon','bowl','banana','apple','sandwich','orange',
            'broccoli','carrot','hot dog','pizza','donut','cake','chair','couch',
            'potted plant','bed','dining table','toilet','tv','laptop','mouse',
            'remote','keyboard','cell phone','microwave','oven','toaster','sink',
            'refrigerator','book','clock','vase','scissors','teddy bear','hair drier',
            'toothbrush'
        ]

        # Surveillance relevant classes
        self.surveillance_classes = {
            'person','car','truck','bus','motorcycle','bicycle',
            'dog','cat','backpack','handbag','suitcase'
        }

        self.model = None
        self.inference_times = []

        try:
            from ultralytics import YOLO
            self.model = YOLO(model_path)
            print(f"Loaded YOLO model: {model_path}")
        except Exception as e:
            print("Warning: Could not load YOLO model.")
            print("Using mock detector instead.")
            print("Error:", e)


    # ========================================================
    # Detection Function
    # ========================================================

    def detect(self, frame: np.ndarray, filter_classes: bool = True) -> List[Detection]:

        start_time = time.time()
        detections = []

        if self.model is not None:

            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )

            for result in results:
                if result.boxes is None:
                    continue

                for box in result.boxes:

                    class_id = int(box.cls[0])
                    class_name = self.class_names[class_id]
                    confidence = float(box.conf[0])

                    if filter_classes and class_name not in self.surveillance_classes:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    center = ((x1 + x2) // 2, (y1 + y2) // 2)
                    area = (x2 - x1) * (y2 - y1)

                    detections.append(
                        Detection(
                            class_id=class_id,
                            class_name=class_name,
                            confidence=confidence,
                            bbox=(x1, y1, x2, y2),
                            center=center,
                            area=area,
                            timestamp=time.time()
                        )
                    )

        else:
            detections = self._mock_detect(frame)

        # Performance tracking
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)

        if len(self.inference_times) > 100:
            self.inference_times.pop(0)

        return detections


    # ========================================================
    # Mock Detection (Fallback)
    # ========================================================

    def _mock_detect(self, frame: np.ndarray) -> List[Detection]:

        height, width = frame.shape[:2]
        detections = []

        for _ in range(np.random.randint(0, 4)):

            class_name = np.random.choice(list(self.surveillance_classes))
            class_id = self.class_names.index(class_name)

            x1 = np.random.randint(0, width - 100)
            y1 = np.random.randint(0, height - 100)
            w = np.random.randint(50, 200)
            h = np.random.randint(80, 250)

            x2 = min(x1 + w, width)
            y2 = min(y1 + h, height)

            detections.append(
                Detection(
                    class_id=class_id,
                    class_name=class_name,
                    confidence=np.random.uniform(0.6, 0.95),
                    bbox=(x1, y1, x2, y2),
                    center=((x1 + x2)//2, (y1 + y2)//2),
                    area=(x2 - x1) * (y2 - y1),
                    timestamp=time.time()
                )
            )

        return detections


    # ========================================================
    # Draw Detections
    # ========================================================

    def draw_detections(self, frame: np.ndarray,
                        detections: List[Detection],
                        show_confidence: bool = True):

        output = frame.copy()

        for det in detections:

            x1, y1, x2, y2 = det.bbox
            color = self._get_class_color(det.class_name)

            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            label = det.class_name
            if show_confidence:
                label += f" {det.confidence:.2f}"

            (tw, th), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            cv2.rectangle(output, (x1, y1 - th - 5),
                          (x1 + tw, y1), color, -1)

            cv2.putText(output, label,
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 255, 255), 2)

            cv2.circle(output, det.center, 4, color, -1)

        # FPS display
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0
            cv2.putText(output, f"FPS: {fps:.1f}",
                        (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2)

        return output


    # ========================================================
    # Analytics
    # ========================================================

    def analyze_detections(self, detections: List[Detection]) -> Dict:

        stats = {
            "total_count": len(detections),
            "class_counts": {},
            "average_confidence": 0.0,
            "total_area": 0,
            "largest_object": None
        }

        if not detections:
            return stats

        stats["average_confidence"] = np.mean([d.confidence for d in detections])
        stats["total_area"] = sum(d.area for d in detections)

        for d in detections:
            stats["class_counts"][d.class_name] = \
                stats["class_counts"].get(d.class_name, 0) + 1

        largest = max(detections, key=lambda d: d.area)

        stats["largest_object"] = {
            "class": largest.class_name,
            "area": largest.area,
            "confidence": largest.confidence
        }

        return stats


    # ========================================================
    # Performance Stats
    # ========================================================

    def get_performance_stats(self) -> Dict:

        if not self.inference_times:
            return {"avg_fps": 0, "avg_latency": 0}

        avg_time = np.mean(self.inference_times)

        return {
            "avg_fps": 1.0 / avg_time if avg_time > 0 else 0,
            "avg_latency": avg_time * 1000,
            "min_latency": min(self.inference_times) * 1000,
            "max_latency": max(self.inference_times) * 1000
        }


    # ========================================================
    # Color Map
    # ========================================================

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:

        colors = {
            "person": (0, 255, 0),
            "car": (255, 0, 0),
            "truck": (255, 100, 0),
            "bus": (255, 150, 0),
            "motorcycle": (0, 165, 255),
            "bicycle": (0, 255, 255)
        }

        return colors.get(class_name, (255, 255, 255))


# ============================================================
# Test Block
# ============================================================

if __name__ == "__main__":

    print("Testing Object Detection Module...")

    detector = ObjectDetector()

    test_frame = np.random.randint(
        0, 255, (720, 1280, 3), dtype=np.uint8
    )

    detections = detector.detect(test_frame)

    print(f"Detected {len(detections)} objects")

    stats = detector.analyze_detections(detections)
    print("Stats:", stats)

    perf = detector.get_performance_stats()
    print("Performance:", perf)

    output = detector.draw_detections(test_frame, detections)
    print("Output frame shape:", output.shape)
