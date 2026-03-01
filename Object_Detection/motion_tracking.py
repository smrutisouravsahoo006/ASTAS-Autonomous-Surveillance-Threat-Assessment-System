import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import deque


@dataclass
class Track:
    track_id: int
    class_name: str
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    velocity: Tuple[float, float] = (0.0, 0.0)
    trajectory: deque = field(default_factory=lambda: deque(maxlen=30))
    age: int = 0
    hits: int = 0
    time_since_update: int = 0
    state: str = "tentative"   # tentative, confirmed, deleted
    confidence: float = 0.0

    def __post_init__(self):
        # FIX: was appending self (Track object) instead of center coords
        self.trajectory.append(self.center)


def _iou(boxA: Tuple, boxB: Tuple) -> float:
    """Compute Intersection-over-Union between two bboxes (x1,y1,x2,y2)."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0, xB - xA) * max(0, yB - yA)
    if inter == 0:
        return 0.0
    areaA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    areaB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    return inter / float(areaA + areaB - inter)


class MotionTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3,
                 iou_threshold: float = 0.3):
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: List[Track] = []
        self.next_track_id = 1
        self.frame_count = 0
        self.prev_gray = None
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracks with a new set of detections.

        Args:
            detections: list of dicts with keys:
                        'bbox' (x1,y1,x2,y2), 'class_name', 'confidence'
        Returns:
            List of active (confirmed + tentative) Track objects.
        """
        self.frame_count += 1

        # --- Step 1: predict / age existing tracks -----------------------
        for t in self.tracks:
            t.age += 1
            t.time_since_update += 1
            # Simple constant-velocity prediction
            cx, cy = t.center
            vx, vy = t.velocity
            pred_cx = int(cx + vx)
            pred_cy = int(cy + vy)
            x1, y1, x2, y2 = t.bbox
            w, h = x2 - x1, y2 - y1
            t.bbox = (pred_cx - w // 2, pred_cy - h // 2,
                      pred_cx + w // 2, pred_cy + h // 2)
            t.center = (pred_cx, pred_cy)

        # --- Step 2: greedy IoU matching ---------------------------------
        matched_track_ids = set()
        matched_det_ids = set()

        if self.tracks and detections:
            iou_matrix = np.zeros((len(self.tracks), len(detections)))
            for ti, trk in enumerate(self.tracks):
                for di, det in enumerate(detections):
                    iou_matrix[ti, di] = _iou(trk.bbox, det['bbox'])

            # Greedy match: best IoU first
            flat_order = np.argsort(-iou_matrix, axis=None)
            for idx in flat_order:
                ti, di = divmod(idx, len(detections))
                if iou_matrix[ti, di] < self.iou_threshold:
                    break
                if ti in matched_track_ids or di in matched_det_ids:
                    continue
                # Update matched track
                det = detections[di]
                trk = self.tracks[ti]
                x1, y1, x2, y2 = det['bbox']
                new_center = ((x1 + x2) // 2, (y1 + y2) // 2)
                trk.velocity = (new_center[0] - trk.center[0],
                                new_center[1] - trk.center[1])
                trk.bbox = det['bbox']
                trk.center = new_center
                trk.hits += 1
                trk.time_since_update = 0
                trk.confidence = det.get('confidence', 0.0)
                trk.trajectory.append(new_center)
                if trk.hits >= self.min_hits:
                    trk.state = "confirmed"
                matched_track_ids.add(ti)
                matched_det_ids.add(di)

        # --- Step 3: spawn new tracks for unmatched detections -----------
        for di, det in enumerate(detections):
            if di not in matched_det_ids:
                x1, y1, x2, y2 = det['bbox']
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                new_track = Track(
                    track_id=self.next_track_id,
                    class_name=det.get('class_name', 'unknown'),
                    bbox=det['bbox'],
                    center=center,
                    confidence=det.get('confidence', 0.0)
                )
                new_track.hits = 1
                self.tracks.append(new_track)
                self.next_track_id += 1

        # --- Step 4: delete stale tracks ---------------------------------
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.max_age
        ]

        return [t for t in self.tracks if t.state in ("confirmed", "tentative")]

    def analyze_track_behavior(self, track: Track) -> Dict:
        """
        Analyse a single track's historical trajectory.

        Returns:
            dict with keys: loitering, rapid_movement, direction_changes,
                            avg_speed, total_distance
        """
        traj = list(track.trajectory)
        if len(traj) < 2:
            return {
                "loitering": False,
                "rapid_movement": False,
                "direction_changes": 0,
                "avg_speed": 0.0,
                "total_distance": 0.0
            }

        speeds = []
        directions = []
        total_dist = 0.0
        for i in range(1, len(traj)):
            dx = traj[i][0] - traj[i - 1][0]
            dy = traj[i][1] - traj[i - 1][1]
            dist = np.hypot(dx, dy)
            total_dist += dist
            speeds.append(dist)
            if dist > 0:
                directions.append(np.arctan2(dy, dx))

        avg_speed = float(np.mean(speeds)) if speeds else 0.0

        # Count significant direction changes
        direction_changes = 0
        for i in range(1, len(directions)):
            angle_diff = abs(directions[i] - directions[i - 1])
            if angle_diff > np.pi:
                angle_diff = 2 * np.pi - angle_diff
            if angle_diff > np.radians(45):
                direction_changes += 1

        # Bounding box of trajectory → loitering heuristic
        xs = [p[0] for p in traj]
        ys = [p[1] for p in traj]
        spread = np.hypot(max(xs) - min(xs), max(ys) - min(ys))
        loitering = spread < 50 and len(traj) > 15

        rapid_movement = avg_speed > 20.0  # pixels per frame

        return {
            "loitering": loitering,
            "rapid_movement": rapid_movement,
            "direction_changes": direction_changes,
            "avg_speed": avg_speed,
            "total_distance": total_dist
        }

    def draw_tracks(self, frame: np.ndarray,
                    tracks: Optional[List[Track]] = None,
                    draw_trajectory: bool = True) -> np.ndarray:
        """Draw bounding boxes and trajectories on frame."""
        output = frame.copy()
        if tracks is None:
            tracks = self.tracks

        colors = {
            "confirmed": (0, 255, 0),
            "tentative": (0, 165, 255),
            "deleted":   (0, 0, 255),
        }

        for track in tracks:
            color = colors.get(track.state, (255, 255, 255))
            x1, y1, x2, y2 = track.bbox

            # Bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Label
            label = f"ID:{track.track_id} {track.class_name}"
            cv2.putText(output, label, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)

            # Trajectory
            if draw_trajectory:
                traj = list(track.trajectory)
                for i in range(1, len(traj)):
                    alpha = i / len(traj)
                    fade = tuple(int(c * alpha) for c in color)
                    cv2.line(output, traj[i - 1], traj[i], fade, 1)

        # HUD
        cv2.putText(output, f"Tracks: {len(tracks)}  Frame: {self.frame_count}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        return output


if __name__ == "__main__":
    print("Testing MotionTracker...")
    tracker = MotionTracker(max_age=10, min_hits=2)

    # Simulate 30 frames with two moving detections
    for frame_idx in range(30):
        dets = [
            {
                'bbox': (100 + frame_idx * 5, 100, 160 + frame_idx * 5, 200),
                'class_name': 'person',
                'confidence': 0.85
            },
            {
                'bbox': (400, 200 + frame_idx * 3, 480, 320 + frame_idx * 3),
                'class_name': 'car',
                'confidence': 0.90
            }
        ]
        active = tracker.update(dets)
        if frame_idx % 10 == 0:
            print(f"Frame {frame_idx}: {len(active)} active tracks")

    for t in tracker.tracks:
        behavior = tracker.analyze_track_behavior(t)
        print(f"Track {t.track_id} ({t.class_name}): {behavior}")

    # Draw on blank frame
    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    output = tracker.draw_tracks(blank)
    print(f"Output frame shape: {output.shape}")
    print("Motion tracker test complete.")