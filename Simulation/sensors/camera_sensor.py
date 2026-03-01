"""
Camera Sensor Module
PyBullet-based camera with RGB/depth capture and FOV-based detection.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


@dataclass
class CameraIntrinsics:
    """Pinhole camera parameters."""
    width:  int   = 640
    height: int   = 480
    fov:    float = 60.0
    near:   float = 0.1
    far:    float = 100.0

    @property
    def aspect(self) -> float:
        return self.width / self.height

    def projection_matrix(self):
        if not PYBULLET_AVAILABLE:
            return None
        return p.computeProjectionMatrixFOV(
            fov=self.fov, aspect=self.aspect,
            nearVal=self.near, farVal=self.far)


class CameraSensor:
    """
    Simulated camera sensor.
    Renders RGB+depth in PyBullet-GUI mode; falls back to FOV-based
    angular detection in headless / no-PyBullet mode.
    """

    def __init__(self, name: str,
                 position: np.ndarray,
                 target:    np.ndarray = None,
                 up_vector: np.ndarray = None,
                 intrinsics: CameraIntrinsics = None,
                 physics_client: int = None):
        self.name          = name
        self.position      = np.asarray(position, dtype=float)
        self.target        = np.asarray(target    if target    is not None else [0,0,0], dtype=float)
        self.up_vector     = np.asarray(up_vector if up_vector is not None else [0,0,1], dtype=float)
        self.intrinsics    = intrinsics or CameraIntrinsics()
        self.physics_client = physics_client
        self.rgb_image: Optional[np.ndarray]  = None
        self.depth_image: Optional[np.ndarray] = None
        self.detections: List[Dict]             = []
        self.max_range = 50.0
        self.fov_deg   = self.intrinsics.fov * self.intrinsics.aspect

    def capture(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Render a frame. Returns (rgb, depth) or (None, None)."""
        if not PYBULLET_AVAILABLE or self.physics_client is None:
            return None, None
        try:
            vm = p.computeViewMatrix(
                cameraEyePosition=self.position.tolist(),
                cameraTargetPosition=self.target.tolist(),
                cameraUpVector=self.up_vector.tolist(),
                physicsClientId=self.physics_client)
            pm = self.intrinsics.projection_matrix()
            w, h = self.intrinsics.width, self.intrinsics.height
            _, _, rgb, depth, _ = p.getCameraImage(
                width=w, height=h, viewMatrix=vm,
                projectionMatrix=pm, physicsClientId=self.physics_client)
            self.rgb_image   = np.reshape(rgb,   (h, w, 4))[:, :, :3]
            self.depth_image = np.reshape(depth, (h, w))
            return self.rgb_image, self.depth_image
        except Exception as exc:
            print(f"⚠  CameraSensor '{self.name}' capture failed: {exc}")
            return None, None

    def detect_objects(self, objects: List) -> List[Dict]:
        """FOV-based detection (works without PyBullet)."""
        detections = []
        fwd = self.target - self.position
        fwd_len = np.linalg.norm(fwd[:2])
        fwd_n = fwd / np.linalg.norm(fwd) if fwd_len > 1e-6 else np.array([1,0,0], dtype=float)
        half_fov = np.radians(self.fov_deg / 2.0)
        for obj in objects:
            pos = getattr(obj, 'position', None)
            if pos is None:
                continue
            rel  = np.asarray(pos, dtype=float) - self.position
            dist = np.linalg.norm(rel)
            if dist < 1e-3 or dist > self.max_range:
                continue
            cos_a = np.clip(np.dot(rel / dist, fwd_n), -1.0, 1.0)
            angle = np.arccos(cos_a)
            if angle <= half_fov:
                detections.append({
                    'sensor':     self.name,
                    'name':       getattr(obj, 'name', 'unknown'),
                    'type':       type(obj).__name__.lower(),
                    'position':   np.asarray(pos, dtype=float).copy(),
                    'distance':   float(dist),
                    'angle_deg':  float(np.degrees(angle)),
                    'confidence': float(np.clip(1.0 - dist / self.max_range, 0.1, 1.0)),
                })
        self.detections = detections
        return detections

    def set_position(self, position, target=None):
        self.position = np.asarray(position, dtype=float)
        if target is not None:
            self.target = np.asarray(target, dtype=float)

    def get_stats(self) -> Dict:
        return {'name': self.name, 'position': self.position.tolist(),
                'resolution': f"{self.intrinsics.width}x{self.intrinsics.height}",
                'fov': self.intrinsics.fov, 'detections': len(self.detections)}

    def __repr__(self):
        return f"CameraSensor('{self.name}', pos={self.position.tolist()})"


if __name__ == "__main__":
    print("Testing CameraSensor...")
    class MockObj:
        def __init__(self, n, pos):
            self.name = n; self.position = np.array(pos, dtype=float)
    cam = CameraSensor("front_cam", np.array([0,-20,5]), target=np.array([0,0,0]))
    dets = cam.detect_objects([MockObj("p1",[0,5,0]), MockObj("v1",[10,10,0]), MockObj("behind",[0,-30,0])])
    print(f"Detections: {len(dets)}")
    for d in dets:
        print(f"  {d['name']:12s}  dist={d['distance']:.1f}m  conf={d['confidence']:.2f}")
    print("✔  CameraSensor test complete")