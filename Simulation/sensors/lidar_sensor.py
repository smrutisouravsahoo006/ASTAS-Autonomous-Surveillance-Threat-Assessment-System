"""
LiDAR Sensor Module
Spinning 2D/3D LiDAR that produces point clouds and cluster detections.
Works in headless mode (no PyBullet required) and with PyBullet ray-tests.
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


@dataclass
class LiDARConfig:
    """LiDAR hardware parameters."""
    max_range:          float = 50.0    # metres
    angular_resolution: float = 0.5    # degrees per ray (horizontal)
    vertical_layers:    int   = 1       # 1 = 2-D scan, >1 = 3-D
    vertical_fov:       float = 30.0   # total vertical FOV (degrees, used when layers > 1)
    noise_std:          float = 0.02   # Gaussian range noise (metres)


class LiDARSensor:
    """
    Simulated spinning LiDAR sensor.

    Modes
    -----
    • PyBullet ray-test  — accurate hit detection through physics scene
    • Headless / fallback — geometric sphere-intersection approximation

    Parameters
    ----------
    name           : unique sensor name
    position       : [x, y, z] world position
    config         : LiDARConfig instance
    physics_client : PyBullet client id (None → headless)
    """

    def __init__(self, name: str,
                 position: np.ndarray,
                 config: LiDARConfig = None,
                 physics_client: int = None):
        self.name           = name
        self.position       = np.asarray(position, dtype=float)
        self.config         = config or LiDARConfig()
        self.physics_client = physics_client
        self.point_cloud:   np.ndarray = np.empty((0, 3))
        self.detections:    List[Dict] = []

    # ------------------------------------------------------------------
    # Scanning
    # ------------------------------------------------------------------

    def scan(self, objects: List = None) -> np.ndarray:
        """
        Perform a full 360° scan.

        Returns
        -------
        np.ndarray of shape (N, 3) — hit points in world coordinates.
        """
        if PYBULLET_AVAILABLE and self.physics_client is not None:
            return self._scan_pybullet()
        else:
            return self._scan_geometric(objects or [])

    def _scan_pybullet(self) -> np.ndarray:
        """Use PyBullet batch ray-tests."""
        cfg   = self.config
        h_angles = np.arange(0, 360, cfg.angular_resolution)
        if cfg.vertical_layers <= 1:
            v_angles = [0.0]
        else:
            v_angles = np.linspace(-cfg.vertical_fov / 2,
                                    cfg.vertical_fov / 2,
                                    cfg.vertical_layers)

        ray_froms, ray_tos = [], []
        for v_deg in v_angles:
            v_rad = np.radians(v_deg)
            cos_v = np.cos(v_rad)
            sin_v = np.sin(v_rad)
            for h_deg in h_angles:
                h_rad = np.radians(h_deg)
                direction = np.array([cos_v * np.cos(h_rad),
                                      cos_v * np.sin(h_rad),
                                      sin_v])
                ray_froms.append(self.position.tolist())
                ray_tos.append((self.position + direction * cfg.max_range).tolist())

        try:
            results = p.rayTestBatch(ray_froms, ray_tos,
                                     physicsClientId=self.physics_client)
        except Exception:
            return np.empty((0, 3))

        points = []
        for res in results:
            if res[0] >= 0:           # hit something
                hit_pos = np.array(res[3])
                # Add noise
                noise = np.random.normal(0, cfg.noise_std, 3)
                points.append(hit_pos + noise)

        self.point_cloud = np.array(points) if points else np.empty((0, 3))
        return self.point_cloud

    def _scan_geometric(self, objects: List) -> np.ndarray:
        """Geometric sphere-intersection fallback (no PyBullet needed)."""
        cfg      = self.config
        h_angles = np.arange(0, 360, cfg.angular_resolution)
        if cfg.vertical_layers <= 1:
            v_angles = [0.0]
        else:
            v_angles = np.linspace(-cfg.vertical_fov / 2,
                                    cfg.vertical_fov / 2,
                                    cfg.vertical_layers)

        points = []
        for v_deg in v_angles:
            v_rad = np.radians(v_deg)
            cos_v = np.cos(v_rad)
            sin_v = np.sin(v_rad)
            for h_deg in h_angles:
                h_rad = np.radians(h_deg)
                ray_dir = np.array([cos_v * np.cos(h_rad),
                                    cos_v * np.sin(h_rad),
                                    sin_v])
                min_dist = cfg.max_range
                hit      = None
                for obj in objects:
                    pos = getattr(obj, 'position', None)
                    if pos is None:
                        continue
                    to_obj   = np.asarray(pos, dtype=float) - self.position
                    proj     = np.dot(to_obj, ray_dir)
                    if proj <= 0:
                        continue
                    closest  = self.position + ray_dir * proj
                    r        = getattr(obj, 'radius',
                                       np.linalg.norm(
                                           getattr(obj, 'size', np.array([1,1,1]))) / 2)
                    if np.linalg.norm(closest - np.asarray(pos, dtype=float)) < r:
                        dist = proj
                        if dist < min_dist:
                            min_dist = dist
                            hit      = self.position + ray_dir * dist
                if hit is not None:
                    noise = np.random.normal(0, cfg.noise_std, 3)
                    points.append(hit + noise)

        self.point_cloud = np.array(points) if points else np.empty((0, 3))
        return self.point_cloud

    # ------------------------------------------------------------------
    # Cluster detection
    # ------------------------------------------------------------------

    def detect_clusters(self, min_points: int = 3,
                         cluster_radius: float = 1.5) -> List[Dict]:
        """
        Simple DBSCAN-like cluster detection on the last point cloud.
        Returns a list of cluster dicts with centroid and point count.
        """
        if len(self.point_cloud) == 0:
            self.detections = []
            return []

        unassigned = list(range(len(self.point_cloud)))
        clusters   = []

        while unassigned:
            seed_idx  = unassigned.pop(0)
            seed_pt   = self.point_cloud[seed_idx]
            cluster   = [seed_idx]
            neighbors = [i for i in unassigned
                         if np.linalg.norm(self.point_cloud[i] - seed_pt) < cluster_radius]
            cluster.extend(neighbors)
            for n in neighbors:
                unassigned.remove(n)

            if len(cluster) >= min_points:
                pts      = self.point_cloud[cluster]
                centroid = pts.mean(axis=0)
                clusters.append({
                    'centroid':    centroid,
                    'num_points':  len(cluster),
                    'distance':    float(np.linalg.norm(centroid - self.position)),
                    'extent':      float(np.max(np.linalg.norm(
                                       pts - centroid, axis=1))),
                })

        self.detections = clusters
        return clusters

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def set_position(self, position: np.ndarray):
        self.position = np.asarray(position, dtype=float)

    def get_stats(self) -> Dict:
        return {
            'name':         self.name,
            'position':     self.position.tolist(),
            'max_range':    self.config.max_range,
            'resolution':   self.config.angular_resolution,
            'layers':       self.config.vertical_layers,
            'points':       len(self.point_cloud),
            'clusters':     len(self.detections),
        }

    def __repr__(self):
        return (f"LiDARSensor('{self.name}', "
                f"pos={self.position.tolist()}, "
                f"range={self.config.max_range}m)")


# =============================================================================
# Quick test
# =============================================================================
if __name__ == "__main__":
    print("Testing LiDARSensor...")

    class MockObj:
        def __init__(self, n, pos, r=1.0):
            self.name = n
            self.position = np.array(pos, dtype=float)
            self.radius   = r

    lidar = LiDARSensor("lidar_1", np.array([0, 0, 1.0]),
                        config=LiDARConfig(max_range=30.0, angular_resolution=1.0))
    print(lidar)

    objs = [MockObj("person", [10, 0, 0]), MockObj("car", [0, 15, 0], r=2.0)]
    pts  = lidar.scan(objs)
    print(f"Points: {len(pts)}")

    clusters = lidar.detect_clusters()
    print(f"Clusters: {len(clusters)}")
    for c in clusters:
        print(f"  centroid={c['centroid'][:2]}  pts={c['num_points']}  dist={c['distance']:.1f}m")

    # 3-D scan
    lidar3d = LiDARSensor("lidar_3d", np.array([0, 0, 2.0]),
                          config=LiDARConfig(vertical_layers=4, vertical_fov=20.0,
                                             angular_resolution=2.0))
    pts3d = lidar3d.scan(objs)
    print(f"3-D scan points: {len(pts3d)}")

    print("✔  LiDARSensor test complete")
