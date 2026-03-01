"""
LiDAR Sensor - Advanced 3D Scanning
Realistic LiDAR simulation with point cloud generation
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from enum import Enum


class LiDARType(Enum):
    """LiDAR sensor types"""
    MECHANICAL = "mechanical"  # Rotating
    SOLID_STATE = "solid_state"  # No moving parts
    FLASH = "flash"  # Captures entire scene at once


class ScanPattern(Enum):
    """Scanning patterns"""
    HORIZONTAL_360 = "horizontal_360"
    VERTICAL_SWEEP = "vertical_sweep"
    RASTER = "raster"
    SPIRAL = "spiral"


@dataclass
class PointCloudData:
    """Point cloud scan result"""
    points: np.ndarray  # Nx3 array of [x, y, z]
    intensities: np.ndarray  # N array of return intensity
    timestamps: np.ndarray  # N array of timestamps
    ranges: np.ndarray  # N array of distances
    
    def get_point_count(self) -> int:
        """Get number of points"""
        return len(self.points)
    
    def filter_by_range(self, min_range: float, max_range: float) -> 'PointCloudData':
        """Filter points by distance"""
        mask = (self.ranges >= min_range) & (self.ranges <= max_range)
        return PointCloudData(
            points=self.points[mask],
            intensities=self.intensities[mask],
            timestamps=self.timestamps[mask],
            ranges=self.ranges[mask]
        )
    
    def filter_by_height(self, min_z: float, max_z: float) -> 'PointCloudData':
        """Filter points by height"""
        mask = (self.points[:, 2] >= min_z) & (self.points[:, 2] <= max_z)
        return PointCloudData(
            points=self.points[mask],
            intensities=self.intensities[mask],
            timestamps=self.timestamps[mask],
            ranges=self.ranges[mask]
        )
    
    def downsample(self, factor: int) -> 'PointCloudData':
        """Downsample point cloud"""
        indices = np.arange(0, len(self.points), factor)
        return PointCloudData(
            points=self.points[indices],
            intensities=self.intensities[indices],
            timestamps=self.timestamps[indices],
            ranges=self.ranges[indices]
        )


class VirtualLiDAR:
    """
    Advanced LiDAR sensor simulation
    
    Features:
    - Multiple scan patterns
    - Configurable resolution
    - Ray casting
    - Point cloud generation
    - Intensity modeling
    - Noise simulation
    - Range filtering
    """
    
    def __init__(self,
                 name: str,
                 position: np.ndarray,
                 rotation: np.ndarray = None,
                 lidar_type: LiDARType = LiDARType.MECHANICAL,
                 scan_pattern: ScanPattern = ScanPattern.HORIZONTAL_360,
                 max_range: float = 100.0,
                 angular_resolution: float = 0.25,  # degrees
                 vertical_fov: Tuple[float, float] = (-15, 15),  # degrees
                 vertical_resolution: float = 2.0):  # degrees
        """
        Initialize LiDAR sensor
        
        Args:
            name: Sensor identifier
            position: [x, y, z] position
            rotation: [roll, pitch, yaw] in radians
            lidar_type: Type of LiDAR
            scan_pattern: Scanning pattern
            max_range: Maximum detection range (meters)
            angular_resolution: Horizontal angular resolution (degrees)
            vertical_fov: (min_angle, max_angle) vertical field of view
            vertical_resolution: Vertical angular resolution (degrees)
        """
        self.name = name
        self.position = np.array(position)
        self.rotation = np.array(rotation) if rotation is not None else np.zeros(3)
        self.lidar_type = lidar_type
        self.scan_pattern = scan_pattern
        self.max_range = max_range
        self.angular_resolution = angular_resolution
        self.vertical_fov = vertical_fov
        self.vertical_resolution = vertical_resolution
        
        # Noise parameters
        self.range_noise_std = 0.02  # meters
        self.angular_noise_std = 0.1  # degrees
        
        # Performance
        self.scan_count = 0
        self.total_points = 0
    
    def _cast_ray(self, direction: np.ndarray, objects: List) -> Optional[Tuple[np.ndarray, float, float]]:
        """
        Cast a ray and find intersection
        
        Args:
            direction: Ray direction (normalized)
            objects: List of objects to check
            
        Returns:
            (hit_point, distance, intensity) or None
        """
        min_distance = self.max_range
        hit_point = None
        hit_intensity = 0.0
        
        for obj in objects:
            obj_pos = getattr(obj, 'position', None)
            if obj_pos is None:
                continue
            
            obj_pos = np.array(obj_pos)
            
            # Vector to object
            to_obj = obj_pos - self.position
            
            # Distance along ray
            ray_dist = np.dot(to_obj, direction)
            
            if ray_dist < 0:
                continue  # Behind sensor
            
            # Closest point on ray
            closest_point = self.position + direction * ray_dist
            
            # Distance from ray to object
            perp_dist = np.linalg.norm(closest_point - obj_pos)
            
            # Object radius (simplified)
            obj_size = getattr(obj, 'size', np.array([1, 1, 1]))
            obj_radius = np.linalg.norm(obj_size) / 2
            
            # Check if ray hits object
            if perp_dist < obj_radius:
                # Calculate actual hit distance
                hit_dist = ray_dist - np.sqrt(max(0, obj_radius**2 - perp_dist**2))
                
                if hit_dist < min_distance and hit_dist > 0:
                    min_distance = hit_dist
                    hit_point = self.position + direction * hit_dist
                    
                    # Calculate intensity (based on surface normal alignment)
                    surface_normal = (hit_point - obj_pos) / np.linalg.norm(hit_point - obj_pos)
                    alignment = abs(np.dot(surface_normal, -direction))
                    hit_intensity = alignment * 255  # 0-255 range
        
        if hit_point is not None:
            # Add noise
            if self.range_noise_std > 0:
                min_distance += np.random.normal(0, self.range_noise_std)
            
            return (hit_point, min_distance, hit_intensity)
        
        return None
    
    def scan_horizontal_360(self, objects: List) -> PointCloudData:
        """Perform 360-degree horizontal scan"""
        points = []
        intensities = []
        ranges = []
        timestamps = []
        
        angles = np.arange(0, 360, self.angular_resolution)
        time_per_angle = 1.0 / len(angles)  # Normalize to 1 second
        
        for i, angle_deg in enumerate(angles):
            angle = np.radians(angle_deg)
            
            # Add angular noise
            if self.angular_noise_std > 0:
                angle += np.radians(np.random.normal(0, self.angular_noise_std))
            
            # Ray direction (horizontal plane)
            direction = np.array([
                np.cos(angle),
                np.sin(angle),
                0.0
            ])
            
            # Cast ray
            result = self._cast_ray(direction, objects)
            
            if result is not None:
                point, distance, intensity = result
                points.append(point)
                intensities.append(intensity)
                ranges.append(distance)
                timestamps.append(i * time_per_angle)
        
        return PointCloudData(
            points=np.array(points) if points else np.zeros((0, 3)),
            intensities=np.array(intensities) if intensities else np.zeros(0),
            timestamps=np.array(timestamps) if timestamps else np.zeros(0),
            ranges=np.array(ranges) if ranges else np.zeros(0)
        )
    
    def scan_vertical_sweep(self, objects: List) -> PointCloudData:
        """Perform vertical sweep scan"""
        points = []
        intensities = []
        ranges = []
        timestamps = []
        
        # Vertical angles
        v_angles = np.arange(self.vertical_fov[0], self.vertical_fov[1], self.vertical_resolution)
        
        # Horizontal angles
        h_angles = np.arange(0, 360, self.angular_resolution)
        
        total_rays = len(v_angles) * len(h_angles)
        time_per_ray = 1.0 / total_rays
        ray_idx = 0
        
        for v_angle_deg in v_angles:
            v_angle = np.radians(v_angle_deg)
            
            for h_angle_deg in h_angles:
                h_angle = np.radians(h_angle_deg)
                
                # 3D ray direction
                direction = np.array([
                    np.cos(v_angle) * np.cos(h_angle),
                    np.cos(v_angle) * np.sin(h_angle),
                    np.sin(v_angle)
                ])
                
                # Cast ray
                result = self._cast_ray(direction, objects)
                
                if result is not None:
                    point, distance, intensity = result
                    points.append(point)
                    intensities.append(intensity)
                    ranges.append(distance)
                    timestamps.append(ray_idx * time_per_ray)
                
                ray_idx += 1
        
        return PointCloudData(
            points=np.array(points) if points else np.zeros((0, 3)),
            intensities=np.array(intensities) if intensities else np.zeros(0),
            timestamps=np.array(timestamps) if timestamps else np.zeros(0),
            ranges=np.array(ranges) if ranges else np.zeros(0)
        )
    
    def scan(self, objects: List) -> PointCloudData:
        """
        Perform scan based on configured pattern
        
        Args:
            objects: List of objects to scan
            
        Returns:
            Point cloud data
        """
        if self.scan_pattern == ScanPattern.HORIZONTAL_360:
            result = self.scan_horizontal_360(objects)
        elif self.scan_pattern == ScanPattern.VERTICAL_SWEEP:
            result = self.scan_vertical_sweep(objects)
        else:
            result = self.scan_horizontal_360(objects)
        
        # Update statistics
        self.scan_count += 1
        self.total_points += result.get_point_count()
        
        return result
    
    def detect_objects(self, point_cloud: PointCloudData,
                      min_points: int = 5) -> List[Dict]:
        """
        Detect objects from point cloud using clustering
        
        Args:
            point_cloud: Point cloud data
            min_points: Minimum points to form an object
            
        Returns:
            List of detected objects
        """
        if len(point_cloud.points) < min_points:
            return []
        
        # Simple clustering by distance
        detections = []
        points = point_cloud.points
        
        # Sort by distance
        indices = np.argsort(point_cloud.ranges)
        
        clusters = []
        current_cluster = [indices[0]]
        
        for i in range(1, len(indices)):
            prev_point = points[indices[i-1]]
            curr_point = points[indices[i]]
            
            dist = np.linalg.norm(curr_point - prev_point)
            
            if dist < 2.0:  # Points within 2m belong to same object
                current_cluster.append(indices[i])
            else:
                if len(current_cluster) >= min_points:
                    clusters.append(current_cluster)
                current_cluster = [indices[i]]
        
        # Add last cluster
        if len(current_cluster) >= min_points:
            clusters.append(current_cluster)
        
        # Create detections
        for cluster_idx, cluster in enumerate(clusters):
            cluster_points = points[cluster]
            
            # Calculate centroid
            centroid = np.mean(cluster_points, axis=0)
            
            # Calculate bounding box
            min_bounds = np.min(cluster_points, axis=0)
            max_bounds = np.max(cluster_points, axis=0)
            size = max_bounds - min_bounds
            
            # Distance to centroid
            distance = np.linalg.norm(centroid - self.position)
            
            detections.append({
                'id': f"object_{cluster_idx}",
                'centroid': centroid,
                'size': size,
                'distance': distance,
                'point_count': len(cluster),
                'points': cluster_points
            })
        
        return detections
    
    def get_statistics(self) -> Dict:
        """Get LiDAR statistics"""
        avg_points = self.total_points / self.scan_count if self.scan_count > 0 else 0
        
        return {
            'scan_count': self.scan_count,
            'total_points': self.total_points,
            'avg_points_per_scan': avg_points,
            'lidar_type': self.lidar_type.value,
            'scan_pattern': self.scan_pattern.value,
            'max_range': self.max_range,
            'angular_resolution': self.angular_resolution
        }
    
    def reset_statistics(self):
        """Reset statistics"""
        self.scan_count = 0
        self.total_points = 0


if __name__ == "__main__":
    print("Testing Advanced LiDAR Sensor...")
    
    # Create LiDAR
    lidar = VirtualLiDAR(
        name="perimeter_lidar",
        position=np.array([0, 0, 5]),
        scan_pattern=ScanPattern.HORIZONTAL_360,
        max_range=50.0,
        angular_resolution=1.0  # 1 degree
    )
    
    # Mock objects
    class MockObject:
        def __init__(self, name, pos, size):
            self.name = name
            self.position = np.array(pos)
            self.size = np.array(size)
    
    obj1 = MockObject("box1", [10, 0, 0], [2, 2, 2])
    obj2 = MockObject("box2", [0, 15, 0], [3, 3, 3])
    obj3 = MockObject("person1", [20, 20, 0], [0.6, 0.6, 1.75])
    
    # Scan
    print("\nScanning...")
    point_cloud = lidar.scan([obj1, obj2, obj3])
    
    print(f"Points captured: {point_cloud.get_point_count()}")
    print(f"Range: {np.min(point_cloud.ranges):.1f}m - {np.max(point_cloud.ranges):.1f}m")
    
    # Detect objects
    print("\nDetecting objects...")
    detections = lidar.detect_objects(point_cloud, min_points=3)
    
    print(f"Objects detected: {len(detections)}")
    for d in detections:
        print(f"  {d['id']}: {d['distance']:.1f}m, {d['point_count']} points, size={d['size']}")
    
    # Filter
    print("\nFiltering by range...")
    filtered = point_cloud.filter_by_range(5, 25)
    print(f"Points after filter: {filtered.get_point_count()}")
    
    # Statistics
    stats = lidar.get_statistics()
    print(f"\nStatistics: {stats}")
    
    print("\n✓ LiDAR sensor test complete")