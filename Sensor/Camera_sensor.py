"""
Camera Sensor - Advanced Virtual Camera
Realistic camera simulation with image processing capabilities
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from enum import Enum


class CameraType(Enum):
    """Camera types"""
    RGB = "rgb"
    THERMAL = "thermal"
    NIGHT_VISION = "night_vision"
    DEPTH = "depth"


class DetectionConfidence(Enum):
    """Detection confidence levels"""
    HIGH = "high"      # >0.8
    MEDIUM = "medium"  # 0.5-0.8
    LOW = "low"        # <0.5


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters"""
    focal_length: float = 50.0  # mm
    sensor_width: float = 36.0  # mm (full frame)
    sensor_height: float = 24.0  # mm
    
    def get_fov_horizontal(self) -> float:
        """Calculate horizontal field of view in degrees"""
        return 2 * np.degrees(np.arctan(self.sensor_width / (2 * self.focal_length)))
    
    def get_fov_vertical(self) -> float:
        """Calculate vertical field of view in degrees"""
        return 2 * np.degrees(np.arctan(self.sensor_height / (2 * self.focal_length)))


@dataclass
class Detection:
    """Object detection result"""
    object_id: str
    object_type: str
    position_3d: np.ndarray
    position_2d: Tuple[int, int]  # Pixel coordinates
    distance: float
    confidence: float
    bounding_box: Tuple[int, int, int, int]  # x, y, width, height
    timestamp: float = 0.0
    
    def get_confidence_level(self) -> DetectionConfidence:
        """Get confidence level"""
        if self.confidence > 0.8:
            return DetectionConfidence.HIGH
        elif self.confidence > 0.5:
            return DetectionConfidence.MEDIUM
        else:
            return DetectionConfidence.LOW


class VirtualCamera:
    """
    Advanced virtual camera sensor
    
    Features:
    - Realistic optics simulation
    - Multiple camera types
    - Field of view calculations
    - Distance estimation
    - Bounding box generation
    - Image coordinates projection
    - Environmental effects (fog, lighting)
    """
    
    def __init__(self,
                 name: str,
                 position: np.ndarray,
                 rotation: np.ndarray = None,
                 camera_type: CameraType = CameraType.RGB,
                 resolution: Tuple[int, int] = (1920, 1080),
                 intrinsics: Optional[CameraIntrinsics] = None):
        """
        Initialize camera
        
        Args:
            name: Camera identifier
            position: [x, y, z] position
            rotation: [roll, pitch, yaw] in radians
            camera_type: Type of camera
            resolution: (width, height) in pixels
            intrinsics: Camera intrinsic parameters
        """
        self.name = name
        self.position = np.array(position)
        self.rotation = np.array(rotation) if rotation is not None else np.zeros(3)
        self.camera_type = camera_type
        self.resolution = resolution
        self.intrinsics = intrinsics or CameraIntrinsics()
        
        # Detection parameters
        self.max_range = 100.0  # meters
        self.min_confidence = 0.3
        
        # Environmental factors
        self.fog_density = 0.0
        self.ambient_light = 1.0  # 0 = dark, 1 = bright
        
        # Performance
        self.detections_per_frame = []
        self.total_detections = 0
    
    def get_fov(self) -> Tuple[float, float]:
        """Get field of view (horizontal, vertical) in degrees"""
        return (self.intrinsics.get_fov_horizontal(),
                self.intrinsics.get_fov_vertical())
    
    def get_forward_vector(self) -> np.ndarray:
        """Get camera forward direction"""
        yaw = self.rotation[2]
        pitch = self.rotation[1]
        
        forward = np.array([
            np.cos(pitch) * np.cos(yaw),
            np.cos(pitch) * np.sin(yaw),
            np.sin(pitch)
        ])
        return forward
    
    def is_in_fov(self, point: np.ndarray) -> bool:
        """
        Check if point is in camera field of view
        
        Args:
            point: 3D point to check
            
        Returns:
            True if point is visible
        """
        # Vector from camera to point
        to_point = point - self.position
        distance = np.linalg.norm(to_point)
        
        if distance > self.max_range:
            return False
        
        to_point_normalized = to_point / distance
        
        # Forward vector
        forward = self.get_forward_vector()
        
        # Angle to point
        cos_angle = np.dot(forward, to_point_normalized)
        angle = np.arccos(np.clip(cos_angle, -1, 1))
        
        # Check against FOV
        fov_h, fov_v = self.get_fov()
        max_angle = np.radians(max(fov_h, fov_v) / 2)
        
        return angle < max_angle
    
    def project_to_image(self, point_3d: np.ndarray) -> Tuple[int, int]:
        """
        Project 3D point to 2D image coordinates
        
        Args:
            point_3d: 3D point in world coordinates
            
        Returns:
            (x, y) pixel coordinates
        """
        # Transform to camera space
        to_point = point_3d - self.position
        
        # Simplified projection (assumes camera looking along Y axis)
        # In real implementation, would use full camera matrix
        x_3d = to_point[0]
        y_3d = to_point[1]
        z_3d = to_point[2]
        
        if y_3d <= 0:
            return (-1, -1)  # Behind camera
        
        # Perspective projection
        focal_px = (self.intrinsics.focal_length / self.intrinsics.sensor_width) * self.resolution[0]
        
        x_2d = int((x_3d / y_3d) * focal_px + self.resolution[0] / 2)
        y_2d = int((-z_3d / y_3d) * focal_px + self.resolution[1] / 2)
        
        return (x_2d, y_2d)
    
    def estimate_bounding_box(self, object_position: np.ndarray,
                             object_size: np.ndarray) -> Tuple[int, int, int, int]:
        """
        Estimate 2D bounding box for object
        
        Args:
            object_position: 3D position
            object_size: 3D size [width, depth, height]
            
        Returns:
            (x, y, width, height) in pixels
        """
        # Project 8 corners of bounding box
        corners = []
        for dx in [-object_size[0]/2, object_size[0]/2]:
            for dy in [-object_size[1]/2, object_size[1]/2]:
                for dz in [0, object_size[2]]:
                    corner = object_position + np.array([dx, dy, dz])
                    x, y = self.project_to_image(corner)
                    if x >= 0 and y >= 0:
                        corners.append((x, y))
        
        if not corners:
            return (0, 0, 0, 0)
        
        # Find min/max
        xs = [c[0] for c in corners]
        ys = [c[1] for c in corners]
        
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        
        return (x_min, y_min, x_max - x_min, y_max - y_min)
    
    def calculate_confidence(self, distance: float, object_type: str) -> float:
        """
        Calculate detection confidence
        
        Args:
            distance: Distance to object
            object_type: Type of object
            
        Returns:
            Confidence score 0-1
        """
        # Base confidence from distance
        distance_factor = 1.0 - (distance / self.max_range)
        
        # Environmental factors
        fog_factor = 1.0 - self.fog_density
        light_factor = self.ambient_light if self.camera_type == CameraType.RGB else 1.0
        
        # Camera type bonuses
        type_bonus = 1.0
        if self.camera_type == CameraType.THERMAL and object_type == "person":
            type_bonus = 1.2  # Better at detecting people
        elif self.camera_type == CameraType.NIGHT_VISION and self.ambient_light < 0.3:
            type_bonus = 1.3  # Better in low light
        
        confidence = distance_factor * fog_factor * light_factor * type_bonus
        
        return np.clip(confidence, 0.0, 1.0)
    
    def detect_objects(self, objects: List) -> List[Detection]:
        """
        Detect objects in camera view
        
        Args:
            objects: List of objects to detect
            
        Returns:
            List of Detection objects
        """
        detections = []
        
        for obj in objects:
            # Get object properties
            obj_pos = getattr(obj, 'position', None)
            if obj_pos is None:
                continue
            
            obj_pos = np.array(obj_pos)
            
            # Check if in FOV
            if not self.is_in_fov(obj_pos):
                continue
            
            # Calculate distance
            distance = np.linalg.norm(obj_pos - self.position)
            
            if distance > self.max_range:
                continue
            
            # Get object info
            obj_name = getattr(obj, 'name', 'unknown')
            obj_type = obj.__class__.__name__.lower()
            obj_size = getattr(obj, 'size', np.array([1, 1, 1]))
            
            # Calculate confidence
            confidence = self.calculate_confidence(distance, obj_type)
            
            if confidence < self.min_confidence:
                continue
            
            # Project to image
            pos_2d = self.project_to_image(obj_pos)
            
            # Estimate bounding box
            bbox = self.estimate_bounding_box(obj_pos, obj_size)
            
            # Create detection
            detection = Detection(
                object_id=obj_name,
                object_type=obj_type,
                position_3d=obj_pos.copy(),
                position_2d=pos_2d,
                distance=distance,
                confidence=confidence,
                bounding_box=bbox
            )
            
            detections.append(detection)
        
        # Update statistics
        self.detections_per_frame.append(len(detections))
        self.total_detections += len(detections)
        
        return detections
    
    def set_fog(self, density: float):
        """Set fog density (0 = clear, 1 = heavy fog)"""
        self.fog_density = np.clip(density, 0.0, 1.0)
    
    def set_ambient_light(self, level: float):
        """Set ambient light level (0 = dark, 1 = bright)"""
        self.ambient_light = np.clip(level, 0.0, 1.0)
    
    def get_statistics(self) -> Dict:
        """Get camera statistics"""
        if not self.detections_per_frame:
            avg_detections = 0
        else:
            avg_detections = np.mean(self.detections_per_frame)
        
        return {
            'total_detections': self.total_detections,
            'avg_detections_per_frame': avg_detections,
            'frames_processed': len(self.detections_per_frame),
            'camera_type': self.camera_type.value,
            'fov': self.get_fov(),
            'max_range': self.max_range
        }
    
    def reset_statistics(self):
        """Reset detection statistics"""
        self.detections_per_frame = []
        self.total_detections = 0


if __name__ == "__main__":
    print("Testing Advanced Camera Sensor...")
    
    # Create camera
    camera = VirtualCamera(
        name="surveillance_cam_1",
        position=np.array([0, 0, 5]),
        rotation=np.array([0, -np.pi/6, 0]),  # Looking down
        camera_type=CameraType.RGB,
        resolution=(1920, 1080)
    )
    
    print(f"FOV: {camera.get_fov()}")
    print(f"Forward: {camera.get_forward_vector()}")
    
    # Mock object
    class MockObject:
        def __init__(self, name, pos):
            self.name = name
            self.position = np.array(pos)
            self.size = np.array([1, 1, 2])
    
    # Test detection
    obj1 = MockObject("person1", [10, 10, 0])
    obj2 = MockObject("vehicle1", [50, 50, 0])
    obj3 = MockObject("person2", [200, 200, 0])  # Too far
    
    detections = camera.detect_objects([obj1, obj2, obj3])
    
    print(f"\nDetections: {len(detections)}")
    for d in detections:
        print(f"  {d.object_id}: {d.distance:.1f}m, confidence={d.confidence:.2f}")
        print(f"    Position 2D: {d.position_2d}")
        print(f"    Bbox: {d.bounding_box}")
    
    # Test environmental effects
    print("\nTesting fog...")
    camera.set_fog(0.5)
    detections = camera.detect_objects([obj1])
    print(f"  Confidence with fog: {detections[0].confidence:.2f}")
    
    # Statistics
    stats = camera.get_statistics()
    print(f"\nStatistics: {stats}")
    
    print("\n✓ Camera sensor test complete")