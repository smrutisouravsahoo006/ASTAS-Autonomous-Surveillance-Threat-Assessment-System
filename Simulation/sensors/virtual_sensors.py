"""
Virtual Sensors Module
Camera, LiDAR, IMU simulation
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict

@dataclass
class VirtualCamera:
    """Virtual camera for object detection"""
    name: str
    position: np.ndarray
    rotation: float = 0.0  # yaw
    fov: float = 90.0  # degrees
    max_range: float = 50.0
    resolution: tuple = (640, 480)
    
    def detect_objects(self, objects: List) -> List[Dict]:
        """Detect objects in camera view"""
        detections = []
        
        for obj in objects:
            # Get object position
            obj_pos = obj.position
            
            # Calculate relative position
            rel_pos = obj_pos - self.position
            dist = np.linalg.norm(rel_pos[:2])
            
            if dist > self.max_range:
                continue
            
            # Check if in FOV
            angle_to_obj = np.arctan2(rel_pos[1], rel_pos[0])
            angle_diff = abs(np.arctan2(np.sin(angle_to_obj - self.rotation),
                                       np.cos(angle_to_obj - self.rotation)))
            
            if angle_diff <= np.radians(self.fov / 2):
                detections.append({
                    'name': obj.name,
                    'type': obj.__class__.__name__.lower(),
                    'position': obj_pos.copy(),
                    'distance': dist,
                    'confidence': max(0.5, 1.0 - dist/self.max_range)
                })
        
        return detections

@dataclass
class VirtualLiDAR:
    """Virtual LiDAR scanner"""
    name: str
    position: np.ndarray
    max_range: float = 50.0
    angular_resolution: float = 0.5  # degrees
    
    def scan(self, objects: List) -> np.ndarray:
        """Generate point cloud"""
        points = []
        
        # Scan 360 degrees
        angles = np.arange(0, 360, self.angular_resolution)
        
        for angle_deg in angles:
            angle = np.radians(angle_deg)
            ray_dir = np.array([np.cos(angle), np.sin(angle), 0])
            
            # Cast ray and find nearest intersection
            min_dist = self.max_range
            hit_point = None
            
            for obj in objects:
                # Simple sphere collision check
                to_obj = obj.position - self.position
                dist = np.linalg.norm(to_obj)
                
                if dist < min_dist:
                    # Check if ray hits object (simplified)
                    ray_to_obj = np.dot(to_obj[:2], ray_dir[:2])
                    if ray_to_obj > 0:
                        closest_point = self.position + ray_dir * ray_to_obj
                        if np.linalg.norm(closest_point - obj.position) < 1.0:
                            min_dist = dist
                            hit_point = obj.position.copy()
            
            if hit_point is not None:
                points.append(hit_point)
        
        return np.array(points) if points else np.array([]).reshape(0, 3)

@dataclass
class VirtualIMU:
    """Virtual IMU sensor"""
    name: str
    attached_to: object = None
    
    last_velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0]))
    
    def get_data(self, dt: float = 0.033) -> Dict:
        """Get IMU readings"""
        if self.attached_to is None:
            return {
                'acceleration': np.zeros(3),
                'angular_velocity': np.zeros(3),
                'orientation': np.zeros(3)
            }
        
        # Calculate acceleration
        current_vel = self.attached_to.velocity
        accel = (current_vel - self.last_velocity) / dt
        self.last_velocity = current_vel.copy()
        
        # Get orientation
        yaw = getattr(self.attached_to, 'rotation', 0.0)
        
        return {
            'acceleration': accel,
            'angular_velocity': np.array([0, 0, (yaw - 0) / dt]),
            'orientation': np.array([0, 0, yaw])
        }

@dataclass
class AudioSensor:
    """Virtual microphone"""
    name: str
    position: np.ndarray
    sensitivity: float = 50.0  # meters
    
    def detect_events(self, events: List[Dict]) -> List[str]:
        """Detect audio events"""
        detected = []
        
        for event in events:
            dist = np.linalg.norm(event['position'] - self.position)
            if dist <= self.sensitivity:
                detected.append(event['type'])
        
        return detected

if __name__ == "__main__":
    print("Testing Virtual Sensors...")
    
    # Mock object
    class MockObj:
        def __init__(self, name, pos):
            self.name = name
            self.position = np.array(pos)
            self.velocity = np.array([1.0, 0, 0])
    
    obj1 = MockObj("obj1", [10, 0, 0])
    obj2 = MockObj("obj2", [0, 10, 0])
    
    # Camera
    cam = VirtualCamera("cam1", np.array([0, 0, 2]), rotation=0.0)
    detections = cam.detect_objects([obj1, obj2])
    print(f"Camera detections: {len(detections)}")
    
    # LiDAR
    lidar = VirtualLiDAR("lidar1", np.array([0, 0, 1]))
    points = lidar.scan([obj1, obj2])
    print(f"LiDAR points: {len(points)}")
    
    # IMU
    imu = VirtualIMU("imu1", attached_to=obj1)
    data = imu.get_data()
    print(f"IMU accel: {data['acceleration']}")
    
    print("✓ Sensors test complete")