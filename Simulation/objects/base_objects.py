"""
Base Simulation Object
Foundation class for all objects in the simulation world
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict
from enum import Enum


class ObjectType(Enum):
    """Types of simulation objects"""
    STATIC = "static"      # Buildings, walls, obstacles
    DYNAMIC = "dynamic"    # People, vehicles, animals
    SENSOR = "sensor"      # Cameras, LiDAR, IMU
    AGENT = "agent"        # AI-controlled entities


@dataclass
class Transform:
    """3D transformation (position + rotation)"""
    position: np.ndarray = None  # [x, y, z]
    rotation: np.ndarray = None  # [roll, pitch, yaw] in radians
    scale: np.ndarray = None     # [sx, sy, sz]
    
    def __post_init__(self):
        if self.position is None:
            self.position = np.array([0.0, 0.0, 0.0], dtype=float)  # FIXED: explicit dtype
        else:
            self.position = np.array(self.position, dtype=float)  # FIXED: ensure float
            
        if self.rotation is None:
            self.rotation = np.array([0.0, 0.0, 0.0], dtype=float)  # FIXED
        else:
            self.rotation = np.array(self.rotation, dtype=float)  # FIXED
            
        if self.scale is None:
            self.scale = np.array([1.0, 1.0, 1.0], dtype=float)  # FIXED
        else:
            self.scale = np.array(self.scale, dtype=float)  # FIXED
    
    def get_forward_vector(self) -> np.ndarray:
        """Get forward direction vector"""
        yaw = self.rotation[2]
        return np.array([
            np.cos(yaw),
            np.sin(yaw),
            0.0
        ], dtype=float)  # FIXED
    
    def get_right_vector(self) -> np.ndarray:
        """Get right direction vector"""
        yaw = self.rotation[2]
        return np.array([
            -np.sin(yaw),
            np.cos(yaw),
            0.0
        ], dtype=float)  # FIXED


class SimulationObject:
    """Base class for all simulation objects"""
    
    def __init__(self, 
                 name: str,
                 object_type: ObjectType,
                 transform: Optional[Transform] = None):
        """
        Initialize simulation object
        
        Args:
            name: Unique identifier
            object_type: Type of object
            transform: Initial transform
        """
        self.name = name
        self.object_type = object_type
        self.transform = transform or Transform()
        
        # Physics properties - FIXED: explicit dtype
        self.mass = 1.0
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=float)
        
        # Rendering
        self.visible = True
        self.color = (1.0, 1.0, 1.0, 1.0)  # RGBA
        
        # Physics handles (set by physics engine)
        self.physics_id = None
        self.collision_shape = None
        
        # Rendering handles (set by render engine)
        self.render_node = None
        
        # Metadata
        self.metadata = {}
        self.active = True
    
    def update(self, dt: float):
        """
        Update object state (called every frame)
        
        Args:
            dt: Time step in seconds
        """
        pass
    
    def get_position(self) -> np.ndarray:
        """Get current position"""
        return self.transform.position.copy()
    
    def set_position(self, position: np.ndarray):
        """Set position"""
        self.transform.position = np.array(position, dtype=float)  # FIXED
    
    def get_rotation(self) -> np.ndarray:
        """Get current rotation (roll, pitch, yaw)"""
        return self.transform.rotation.copy()
    
    def set_rotation(self, rotation: np.ndarray):
        """Set rotation"""
        self.transform.rotation = np.array(rotation, dtype=float)  # FIXED
    
    def get_velocity(self) -> np.ndarray:
        """Get current velocity"""
        return self.velocity.copy()
    
    def set_velocity(self, velocity: np.ndarray):
        """Set velocity"""
        self.velocity = np.array(velocity, dtype=float)  # FIXED
    
    def apply_force(self, force: np.ndarray):
        """
        Apply force to object
        
        Args:
            force: Force vector [fx, fy, fz]
        """
        # F = ma -> a = F/m
        force = np.array(force, dtype=float)  # FIXED
        acceleration = force / self.mass
        self.velocity = self.velocity + acceleration  # FIXED: assignment
    
    def get_bounding_box(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get axis-aligned bounding box
        
        Returns:
            (min_corner, max_corner)
        """
        # Override in subclasses
        half_size = self.transform.scale * 0.5
        min_corner = self.transform.position - half_size
        max_corner = self.transform.position + half_size
        return min_corner, max_corner
    
    def distance_to(self, other: 'SimulationObject') -> float:
        """Calculate distance to another object"""
        return np.linalg.norm(
            self.transform.position - other.transform.position
        )
    
    def to_dict(self) -> Dict:
        """Serialize to dictionary"""
        return {
            'name': self.name,
            'type': self.object_type.value,
            'position': self.transform.position.tolist(),
            'rotation': self.transform.rotation.tolist(),
            'scale': self.transform.scale.tolist(),
            'velocity': self.velocity.tolist(),
            'mass': self.mass,
            'visible': self.visible,
            'color': self.color,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SimulationObject':
        """Deserialize from dictionary"""
        transform = Transform(
            position=np.array(data['position'], dtype=float),  # FIXED
            rotation=np.array(data['rotation'], dtype=float),  # FIXED
            scale=np.array(data['scale'], dtype=float)  # FIXED
        )
        
        obj = cls(
            name=data['name'],
            object_type=ObjectType(data['type']),
            transform=transform
        )
        
        obj.velocity = np.array(data['velocity'], dtype=float)  # FIXED
        obj.mass = data['mass']
        obj.visible = data['visible']
        obj.color = tuple(data['color'])
        obj.metadata = data.get('metadata', {})
        
        return obj
    
    def __repr__(self):
        pos = self.transform.position
        return f"{self.__class__.__name__}(name='{self.name}', pos=[{pos[0]:.1f}, {pos[1]:.1f}, {pos[2]:.1f}])"


if __name__ == "__main__":
    # Test
    print("Testing SimulationObject...")
    
    obj = SimulationObject(
        name="test_object",
        object_type=ObjectType.DYNAMIC,
        transform=Transform(
            position=np.array([1.0, 2.0, 3.0]),
            rotation=np.array([0.0, 0.0, np.pi/4])
        )
    )
    
    print(f"Object: {obj}")
    print(f"Position: {obj.get_position()}")
    print(f"Position dtype: {obj.transform.position.dtype}")
    print(f"Forward vector: {obj.transform.get_forward_vector()}")
    print(f"Bounding box: {obj.get_bounding_box()}")
    
    # Serialize
    data = obj.to_dict()
    print(f"Serialized: {data}")
    
    # Deserialize
    obj2 = SimulationObject.from_dict(data)
    print(f"Deserialized: {obj2}")
    
    print("✓ Test complete")