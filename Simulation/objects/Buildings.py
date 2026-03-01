"""
Building Module - Static Structures
Creates 3D buildings with collision
"""
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False

@dataclass
class Building:
    """Static building object with physics"""
    name: str
    position: np.ndarray
    size: np.ndarray  # [width, depth, height]
    color: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)
    
    physics_id: Optional[int] = None
    collision_id: Optional[int] = None
    visual_id: Optional[int] = None
    
    def create_physics(self, physics_client):
        """Create physics body in PyBullet"""
        if not PYBULLET_AVAILABLE:
            print("⚠  PyBullet not available, skipping physics creation")
            return None

        half_extents = self.size / 2.0
        
        # Collision shape
        self.collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents.tolist(),
            physicsClientId=physics_client
        )
        
        # Visual shape
        self.visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents.tolist(),
            rgbaColor=self.color,
            physicsClientId=physics_client
        )
        
        # Create body (static)
        self.physics_id = p.createMultiBody(
            baseMass=0,  # Static
            baseCollisionShapeIndex=self.collision_id,
            baseVisualShapeIndex=self.visual_id,
            basePosition=self.position.tolist(),
            physicsClientId=physics_client
        )
        
        return self.physics_id
    
    def get_corners(self):
        """Get 8 corner points for rendering"""
        x, y, z = self.position
        w, d, h = self.size
        
        corners = []
        for dx in [-w/2, w/2]:
            for dy in [-d/2, d/2]:
                for dz in [0, h]:
                    corners.append([x + dx, y + dy, z + dz])
        
        return np.array(corners)

@dataclass
class Wall:
    """Wall/Fence structure"""
    name: str
    start: np.ndarray
    end: np.ndarray
    height: float
    thickness: float = 0.2
    color: Tuple[float, float, float, float] = (0.5, 0.3, 0.1, 1.0)
    
    physics_id: Optional[int] = None
    
    def create_physics(self, physics_client):
        """Create wall physics"""
        if not PYBULLET_AVAILABLE:
            print("⚠  PyBullet not available, skipping physics creation")
            return None

        # Calculate center and dimensions
        center = (self.start + self.end) / 2.0
        center[2] = self.height / 2.0
        
        length = np.linalg.norm(self.end - self.start)
        direction = self.end - self.start
        angle = np.arctan2(direction[1], direction[0])
        
        # Create box shape
        half_extents = [length/2, self.thickness/2, self.height/2]
        
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=physics_client
        )
        
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=self.color,
            physicsClientId=physics_client
        )
        
        # Orientation quaternion
        orientation = p.getQuaternionFromEuler([0, 0, angle])
        
        self.physics_id = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=center.tolist(),
            baseOrientation=orientation,
            physicsClientId=physics_client
        )
        
        return self.physics_id

if __name__ == "__main__":
    print("Testing Building module...")
    
    # Test without physics (just data structures)
    building = Building(
        "warehouse",
        np.array([10, 10, 0]),
        np.array([20, 15, 8])
    )
    print(f"Building: {building.name}")
    print(f"Corners: {building.get_corners().shape}")
    
    wall = Wall("fence", np.array([0, 0, 0]), np.array([10, 0, 0]), height=3.0)
    print(f"Wall: {wall.name}, length: {np.linalg.norm(wall.end - wall.start):.1f}m")
    
    print("✓ Building module OK")