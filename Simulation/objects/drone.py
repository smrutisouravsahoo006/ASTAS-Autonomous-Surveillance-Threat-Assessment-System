"""
Drone Module
Flying surveillance drone with patrol patterns
"""

import numpy as np 
from dataclasses import dataclass, field
from typing import Optional, List

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False


@dataclass
class Drone:
    """Flying surveillance drone"""
    name: str
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 10.0], dtype=float))  # FIXED
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # FIXED

    size: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.2], dtype=float))  # FIXED
    mass: float = 2.0
    max_speed: float = 15.0
    hover_altitude: float = 10.0  # FIXED: was hover_altidude (typo)

    patrol_points: List[np.ndarray] = field(default_factory=list)
    current_point: int = 0

    physics_id: Optional[int] = None

    color: tuple = (0.2, 0.2, 0.8, 1.0)
    
    def create_physics(self, physics_client):  # FIXED: was phsics_client
        """Create physics body in PyBullet"""
        if not PYBULLET_AVAILABLE:
            print("⚠ PyBullet not available, skipping physics creation")
            return None
            
        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=(self.size / 2).tolist(),
            physicsClientId=physics_client
        )
        
        # Visual shape (box for drone body)
        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=(self.size / 2).tolist(),
            rgbaColor=self.color,
            physicsClientId=physics_client
        )
        
        self.physics_id = p.createMultiBody(
            baseMass=self.mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=self.position.tolist(),
            physicsClientId=physics_client
        )
        
        return self.physics_id
    
    def update(self, dt: float):
        """Update drone position and patrol logic"""
        if len(self.patrol_points) == 0:
            return
        
        if self.current_point >= len(self.patrol_points):
            self.current_point = 0
        
        target = self.patrol_points[self.current_point]
        direction = target - self.position
        dist = np.linalg.norm(direction)
        
        if dist > 0.5:
            direction = direction / dist
            self.velocity = direction * self.max_speed
        else:
            self.current_point += 1
        
        # FIXED: assignment instead of +=
        self.position = self.position + self.velocity * dt
    
    def set_patrol_pattern(self, points: List, altitude: float = None):
        """
        Set patrol pattern for drone
        
        Args:
            points: List of [x, y] or [x, y, z] waypoints
            altitude: Default altitude if points don't specify z
        """
        if altitude is None:
            altitude = self.hover_altitude
        
        self.patrol_points = []
        for p in points:
            point = np.array(p, dtype=float)  # FIXED: explicit dtype
            
            # FIXED: Check point length, not points
            if len(point) == 2:
                # Add altitude for 2D points
                point = np.array([point[0], point[1], altitude], dtype=float)
            elif len(point) == 3:
                # Use provided 3D point
                point = np.array([point[0], point[1], point[2]], dtype=float)
            
            self.patrol_points.append(point)
        
        self.current_point = 0


if __name__ == "__main__":
    print("Testing Drone module...")
    
    # Create drone
    drone = Drone("surveillance_1", position=np.array([0, 0, 15]))
    print(f"Drone: {drone.name}")
    print(f"  Position dtype: {drone.position.dtype}")
    print(f"  Velocity dtype: {drone.velocity.dtype}")
    
    # Set patrol pattern
    drone.set_patrol_pattern(
        [[20, 20], [-20, 20], [-20, -20], [20, -20]], 
        altitude=15
    )
    print(f"  Patrol points: {len(drone.patrol_points)}")
    
    # Test update
    for i in range(5):
        drone.update(0.1)
        if i == 0 or i == 4:
            print(f"  Step {i}: pos={drone.position}")
    
    # Test 3D patrol points
    drone2 = Drone("surveillance_2")
    drone2.set_patrol_pattern([
        [10, 10, 15],
        [20, 20, 20],
        [10, 30, 10]
    ])
    print(f"\nDrone 2: {len(drone2.patrol_points)} 3D waypoints")
    
    print("\n✓ Drone module OK")
