"""
Static Objects Module
Buildings, walls, and ground planes
"""

import numpy as np 
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Building:
    """Static building structure"""
    name: str
    position: np.ndarray
    size: np.ndarray
    color: Tuple[float, float, float, float] = (0.6, 0.6, 0.6, 1.0)

    def get_corners(self):
        """Get corner points of building footprint"""
        x, y, z = self.position
        w, d, h = self.size
        return [
            [x - w/2, y - d/2, z],
            [x + w/2, y - d/2, z],
            [x + w/2, y + d/2, z],
            [x - w/2, y + d/2, z]
        ]


@dataclass
class Wall:
    """Wall or fence structure"""
    name: str
    start: np.ndarray
    end: np.ndarray
    height: float  # FIXED: was 'heigth'
    thickness: float = 0.2  # FIXED: Added default value
    color: Tuple[float, float, float, float] = (0.5, 0.3, 0.1, 1.0)
    
    def get_length(self):
        """Get wall length in meters"""
        return np.linalg.norm(self.end - self.start)
    
    def get_center(self):
        """Get center point of wall"""
        return (self.start + self.end) / 2.0


@dataclass
class Ground:
    """Ground plane with grid"""
    name: str = "ground"
    size: float = 100.0
    color: Tuple[float, float, float, float] = (0.3, 0.5, 0.3, 1.0)
    grid_spacing: float = 5.0


if __name__ == "__main__":
    print("Testing Static Objects...")
    
    # Test Building
    building = Building(
        name="warehouse",
        position=np.array([10, 10, 0]),
        size=np.array([20, 15, 8])
    )
    print(f"Building: {building.name}")
    print(f"  Corners: {len(building.get_corners())} points")
    
    # Test Wall - FIXED: using 'height' not 'heigth'
    wall = Wall(
        name="fence",
        start=np.array([0, 0, 0]),
        end=np.array([10, 0, 0]),
        height=3.0  # FIXED: correct parameter name
        # thickness will use default value 0.2
    )
    print(f"\nWall: {wall.name}")
    print(f"  Length: {wall.get_length():.1f}m")
    print(f"  Height: {wall.height}m")
    print(f"  Thickness: {wall.thickness}m")
    print(f"  Center: {wall.get_center()}")
    
    # Test Ground
    ground = Ground()
    print(f"\nGround: {ground.name}")
    print(f"  Size: {ground.size}m")
    
    print("\n✓ Static objects module OK")