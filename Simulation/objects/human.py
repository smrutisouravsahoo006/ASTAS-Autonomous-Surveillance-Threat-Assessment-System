"""
Human Object
Simulates a person with realistic movement and behavior
"""

import numpy as np
from typing import Optional, List
from enum import Enum

# FIXED: Correct import path
from objects.base_objects import SimulationObject, ObjectType, Transform


class HumanState(Enum):
    """Human behavioral states"""
    IDLE = "idle"
    WALKING = "walking"
    RUNNING = "running"
    LOITERING = "loitering"
    SITTING = "sitting"


class Human(SimulationObject):
    """Human/Person simulation object"""
    
    def __init__(self, 
                 name: str,
                 position: Optional[np.ndarray] = None,
                 age: int = 30,
                 height: float = 1.75):
        """
        Initialize human
        
        Args:
            name: Unique identifier
            position: Starting position [x, y, z]
            age: Age in years
            height: Height in meters
        """
        # Default position - FIXED: explicit dtype
        if position is None:
            position = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            position = np.array(position, dtype=float)  # FIXED: ensure float
        
        # Transform
        transform = Transform(
            position=position,
            rotation=np.array([0.0, 0.0, 0.0], dtype=float),  # FIXED: explicit dtype
            scale=np.array([0.3, 0.3, height], dtype=float)  # FIXED: explicit dtype
        )
        
        super().__init__(name, ObjectType.DYNAMIC, transform)
        
        # Physical properties
        self.mass = 70.0  # kg
        self.height = height
        self.age = age
        
        # Movement properties
        self.walk_speed = 1.4  # m/s (normal walking)
        self.run_speed = 5.0   # m/s (running)
        self.max_acceleration = 2.0  # m/s²
        
        # Behavior
        self.state = HumanState.IDLE
        self.destination = None
        self.path = []
        self.loiter_time = 0.0
        self.direction_change_timer = 0.0
        self.direction_change_interval = 3.0  # Change direction every 3s when loitering
        
        # Visual properties
        self.color = (0.2, 0.8, 0.2, 1.0)  # Green
        
        # Metadata
        self.metadata = {
            'age': age,
            'height': height,
            'threat_level': 0.0
        }
        
        # Ensure velocity is float type - FIXED
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=float)
    
    def update(self, dt: float):
        """Update human behavior"""
        if not self.active:
            return
        
        if self.state == HumanState.WALKING:
            self._update_walking(dt)
        elif self.state == HumanState.RUNNING:
            self._update_running(dt)
        elif self.state == HumanState.LOITERING:
            self._update_loitering(dt)
        elif self.state == HumanState.IDLE:
            self._update_idle(dt)
        
        # Apply velocity - FIXED: use assignment instead of +=
        self.transform.position = self.transform.position + self.velocity * dt
        
        # Keep on ground
        self.transform.position[2] = max(0.0, self.transform.position[2])
    
    def _update_walking(self, dt: float):
        """Update walking behavior"""
        if self.destination is not None:
            # Move toward destination
            direction = self.destination - self.transform.position
            direction[2] = 0  # Ignore Z
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                # Normalize and apply speed
                direction = direction / distance
                target_velocity = direction * self.walk_speed
                
                # Smooth acceleration
                accel_factor = min(self.max_acceleration * dt, 1.0)
                self.velocity = self.velocity + (target_velocity - self.velocity) * accel_factor
                
                # Update rotation to face movement direction
                if np.linalg.norm(self.velocity[:2]) > 0.1:
                    self.transform.rotation[2] = np.arctan2(self.velocity[1], self.velocity[0])
            else:
                # Reached destination
                self.destination = None
                self.state = HumanState.IDLE
                self.velocity *= 0.9  # Slow down
        else:
            # No destination, slow down
            self.velocity *= 0.9
            if np.linalg.norm(self.velocity) < 0.1:
                self.state = HumanState.IDLE
    
    def _update_running(self, dt: float):
        """Update running behavior"""
        if self.destination is not None:
            direction = self.destination - self.transform.position
            direction[2] = 0
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:
                direction = direction / distance
                target_velocity = direction * self.run_speed
                accel_factor = min(self.max_acceleration * dt, 1.0)
                self.velocity = self.velocity + (target_velocity - self.velocity) * accel_factor
                
                if np.linalg.norm(self.velocity[:2]) > 0.1:
                    self.transform.rotation[2] = np.arctan2(self.velocity[1], self.velocity[0])
            else:
                self.destination = None
                self.state = HumanState.WALKING
        else:
            self.velocity *= 0.95
    
    def _update_loitering(self, dt: float):
        """Update loitering behavior (random slow movement)"""
        self.loiter_time += dt
        self.direction_change_timer += dt
        
        # Change direction periodically
        if self.direction_change_timer >= self.direction_change_interval:
            self.direction_change_timer = 0.0
            
            # Random direction
            angle = np.random.uniform(0, 2 * np.pi)
            distance = np.random.uniform(1.0, 3.0)
            
            # FIXED: explicit dtype
            offset = np.array([
                np.cos(angle) * distance,
                np.sin(angle) * distance,
                0.0
            ], dtype=float)
            
            self.destination = self.transform.position + offset
            self.state = HumanState.WALKING
    
    def _update_idle(self, dt: float):
        """Update idle behavior"""
        # Apply friction
        self.velocity *= 0.8
        
        if np.linalg.norm(self.velocity) < 0.01:
            self.velocity = np.zeros(3, dtype=float)  # FIXED: explicit dtype
    
    def walk_to(self, destination: np.ndarray):
        """Start walking to destination"""
        self.destination = np.array(destination, dtype=float)  # FIXED: explicit dtype
        self.state = HumanState.WALKING
    
    def run_to(self, destination: np.ndarray):
        """Start running to destination"""
        self.destination = np.array(destination, dtype=float)  # FIXED: explicit dtype
        self.state = HumanState.RUNNING
    
    def start_loitering(self):
        """Start loitering behavior"""
        self.state = HumanState.LOITERING
        self.loiter_time = 0.0
        self.direction_change_timer = 0.0
    
    def stop(self):
        """Stop all movement"""
        self.state = HumanState.IDLE
        self.destination = None
        self.velocity = np.zeros(3, dtype=float)  # FIXED: explicit dtype
    
    def get_speed(self) -> float:
        """Get current speed in m/s"""
        return np.linalg.norm(self.velocity)
    
    def is_moving(self) -> bool:
        """Check if currently moving"""
        return self.get_speed() > 0.1
    
    def get_bounding_box(self):
        """Get bounding box (capsule approximation)"""
        radius = self.transform.scale[0]
        height = self.transform.scale[2]
        
        pos = self.transform.position
        min_corner = pos + np.array([-radius, -radius, 0], dtype=float)  # FIXED
        max_corner = pos + np.array([radius, radius, height], dtype=float)  # FIXED
        
        return min_corner, max_corner


class HumanGroup:
    """Group of humans moving together"""
    
    def __init__(self, 
                 base_name: str,
                 count: int,
                 center_position: np.ndarray,
                 formation_radius: float = 2.0):
        """
        Create group of humans
        
        Args:
            base_name: Base name for humans (will add _1, _2, etc)
            count: Number of humans
            center_position: Center of formation
            formation_radius: Radius of circular formation
        """
        self.humans: List[Human] = []
        self.center_position = np.array(center_position, dtype=float)  # FIXED
        
        for i in range(count):
            # Place in circular formation
            angle = (2 * np.pi * i) / count
            
            # FIXED: explicit dtype
            offset = np.array([
                np.cos(angle) * formation_radius,
                np.sin(angle) * formation_radius,
                0.0
            ], dtype=float)
            
            human = Human(
                name=f"{base_name}_{i+1}",
                position=center_position + offset
            )
            self.humans.append(human)
    
    def update(self, dt: float):
        """Update all humans in group"""
        for human in self.humans:
            human.update(dt)
    
    def move_to(self, destination: np.ndarray, formation_radius: float = 2.0):
        """Move entire group to destination"""
        destination = np.array(destination, dtype=float)  # FIXED
        count = len(self.humans)
        
        for i, human in enumerate(self.humans):
            # Maintain formation around destination
            angle = (2 * np.pi * i) / count
            
            # FIXED: explicit dtype
            offset = np.array([
                np.cos(angle) * formation_radius,
                np.sin(angle) * formation_radius,
                0.0
            ], dtype=float)
            
            human.walk_to(destination + offset)


if __name__ == "__main__":
    print("Testing Human object...")
    
    # Single human
    human = Human("person_1", position=np.array([5.0, 5.0, 0.0]))
    print(f"Created: {human}")
    print(f"Height: {human.height}m, Mass: {human.mass}kg")
    print(f"Position dtype: {human.transform.position.dtype}")
    print(f"Velocity dtype: {human.velocity.dtype}")
    
    # Test movement
    human.walk_to(np.array([10.0, 10.0, 0.0]))
    print(f"\nWalking to [10, 10, 0]...")
    
    for i in range(10):
        human.update(0.1)
        if i % 3 == 0:
            print(f"  Step {i}: pos={human.get_position()[:2]}, speed={human.get_speed():.2f}m/s")
    
    # Test group
    print("\nTesting HumanGroup...")
    group = HumanGroup("group", count=5, center_position=np.array([0, 0, 0]))
    print(f"Created group with {len(group.humans)} humans")
    
    group.move_to(np.array([10, 10, 0]))
    print("Group moving to [10, 10, 0]")
    
    print("\n✓ Human module test complete")