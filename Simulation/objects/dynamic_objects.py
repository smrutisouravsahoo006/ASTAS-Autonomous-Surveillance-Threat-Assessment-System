"""
Dynamic Objects Module
Humans, vehicles, drones - all moving entities
"""
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List
from enum import Enum


class MovementState(Enum):
    IDLE = 0
    WALKING = 1
    RUNNING = 2
    LOITERING = 3


@dataclass
class Person:
    """Human/person with realistic movement"""
    name: str
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # FIXED: explicit dtype
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # FIXED: explicit dtype
    rotation: float = 0.0  # yaw angle in radians
    
    # Properties
    height: float = 1.75
    mass: float = 70.0
    walk_speed: float = 1.4  # m/s
    run_speed: float = 5.0
    
    # State
    state: MovementState = MovementState.IDLE
    destination: Optional[np.ndarray] = None
    loiter_time: float = 0.0
    direction_change_timer: float = 0.0
    
    # Rendering
    color: tuple = (0.2, 0.8, 0.2, 1.0)
    
    def update(self, dt: float):
        """Update person state"""
        if self.state == MovementState.WALKING and self.destination is not None:
            direction = self.destination - self.position
            direction[2] = 0  # Ignore Z
            dist = np.linalg.norm(direction)
            
            if dist > 0.1:
                direction = direction / dist
                self.velocity = direction * self.walk_speed
                self.rotation = np.arctan2(direction[1], direction[0])
            else:
                self.destination = None
                self.state = MovementState.IDLE
                self.velocity = np.zeros(3, dtype=float)  # FIXED: explicit zeros
        
        elif self.state == MovementState.RUNNING and self.destination is not None:
            direction = self.destination - self.position
            direction[2] = 0
            dist = np.linalg.norm(direction)
            
            if dist > 0.1:
                direction = direction / dist
                self.velocity = direction * self.run_speed
                self.rotation = np.arctan2(direction[1], direction[0])
            else:
                self.state = MovementState.WALKING
        
        elif self.state == MovementState.LOITERING:
            self.loiter_time += dt
            self.direction_change_timer += dt
            
            if self.direction_change_timer > 3.0:
                self.direction_change_timer = 0.0
                angle = np.random.uniform(0, 2*np.pi)
                dist = np.random.uniform(1.0, 3.0)
                offset = np.array([np.cos(angle)*dist, np.sin(angle)*dist, 0.0], dtype=float)  # FIXED: explicit dtype
                self.destination = self.position + offset
                self.state = MovementState.WALKING
        
        else:  # IDLE
            self.velocity *= 0.8
        
        # Apply velocity
        self.position = self.position + self.velocity * dt  # FIXED: assignment instead of +=
        self.position[2] = max(0, self.position[2])  # Stay on ground
    
    def walk_to(self, target: np.ndarray):
        self.destination = np.array(target, dtype=float)  # FIXED: explicit dtype
        self.state = MovementState.WALKING
    
    def run_to(self, target: np.ndarray):
        self.destination = np.array(target, dtype=float)  # FIXED: explicit dtype
        self.state = MovementState.RUNNING
    
    def start_loitering(self):
        self.state = MovementState.LOITERING
        self.loiter_time = 0.0


@dataclass
class Vehicle:
    """Car/truck with path following"""
    name: str
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # FIXED
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # FIXED
    rotation: float = 0.0
    
    # Properties  
    size: np.ndarray = field(default_factory=lambda: np.array([2.0, 4.5, 1.5], dtype=float))  # car  # FIXED
    mass: float = 1500.0
    max_speed: float = 20.0
    turn_rate: float = np.pi/4
    
    # Path
    waypoints: List[np.ndarray] = field(default_factory=list)
    current_waypoint: int = 0
    target_speed: float = 0.0
    
    # Rendering
    color: tuple = (0.8, 0.2, 0.2, 1.0)
    
    def update(self, dt: float):
        """Update vehicle"""
        if len(self.waypoints) == 0:
            return
        
        if self.current_waypoint >= len(self.waypoints):
            self.current_waypoint = 0
        
        target = self.waypoints[self.current_waypoint]
        direction = target[:2] - self.position[:2]
        dist = np.linalg.norm(direction)
        
        if dist > 0.5:
            direction = direction / dist
            
            # Steer
            target_angle = np.arctan2(direction[1], direction[0])
            angle_diff = np.arctan2(np.sin(target_angle - self.rotation), 
                                   np.cos(target_angle - self.rotation))
            self.rotation += np.clip(angle_diff, -self.turn_rate*dt, self.turn_rate*dt)
            
            # Accelerate
            forward = np.array([np.cos(self.rotation), np.sin(self.rotation), 0.0], dtype=float)  # FIXED
            speed = np.linalg.norm(self.velocity)
            
            if speed < self.target_speed:
                self.velocity = self.velocity + forward * 3.0 * dt  # FIXED: assignment
            else:
                self.velocity *= 0.95
            
            # Limit
            speed = np.linalg.norm(self.velocity)
            if speed > self.max_speed:
                self.velocity = (self.velocity / speed) * self.max_speed
        else:
            self.current_waypoint += 1
        
        self.position = self.position + self.velocity * dt  # FIXED: assignment
    
    def set_path(self, waypoints: List, speed: float = 10.0):
        self.waypoints = [np.array(wp, dtype=float) for wp in waypoints]  # FIXED: explicit dtype
        self.target_speed = min(speed, self.max_speed)
        self.current_waypoint = 0


@dataclass
class Drone:
    """Flying surveillance drone"""
    name: str
    position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 10.0], dtype=float))  # FIXED
    velocity: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 0.0], dtype=float))  # FIXED
    
    size: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.2], dtype=float))  # FIXED
    max_speed: float = 15.0
    altitude: float = 10.0
    
    patrol_points: List[np.ndarray] = field(default_factory=list)
    current_point: int = 0
    
    color: tuple = (0.2, 0.2, 0.8, 1.0)
    
    def update(self, dt: float):
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
        
        self.position = self.position + self.velocity * dt  # FIXED: assignment


if __name__ == "__main__":
    print("Testing Dynamic Objects...")
    
    # Test Person
    person = Person("person_1")
    person.walk_to(np.array([10, 10, 0]))
    
    print(f"Person: {person.name}")
    print(f"  Position dtype: {person.position.dtype}")
    print(f"  Velocity dtype: {person.velocity.dtype}")
    
    for _ in range(5):
        person.update(0.1)
    
    print(f"  Moved to: {person.position[:2]}")
    print(f"  Speed: {np.linalg.norm(person.velocity):.2f}m/s")
    
    # Test Vehicle
    car = Vehicle("car_1")
    car.set_path([[10, 0, 0], [10, 10, 0]], speed=15)
    
    print(f"\nCar: {car.name}")
    print(f"  Position dtype: {car.position.dtype}")
    
    for _ in range(5):
        car.update(0.1)
    
    print(f"  Moved to: {car.position[:2]}")
    
    # Test Drone
    drone = Drone("drone_1")
    drone.patrol_points = [
        np.array([10, 10, 10], dtype=float),
        np.array([-10, -10, 10], dtype=float)
    ]
    
    print(f"\nDrone: {drone.name}")
    drone.update(0.1)
    print(f"  Position: {drone.position}")
    
    print("\n✓ Dynamic objects test complete")


