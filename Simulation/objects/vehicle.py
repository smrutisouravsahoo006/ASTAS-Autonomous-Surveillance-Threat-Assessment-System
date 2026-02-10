import numpy as np
from typing import Optional, List
from enum import Enum
from base_objects import SimulationObject, ObjectType, Transform


class VehicleType(Enum):
    CAR = "car"
    TRUCK = "truck"
    MOTORCYCLE = "motorcycle"
    BUS = "bus"


class Vehicle(SimulationObject):
    """Vehicle simulation object"""

    def __init__(
        self,
        name: str,
        vehicle_type: VehicleType,
        position: Optional[np.ndarray] = None
    ):
        # --- Ensure float position ---
        if position is None:
            position = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            position = np.asarray(position, dtype=float)

        sizes = {
            VehicleType.CAR: np.array([2.0, 4.5, 1.5], dtype=float),
            VehicleType.TRUCK: np.array([2.5, 7.0, 3.0], dtype=float),
            VehicleType.MOTORCYCLE: np.array([0.8, 2.0, 1.2], dtype=float),
            VehicleType.BUS: np.array([2.5, 12.0, 3.5], dtype=float),
        }

        transform = Transform(position=position, scale=sizes[vehicle_type])
        super().__init__(name, ObjectType.DYNAMIC, transform)

        # --- Vehicle properties ---
        self.vehicle_type = vehicle_type
        self.mass = 1500.0 if vehicle_type == VehicleType.CAR else 3000.0
        self.max_speed = 30.0           # m/s
        self.acceleration = 3.0         # m/s²
        self.turn_rate = np.pi / 4      # rad/s

        self.color = (0.8, 0.2, 0.2, 1.0)

        # --- Path following ---
        self.target_speed = 0.0
        self.waypoints: List[np.ndarray] = []
        self.current_waypoint_idx = 0

    def update(self, dt: float):
        if not self.active or not self.waypoints:
            return

        # Loop waypoints
        if self.current_waypoint_idx >= len(self.waypoints):
            self.current_waypoint_idx = 0

        target = self.waypoints[self.current_waypoint_idx]
        position_2d = self.transform.position[:2]

        direction = target - position_2d
        distance = np.linalg.norm(direction)

        if distance > 0.5:
            direction /= distance

            # --- Steering ---
            target_angle = np.arctan2(direction[1], direction[0])
            current_angle = self.transform.rotation[2]

            angle_error = np.arctan2(
                np.sin(target_angle - current_angle),
                np.cos(target_angle - current_angle),
            )

            self.transform.rotation[2] += np.clip(
                angle_error,
                -self.turn_rate * dt,
                self.turn_rate * dt,
            )

            # --- Acceleration ---
            forward = self.transform.get_forward_vector()[:2]
            speed = np.linalg.norm(self.velocity[:2])

            if speed < self.target_speed:
                self.velocity[:2] += forward * self.acceleration * dt
            else:
                self.velocity[:2] *= 0.95  # simple drag

            # --- Speed limit ---
            speed = np.linalg.norm(self.velocity[:2])
            if speed > self.max_speed:
                self.velocity[:2] *= self.max_speed / speed

        else:
            # Waypoint reached
            self.current_waypoint_idx += 1

        # --- Integrate position ---
        self.transform.position[:2] += self.velocity[:2] * dt

    def set_path(self, waypoints: List, speed: float = 10.0):
        """Assign waypoint path"""
        self.waypoints = [np.asarray(wp, dtype=float) for wp in waypoints]
        self.target_speed = min(speed, self.max_speed)
        self.current_waypoint_idx = 0

if __name__ == "__main__":
    print("Testing Vehicle...")

    car = Vehicle(
        "car_1",
        VehicleType.CAR,
        position=np.array([0.0, 0.0, 0.0])
    )

    car.set_path(
        [[10, 0], [10, 10], [0, 10], [0, 0]],
        speed=15.0
    )

    for i in range(20):
        car.update(0.1)
        pos = car.get_position()[:2]
        speed = np.linalg.norm(car.velocity[:2])
        print(f"Step {i:02d}: pos={pos}, speed={speed:.2f} m/s")

    print("✓ Test complete")
