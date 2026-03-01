import numpy as np
from enum import Enum

try:
    # When used as part of the Simulation.objects package
    from .base_objects import SimulationObject, ObjectType, Transform
except ImportError:
    # Fallback: running standalone or base_objects is in same directory
    from base_objects import SimulationObject, ObjectType, Transform


# =========================================================
# MOVEMENT STATE
# =========================================================

class MovementState(Enum):
    IDLE       = "idle"
    WALKING    = "walking"
    RUNNING    = "running"
    PATROLLING = "patrolling"
    LOITERING  = "loitering"


# =========================================================
# BASE DYNAMIC OBJECT
# =========================================================

class DynamicObject(SimulationObject):
    def __init__(self, name: str, position=None, color=None, **kwargs):
        if position is None:
            position = [0.0, 0.0, 0.0]
        transform = Transform(
            position=np.array(position, dtype=float),
            rotation=np.zeros(3, dtype=float),
        )
        super().__init__(name, ObjectType.DYNAMIC, transform)

        self.speed    = 2.0
        self.velocity = np.zeros(3, dtype=float)
        self.state    = MovementState.IDLE
        self.target   = None

        # Patrol system
        self.patrol_points = []
        self.patrol_index  = 0

        # Rendering / 3D attributes
        self.color             = color
        self.render_properties = kwargs

    # ----------------------------------------------------------
    # Convenience property: obj.position  (read / write)
    # ----------------------------------------------------------
    @property
    def position(self) -> np.ndarray:
        return self.transform.position

    @position.setter
    def position(self, value):
        self.transform.position = np.array(value, dtype=float)

    # ----------------------------------------------------------
    # BASIC MOVEMENT
    # ----------------------------------------------------------

    def walk_to(self, target):
        self.target = np.array(target, dtype=float)
        self.speed  = 1.4
        self.state  = MovementState.WALKING

    def run_to(self, target):
        self.target = np.array(target, dtype=float)
        self.speed  = 5.0
        self.state  = MovementState.RUNNING

    def start_loitering(self):
        self.state  = MovementState.LOITERING
        self.target = None

    # ----------------------------------------------------------
    # PATROL / PATH SYSTEM
    # ----------------------------------------------------------

    def set_patrol(self, points, speed=2.0):
        self.patrol_points = [np.array(p, dtype=float) for p in points]
        self.patrol_index  = 0
        self.speed         = speed
        if self.patrol_points:
            self.target = self.patrol_points[0]
            self.state  = MovementState.PATROLLING

    def set_path(self, points, speed=5.0, altitude=None):
        """Alias for set_patrol (altitude accepted but ignored here)."""
        self.set_patrol(points, speed=speed)

    # ----------------------------------------------------------
    # UPDATE LOOP
    # ----------------------------------------------------------

    def update(self, dt: float):
        if self.target is not None:
            direction = self.target - self.transform.position
            dist = np.linalg.norm(direction)

            if dist > 0.1:
                direction        = direction / dist
                self.velocity    = direction * self.speed
                self.transform.position += self.velocity * dt
            else:
                if self.state == MovementState.PATROLLING and self.patrol_points:
                    self.patrol_index = (self.patrol_index + 1) % len(self.patrol_points)
                    self.target       = self.patrol_points[self.patrol_index]
                else:
                    self.velocity = np.zeros(3, dtype=float)
                    self.state    = MovementState.IDLE
                    self.target   = None
        else:
            self.velocity = np.zeros(3, dtype=float)

    # ----------------------------------------------------------
    # HELPERS
    # ----------------------------------------------------------

    def get_speed(self) -> float:
        return float(np.linalg.norm(self.velocity))

    def is_moving(self) -> bool:
        return self.state not in (MovementState.IDLE, MovementState.LOITERING)

    def __repr__(self):
        return (f"<{self.__class__.__name__} name={self.name!r} "
                f"pos={np.round(self.transform.position, 2)} state={self.state.name}>")


# =========================================================
# PERSON
# =========================================================

class Person(DynamicObject):
    def __init__(self, name: str, position=None, color=None, **kwargs):
        super().__init__(name, position, color=color, **kwargs)


# =========================================================
# VEHICLE
# =========================================================

class Vehicle(DynamicObject):
    def __init__(self, name: str, position=None, color=None, **kwargs):
        super().__init__(name, position, color=color, **kwargs)
        self.speed = 5.0
        self.size  = np.array([2.0, 4.5, 1.5], dtype=float)  # W x L x H

    def set_path(self, points, speed=5.0, altitude=None):
        self.set_patrol(points, speed=speed)


# =========================================================
# DRONE  (lightweight — used by world / environment files)
# =========================================================

class Drone(DynamicObject):
    def __init__(self, name: str = "drone", position=None, color=None, **kwargs):
        if position is None:
            position = [0.0, 0.0, 10.0]
        super().__init__(name, position, color=color, **kwargs)
        self.speed = 8.0
        self.size  = np.array([0.6, 0.6, 0.2], dtype=float)  # W x L x H

    def set_path(self, points, speed=8.0, altitude=None):
        """Accept 2-D or 3-D waypoints; inject altitude for 2-D points."""
        processed = []
        for p in points:
            pt = np.array(p, dtype=float)
            if pt.shape[0] == 2:
                z  = float(altitude) if altitude is not None else self.transform.position[2]
                pt = np.array([pt[0], pt[1], z], dtype=float)
            processed.append(pt)
        self.set_patrol(processed, speed=speed)