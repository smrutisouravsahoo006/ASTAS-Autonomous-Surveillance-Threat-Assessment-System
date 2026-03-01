import numpy as np
from enum import Enum

try:
    from .dynamic_objects import Person, MovementState
except ImportError:
    from dynamic_objects import Person, MovementState


class HumanState(Enum):
    IDLE      = "idle"
    WALKING   = "walking"
    RUNNING   = "running"
    LOITERING = "loitering"
    GROUPED   = "grouped"


class Human(Person):
    """
    Extended person with biometric attributes and richer behavioural API.

    Attributes exposed by main.py:
        h.height, h.mass, h.walk_speed, h.run_speed
        h.get_speed(), h.get_position(), h.is_moving()
        h.walk_to(), h.run_to(), h.start_loitering(), h.update(dt)
    """

    def __init__(self, name: str, position=None, age: int = 30,
                 height: float = 1.75, mass: float = 75.0):
        position = np.array(
            position if position is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )
        super().__init__(name, position)

        self.age         = age
        self.height      = height       # metres
        self.mass        = mass         # kg
        self.walk_speed  = 1.4          # m/s
        self.run_speed   = 5.0          # m/s
        self.human_state = HumanState.IDLE

    # ----------------------------------------------------------
    # Overrides — keep human_state in sync with parent state
    # ----------------------------------------------------------

    def walk_to(self, target):
        super().walk_to(target)
        self.human_state = HumanState.WALKING

    def run_to(self, target):
        super().run_to(target)
        self.human_state = HumanState.RUNNING

    def start_loitering(self):
        super().start_loitering()
        self.human_state = HumanState.LOITERING

    def update(self, dt: float):
        super().update(dt)
        _map = {
            MovementState.IDLE:       HumanState.IDLE,
            MovementState.WALKING:    HumanState.WALKING,
            MovementState.RUNNING:    HumanState.RUNNING,
            MovementState.LOITERING:  HumanState.LOITERING,
            MovementState.PATROLLING: HumanState.WALKING,
        }
        self.human_state = _map.get(self.state, HumanState.IDLE)

    def __repr__(self):
        return (f"<Human name={self.name!r} age={self.age} "
                f"height={self.height}m mass={self.mass}kg "
                f"state={self.human_state.name}>")


# =========================================================
# HUMAN GROUP
# =========================================================

class HumanGroup:
    """
    A collection of Human objects spread around a centre position.

    Attributes / methods used by main.py:
        group.humans          — list of Human objects
        group.move_to(target) — walk all members toward target
        group.update(dt)
    """

    def __init__(self, name: str, count: int,
                 center_position=None,
                 formation_radius: float = 2.0):
        center = np.array(
            center_position if center_position is not None else [0.0, 0.0, 0.0],
            dtype=float,
        )
        self.name             = name
        self.formation_radius = formation_radius

        # Spread members evenly in a circle around center
        self.humans: list = []
        for i in range(count):
            angle  = 2 * np.pi * i / max(count, 1)
            offset = np.array([
                formation_radius * np.cos(angle),
                formation_radius * np.sin(angle),
                0.0,
            ], dtype=float)
            self.humans.append(Human(f"{name}_{i}", position=center + offset))

        # Legacy alias
        self.members = self.humans

    def move_to(self, target):
        """Walk all members toward target."""
        target = np.array(target, dtype=float)
        for h in self.humans:
            h.walk_to(target)

    def update(self, dt: float):
        for h in self.humans:
            h.update(dt)