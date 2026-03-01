"""
Border Patrol Environment  (re-export from uploaded border_patrol.py logic)
Builds a 200 m perimeter with fence segments, guard posts, patrol vehicles,
cameras and intruders.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from objects.dynamic_objects import Person, Vehicle, Drone
from objects.static_objects import Building, Wall
from world.world import SimulationWorld


@dataclass
class BorderPatrolConfig:
    border_length: float = 200.0
    fence_height: float = 4.0
    num_guard_posts: int = 4
    num_cameras: int = 6
    num_intruders: int = 3


class BorderPatrolEnvironment:

    def __init__(self, config: Optional[BorderPatrolConfig] = None):
        self.config = config or BorderPatrolConfig()

    def create_world(self) -> SimulationWorld:
        cfg = self.config
        world = SimulationWorld()
        length = cfg.border_length

        # ── Fence (10 segments) ────────────────────────────────────────
        for i in range(10):
            x0 = -length / 2 + i * (length / 10)
            x1 = x0 + length / 10
            world.add_static_object(Wall(
                name=f"fence_{i+1}",
                start=np.array([x0, 0.0, 0.0]),
                end=np.array([x1, 0.0, 0.0]),
                height=cfg.fence_height
            ))

        # ── Guard posts (buildings + stationary guards) ─────────────────
        spacing = length / (cfg.num_guard_posts + 1)
        for i in range(cfg.num_guard_posts):
            x = -length / 2 + spacing * (i + 1)
            world.add_static_object(Building(
                name=f"post_{i+1}",
                position=np.array([x, -10.0, 0.0]),
                size=np.array([4.0, 4.0, 4.0]),
                color=(0.6, 0.5, 0.4, 1.0)
            ))
            guard = Person(name=f"guard_{i+1}",
                           position=np.array([x, -12.0, 0.0]),
                           color=(0.2, 0.4, 0.9, 1.0))
            guard.set_patrol([
                np.array([x - 3, -12, 0]),
                np.array([x + 3, -12, 0]),
            ])
            world.add_dynamic_object(guard)

        # ── Zones ──────────────────────────────────────────────────────
        world.add_zone("border_zone", "restricted",
                       [[-length/2, -5], [length/2, -5],
                        [length/2, 5], [-length/2, 5]])
        world.add_zone("safe_side", "safe",
                       [[-length/2, -40], [length/2, -40],
                        [length/2, -10], [-length/2, -10]])

        # ── Patrol vehicles ────────────────────────────────────────────
        v1 = Vehicle(name="patrol_1",
                     position=np.array([-length/2 + 10, -20.0, 0.0]),
                     color=(0.2, 0.5, 1.0, 1.0))
        v1.set_path([[-length/2 + 10, -20, 0], [length/2 - 10, -20, 0]], speed=15.0)
        world.add_dynamic_object(v1)

        v2 = Vehicle(name="patrol_2",
                     position=np.array([length/2 - 10, -20.0, 0.0]),
                     color=(0.2, 0.5, 1.0, 1.0))
        v2.set_path([[length/2 - 10, -20, 0], [-length/2 + 10, -20, 0]], speed=15.0)
        world.add_dynamic_object(v2)

        # ── Intruders ─────────────────────────────────────────────────
        for i in range(cfg.num_intruders):
            x = np.random.uniform(-length / 3, length / 3)
            y = np.random.uniform(8, 20)
            intruder = Person(name=f"intruder_{i+1}",
                              position=np.array([x, y, 0.0]),
                              color=(1.0, 0.2, 0.2, 1.0))
            if i % 2 == 0:
                intruder.run_to(np.array([x, -8.0, 0.0]))
            else:
                intruder.walk_to(np.array([x, -8.0, 0.0]))
            world.add_dynamic_object(intruder)

        return world