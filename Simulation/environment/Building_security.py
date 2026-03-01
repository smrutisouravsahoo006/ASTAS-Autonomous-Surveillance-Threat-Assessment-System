"""
Building Security Environment
Perimeter + interior security for a multi-building campus.
"""
import numpy as np
from dataclasses import dataclass
from typing import Optional

from objects.dynamic_objects import Person, Vehicle, Drone
from objects.static_objects import Building, Wall, Obstacle
from world.world import SimulationWorld


@dataclass
class BuildingSecurityConfig:
    campus_size: float = 80.0          # metres
    num_buildings: int = 3
    num_guards: int = 4
    num_drones: int = 2
    num_intruders: int = 3
    perimeter_wall: bool = True


class BuildingSecurityEnvironment:

    def __init__(self, config: Optional[BuildingSecurityConfig] = None):
        self.config = config or BuildingSecurityConfig()

    def create_world(self) -> SimulationWorld:
        cfg = self.config
        world = SimulationWorld()
        sz = cfg.campus_size

        # ── Perimeter wall (4 segments) ─────────────────────────────────
        if cfg.perimeter_wall:
            corners = [(-sz/2, -sz/2), (sz/2, -sz/2),
                       (sz/2,  sz/2), (-sz/2,  sz/2)]
            for i in range(4):
                x0, y0 = corners[i]
                x1, y1 = corners[(i+1) % 4]
                world.add_static_object(Wall(
                    name=f"wall_{i+1}",
                    start=np.array([x0, y0, 0]),
                    end=np.array([x1, y1, 0]),
                    height=3.0
                ))

        # ── Buildings ──────────────────────────────────────────────────
        building_positions = [
            np.array([-20.0,  10.0, 0]),
            np.array([ 15.0,  15.0, 0]),
            np.array([-10.0, -20.0, 0]),
        ]
        for i in range(min(cfg.num_buildings, len(building_positions))):
            world.add_static_object(Building(
                name=f"building_{i+1}",
                position=building_positions[i],
                size=np.array([12.0, 10.0, 6.0]),
                color=(0.55, 0.55, 0.6, 1.0)
            ))

        # ── Zones ──────────────────────────────────────────────────────
        world.add_zone("campus_restricted",   "restricted",
                       [[-sz/2, -sz/2], [sz/2, -sz/2], [sz/2, sz/2], [-sz/2, sz/2]])
        world.add_zone("server_room",         "restricted",
                       [[-25, 5], [-15, 5], [-15, 15], [-25, 15]])
        world.add_zone("public_entrance",     "safe",
                       [[-5, -sz/2], [5, -sz/2], [5, -sz/2+8], [-5, -sz/2+8]])

        # ── Guards on patrol ──────────────────────────────────────────
        patrol_routes = [
            [[-35, -35, 0], [35, -35, 0], [35, 35, 0], [-35, 35, 0]],
            [[-20, 0, 0], [20, 0, 0], [20, 20, 0], [-20, 20, 0]],
            [[-30, 30, 0], [30, 30, 0], [30, -30, 0], [-30, -30, 0]],
            [[-10, -10, 0], [10, -10, 0], [10, 10, 0], [-10, 10, 0]],
        ]
        for i in range(min(cfg.num_guards, len(patrol_routes))):
            guard = Person(name=f"guard_{i+1}",
                           position=np.array(patrol_routes[i][0]),
                           color=(0.2, 0.4, 0.9, 1.0))
            guard.set_patrol(patrol_routes[i], speed=1.2)
            world.add_dynamic_object(guard)

        # ── Surveillance drones ────────────────────────────────────────
        drone_routes = [
            [[-30, -30], [30, -30], [30, 30], [-30, 30]],
            [[0, -30],   [30, 0],   [0, 30],  [-30, 0]],
        ]
        for i in range(min(cfg.num_drones, len(drone_routes))):
            drone = Drone(name=f"drone_{i+1}",
                          position=np.array([drone_routes[i][0][0],
                                             drone_routes[i][0][1], 20.0]),
                          color=(0.9, 0.7, 0.1, 1.0))
            drone.set_path(drone_routes[i], speed=5.0, altitude=20.0)
            world.add_dynamic_object(drone)

        # ── Intruders ─────────────────────────────────────────────────
        intruder_scenarios = [
            (np.array([-22.0, 12.0, 0]), "loiter"),   # near server room
            (np.array([  5.0, 30.0, 0]), "walk",  np.array([5, -10, 0])),
            (np.array([ 30.0, -5.0, 0]), "run",   np.array([-30, -5, 0])),
        ]
        colors = [(1.0, 0.2, 0.2, 1.0), (1.0, 0.5, 0.1, 1.0), (0.9, 0.1, 0.5, 1.0)]
        for i in range(min(cfg.num_intruders, len(intruder_scenarios))):
            sc = intruder_scenarios[i]
            intruder = Person(name=f"intruder_{i+1}",
                              position=sc[0],
                              color=colors[i % len(colors)])
            if sc[1] == "loiter":
                intruder.start_loitering()
            elif sc[1] == "walk":
                intruder.walk_to(sc[2])
            elif sc[1] == "run":
                intruder.run_to(sc[2])
            world.add_dynamic_object(intruder)

        return world