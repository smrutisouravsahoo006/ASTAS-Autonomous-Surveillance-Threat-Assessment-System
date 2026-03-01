#!/usr/bin/env python3
"""
ASTAS 3D Visualization - PyBullet
Robust, safe 3D visualization for all ASTAS scenarios
"""

import sys
import time
import numpy as np
from pathlib import Path

# -----------------------------------------------------------------------------
# PyBullet
# -----------------------------------------------------------------------------
try:
    import pybullet as p
    import pybullet_data
except ImportError:
    print("❌ PyBullet not installed. Run: pip install pybullet")
    sys.exit(1)

# -----------------------------------------------------------------------------
# Project imports
# -----------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from world.world import SimulationWorld
from objects.dynamic_objects import Person, Vehicle, Drone
from objects.static_objects import Building, Wall


# =============================================================================
# ASTAS 3D VIEWER
# =============================================================================
class ASTAS3DSimulation:

    def __init__(self, scenario="simple",fps=240,gui=True):
        
        if p.isConnected():
           p.disconnect()
        print("\n" + "=" * 70)
        print("ASTAS 3D VISUALIZATION - PYBULLET")
        print("=" * 70)
        print(f"Loading scenario: {scenario}")
        self.scenario = scenario
        self.gui = gui

        if self.gui:
          self.physics_client = p.connect(p.GUI)
        else:
          self.physics_client = p.connect(p.DIRECT)

        if self.physics_client < 0:
          raise RuntimeError("Failed to connect to PyBullet")

        # --- Connect to PyBullet GUI ---

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)
        p.setRealTimeSimulation(0)

        if self.gui:
           p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
           p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1)

        # --- Simulation state ---
        self.scenario_name = scenario
        self.world = self._create_scenario(scenario)

        self.fps=fps
        self.dt = 1.0 / float(self.fps)
        self.time = 0.0
        self.speed = 1.0
        self.paused = False

        self.visual_ids = {}

        # --- Build scene ---
        self._create_environment()
        if self.gui:
            self._setup_camera()
            self._setup_ui()

        print(f"✓ Loaded: {scenario}")
        print(f"  Static objects : {len(self.world.static_objects)}")
        print(f"  Dynamic objects: {len(self.world.dynamic_objects)}")
        print(f"  Sensors        : {len(self.world.sensors)}")
        print(f"  Zones          : {len(self.world.zones)}")
        print("=" * 70)
        self._show_controls()

    # -------------------------------------------------------------------------
    # UI & Camera
    # -------------------------------------------------------------------------
    def _show_controls(self):
        print("\nCONTROLS:")
        print("  Mouse Drag   - Rotate camera")
        print("  Ctrl+Drag    - Pan camera")
        print("  Mouse Wheel  - Zoom")
        print("  Speed Slider - Simulation speed")
        print("  Close Window - Exit\n")

    def _setup_ui(self):
        self.speed_slider = p.addUserDebugParameter(
            "Speed", 0.1, 4.0, 1.0
        )

    def _setup_camera(self):
        presets = {
            "simple": (60, 45, -30),
            "border_patrol": (150, 45, -40),
            "building_security": (100, 45, -35),
        }
        dist, yaw, pitch = presets.get(self.scenario_name, (60, 45, -30))
        p.resetDebugVisualizerCamera(dist, yaw, pitch, [0, 0, 0])

    # -------------------------------------------------------------------------
    # Scenario creation
    # -------------------------------------------------------------------------
    def _create_scenario(self, scenario):
        if scenario == "simple":
            return self._simple_scenario()

        if scenario == "border_patrol":
            try:
                from environment.border_patrol import (
                    BorderPatrolEnvironment,
                    BorderPatrolConfig,
                )
                return BorderPatrolEnvironment(BorderPatrolConfig()).create_world()
            except Exception as e:
                print(f"⚠ Border patrol failed: {e}")

        if scenario == "building_security":
            try:
                from environment.Building_security import (
                    BuildingSecurityEnvironment,
                    BuildingSecurityConfig,
                )
                return BuildingSecurityEnvironment(
                    BuildingSecurityConfig()
                ).create_world()
            except Exception as e:
                print(f"⚠ Building security failed: {e}")

        print("⚠ Falling back to simple scenario")
        return self._simple_scenario()

    def _simple_scenario(self):
        world = SimulationWorld()

        world.add_zone(
            "restricted_area",
            "restricted",
            [[15, 15], [25, 15], [25, 25], [15, 25]],
        )

        intruder = Person("intruder", np.array([20, 20, 0]))
        intruder.start_loitering()
        intruder.color = (1.0, 0.2, 0.2, 1.0)
        world.add_dynamic_object(intruder)

        patrol = Vehicle("patrol_car", np.array([0, 0, 0]))
        patrol.set_path(
            [[10, 10, 0], [30, 10, 0], [30, 30, 0], [10, 30, 0]],
            speed=10,
        )
        patrol.color = (0.2, 0.4, 1.0, 1.0)
        world.add_dynamic_object(patrol)

        return world

    # -------------------------------------------------------------------------
    # Environment creation
    # -------------------------------------------------------------------------
    def _create_environment(self):
        # Ground
        p.loadURDF("plane.urdf")

        # Static objects
        for obj in self.world.static_objects:
            if isinstance(obj, Building):
                self._create_box(obj.name, obj.position, obj.size, obj.color, static=True)
            elif isinstance(obj, Wall):
                self._create_wall(obj)

        # Dynamic objects
        for obj in self.world.dynamic_objects:
            if isinstance(obj, Person):
                self._create_person(obj)
            elif isinstance(obj, Vehicle):
                self._create_box(obj.name, obj.position, obj.size, obj.color)
            elif isinstance(obj, Drone):
                self._create_box(obj.name, obj.position, obj.size, obj.color)

        # Zones
        for zone in self.world.zones:
            self._draw_zone(zone)

    # -------------------------------------------------------------------------
    # Object builders
    # -------------------------------------------------------------------------
    def _create_box(self, name, pos, size, color, static=False):
        cid = p.createCollisionShape(p.GEOM_BOX, halfExtents=size / 2)
        vid = p.createVisualShape(p.GEOM_BOX, halfExtents=size / 2, rgbaColor=color)
        bid = p.createMultiBody(
            baseMass=0 if static else 1,
            baseCollisionShapeIndex=cid,
            baseVisualShapeIndex=vid,
            basePosition=[pos[0], pos[1], size[2] / 2],
        )
        p.changeDynamics(bid, -1, mass=0)
        self.visual_ids[name] = bid

    def _create_person(self, person):
        h = getattr(person, "height", 1.75)
        cid = p.createCollisionShape(p.GEOM_CAPSULE, radius=0.3, height=h * 0.6)
        vid = p.createVisualShape(
            p.GEOM_CAPSULE, radius=0.3, length=h * 0.6, rgbaColor=person.color
        )
        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cid,
            baseVisualShapeIndex=vid,
            basePosition=[person.position[0], person.position[1], h / 2],
        )
        self.visual_ids[person.name] = bid

    def _create_wall(self, wall):
        center = (wall.start + wall.end) / 2
        length = np.linalg.norm(wall.end - wall.start)

        dx = wall.end[0] - wall.start[0]
        dy = wall.end[1] - wall.start[1]
        angle = np.arctan2(dy, dx)

        orientation = p.getQuaternionFromEuler([0, 0, angle])

        cid = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=[length / 2, wall.thickness / 2, wall.height / 2],
        )
        vid = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=[length / 2, wall.thickness / 2, wall.height / 2],
            rgbaColor=wall.color,
        )

        bid = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=cid,
            baseVisualShapeIndex=vid,
            basePosition=[center[0], center[1], wall.height / 2],
            baseOrientation=orientation,
        )
        self.visual_ids[wall.name] = bid

    def _draw_zone(self, zone):
        for i in range(len(zone.polygon)):
            p1 = zone.polygon[i]
            p2 = zone.polygon[(i + 1) % len(zone.polygon)]
            p.addUserDebugLine(
                [p1[0], p1[1], 0.05],
                [p2[0], p2[1], 0.05],
                zone.color[:3],
                3,
                0,
            )

    # -------------------------------------------------------------------------
    # Update loop
    # -------------------------------------------------------------------------
    def _update(self):
        if self.gui:
            self.speed = p.readUserDebugParameter(self.speed_slider)

        dt = self.dt * self.speed
        self.world.update(dt)
        self.time += dt

        for obj in self.world.dynamic_objects:
            bid = self.visual_ids.get(obj.name)
            if bid is None:
                continue

            if isinstance(obj, Person):
                z = getattr(obj, "height", 1.75) / 2
            elif isinstance(obj, Drone):
                z = obj.position[2]
            else:
                z = obj.size[2] / 2

            p.resetBasePositionAndOrientation(
                bid,
                [obj.position[0], obj.position[1], z],
                [0, 0, 0, 1],
            )

        p.stepSimulation()
    def get_astas_context(self):
       """Collect simulation state for LLM threat engine"""

       context = {
          "time": round(self.time, 2),
          "scenario": self.scenario_name,
          "dynamic_objects": [],
           "zones": [],
    }

    # Collect dynamic object states
       for obj in getattr(self.world, "dynamic_objects", []):
        try:
            pos, _ = p.getBasePositionAndOrientation(obj.body_id)
            context["dynamic_objects"].append({
                "id": obj.body_id,
                "position": [round(x, 2) for x in pos],
                "type": getattr(obj, "type", "unknown")
            })
        except:
            pass

    # Collect zones
       for zone in getattr(self.world, "zones", []):
         context["zones"].append({
            "name": getattr(zone, "name", "zone"),
            "type": getattr(zone, "type", "unknown")
        })

       return context

    def _debug_text(self):
        if not p.isConnected():
            return

        lines = [
            f"Scenario: {self.scenario_name}",
            f"Time: {self.time:.1f}s | Speed: {self.speed:.1f}x",
            f"Objects: {len(self.world.dynamic_objects)}",
            f"Violations: {len(self.world.check_zone_violations())}",
        ]

        for i, line in enumerate(lines):
            p.addUserDebugText(
                line,
                [0, 0, 25 - i * 2],
                [1, 1, 1],
                1.4,
                0.1,
            )

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    def step(self):
      """
      Single simulation step (for AI-integrated modes)
      """
      if not p.isConnected():
         return

      self._update()

      if self.gui:
          self._debug_text()

      time.sleep(self.dt)
    
    
    def run(self):
        print("\n✓ 3D Viewer running (close window to exit)\n")

        try:
            while p.isConnected():
                self._update()
                self._debug_text()
                time.sleep(self.dt)
        finally:
            if p.isConnected():
                p.disconnect()
            print("✓ Viewer closed safely")


# =============================================================================
# ENTRY POINT
# =============================================================================
def main():
    import argparse

    parser = argparse.ArgumentParser(description="ASTAS 3D PyBullet Viewer")
    parser.add_argument(
        "--scenario",
        default="simple",
        choices=["simple", "border_patrol", "building_security"],
        help="Scenario to run",
    )
    args = parser.parse_args()

    viewer = ASTAS3DSimulation(args.scenario)
    viewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
