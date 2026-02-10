#!/usr/bin/env python3
"""
ASTAS 2D Visualization - PyGame
Top-down interactive simulation with threat heatmaps
Scenarios:
  - simple
  - border_patrol
  - building_security
"""

import sys
import argparse
from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# PyGame
# ---------------------------------------------------------------------
try:
    import pygame
    from pygame.locals import *
except ImportError:
    print("❌ PyGame not installed. Run: pip install pygame")
    sys.exit(1)

# ---------------------------------------------------------------------
# Project path
# ---------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------
# ASTAS imports
# ---------------------------------------------------------------------
from world.world import SimulationWorld
from objects.dynamic_objects import Person, Vehicle, Drone
from objects.static_objects import Building, Wall


# =====================================================================
# ASTAS 2D VIEWER
# =====================================================================
class ASTAS2DViewer:

    def __init__(self, scenario="simple"):
        pygame.init()

        self.scenario = scenario
        self.width, self.height = 1400, 900
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption(
            f"ASTAS 2D | {scenario.replace('_', ' ').title()}"
        )

        # Timing
        self.clock = pygame.time.Clock()
        self.fps = 60
        self.sim_time = 0.0
        self.speed = 1.0
        self.paused = False

        # Fonts
        self.font_lg = pygame.font.Font(None, 34)
        self.font_md = pygame.font.Font(None, 24)
        self.font_sm = pygame.font.Font(None, 18)

        # Camera
        self.camera_x = 0.0
        self.camera_y = 0.0
        self.zoom = self._initial_zoom()
        self.min_zoom = 1.0
        self.max_zoom = 40.0

        # Mouse
        self.dragging = False
        self.last_mouse = (0, 0)

        # Toggles
        self.show_ui = True
        self.show_labels = True
        self.show_sensors = True
        self.show_heatmap = True

        # Heatmap
        self.heatmap_resolution = 2.0   # meters per cell
        self.heatmap_radius = 8.0       # meters
        self.heatmap_decay = 0.95
        self.heatmap = {}

        # Colors
        self.colors = {
            "bg": (35, 50, 35),
            "grid": (70, 90, 70),
            "grid_major": (100, 120, 100),
            "building": (110, 110, 140),
            "wall": (90, 70, 50),
            "person": (230, 90, 90),
            "vehicle": (80, 130, 255),
            "drone": (240, 240, 120),
            "sensor": (200, 200, 255),
            "ui_bg": (20, 20, 20, 200),
            "ui_text": (255, 255, 255),
            "ok": (100, 255, 100),
            "warn": (255, 200, 0),
            "err": (255, 80, 80),
        }

        # Load world
        self.world = self._create_world()
        self._auto_center()
        self._print_banner()

    # ------------------------------------------------------------------
    # Scenario creation
    # ------------------------------------------------------------------
    def _create_world(self):
        print("\n" + "=" * 70)
        print("ASTAS 2D VISUALIZATION - PYGAME")
        print("=" * 70)
        print(f"Loading scenario: {self.scenario}")

        if self.scenario == "simple":
            return self._simple_scenario()
        if self.scenario == "border_patrol":
            return self._border_patrol()
        if self.scenario == "building_security":
            return self._building_security()

        print("⚠ Unknown scenario, using simple")
        return self._simple_scenario()

    def _simple_scenario(self):
        world = SimulationWorld()

        world.add_zone(
            "restricted_zone",
            "restricted",
            [[15, 15], [25, 15], [25, 25], [15, 25]],
        )

        intruder = Person("intruder", position=np.array([20, 20, 0]))
        intruder.start_loitering()
        intruder.color = (1.0, 0.2, 0.2, 1.0)
        world.add_dynamic_object(intruder)

        patrol = Vehicle("patrol_car", position=np.array([0, 0, 0]))
        patrol.set_path(
            [[10, 10, 0], [30, 10, 0], [30, 30, 0], [10, 30, 0]],
            speed=10,
        )
        patrol.color = (0.3, 0.5, 1.0, 1.0)
        world.add_dynamic_object(patrol)

        return world

    def _border_patrol(self):
        try:
            from environment.border_patrol import BorderPatrolEnvironment, BorderPatrolConfig
            return BorderPatrolEnvironment(BorderPatrolConfig()).create_world()
        except Exception as e:
            print(f"⚠ Border patrol failed: {e}")
            return self._simple_scenario()

    def _building_security(self):
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
            return self._simple_scenario()

    # ------------------------------------------------------------------
    # Camera helpers
    # ------------------------------------------------------------------
    def _initial_zoom(self):
        return {"simple": 12.0, "border_patrol": 3.0, "building_security": 6.0}.get(
            self.scenario, 8.0
        )

    def _auto_center(self):
        pts = []

        for o in self.world.dynamic_objects:
            if hasattr(o, "position"):
                pts.append(o.position[:2])

        for o in self.world.static_objects:
            if isinstance(o, Building):
                pts.append(o.position[:2])
            elif isinstance(o, Wall):
                pts.append(o.start[:2])
                pts.append(o.end[:2])

        for z in self.world.zones:
            for p in z.polygon:
                pts.append(p[:2])

        if pts:
            center = np.mean(pts, axis=0)
            self.camera_x = -center[0]
            self.camera_y = -center[1]

    def world_to_screen(self, x, y):
        sx = (x + self.camera_x) * self.zoom + self.width / 2
        sy = (-y - self.camera_y) * self.zoom + self.height / 2
        return int(sx), int(sy)

    # ------------------------------------------------------------------
    # Heatmap logic
    # ------------------------------------------------------------------
    def _heat_idx(self, x, y):
        return int(x // self.heatmap_resolution), int(y // self.heatmap_resolution)

    def update_heatmap(self):
        # decay
        for k in list(self.heatmap.keys()):
            self.heatmap[k] *= self.heatmap_decay
            if self.heatmap[k] < 0.01:
                del self.heatmap[k]

        violations = self.world.check_zone_violations()
        viol_objs = {v["object"] for v in violations}

        for obj in self.world.dynamic_objects:
            if not hasattr(obj, "position"):
                continue

            threat = 0.0

            if isinstance(obj, Person) and "intruder" in obj.name.lower():
                threat += 5.0

            if obj in viol_objs:
                threat += 10.0

            if threat <= 0:
                continue

            x, y = obj.position[0], obj.position[1]
            cells = int(self.heatmap_radius // self.heatmap_resolution)

            for dx in range(-cells, cells + 1):
                for dy in range(-cells, cells + 1):
                    wx = x + dx * self.heatmap_resolution
                    wy = y + dy * self.heatmap_resolution
                    dist = np.hypot(wx - x, wy - y)
                    if dist > self.heatmap_radius:
                        continue
                    falloff = 1.0 - dist / self.heatmap_radius
                    idx = self._heat_idx(wx, wy)
                    self.heatmap[idx] = self.heatmap.get(idx, 0.0) + threat * falloff

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------
    def update(self):
        if not self.paused:
            dt = (1.0 / self.fps) * self.speed
            self.world.update(dt)
            self.sim_time += dt
            self.update_heatmap()

    # ------------------------------------------------------------------
    # Drawing
    # ------------------------------------------------------------------
    def draw_heatmap(self):
        if not self.show_heatmap:
            return

        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)

        for (ix, iy), v in self.heatmap.items():
            intensity = min(1.0, v / 15.0)
            r = int(255 * intensity)
            g = int(255 * (1 - intensity))
            a = int(120 * intensity)

            wx = ix * self.heatmap_resolution
            wy = iy * self.heatmap_resolution
            sx, sy = self.world_to_screen(wx, wy)

            size = max(2, int(self.heatmap_resolution * self.zoom))
            rect = pygame.Rect(sx - size // 2, sy - size // 2, size, size)
            pygame.draw.rect(overlay, (r, g, 0, a), rect)

        self.screen.blit(overlay, (0, 0))

    def draw(self):
        self.screen.fill(self.colors["bg"])
        self.draw_heatmap()
        pygame.display.flip()

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------
    def handle_events(self):
        for e in pygame.event.get():
            if e.type == QUIT:
                return False

            if e.type == KEYDOWN:
                if e.key == K_ESCAPE:
                    return False
                if e.key == K_SPACE:
                    self.paused = not self.paused
                if e.key in (K_EQUALS, K_PLUS):
                    self.speed = min(8.0, self.speed * 2)
                if e.key == K_MINUS:
                    self.speed = max(0.125, self.speed / 2)
                if e.key == K_h:
                    self.show_heatmap = not self.show_heatmap

        return True

    # ------------------------------------------------------------------
    # Run loop
    # ------------------------------------------------------------------
    def run(self):
        running = True
        while running:
            running = self.handle_events()
            self.update()
            self.draw()
            self.clock.tick(self.fps)

        pygame.quit()
        print("✓ 2D viewer closed")

    # ------------------------------------------------------------------
    def _print_banner(self):
        print(f"✓ Loaded {self.scenario}")
        print(f"  Static  : {len(self.world.static_objects)}")
        print(f"  Dynamic : {len(self.world.dynamic_objects)}")
        print(f"  Sensors : {len(self.world.sensors)}")
        print(f"  Zones   : {len(self.world.zones)}")
        print("=" * 70)


# =====================================================================
# ENTRY POINT
# =====================================================================
def main():
    parser = argparse.ArgumentParser("ASTAS 2D Launcher")
    parser.add_argument(
        "--scenario",
        choices=["simple", "border_patrol", "building_security"],
        default="simple",
    )
    args = parser.parse_args()

    viewer = ASTAS2DViewer(args.scenario)
    viewer.run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
