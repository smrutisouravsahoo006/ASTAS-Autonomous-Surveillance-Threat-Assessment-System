#!/usr/bin/env python3
"""
ASTAS — Autonomous Surveillance & Threat Assessment System
==========================================================
Unified entry point for every subsystem.

Usage
-----
  python main.py                            # Full pipeline demo
  python main.py --mode world               # SimulationWorld + zones + violations
  python main.py --mode objects             # Person / Vehicle / Drone movement
  python main.py --mode human               # Human + HumanGroup behaviour
  python main.py --mode buildings           # Building + Wall physics shapes
  python main.py --mode sensors             # CameraSensor + LiDARSensor + VirtualSensors
  python main.py --mode camera_ctrl         # CameraController orbit/pan/zoom
  python main.py --mode physics             # PhysicsEngine (PyBullet) demo
  python main.py --mode collision           # CollisionHandler detection & callbacks
  python main.py --mode render              # RenderEngine (Panda3D / headless)
  python main.py --mode drone               # Drone patrol pattern
  python main.py --mode detection           # Object detection (YOLOv8 or mock)
  python main.py --mode tracking            # Motion tracking (SORT-like)
  python main.py --mode thermal             # Thermal / infrared analysis
  python main.py --mode lidar               # Legacy VirtualLiDAR scan
  python main.py --mode pid                 # PID controller & trajectory planner
  python main.py --mode decision            # Rule-based threat decision engine
  python main.py --mode llm                 # LLM model loader overview
  python main.py --mode finetune            # QLoRA fine-tune config walkthrough
  python main.py --mode dataprep            # Dataset generation & prep
  python main.py --mode viz                 # Map visualiser (OpenCV window)
  python main.py --mode sim3d               # 3-D PyBullet sim — simple
  python main.py --mode sim3d-border        # 3-D PyBullet sim — border patrol
  python main.py --mode sim3d-campus        # 3-D PyBullet sim — building security
  python main.py --list                     # Print all modes
"""

import argparse
import sys
import time
import numpy as np
from typing import Dict


# ─────────────────────────────────────────────────────────────────────────────
# Pretty-print helpers
# ─────────────────────────────────────────────────────────────────────────────

def _banner(title: str):
    line = "=" * 70
    print(f"\n{line}\n  {title}\n{line}")


def _section(title: str):
    print(f"\n{'─' * 50}\n  {title}\n{'─' * 50}")


def _ok(msg: str):   print(f"  ✔  {msg}")
def _info(msg: str): print(f"  •  {msg}")
def _warn(msg: str): print(f"  ⚠  {msg}")


# ─────────────────────────────────────────────────────────────────────────────
# NEW — simulation layer demos
# ─────────────────────────────────────────────────────────────────────────────

def demo_world():
    """SimulationWorld — zones, detection, violations, ASTAS context."""
    _banner("Simulation World — world.py")
    from Simulation.world.world import SimulationWorld

    world = SimulationWorld()

    _section("Adding zones")
    world.add_zone("restricted_perimeter", "restricted",
                   [[0, 0], [20, 0], [20, 20], [0, 20]])
    world.add_zone("caution_buffer", "caution",
                   [[-5, -5], [25, -5], [25, 25], [-5, 25]])
    world.add_zone("safe_outer", "safe",
                   [[-15, -15], [35, -15], [35, 35], [-15, 35]])

    _section("Adding dynamic objects")
    from Simulation.objects.dynamic_objects import Person, Vehicle, Drone

    intruder = Person("intruder_1", position=np.array([10.0, 10.0, 0.0]))
    intruder.start_loitering()
    world.add_dynamic_object(intruder)

    guard = Person("guard_1", position=np.array([-10.0, -10.0, 0.0]))
    guard.walk_to(np.array([5.0, 5.0, 0.0]))
    world.add_dynamic_object(guard)

    patrol = Vehicle("patrol_car", position=np.array([30.0, 30.0, 0.0]))
    patrol.set_path([[30,30,0], [30,-10,0], [-10,-10,0], [-10,30,0]], speed=8.0)
    world.add_dynamic_object(patrol)

    uav = Drone("drone_1")
    uav.patrol_points = [np.array([0,0,15],dtype=float),
                         np.array([20,0,15],dtype=float),
                         np.array([20,20,15],dtype=float)]
    world.add_dynamic_object(uav)

    _section("Simulating 90 frames at 30 fps")
    for i in range(90):
        world.update(1/30)

    detections = world.get_detections()
    violations = world.check_zone_violations()
    _ok(f"Detections  : {len(detections)}")
    _ok(f"Violations  : {len(violations)}")
    for v in violations:
        _info(f"  {v['object']:<14}  zone={v['zone']:<22}  type={v['zone_type']}")

    _section("ASTAS context snapshot")
    ctx = world.get_astas_context()
    for k in ('zone', 'num_detections', 'loitering',
              'rapid_movement', 'speed', 'time_of_day'):
        _info(f"{k}: {ctx[k]}")

    world.print_summary()


def demo_objects():
    """Person, Vehicle, Drone movement and state transitions."""
    _banner("Dynamic Objects — dynamic_objects.py")
    from Simulation.objects.dynamic_objects import Person, Vehicle, Drone, MovementState

    _section("Person — walk → loiter → run")
    p = Person("alice", position=np.array([0.0, 0.0, 0.0]))
    p.walk_to(np.array([5.0, 5.0, 0.0]))
    for _ in range(30): p.update(0.1)
    _info(f"After walk  : pos={np.round(p.position[:2],2)}  state={p.state.name}")

    p.start_loitering()
    for _ in range(60): p.update(0.1)
    _info(f"After loiter: pos={np.round(p.position[:2],2)}  state={p.state.name}")

    p.run_to(np.array([20.0, 0.0, 0.0]))
    for _ in range(30): p.update(0.1)
    speed = float(np.linalg.norm(p.velocity))
    _info(f"After run   : pos={np.round(p.position[:2],2)}  speed={speed:.2f} m/s")

    _section("Vehicle — path following")
    car = Vehicle("patrol_car")
    car.set_path([[0,0,0],[20,0,0],[20,20,0],[0,20,0]], speed=12.0)
    for i in range(150):
        car.update(0.1)
        if i % 30 == 0:
            _info(f"Step {i:3d}: pos={np.round(car.position[:2],2)}")

    _section("Drone — 3-D patrol")
    uav = Drone("uav_1", position=np.array([0.0, 0.0, 20.0]))
    uav.patrol_points = [
        np.array([ 10, 10, 20], dtype=float),
        np.array([-10, 10, 20], dtype=float),
        np.array([-10,-10, 20], dtype=float),
        np.array([ 10,-10, 20], dtype=float),
    ]
    for i in range(80):
        uav.update(0.1)
        if i % 20 == 0:
            _info(f"Step {i:3d}: pos={np.round(uav.position,1)}")


def demo_human():
    """Human object — behavioural states, HumanGroup formation."""
    _banner("Human Object — human.py")
    from Simulation.objects.human import Human, HumanGroup, HumanState

    _section("Individual Human — walk → run → loiter")
    h = Human("alice", position=np.array([0.0, 0.0, 0.0]))
    _info(f"Created: {h}")
    _info(f"Height={h.height}m  Mass={h.mass}kg  Walk={h.walk_speed} m/s  Run={h.run_speed} m/s")

    h.walk_to(np.array([10.0, 10.0, 0.0]))
    for _ in range(50): h.update(0.1)
    _info(f"After walk : pos={np.round(h.get_position()[:2], 2)}  speed={h.get_speed():.2f} m/s")

    h.run_to(np.array([30.0, 30.0, 0.0]))
    for _ in range(20): h.update(0.1)
    _info(f"After run  : pos={np.round(h.get_position()[:2], 2)}  speed={h.get_speed():.2f} m/s")

    h.start_loitering()
    for _ in range(100): h.update(0.1)
    _info(f"After loiter: pos={np.round(h.get_position()[:2], 2)}  moving={h.is_moving()}")

    _section("HumanGroup — 5 people in formation")
    group = HumanGroup("crowd", count=5,
                       center_position=np.array([0.0, 0.0, 0.0]),
                       formation_radius=2.0)
    _ok(f"Group created with {len(group.humans)} humans")
    for hm in group.humans:
        _info(f"  {hm.name:<12}  pos={np.round(hm.get_position()[:2], 2)}")

    group.move_to(np.array([20.0, 15.0, 0.0]))
    for _ in range(80): group.update(0.1)
    _section("Group positions after moving to [20, 15]")
    for hm in group.humans:
        _info(f"  {hm.name:<12}  pos={np.round(hm.get_position()[:2], 2)}"
              f"  speed={hm.get_speed():.2f}")


def demo_buildings():
    """Buildings.py — Building and Wall data structures + PyBullet physics."""
    _banner("Buildings & Walls — Buildings.py")
    from Simulation.objects.Buildings import Building, Wall

    _section("Building data")
    b = Building("warehouse", np.array([10.0, 10.0, 0.0]),
                 np.array([20.0, 15.0, 8.0]), color=(0.6, 0.6, 0.65, 1.0))
    _info(f"Name    : {b.name}")
    _info(f"Position: {b.position}")
    _info(f"Size    : {b.size}  (W×D×H)")
    corners = b.get_corners()
    _info(f"Corners : {corners.shape[0]} points")
    for i, c in enumerate(corners):
        _info(f"  corner {i}: {np.round(c, 2)}")

    _section("Wall data")
    walls = [
        Wall("north_fence", np.array([-50, 0, 0]), np.array([50, 0, 0]),  height=3.0),
        Wall("east_fence",  np.array([50, 0, 0]),  np.array([50, 50, 0]), height=3.0),
        Wall("gate_section",np.array([-10, 0, 0]), np.array([10, 0, 0]),  height=2.0, thickness=0.3),
    ]
    for w in walls:
        length = float(np.linalg.norm(w.end - w.start))
        _info(f"{w.name:<14}  length={length:6.1f}m  h={w.height}m  t={w.thickness}m")

    _section("PyBullet physics (skipped — requires pybullet)")
    _info("Call  building.create_physics(client_id)  to spawn collision body.")
    _info("Call  wall.create_physics(client_id)      to spawn wall body.")
    _ok("Buildings module OK")


def demo_sensors():
    """CameraSensor, LiDARSensor, VirtualCamera/LiDAR/IMU."""
    _banner("Sensor Suite — camera_sensor.py / lidar_sensor.py / virtual_sensors.py")

    # ── mock scene objects ──────────────────────────────────────────────
    class SceneObj:
        def __init__(self, name, pos, r=0.8):
            self.name     = name
            self.position = np.array(pos, dtype=float)
            self.velocity = np.zeros(3)
            self.radius   = r
            self.size     = np.array([r*2, r*2, r*2])

    scene = [
        SceneObj("person_1",  [5,  2, 0], r=0.4),
        SceneObj("person_2",  [8, -3, 0], r=0.4),
        SceneObj("vehicle_1", [0, 15, 0], r=2.0),
        SceneObj("drone_1",   [3,  3, 8], r=0.4),
    ]

    # ── CameraSensor ────────────────────────────────────────────────────
    _section("CameraSensor — FOV detection (headless)")
    from Simulation.sensors.camera_sensor import CameraSensor, CameraIntrinsics

    cam = CameraSensor(
        name="front_cam",
        position=np.array([0.0, -20.0, 5.0]),
        target=np.array([0.0, 5.0, 0.0]),
        intrinsics=CameraIntrinsics(width=1280, height=720, fov=60.0),
    )
    _info(repr(cam))
    dets = cam.detect_objects(scene)
    _ok(f"Camera detections: {len(dets)}")
    for d in dets:
        _info(f"  {d['name']:<12}  dist={d['distance']:5.1f}m  "
              f"angle={d['angle_deg']:5.1f}°  conf={d['confidence']:.2f}")

    # ── LiDARSensor ─────────────────────────────────────────────────────
    _section("LiDARSensor — geometric scan + clustering (headless)")
    from Simulation.sensors.lidar_sensor import LiDARSensor, LiDARConfig

    lidar = LiDARSensor(
        name="roof_lidar",
        position=np.array([0.0, 0.0, 5.0]),
        config=LiDARConfig(max_range=30.0, angular_resolution=1.0, noise_std=0.01),
    )
    _info(repr(lidar))
    pts = lidar.scan(scene)
    _ok(f"Point cloud: {len(pts)} points")

    clusters = lidar.detect_clusters(min_points=2, cluster_radius=2.0)
    _ok(f"Clusters: {len(clusters)}")
    for c in clusters:
        _info(f"  centroid={np.round(c['centroid'][:2],2)}  "
              f"pts={c['num_points']}  "
              f"dist={c['distance']:.1f}m  "
              f"extent={c['extent']:.2f}m")

    # 3-D multi-layer scan
    lidar3d = LiDARSensor(
        "lidar_3d", np.array([0.0, 0.0, 2.0]),
        config=LiDARConfig(vertical_layers=4, vertical_fov=20.0,
                           angular_resolution=2.0, max_range=20.0),
    )
    pts3d = lidar3d.scan(scene)
    _ok(f"3-D scan: {len(pts3d)} points across 4 layers")

    # ── VirtualCamera / LiDAR / IMU (virtual_sensors.py) ───────────────
    _section("Virtual Sensors — virtual_sensors.py")
    from Simulation.sensors.virtual_sensors import VirtualCamera, VirtualLiDAR, VirtualIMU, AudioSensor

    vcam = VirtualCamera("vcam_1", np.array([0, 0, 3]), rotation=0.0,
                         fov=90.0, max_range=40.0)
    vdets = vcam.detect_objects(scene)
    _ok(f"VirtualCamera detections: {len(vdets)}")
    for d in vdets:
        _info(f"  {d['name']:<12}  dist={d['distance']:5.1f}m  conf={d['confidence']:.2f}")

    vlidar = VirtualLiDAR("vlidar_1", np.array([0, 0, 2]), max_range=30.0)
    vpts = vlidar.scan(scene)
    _ok(f"VirtualLiDAR points: {len(vpts)}")

    imu = VirtualIMU("imu_1", attached_to=scene[0])
    imu_data = imu.get_data(dt=0.033)
    _ok(f"IMU accel: {np.round(imu_data['acceleration'],3)}")

    audio = AudioSensor("mic_1", np.array([0.0, 0.0, 1.0]), sensitivity=20.0)
    events = [{'position': np.array([3, 3, 0]), 'type': 'footsteps'},
              {'position': np.array([80, 80, 0]), 'type': 'vehicle_engine'}]
    heard = audio.detect_events(events)
    _ok(f"Audio events detected: {heard}")


def demo_camera_ctrl():
    """CameraController — orbit, zoom, pan, state save/load."""
    _banner("Camera Controller — camera_controller.py")
    from Simulation.rendering.camera_controller import CameraController, CameraMode

    _section("Creating controller (no Panda3D window needed)")
    ctrl = CameraController(camera=None,
                            initial_position=np.array([0.0, -60.0, 40.0]),
                            initial_target=np.array([0.0, 0.0, 0.0]))
    _info(f"Initial state: {ctrl.get_state()}")
    _info(f"Position dtype: {ctrl.position.dtype}")

    _section("Orbit  (Δh=45°, Δv=15°)")
    ctrl.orbit(45, 15)
    _info(f"angle_h={ctrl.angle_h:.1f}°  angle_v={ctrl.angle_v:.1f}°")
    _info(f"position={np.round(ctrl.position,2)}")

    _section("Zoom  (Δdist=+20)")
    ctrl.zoom(20)
    _info(f"distance={ctrl.distance:.1f}m")

    _section("Pan  (Δx=5, Δy=3)")
    ctrl.pan(5, 3)
    _info(f"position={np.round(ctrl.position,2)}")
    _info(f"target={np.round(ctrl.target,2)}")

    _section("Follow mode")
    ctrl.set_mode(CameraMode.FOLLOW)
    for i in range(5):
        obj_pos = np.array([i * 3.0, 0.0, 0.0])
        ctrl.follow_object_position(obj_pos)
        _info(f"  step {i}: cam_pos={np.round(ctrl.position,2)}")

    _section("Direction vectors")
    _info(f"Forward: {np.round(ctrl.get_forward_vector(),3)}")
    _info(f"Right  : {np.round(ctrl.get_right_vector(),3)}")
    _info(f"Up     : {np.round(ctrl.get_up_vector(),3)}")

    _section("Screen→World ray  (centre of 1920×1080)")
    ray = ctrl.screen_to_world(960, 540, 1920, 1080)
    _info(f"Ray direction: {np.round(ray,3)}")

    _section("State serialisation")
    state = ctrl.get_state()
    _info(f"Serialised: {state}")
    ctrl.set_state(state)
    _ok("State round-trip OK")


def demo_physics():
    """PhysicsEngine — create bodies, step simulation, read positions."""
    _banner("Physics Engine — physics_engine.py")
    from Simulation.physics.physics_engine import PhysicsEngine, PYBULLET_AVAILABLE

    if not PYBULLET_AVAILABLE:
        _warn("PyBullet not installed — physics engine cannot run.")
        _info("Install with:  pip install pybullet")
        _info("Module structure and API are still verified below.")

    _section("Engine instantiation (DIRECT / headless mode)")
    engine = PhysicsEngine(gui=False)
    if engine.client_id is None:
        _warn("Physics disabled (no PyBullet).  Skipping simulation steps.")
        return

    _section("Spawning objects")
    box_id    = engine.create_box(np.array([0.0, 0.0, 5.0]),
                                  np.array([1.0, 1.0, 1.0]), mass=1.0)
    sphere_id = engine.create_sphere(np.array([3.0, 0.0, 5.0]),
                                     radius=0.5, mass=0.5)
    box2_id   = engine.create_box(np.array([-3.0, 0.0, 8.0]),
                                  np.array([0.5, 0.5, 0.5]), mass=2.0,
                                  color=(0.2, 0.8, 0.2, 1.0))
    _ok(f"Box body IDs: {box_id}, {box2_id}")
    _ok(f"Sphere body ID: {sphere_id}")

    _section("Applying force and stepping 240 frames (1 s)")
    engine.apply_force(box_id, np.array([5.0, 0.0, 0.0]))
    for _ in range(240):
        engine.step()

    _info(f"Box    final pos: {np.round(engine.get_position(box_id),   3)}")
    _info(f"Sphere final pos: {np.round(engine.get_position(sphere_id),3)}")
    _info(f"Box2   final pos: {np.round(engine.get_position(box2_id),  3)}")
    _info(f"Box    velocity : {np.round(engine.get_velocity(box_id),   3)}")

    _section("set_velocity override")
    engine.set_velocity(sphere_id, np.array([2.0, 1.0, 0.0]))
    engine.step()
    _info(f"Sphere vel after override: {np.round(engine.get_velocity(sphere_id),3)}")

    engine.disconnect()
    _ok("Physics engine disconnected cleanly")


def demo_collision():
    """CollisionHandler — callbacks, ignore pairs, filter, statistics."""
    _banner("Collision Handler — collision_handler.py")
    from Simulation.physics.collision_handler import (CollisionHandler, CollisionEvent,
                                   CollisionType, CollisionFilter,
                                   CollisionDebugger)

    _section("Initialisation")
    handler = CollisionHandler()

    _section("Registering callbacks")
    log = []
    def on_hit(event: CollisionEvent):
        log.append(f"HIT  {event.body_a} ↔ {event.body_b}  impulse={event.impulse:.2f}")

    handler.register_callback(10, on_hit)
    handler.register_callback(20, on_hit)
    _ok("Callbacks registered for body IDs 10 and 20")

    _section("Ignore pairs")
    handler.ignore_collision(10, 30)
    handler.ignore_collision(20, 40)
    _info(f"Ignored pairs: {handler.ignore_pairs}")

    _section("Collision type assignment")
    handler.set_collision_type(10, CollisionType.BOUNCE)
    handler.set_collision_type(20, CollisionType.STOP)
    handler.set_collision_type(30, CollisionType.TRIGGER)
    _info(f"Types: {handler.collision_types}")

    _section("Injecting synthetic collision events")
    for i in range(6):
        ev = CollisionEvent(
            body_a=10, body_b=20,
            position=np.array([float(i), 0.0, 0.0]),
            normal=np.array([0.0, 1.0, 0.0]),
            impulse=float(i * 3),
            timestamp=float(i * 0.1),
        )
        handler._add_to_history(ev)
        handler._trigger_callbacks(ev)

    _ok(f"History length: {len(handler.collision_history)}")
    _ok(f"Callback log entries: {len(log)}")
    for entry in log[:4]:
        _info(f"  {entry}")

    _section("Body-specific history lookup")
    events_for_10 = handler.get_collisions_for_body(10)
    _ok(f"Events involving body 10: {len(events_for_10)}")

    _section("Statistics")
    stats = handler.get_collision_statistics()
    for k, v in stats.items():
        _info(f"  {k}: {v}")

    _section("CollisionFilter — only impulse > 10")
    cf = CollisionFilter()
    cf.add_rule(lambda c: c.impulse > 10.0)
    strong = cf.filter(handler.collision_history)
    _ok(f"Events with impulse > 10: {len(strong)}")
    for ev in strong:
        _info(f"  impulse={ev.impulse:.2f}")

    _section("Manual bounding-box detection on mock objects")
    class MockBody:
        def __init__(self, name, pos, r):
            self.name = name
            self.position = np.array(pos, dtype=float)
            self.radius = r
    bodies = [MockBody("A", [0,0,0], 1.5),
              MockBody("B", [1,0,0], 1.5),   # overlaps A
              MockBody("C", [10,0,0], 1.5)]  # far away
    manual = handler.detect_collisions_manual(bodies)
    _ok(f"Manual collisions detected: {len(manual)}")
    for c in manual:
        _info(f"  {c}")

    handler.clear_history()
    _ok("History cleared")


def demo_render():
    """RenderEngine / SimpleRenderEngine — object management, stats."""
    _banner("Render Engine — render_engine.py")
    from Simulation.rendering.render_engine import SimpleRenderEngine, RenderMode, PANDA3D_AVAILABLE

    if PANDA3D_AVAILABLE:
        _info("Panda3D found — using full RenderEngine")
        _warn("GUI window demos are skipped in CLI mode. "
              "Call  engine.run()  to open the Panda3D window.")
    else:
        _info("Panda3D not installed — using SimpleRenderEngine (headless).")
        _info("Install with:  pip install panda3d")

    _section("SimpleRenderEngine — object lifecycle")
    engine = SimpleRenderEngine()

    engine.create_box("building_1",   np.array([0.0,  0.0, 0.0]), np.array([10, 8, 6]))
    engine.create_box("building_2",   np.array([15.0, 0.0, 0.0]), np.array([6,  6, 4]))
    engine.create_sphere("person_1",  np.array([5.0,  5.0, 0.0]), radius=0.4)
    engine.create_capsule("person_2", np.array([-5.0, 3.0, 0.0]), radius=0.3, height=1.75)
    _ok(f"Objects created: {engine.object_count}")

    engine.update_position("person_1", np.array([6.0, 5.5, 0.0]))
    engine.update_rotation("person_2", np.array([0.0, 0.0, np.pi/4]))
    _ok("Positions and rotations updated")

    engine.remove_object("building_2")
    _ok(f"After remove: {engine.object_count} objects  nodes={len(engine.render_nodes)}")

    _section("Stats")
    stats = engine.get_stats()
    for k, v in stats.items():
        _info(f"  {k}: {v}")
    _ok("RenderEngine test complete")


def demo_drone():
    """Drone module — patrol patterns, 2D/3D waypoints, physics body."""
    _banner("Drone — drone.py")
    from Simulation.objects.drone import Drone, PYBULLET_AVAILABLE as PB

    _section("Creating surveillance drone")
    uav = Drone("surveillance_1", position=np.array([0.0, 0.0, 20.0]))
    _info(f"Name       : {uav.name}")
    _info(f"Position   : {uav.position}  dtype={uav.position.dtype}")
    _info(f"Velocity   : {uav.velocity}  dtype={uav.velocity.dtype}")
    _info(f"Max speed  : {uav.max_speed} m/s")
    _info(f"Hover alt  : {uav.hover_altitude} m")

    _section("2-D patrol pattern at altitude 15 m")
    uav.set_patrol_pattern(
        [[20, 20], [-20, 20], [-20, -20], [20, -20]], altitude=15
    )
    _ok(f"Waypoints set: {len(uav.patrol_points)}")
    for i, wp in enumerate(uav.patrol_points):
        _info(f"  wp{i}: {np.round(wp, 2)}")

    _section("Simulating 80 update steps (dt=0.1 s)")
    for i in range(80):
        uav.update(0.1)
        if i % 20 == 0:
            _info(f"  step {i:2d}: pos={np.round(uav.position, 2)}")

    _section("3-D patrol waypoints")
    uav2 = Drone("surveillance_2")
    uav2.set_patrol_pattern(
        [[10, 10, 15], [20, 20, 20], [10, 30, 10]], altitude=15
    )
    _ok(f"3-D waypoints: {len(uav2.patrol_points)}")
    for wp in uav2.patrol_points:
        _info(f"  {np.round(wp, 2)}")

    _section("PyBullet physics body")
    if not PB:
        _warn("PyBullet not installed — physics body skipped.")
        _info("Call  uav.create_physics(client_id)  when PyBullet is available.")
    else:
        _info("Call  uav.create_physics(client_id)  to spawn physics body.")
    _ok("Drone module test complete")


# ─────────────────────────────────────────────────────────────────────────────
# ORIGINAL demos (kept intact from previous version)
# ─────────────────────────────────────────────────────────────────────────────

def demo_detection():
    """Object detection (YOLOv8 or mock fallback)."""
    _banner("Object Detection — object_detection.py")
    from Object_Detection.object_detection import ObjectDetector

    detector = ObjectDetector()
    frame = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)

    _section("Running detection on synthetic frame")
    detections = detector.detect(frame)
    _ok(f"Detected {len(detections)} objects")
    for d in detections:
        _info(f"{d.class_name:<12}  conf={d.confidence:.2f}  "
              f"bbox={d.bbox}  center={d.center}")

    stats = detector.analyze_detections(detections)
    _section("Detection statistics")
    for k, v in stats.items():
        _info(f"{k}: {v}")

    perf = detector.get_performance_stats()
    _section("Performance")
    for k, v in perf.items():
        _info(f"{k}: {v}")


def demo_tracking():
    """Motion tracking — SORT-like multi-object tracker."""
    _banner("Motion Tracking — motion_tracking.py")
    from Object_Detection.motion_tracking import MotionTracker

    tracker = MotionTracker(max_age=15, min_hits=2, iou_threshold=0.3)

    _section("Simulating 40 frames with two moving objects")
    for frame_idx in range(40):
        dets = [
            {'bbox': (50 + frame_idx*8, 80, 130 + frame_idx*8, 200),
             'class_name': 'person',  'confidence': 0.88},
            {'bbox': (600, 300 + frame_idx*4, 720, 420 + frame_idx*4),
             'class_name': 'car',     'confidence': 0.92},
        ]
        active = tracker.update(dets)
        if frame_idx % 10 == 0:
            _info(f"Frame {frame_idx:3d}: {len(active)} active tracks")

    _section("Track behaviour analysis")
    for t in tracker.tracks:
        b = tracker.analyze_track_behavior(t)
        _info(f"Track {t.track_id} ({t.class_name:<8})  "
              f"loitering={b['loitering']}  rapid={b['rapid_movement']}  "
              f"dir_changes={b['direction_changes']}  "
              f"avg_speed={b['avg_speed']:.1f}px/fr")

    blank = np.zeros((480, 640, 3), dtype=np.uint8)
    output = tracker.draw_tracks(blank)
    _ok(f"Rendered frame: {output.shape}")


def demo_thermal():
    """Thermal / infrared analysis."""
    _banner("Thermal Analysis — thermal_analysis.py")
    from Object_Detection.thermal_analysis import ThermalAnalyzer, generate_synthetic_thermal_frame

    analyzer = ThermalAnalyzer(temp_min=0, temp_max=55)

    _section("Generating synthetic thermal frame")
    thermal_frame = generate_synthetic_thermal_frame(width=640, height=480)
    _ok(f"Thermal frame: {thermal_frame.shape}  dtype={thermal_frame.dtype}")

    _section("Detecting heat signatures")
    signatures = analyzer.process_thermal_frame(thermal_frame)
    _ok(f"Found {len(signatures)} signatures")
    for i, sig in enumerate(signatures, 1):
        _info(f"#{i}  type={sig.signature_type:<8}  temp={sig.temperature:.1f}°C  "
              f"conf={sig.confidence:.2f}  area={sig.area}px²  center={sig.center}")

    enhanced = analyzer.enhance_thermal_image(thermal_frame)
    _ok(f"Enhanced frame: {enhanced.shape}")

    rgb_mock = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    fused = analyzer.fusion_with_rgb(rgb_mock, thermal_frame, alpha=0.6)
    _ok(f"RGB-thermal fusion: {fused.shape}")


def demo_lidar():
    """LiDAR simulation (legacy Lidar_sensor.py)."""
    _banner("LiDAR Sensor — Lidar_sensor.py")
    from Simulation.sensors.lidar_sensor import VirtualLiDAR, ScanPattern

    lidar = VirtualLiDAR(name="perimeter_lidar",
                         position=np.array([0.0, 0.0, 5.0]),
                         scan_pattern=ScanPattern.HORIZONTAL_360,
                         max_range=50.0, angular_resolution=1.0)

    class MockObj:
        def __init__(self, name, pos, size):
            self.name = name
            self.position = np.array(pos, dtype=float)
            self.size = np.array(size, dtype=float)

    objects = [MockObj("box1",    [10,  0, 0], [2,2,2]),
               MockObj("box2",    [ 0, 15, 0], [3,3,3]),
               MockObj("person1", [20, 20, 0], [0.6,0.6,1.75])]

    _section("360° horizontal scan")
    pc = lidar.scan(objects)
    _ok(f"Points: {pc.get_point_count()}")
    if pc.get_point_count() > 0:
        _info(f"Range: {pc.ranges.min():.2f}m – {pc.ranges.max():.2f}m")

    _section("Object detection")
    dets = lidar.detect_objects(pc, min_points=3)
    _ok(f"Objects: {len(dets)}")
    for d in dets:
        _info(f"{d['id']:<14}  dist={d['distance']:.1f}m  pts={d['point_count']}")

    _section("Range filter (5–25 m)")
    filtered = pc.filter_by_range(5, 25)
    _ok(f"Points after filter: {filtered.get_point_count()}")

    _section("Statistics")
    for k, v in lidar.get_statistics().items():
        _info(f"{k}: {v}")


def demo_pid():
    """PID controller and trajectory planner."""
    _banner("PID Controller & Trajectory Planner — pid_controller.py")
    from Control_Systems.pid_controller import PIDcontroller, CameraController, TrajectoryPlanner, Waypoint

    _section("PID — step-response (setpoint=100)")
    pid = PIDcontroller(kp=1.0, ki=0.1, kd=0.05)
    pid.setpoint = 100.0
    measurement = 0.0
    for i in range(100):
        out = pid.compute(measurement)
        measurement += out.value * 0.1
        if i % 20 == 0:
            _info(f"Step {i:3d}  SP={out.setpoint:.0f}  "
                  f"PV={measurement:6.2f}  err={out.error:7.2f}  u={out.value:.2f}")

    _section("Camera controller — target tracking")
    cam = CameraController(max_rotation_speed=30.0)
    target = (800, 400)
    fc = (640, 360)
    fs = (1280, 720)
    for i in range(30):
        target = (target[0] + 10, target[1] + 5)
        ps, ts = cam.track_target(target, fc, fs)
        cam.update_position(ps, ts, dt=0.033)
        if i % 10 == 0:
            _info(f"Frame {i:2d}  pan={cam.current_pan:7.2f}°  "
                  f"tilt={cam.current_tilt:6.2f}°  speeds=({ps:.1f},{ts:.1f})°/s")

    _section("Trajectory planner — 4-waypoint path")
    planner = TrajectoryPlanner(max_speed=2.0, max_acceleration=1.0)
    waypoints = [Waypoint(0, 0), Waypoint(5, 5), Waypoint(10, 5), Waypoint(10, 10)]
    path = planner.plan_path(waypoints)
    vels = planner.compute_velocity_profile(path)
    _ok(f"Path: {len(path)} points  max_vel={max(vels):.2f} m/s")

    idx = 0
    for step in range(200):
        vc, idx = planner.follow_path(path, vels, idx, dt=0.1)
        planner.update_state(vc, 0.1)
        if step % 30 == 0:
            pos = planner.current_position
            spd = float(np.linalg.norm(planner.current_velocity))
            _info(f"Step {step:3d}  pos=({pos[0]:5.2f},{pos[1]:5.2f})  |v|={spd:.2f}")
        if idx >= len(path) - 1:
            _ok(f"Goal reached at step {step}")
            break


def demo_decision():
    """Rule-based (+ optional LLM) threat decision engine."""
    _banner("Decision Engine — decision_engine.py")
    from LLM_Enigne.Inference.decision_engine import LLMDecisionEngine

    engine = LLMDecisionEngine()

    scenarios = [
        ("Normal activity", {
            'detections': ['person'], 'num_detections': 1,
            'primary_object': 'person', 'zone': 'green',
            'time_of_day': 'day', 'restricted_area': False,
            'motion_type': 'slow', 'speed': 'slow',
            'loitering': False, 'direction_changes': 0,
            'time_in_area': 10, 'audio_events': [],
            'vibration': False, 'lidar_objects': 1,
            'previous_alerts': 0, 'unusual_pattern': False,
        }),
        ("Suspicious intruder", {
            'detections': ['person'], 'num_detections': 1,
            'primary_object': 'person', 'zone': 'red',
            'time_of_day': 'night', 'restricted_area': True,
            'motion_type': 'moderate', 'speed': 'medium',
            'loitering': True, 'direction_changes': 7,
            'time_in_area': 120, 'audio_events': [],
            'vibration': False, 'lidar_objects': 1,
            'previous_alerts': 2, 'unusual_pattern': True,
        }),
        ("Critical — gunshot + multiple", {
            'detections': ['person', 'vehicle'], 'num_detections': 5,
            'primary_object': 'person', 'zone': 'red',
            'time_of_day': 'night', 'restricted_area': True,
            'motion_type': 'rapid', 'speed': 'fast',
            'loitering': False, 'direction_changes': 3,
            'time_in_area': 60, 'audio_events': ['gunshot'],
            'vibration': True, 'lidar_objects': 5,
            'previous_alerts': 0, 'unusual_pattern': True,
        }),
    ]

    for name, ctx in scenarios:
        _section(f"Scenario: {name}")
        assessment = engine.assess_threat(ctx)
        print(engine.generate_report(assessment, ctx))


def demo_llm():
    """LLM loader — list available models."""
    _banner("LLM Loader — model_loader.py")
    from LLM_Enigne.Base_Model.model_loader import LLMLoader, ALL_MODELS, GATED_MODELS
    import torch

    vram = 0.0
    if torch.cuda.is_available():
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        _ok(f"GPU: {torch.cuda.get_device_name(0)}  ({vram:.1f} GB VRAM)")
    else:
        _warn("No CUDA GPU — running on CPU.")

    _section("Available models")
    print(f"  {'ALIAS':<28} {'4-bit VRAM':<12} {'TYPE':<14} FITS?")
    print("  " + "─" * 62)
    for alias, entry in sorted(ALL_MODELS.items(), key=lambda x: x[1]["vram_4bit"]):
        gated = "🔒 gated" if alias in GATED_MODELS else "🔓 free"
        fits  = "✅" if vram == 0 or vram >= entry["vram_4bit"] else "❌"
        print(f"  {alias:<28} ~{entry['vram_4bit']:.1f} GB{'':<5} {gated:<14} {fits}")

    _section("Loader init (no download)")
    loader = LLMLoader(model_name="qwen2.5-7b", model_path=None)
    print(loader)
    _info("To load:  model, tok = loader.load_model(quantization='4bit')")
    _info("To chat:  loader.chat('Describe the threat situation...')")


def demo_finetune():
    """QLoRA fine-tuning pipeline walkthrough."""
    _banner("QLoRA Fine-Tuning — fine_tune_qlora.py")
    from LLM_Enigne.Finetunning_Model.fine_tune_qlora import QLoRAConfig, ASTASQLoRATrainer

    _section("Configuration")
    cfg = QLoRAConfig(model_name="meta-llama/Llama-2-7b-chat-hf",
                      lora_r=16, lora_alpha=32, num_epochs=3,
                      per_device_train_batch_size=4,
                      gradient_accumulation_steps=4)
    _info(f"Model         : {cfg.model_name}")
    _info(f"LoRA rank     : {cfg.lora_r}  alpha={cfg.lora_alpha}")
    _info(f"Target modules: {cfg.lora_target_modules}")
    _info(f"Epochs        : {cfg.num_epochs}")
    _info(f"Eff. batch    : {cfg.per_device_train_batch_size * cfg.gradient_accumulation_steps}")
    _info(f"Quantisation  : {cfg.bnb_4bit_quant_type} 4-bit  "
          f"double_quant={cfg.bnb_4bit_use_double_quant}")
    _info(f"Output dir    : {cfg.output_dir}")

    _section("Pipeline steps")
    for s in [
        "1  prepare_model()      — load base LLM with 4-bit BitsAndBytes quantisation",
        "2  prepare_tokenizer()  — load tokeniser, set padding token",
        "3  prepare_dataset()    — load CSV, format as Llama-2 chat, tokenise",
        "4  train()              — QLoRA loop with paged-AdamW optimiser",
        "5  test_inference()     — smoke-test the fine-tuned adapters",
    ]:
        _info(s)

    _warn("Requires: pip install transformers peft bitsandbytes accelerate datasets")
    _warn("Minimum GPU: 8 GB VRAM for 7B model.")


def demo_dataprep():
    """Dataset generation for fine-tuning."""
    _banner("Dataset Preparation — data_prep.py")
    from LLM_Enigne.Finetunning_Model.data_prep import ThreatScenarioGenerator, SFTFormatter, DataSplitter
    from collections import Counter

    _section("Generating threat scenarios")
    gen = ThreatScenarioGenerator(seed=42)
    samples = gen.generate(n=200, balance=True)
    _ok(f"Generated {len(samples)} samples")
    levels = Counter(s.threat_level for s in samples)
    for lvl, cnt in sorted(levels.items()):
        _info(f"  {lvl:<10}: {cnt}")

    _section("SFT formatting")
    fmt = SFTFormatter()
    sft = fmt.format_all_sft(samples)
    _ok(f"SFT records: {len(sft)}")
    if sft:
        msg = sft[0]['messages']
        _info(f"Roles: {[m['role'] for m in msg]}")
        _info(f"User msg (truncated): {msg[1]['content'][:120].strip()}…")

    _section("Train / Val / Test split")
    splitter = DataSplitter(train=0.80, val=0.10, test=0.10)
    train, val, test = splitter.split(sft)
    _ok(f"Train: {len(train)}  Val: {len(val)}  Test: {len(test)}")

    _section("DPO pair generation")
    dpo_pairs = fmt.format_all_dpo(samples[:50])
    _ok(f"DPO pairs from 50 samples: {len(dpo_pairs)}")


def demo_sim3d(scenario: str = "simple", fps: int = 240):
    """Launch the full PyBullet 3-D simulation."""
    _banner(f"3-D Simulation — simulation_3d.py  [{scenario}]")
    try:
        from Simulation.simulation_3d_launch import ASTAS3DSimulation
    except ImportError as e:
        _warn(f"simulation_3d import failed: {e}")
        _warn("Install with:  pip install pybullet")
        return
    sim = ASTAS3DSimulation(scenario=scenario, fps=fps)
    sim.run()
def demo_sim_border_ai(scenario="border_patrol",
        fps=60,
        headless=False,
        log=False,
        profile=False):
    """
    Integrated 3D Border Patrol
    + Real-time LLM Threat Assessment
    + Optional logging & profiling
    """
    _banner("ASTAS — Integrated 3D Border AI Mode")

    from Simulation.simulation_3d_launch import ASTAS3DSimulation
    from LLM_Enigne.Inference.decision_engine import LLMDecisionEngine
    import time
    import json

    sim = ASTAS3DSimulation(
        scenario=scenario,
        fps=fps,
        gui=not headless
    )

    decision_engine = LLMDecisionEngine()

    logfile = None
    if log:
        logfile = open("threat_log.jsonl", "a")

    _ok("Simulation + LLM initialized")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            sim.step()
            frame_count += 1

            ctx = sim.get_astas_context()
            assessment = decision_engine.assess_threat(ctx)

            report = decision_engine.generate_report(assessment, ctx)

            print("\033c", end="")  # clear terminal
            print(report)

            if log:
                logfile.write(json.dumps({
                    "timestamp": time.time(),
                    "context": ctx,
                    "threat_level": assessment.threat_level,
                    "score": assessment.score
                }) + "\n")
                logfile.flush()

            if profile and frame_count % fps == 0:
                elapsed = time.time() - t_start
                fps_actual = frame_count / elapsed
                _info(f"Runtime FPS: {fps_actual:.2f}")

            time.sleep(1 / fps)

    except KeyboardInterrupt:
        _warn("Stopping simulation...")
    finally:
        if logfile:
            logfile.close()
def demo_sim_campus_ai(
        scenario="campus",
        fps=60,
        headless=False,
        log=False,
        profile=False):
    """
    Integrated 3D Campus Surveillance
    + Real-time LLM Threat Assessment
    + Optional logging & profiling
    """

    _banner("ASTAS — Integrated 3D Campus AI Mode")

    from Simulation.simulation_3d_launch import ASTAS3DSimulation
    from LLM_Enigne.Inference.decision_engine import LLMDecisionEngine
    import time
    import json

    # 🔹 Initialize 3D Campus Simulation
    sim = ASTAS3DSimulation(
        scenario=scenario,
        fps=fps,
        gui=not headless
    )

    # 🔹 Initialize LLM Threat Engine
    decision_engine = LLMDecisionEngine()

    logfile = None
    if log:
        logfile = open("campus_threat_log.jsonl", "a")

    _ok("Campus Simulation + LLM initialized")

    frame_count = 0
    t_start = time.time()

    try:
        while True:
            sim.step()
            frame_count += 1

            # 🔹 Get simulation context
            ctx = sim.get_astas_context()

            # 🔹 Threat assessment
            assessment = decision_engine.assess_threat(ctx)

            # 🔹 Generate readable report
            report = decision_engine.generate_report(assessment, ctx)

            # 🔹 Clear terminal and print report
            print("\033c", end="")
            print("🎓 CAMPUS SURVEILLANCE AI\n")
            print(report)

            # 🔹 Logging (optional)
            if log:
                logfile.write(json.dumps({
                    "timestamp": time.time(),
                    "scenario": "campus",
                    "context": ctx,
                    "threat_level": assessment.threat_level,
                    "score": assessment.score
                }) + "\n")
                logfile.flush()

            # 🔹 Performance profiling (optional)
            if profile and frame_count % fps == 0:
                elapsed = time.time() - t_start
                fps_actual = frame_count / elapsed
                _info(f"Runtime FPS: {fps_actual:.2f}")

            time.sleep(1 / fps)

    except KeyboardInterrupt:
        _warn("Stopping campus simulation...")

    finally:
        if logfile:
            logfile.close()


def demo_visualizer():
    """Map visualiser (OpenCV window)."""
    _banner("Map Visualiser — map_visualizer.py")
    try:
        import cv2
    except ImportError:
        _warn("OpenCV not installed — skipping.")
        _info("Install with:  pip install opencv-python")
        return

    from Visulaization.map_visualizer import MapVisualizer, VisualizationConfig
    from  Object_Detection.motion_tracking import MotionTracker
    from Object_Detection.object_detection import Detection
    import time as _time

    cfg = VisualizationConfig(width=1280, height=720, show_trajectories=True)
    vis = MapVisualizer(config=cfg)

    zones = [
        {'type': 'restricted', 'polygon': [(400,200),(800,200),(800,500),(400,500)],
         'name': 'Restricted Zone'},
        {'type': 'green',      'polygon': [(50,50),(350,50),(350,680),(50,680)],
         'name': 'Safe Zone'},
    ]
    mock_dets = [
        Detection(class_id=0, class_name='person', confidence=0.93,
                  bbox=(550,280,620,420), center=(585,350),
                  area=9100, timestamp=_time.time()),
        Detection(class_id=2, class_name='car',    confidence=0.87,
                  bbox=(900,400,1100,520), center=(1000,460),
                  area=13200, timestamp=_time.time()),
    ]
    tracker = MotionTracker()
    active_tracks = tracker.update(
        [{'bbox': d.bbox, 'class_name': d.class_name, 'confidence': d.confidence}
         for d in mock_dets])
    threat = {
        'threat_level': 'high', 'threat_score': 0.72, 'confidence': 0.85,
        'reasoning': 'Person in restricted zone at night — loitering detected.',
        'recommended_actions': ['Alert operator', 'Increase surveillance'],
    }
    sensor_data = {'lidar_objects': 2, 'audio_events': ['footsteps'],
                   'vibration': False, 'fps': 28.5}
    stats = {'total_detections': 2, 'active_tracks': len(active_tracks), 'alerts_today': 3}

    _section("Rendering visualisation frame")
    frame = vis.visualize_complete(detections=mock_dets, tracks=active_tracks,
                                   zones=zones, threat_assessment=threat,
                                   sensor_data=sensor_data, stats=stats)
    _ok(f"Frame rendered: {frame.shape}")

    _section("Showing frame (auto-closes after 3 s)")
    cv2.imshow("ASTAS — Map Visualiser", frame)
    cv2.waitKey(3000)
    cv2.destroyAllWindows()
    _ok("Visualiser closed.")


# ─────────────────────────────────────────────────────────────────────────────
# Full pipeline
# ─────────────────────────────────────────────────────────────────────────────

def demo_full():
    """Run every non-interactive module in sequence."""
    _banner("ASTAS — Full System Demo")

    modules = [
        # Simulation layer (new)
        ("SimulationWorld",    demo_world),
        ("Dynamic Objects",    demo_objects),
        ("Human Object",       demo_human),
        ("Buildings & Walls",  demo_buildings),
        ("Sensor Suite",       demo_sensors),
        ("Camera Controller",  demo_camera_ctrl),
        ("Physics Engine",     demo_physics),
        ("Collision Handler",  demo_collision),
        ("Render Engine",      demo_render),
        ("Drone Module",       demo_drone),
        # Original ASTAS pipeline
        ("Object Detection",   demo_detection),
        ("Motion Tracking",    demo_tracking),
        ("Thermal Analysis",   demo_thermal),
        ("PID / Trajectory",   demo_pid),
        ("Decision Engine",    demo_decision),
        ("LLM Loader",         demo_llm),
        ("Dataset Prep",       demo_dataprep),
        # Intentionally skipped in full: LiDAR legacy, finetune, viz, sim3d
        # (require optional deps / open windows)
    ]

    results: Dict[str, bool] = {}
    for name, fn in modules:
        try:
            fn()
            results[name] = True
        except Exception as exc:
            _warn(f"{name} raised: {exc}")
            results[name] = False

    _banner("Summary")
    passed = sum(v for v in results.values())
    total  = len(results)
    for name, ok in results.items():
        print(f"  {'✔  PASS' if ok else '✘  FAIL'}   {name}")
    print(f"\n  {passed}/{total} modules passed")


# ─────────────────────────────────────────────────────────────────────────────
# Mode registry
# ─────────────────────────────────────────────────────────────────────────────

MODES = {

    # ── new simulation layer ───────────────────────────────────────────
    "world":        ("SimulationWorld: zones, violations, ASTAS context",         demo_world),
    "objects":      ("Person / Vehicle / Drone movement & state",                 demo_objects),
    "human":        ("Human behaviour + HumanGroup formation",                    demo_human),
    "buildings":    ("Building & Wall data structures + physics shapes",          demo_buildings),
    "sensors":      ("CameraSensor + LiDARSensor + VirtualSensors",               demo_sensors),
    "camera_ctrl":  ("CameraController orbit/pan/zoom/follow",                    demo_camera_ctrl),
    "physics":      ("PhysicsEngine — PyBullet rigid bodies",                     demo_physics),
    "collision":    ("CollisionHandler — callbacks, filter, statistics",          demo_collision),
    "render":       ("RenderEngine — Panda3D / headless object management",       demo_render),
    "drone":        ("Drone patrol patterns and physics body",                    demo_drone),

    # ── original ASTAS pipeline ────────────────────────────────────────
    "detection":    ("Object detection (YOLOv8 or mock)",                         demo_detection),
    "tracking":     ("Motion tracking (SORT-like)",                                demo_tracking),
    "thermal":      ("Thermal / infrared analysis",                                demo_thermal),
    "lidar":        ("Legacy VirtualLiDAR scan (Lidar_sensor.py)",                 demo_lidar),
    "pid":          ("PID controller & trajectory planner",                        demo_pid),
    "decision":     ("Rule-based threat decision engine",                          demo_decision),
    "llm":          ("LLM model loader overview",                                  demo_llm),
    "finetune":     ("QLoRA fine-tuning config walkthrough",                       demo_finetune),
    "dataprep":     ("Dataset generation & preparation",                           demo_dataprep),
    "viz":          ("Map visualiser (requires opencv-python)",                    demo_visualizer),

    # ── 3-D simulation ─────────────────────────────────────────────────
    "sim3d": (
        "3-D PyBullet simulation — simple scenario",
        lambda args: demo_sim3d("simple", args.fps)
    ),

    "sim3d-border": (
        "3-D PyBullet simulation — border patrol",
        lambda args: demo_sim3d("border_patrol", args.fps)
    ),

    "sim3d-campus": (
        "3-D PyBullet simulation — building security",
        lambda args: demo_sim3d("building_security", args.fps)
    ),

    # ── NEW: Integrated AI Simulation Modes ───────────────────────────

    "sim-border-ai": (
        "Integrated Border Patrol (3D + LLM Threat Engine)",
        lambda args: demo_sim_border_ai(
            scenario="border_patrol",
            fps=args.fps,
            headless=args.headless,
            log=args.log,
            profile=args.profile
        )
    ),

    "sim-campus-ai": (
        "Integrated Building Security (3D + LLM Threat Engine)",
        lambda args: demo_sim_campus_ai(
           scenario="campus",
           fps=args.fps,
           headless=args.headless,
           log=args.log,
           profile=args.profile
        )
    ),

    # ── full pipeline ──────────────────────────────────────────────────
    "full": (
        "Full pipeline (all non-interactive modules)",
        demo_full
    ),
}

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="main.py",
        description="ASTAS — Autonomous Surveillance & Threat Assessment System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="\n".join(
            f"  python main.py --mode {k:<14}  {v}"
            for k, (v, _) in MODES.items()
        ),
    )
    parser.add_argument(
        "--mode", default="full",
        choices=list(MODES.keys()),
        help="Subsystem demo to run  (default: full)",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List all modes and exit",
    )
    parser.add_argument("--fps", type=int, default=60,
                    help="Simulation FPS (for sim modes)")

    parser.add_argument("--headless", action="store_true",
                    help="Run simulation without GUI (DIRECT mode)")

    parser.add_argument("--log", action="store_true",
                    help="Enable threat logging to file")

    parser.add_argument("--profile", action="store_true",
                    help="Show performance metrics")
    return parser


def main() -> int:
    parser = build_parser()
    args   = parser.parse_args()

    if args.list:
        _banner("ASTAS — Available modes")
        col_groups = [
            ("Simulation layer (new)",
             ["world","objects","human","buildings","sensors",
              "camera_ctrl","physics","collision","render","drone"]),
            ("Original ASTAS pipeline",
             ["detection","tracking","thermal","lidar","pid",
              "decision","llm","finetune","dataprep","viz"]),
            ("3-D simulation (PyBullet)",
            ["sim3d","sim3d-border","sim3d-campus",
            "sim-border-ai","sim-campus-ai"]),
            ("Full pipeline",
             ["full"]),
        ]
        for group_name, keys in col_groups:
            print(f"\n  ── {group_name} ──")
            for k in keys:
                desc, _ = MODES[k]
                print(f"    {k:<16}  {desc}")
        print()
        return 0

    desc, fn = MODES[args.mode]

    _banner(f"ASTAS  ·  mode = {args.mode}")
    _info(desc)

    t0 = time.time()

    try:
       import inspect

       if len(inspect.signature(fn).parameters) == 0:
         fn()
       else:
         fn(args)

    except KeyboardInterrupt:
       _warn("Interrupted by user.")
       return 1

    except Exception as e:
       print(f"\n  ⚠  Unhandled exception in mode '{args.mode}': {e}")
       import traceback
       traceback.print_exc()
       return 1

    finally:
       dt = time.time() - t0
       print(f"\n✓ Mode '{args.mode}' finished in {dt:.2f}s\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())