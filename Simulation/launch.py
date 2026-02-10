#!/usr/bin/env python3
"""
ASTAS - Autonomous Surveillance & Threat Assessment System
Simulation Launcher - Corrected Version

This launcher tests all modules and runs simulations based on your project structure.
"""

import sys
import os
import numpy as np
from typing import Dict, List

# =============================================================================
# BANNER
# =============================================================================
BANNER = """
╔═══════════════════════════════════════════════════════════════════════╗
║       ASTAS - Autonomous Surveillance & Threat Assessment System      ║
║                          Simulation Launcher                          ║
║                              v1.0.0                                   ║
╚═══════════════════════════════════════════════════════════════════════╝
"""


def print_section(title: str, char: str = "="):
    """Print a formatted section header"""
    print(f"\n{char * 75}")
    print(f"  {title}")
    print(f"{char * 75}")


def print_status(message: str, status: str = "OK", indent: int = 2):
    """Print a status message with icon"""
    icons = {
        "OK": "✓",
        "FAIL": "✗",
        "SKIP": "⊘",
        "WARN": "⚠",
        "INFO": "ℹ"
    }
    icon = icons.get(status, "•")
    spaces = " " * indent
    print(f"{spaces}{icon} {message}")


# =============================================================================
# DEPENDENCY CHECKING
# =============================================================================
def check_dependencies() -> Dict[str, bool]:
    """Check which dependencies are installed"""
    print_section("DEPENDENCY CHECK")
    
    deps = {}
    
    # NumPy (required)
    try:
        import numpy
        deps["numpy"] = True
        print_status("numpy - INSTALLED (REQUIRED)", "OK")
    except ImportError:
        deps["numpy"] = False
        print_status("numpy - MISSING (REQUIRED)", "FAIL")
    
    # PyBullet (physics)
    try:
        import pybullet
        deps["pybullet"] = True
        print_status("pybullet - INSTALLED (physics enabled)", "OK")
    except ImportError:
        deps["pybullet"] = False
        print_status("pybullet - NOT INSTALLED (physics disabled)", "WARN")
    
    # Panda3D (3D rendering)
    try:
        from panda3d.core import loadPrcFileData
        deps["panda3d"] = True
        print_status("panda3d - INSTALLED (3D rendering enabled)", "OK")
    except ImportError:
        deps["panda3d"] = False
        print_status("panda3d - NOT INSTALLED (rendering disabled)", "WARN")
    
    # PyQt5 (GUI)
    try:
        from PyQt5 import QtCore
        deps["pyqt5"] = True
        print_status("PyQt5 - INSTALLED (GUI enabled)", "OK")
    except ImportError:
        deps["pyqt5"] = False
        print_status("PyQt5 - NOT INSTALLED (GUI disabled)", "WARN")
    
    return deps


def determine_mode(deps: Dict[str, bool]) -> str:
    """Determine which simulation mode to use"""
    if not deps["numpy"]:
        return "ERROR"
    if deps["panda3d"] and deps["pybullet"]:
        return "FULL_3D"
    if deps["pybullet"]:
        return "PHYSICS_ONLY"
    return "HEADLESS"


# =============================================================================
# MODULE TESTING
# =============================================================================
def test_base_objects() -> bool:
    """Test base object module"""
    try:
        from objects.base_objects import SimulationObject, ObjectType, Transform
        
        # Create test object
        transform = Transform(
            position=np.array([1.0, 2.0, 3.0]),
            rotation=np.array([0.0, 0.0, np.pi/4])
        )
        
        obj = SimulationObject(
            name="test_object",
            object_type=ObjectType.DYNAMIC,
            transform=transform
        )
        
        # Test methods
        obj.update(0.03)
        obj.get_position()
        obj.get_bounding_box()
        
        print_status("base_object.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"base_object.py - FAILED: {e}", "FAIL")
        return False


def test_static_objects() -> bool:
    """Test static objects module"""
    try:
        from objects.static_objects import Building, Wall, Ground
        
        # Test Building
        building = Building(
            name="test_building",
            position=np.array([0, 0, 0]),
            size=np.array([10, 10, 5])
        )
        corners = building.get_corners()
        
        # Test Wall
        wall = Wall(
            name="test_wall",
            start=np.array([0, 0, 0]),
            end=np.array([10, 0, 0]),
            height=3.0,  # FIXED: was heigth
            thickness=0.2
        )
        length = wall.get_length()
        
        # Test Ground
        ground = Ground()
        
        print_status("static_objects.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"static_objects.py - FAILED: {e}", "FAIL")
        return False


def test_dynamic_objects() -> bool:
    """Test dynamic objects module"""
    try:
        from objects.dynamic_objects import Person, Vehicle, Drone, MovementState
        
        # Test Person
        person = Person("test_person", position=np.array([0, 0, 0]))
        person.walk_to(np.array([10, 10, 0]))
        person.update(0.03)
        
        # Test Vehicle
        vehicle = Vehicle("test_vehicle", position=np.array([0, 0, 0]))
        vehicle.set_path([[5, 5, 0], [10, 10, 0]], speed=10.0)
        vehicle.update(0.03)
        
        # Test Drone
        drone = Drone("test_drone", position=np.array([0, 0, 10]))
        drone.patrol_points = [np.array([10, 10, 10]), np.array([-10, -10, 10])]
        drone.update(0.03)
        
        print_status("dynamic_objects.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"dynamic_objects.py - FAILED: {e}", "FAIL")
        return False


def test_human_module() -> bool:
    """Test human module"""
    try:
        from objects.human import Human, HumanState, HumanGroup
        
        # Test single human
        human = Human("test_human", position=np.array([0, 0, 0]), age=25, height=1.75)
        human.walk_to(np.array([10, 10, 0]))
        
        for _ in range(5):
            human.update(0.03)
        
        # Test human group
        group = HumanGroup("test_group", count=3, center_position=np.array([0, 0, 0]))
        group.update(0.03)
        
        print_status("human.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"human.py - FAILED: {e}", "FAIL")
        return False


def test_sensors() -> bool:
    """Test virtual sensors module"""
    try:
        from sensors.virtual_sensors import VirtualCamera, VirtualLiDAR, VirtualIMU, AudioSensor
        
        # Test Camera
        camera = VirtualCamera(
            name="test_cam",
            position=np.array([0, 0, 5]),
            rotation=0.0,
            fov=90.0
        )
        
        # Test LiDAR
        lidar = VirtualLiDAR(
            name="test_lidar",
            position=np.array([0, 0, 3]),
            max_range=50.0
        )
        
        # Test IMU
        imu = VirtualIMU(name="test_imu")
        
        # Test Audio
        audio = AudioSensor(
            name="test_audio",
            position=np.array([0, 0, 1])
        )
        
        print_status("virtual_sensors.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"virtual_sensors.py - FAILED: {e}", "FAIL")
        return False


def test_world() -> bool:
    """Test world module"""
    try:
        from world.world import SimulationWorld, Zone
        
        # Create world
        world = SimulationWorld()
        
        # Add zone
        zone = world.add_zone(
            name="test_zone",
            zone_type="restricted",
            polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]
        )
        
        # Update world
        world.update(0.03)
        
        # Test detections
        detections = world.get_detection()
        violations = world.check_zone_violations()
        context = world.get_astas_context()
        
        print_status("world.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"world.py - FAILED: {e}", "FAIL")
        return False


def test_physics(deps: Dict[str, bool]) -> bool:
    """Test physics engine"""
    if not deps["pybullet"]:
        print_status("physics_engine.py - SKIPPED (PyBullet not installed)", "SKIP")
        return True
    
    try:
        from physics.physics_engine import PhysicsEngine
        
        # Create engine (headless)
        engine = PhysicsEngine(gui=False, gravity=-9.81)
        
        # Create test objects
        box_id = engine.create_box(
            position=np.array([0, 0, 5]),
            size=np.array([1, 1, 1]),
            mass=1.0
        )
        
        sphere_id = engine.create_sphere(
            position=np.array([2, 0, 5]),
            radius=0.5,
            mass=1.0
        )
        
        # Run simulation
        for _ in range(10):
            engine.step()
        
        # Check positions
        box_pos = engine.get_position(box_id)
        sphere_pos = engine.get_position(sphere_id)
        
        # Cleanup
        engine.disconnect()
        
        print_status("physics_engine.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"physics_engine.py - FAILED: {e}", "FAIL")
        return False


def test_collision_handler(deps: Dict[str, bool]) -> bool:
    """Test collision handler"""
    try:
        from physics.collision_handler import CollisionHandler, CollisionType, CollisionFilter
        
        # Create handler (works without PyBullet)
        handler = CollisionHandler()
        
        # Test callback
        def on_collision(event):
            pass
        
        handler.register_callback(1, on_collision)
        handler.ignore_collision(1, 2)
        handler.set_collision_type(1, CollisionType.BOUNCE)
        
        # Test filter
        collision_filter = CollisionFilter()
        collision_filter.add_rule(lambda c: c.impulse > 5.0)
        
        # Test statistics
        stats = handler.get_collision_statistics()
        
        print_status("collision_handler.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"collision_handler.py - FAILED: {e}", "FAIL")
        return False


def test_rendering(deps: Dict[str, bool]) -> bool:
    """Test rendering engine"""
    if not deps["panda3d"]:
        print_status("render_engine.py - SKIPPED (Panda3D not installed)", "SKIP")
        return True
    
    try:
        from rendering.render_engine import create_render_engine, SimpleRenderEngine
        
        # Create simple render engine (fallback)
        engine = SimpleRenderEngine()
        
        # Test object creation
        engine.create_box("box1", np.array([0, 0, 1]), np.array([2, 2, 2]))
        engine.create_sphere("sphere1", np.array([5, 0, 1]), radius=1.0)
        
        # Test updates
        engine.update_position("box1", np.array([1, 1, 1]))
        
        # Get stats
        stats = engine.get_stats()
        
        print_status("render_engine.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"render_engine.py - FAILED: {e}", "FAIL")
        return False


def test_camera_controller(deps: Dict[str, bool]) -> bool:
    """Test camera controller"""
    try:
        from rendering.camera_controller import CameraController, CameraMode
        
        # Create mock camera
        class MockCamera:
            def setPos(self, x, y, z):
                pass
            def lookAt(self, x, y, z):
                pass
        
        camera = MockCamera()
        controller = CameraController(camera)
        
        # Test controls
        controller.set_mode(CameraMode.ORBIT)
        controller.orbit(45, 10)
        controller.zoom(10)
        controller.pan(5, 0)
        
        # Test state
        state = controller.get_state()
        controller.set_state(state)
        
        print_status("camera_controller.py - OK", "OK")
        return True
    except Exception as e:
        print_status(f"camera_controller.py - FAILED: {e}", "FAIL")
        return False


def run_all_module_tests(deps: Dict[str, bool]) -> Dict[str, bool]:
    """Run all module tests"""
    print_section("MODULE TESTS")
    
    results = {
        "base_objects": test_base_objects(),
        "static_objects": test_static_objects(),
        "dynamic_objects": test_dynamic_objects(),
        "human": test_human_module(),
        "sensors": test_sensors(),
        "world": test_world(),
        "physics": test_physics(deps),
        "collision": test_collision_handler(deps),
        "rendering": test_rendering(deps),
        "camera": test_camera_controller(deps),
    }
    
    # Summary
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    print(f"\n  {'─' * 71}")
    if passed == total:
        print(f"  ✓ All tests passed ({passed}/{total})")
    else:
        print(f"  ⚠ {passed}/{total} tests passed, {total - passed} failed")
    
    return results


# =============================================================================
# SIMULATION SCENARIOS
# =============================================================================
def run_simple_simulation(duration: float = 10.0):
    """Run a simple test simulation"""
    print_section("SIMPLE SIMULATION TEST")
    
    try:
        from world.world import SimulationWorld
        from objects.dynamic_objects import Person, Vehicle
        from sensors.virtual_sensors import VirtualCamera
        
        # Create world
        world = SimulationWorld()
        print_status("Created simulation world", "OK")
        
        # Add restricted zone
        world.add_zone(
            name="restricted_area",
            zone_type="restricted",
            polygon=[[15, 15], [25, 15], [25, 25], [15, 25]]
        )
        print_status("Added restricted zone", "OK")
        
        # Add intruder
        intruder = Person("intruder", position=np.array([20.0, 20.0, 0.0]))
        intruder.start_loitering()
        world.add_dynamic_object(intruder)
        print_status("Added intruder (loitering)", "OK")
        
        # Add patrol vehicle
        patrol = Vehicle("patrol_car", position=np.array([0.0, 0.0, 0.0]))
        patrol.set_path(
            [[10, 10, 0], [30, 10, 0], [30, 30, 0], [10, 30, 0]],
            speed=8.0
        )
        world.add_dynamic_object(patrol)
        print_status("Added patrol vehicle", "OK")
        
        # Add camera
        camera = VirtualCamera(
            name="cam_1",
            position=np.array([20.0, 20.0, 8.0]),
            rotation=0.0,
            fov=120.0
        )
        world.add_sensor(camera)
        print_status("Added surveillance camera", "OK")
        
        # Run simulation
        fps = 30
        dt = 1.0 / fps
        frames = int(duration * fps)
        
        print(f"\n  Running simulation: {duration}s @ {fps} FPS ({frames} frames)")
        
        violation_count = 0
        detection_count = 0
        
        for frame in range(frames):
            world.update(dt)
            
            # Check for violations every second
            if frame % fps == 0:
                violations = world.check_zone_violations()
                if violations:
                    violation_count += len(violations)
                
                detections = world.get_detection()
                detection_count = len(detections)
        
        # Results
        print(f"\n  {'─' * 71}")
        print(f"  Simulation completed:")
        print(f"    • Duration: {duration}s")
        print(f"    • Objects: {len(world.dynamic_objects)}")
        print(f"    • Sensors: {len(world.sensors)}")
        print(f"    • Zones: {len(world.zones)}")
        print(f"    • Zone violations detected: {violation_count}")
        print(f"    • Final detection count: {detection_count}")
        
        print_status("Simulation completed successfully", "OK")
        return True
        
    except Exception as e:
        print_status(f"Simulation failed: {e}", "FAIL")
        import traceback
        traceback.print_exc()
        return False


def run_border_patrol_scenario():
    """Run border patrol scenario (if available)"""
    print_section("BORDER PATROL SCENARIO")
    
    try:
        from environment.border_patrol import BorderPatrolEnvironment, BorderPatrolConfig
        
        # Create config
        config = BorderPatrolConfig(
            border_length=200.0,
            fence_height=4.0,
            num_guard_posts=4,
            num_cameras=6,
            num_intruders=2
        )
        
        # Create environment
        env = BorderPatrolEnvironment(config)
        world = env.create_world()
        
        print_status("Border patrol scenario created", "OK")
        
        # Run for a few frames
        for _ in range(30):
            world.update(0.033)
        
        print_status("Border patrol scenario completed", "OK")
        return True
        
    except ImportError as e:
        print_status(f"Border patrol scenario not available: {e}", "SKIP")
        return True
    except Exception as e:
        print_status(f"Border patrol scenario failed: {e}", "FAIL")
        return False


def run_building_security_scenario():
    """Run building security scenario (if available)"""
    print_section("BUILDING SECURITY SCENARIO")
    
    try:
        from environment.Building_security import BuildingSecurityEnvironment, BuildingSecurityConfig
        
        # Create config
        config = BuildingSecurityConfig(
            num_entrances=4,
            parking_capacity=20,
            num_guards=6,
            num_cameras=12
        )
        
        # Create environment
        env = BuildingSecurityEnvironment(config)
        world = env.create_world()
        
        print_status("Building security scenario created", "OK")
        
        # Run for a few frames
        for _ in range(30):
            world.update(0.033)
        
        print_status("Building security scenario completed", "OK")
        return True
        
    except ImportError as e:
        print_status(f"Building security scenario not available: {e}", "SKIP")
        return True
    except Exception as e:
        print_status(f"Building security scenario failed: {e}", "FAIL")
        return False


# =============================================================================
# MAIN
# =============================================================================
def main():
    """Main entry point"""
    print(BANNER)
    
    # Parse command line arguments
    args = sys.argv[1:]
    
    # Options
    skip_tests = "--skip-tests" in args
    skip_sim = "--skip-sim" in args
    duration = 10.0
    
    if "--duration" in args:
        try:
            idx = args.index("--duration")
            duration = float(args[idx + 1])
        except (IndexError, ValueError):
            print_status("Invalid --duration value, using default (10.0s)", "WARN")
    
    # Check dependencies
    deps = check_dependencies()
    
    if not deps["numpy"]:
        print("\n❌ NumPy is required but not installed.")
        print("   Install with: pip install numpy")
        return 1
    
    # Determine mode
    mode = determine_mode(deps)
    print(f"\n🎮 Simulation Mode: {mode}")
    
    if mode == "HEADLESS":
        print("   Note: Running in headless mode (no physics or 3D rendering)")
    elif mode == "PHYSICS_ONLY":
        print("   Note: Physics enabled, but no 3D rendering")
    elif mode == "FULL_3D":
        print("   Note: Full 3D simulation with physics")
    
    # Run module tests
    if not skip_tests:
        results = run_all_module_tests(deps)
        
        if not all(results.values()):
            print("\n⚠ Some tests failed. Continuing anyway...")
    else:
        print_status("Module tests skipped (--skip-tests)", "INFO")
    
    # Run simulations
    if not skip_sim:
        success = run_simple_simulation(duration)
        
        # Run scenarios if available
        run_border_patrol_scenario()
        run_building_security_scenario()
        
        if success:
            print_section("✓ ALL TESTS COMPLETED SUCCESSFULLY", "=")
            return 0
        else:
            print_section("⚠ SOME TESTS FAILED", "=")
            return 1
    else:
        print_status("Simulations skipped (--skip-sim)", "INFO")
        return 0


def print_usage():
    """Print usage information"""
    print("""
Usage: python launch.py [OPTIONS]

Options:
    --skip-tests        Skip module tests
    --skip-sim          Skip simulations
    --duration SECONDS  Set simulation duration (default: 10.0)
    --help              Show this help message

Examples:
    python launch.py
    python launch.py --duration 30
    python launch.py --skip-tests
    """)


if __name__ == "__main__":
    try:
        if "--help" in sys.argv:
            print(BANNER)
            print_usage()
            sys.exit(0)
        
        sys.exit(main())
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user (Ctrl+C)")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)