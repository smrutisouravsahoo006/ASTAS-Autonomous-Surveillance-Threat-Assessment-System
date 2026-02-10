import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

try:
    from objects.dynamic_objects import Person, Vehicle, Drone  # FIXED: was dymanics_objects
    from objects.static_objects import Building, Wall  # FIXED: Wall capitalized
    from sensors.virtual_sensors import VirtualCamera, VirtualLiDAR
    from world.world import SimulationWorld  # FIXED: was simulation_world
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_OK = False


@dataclass
class BorderPatrolConfig:
    border_length: float = 200.0
    fence_height: float = 4.0
    num_guard_posts: int = 4
    num_cameras: int = 6
    num_intruders: int = 2


class BorderPatrolEnvironment:
    def __init__(self, config: Optional[BorderPatrolConfig] = None):
        if not IMPORTS_OK:
            raise ImportError("Required modules not available")
        self.config = config or BorderPatrolConfig()

    def create_world(self) -> SimulationWorld:
        print("\n" + "=" * 70)
        print("BORDER PATROL ENVIRONMENT")
        print("=" * 70)
        
        world = SimulationWorld()
        
        print("\n[1/6] Building fence...")
        length = self.config.border_length
        
        for i in range(10):
            start_x = -length / 2 + i * (length / 10)
            end_x = start_x + length / 10
            
            # FIXED: renamed from 'wall' to 'fence_wall' to avoid conflict
            fence_wall = Wall(
                name=f"fence_{i+1}",
                start=np.array([start_x, 0, 0]),
                end=np.array([end_x, 0, 0]),
                height=self.config.fence_height  # FIXED: height (not heigth)
                # thickness will use default value 0.2
            )
            world.add_static_object(fence_wall)
        
        print(f"  ✓ {length}m fence")
        
        print("\n[2/6] Creating guard posts...")
        spacing = length / (self.config.num_guard_posts + 1)
        
        for i in range(self.config.num_guard_posts):
            x = -length / 2 + spacing * (i + 1)
            
            post = Building(
                name=f"post_{i+1}",
                position=np.array([x, -10, 0]),
                size=np.array([4, 4, 3])
            )
            world.add_static_object(post)
            
            guard = Person(
                name=f"guard_{i+1}",
                position=np.array([x, -10, 0])
            )
            guard.walk_to(np.array([x, -5, 0]))
            world.add_dynamic_object(guard)
        
        print(f"  ✓ {self.config.num_guard_posts} posts")
        
        print("\n[3/6] Defining zones...")
        # FIXED: Added proper zone definitions instead of empty calls
        world.add_zone(
            name="border_safe_zone",
            zone_type="safe",
            polygon=[[-length/2 - 20, -30], [length/2 + 20, -30], 
                    [length/2 + 20, -10], [-length/2 - 20, -10]]
        )
        world.add_zone(
            name="border_restricted_zone",
            zone_type="restricted",
            polygon=[[-length/2, -5], [length/2, -5], 
                    [length/2, 5], [-length/2, 5]]
        )
        print("  ✓ 2 zones")

        print("\n[4/6] Deploying sensors...")
        cam_spacing = length / (self.config.num_cameras + 1)
        
        for i in range(self.config.num_cameras):
            x = -length / 2 + cam_spacing * (i + 1)
            
            cam = VirtualCamera(
                name=f"cam_{i+1}",
                position=np.array([x, 0, 6]),
                rotation=np.pi / 2,
                fov=120,
                max_range=50
            )
            world.add_sensor(cam)  # FIXED: was add_sensors (plural)
        
        print(f"  ✓ {self.config.num_cameras} cameras")

        print("\n[5/6] Adding patrol...")
        v1 = Vehicle(
            name="patrol_1",
            position=np.array([-length / 2 + 10, -15, 0])
        )
        v1.set_path(
            [[-length / 2 + 10, -20, 0], [length / 2 - 10, -20, 0]],
            speed=15
        )
        world.add_dynamic_object(v1)
        
        v2 = Vehicle(
            name="patrol_2",
            position=np.array([length / 2 - 10, -20, 0])
        )
        v2.set_path(
            [[length / 2 - 10, -20, 0], [-length / 2 + 10, -20, 0]],
            speed=15
        )
        world.add_dynamic_object(v2)
        
        print("  ✓ 2 vehicles")

        print("\n[6/6] Adding intruders...")
        for i in range(self.config.num_intruders):
            x = np.random.uniform(-length / 3, length / 3)
            y = np.random.uniform(5, 15)
            
            intruder = Person(
                name=f"intruder_{i+1}",
                position=np.array([x, y, 0])
            )
            
            # FIXED: Proper conditional syntax
            if i % 2:
                intruder.run_to(np.array([x, -5, 0]))
            else:
                intruder.walk_to(np.array([x, -5, 0]))
            
            world.add_dynamic_object(intruder)  # FIXED: was add_dynamic_objects (plural)
        
        print(f"  ✓ {self.config.num_intruders} intruders")
        
        print("\n✓ Border Patrol Ready")
        print(f"  Objects: {len(world.dynamic_objects)}")
        print(f"  Sensors: {len(world.sensors)}")
        print("=" * 70 + "\n")
        
        return world


def create_border_patrol_scenario():
    """Factory function to create border patrol scenario"""
    config = BorderPatrolConfig()
    env = BorderPatrolEnvironment(config)
    return env.create_world()


# Test the module
if __name__ == "__main__":
    print("Testing Border Patrol Environment...")
    
    if IMPORTS_OK:
        try:
            world = create_border_patrol_scenario()
            
            # Run simulation for a few frames
            print("\nRunning simulation...")
            for i in range(30):
                world.update(0.033)
                
                if i % 10 == 0:
                    detections = world.get_detection()
                    violations = world.check_zone_violations()
                    print(f"Frame {i}: {len(detections)} detections, {len(violations)} violations")
            
            print("\n✓ Border Patrol Environment test complete")
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ Cannot test - imports failed")




