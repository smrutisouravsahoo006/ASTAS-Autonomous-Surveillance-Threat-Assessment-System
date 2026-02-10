import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

try:
    from objects.dynamic_objects import Person, Vehicle, Drone  # FIXED
    from objects.static_objects import Building, Wall  # FIXED
    from sensors.virtual_sensors import VirtualCamera, VirtualLiDAR, VirtualIMU
    from world.world import SimulationWorld  # FIXED
    IMPORTS_OK = True
except ImportError as e:
    print(f"Import error: {e}")
    IMPORTS_OK = False


@dataclass
class BuildingSecurityConfig:
    building_size: np.ndarray = None
    num_entrances: int = 4
    parking_capacity: int = 20
    num_guards: int = 6
    num_cameras: int = 12
    perimeter_fence: bool = True
    enable_employees: bool = True
    enable_visitors: bool = True
    enable_suspicious_activity: bool = True
    
    def __post_init__(self):
        if self.building_size is None:
            self.building_size = np.array([50, 40, 15])


class BuildingSecurityEnvironment:
    def __init__(self, config: Optional[BuildingSecurityConfig] = None):
        if not IMPORTS_OK:
            raise ImportError("Required modules not available")
        self.config = config or BuildingSecurityConfig()
    
    def create_world(self) -> SimulationWorld:
        print("\n" + "=" * 70)
        print("BUILDING SECURITY ENVIRONMENT")
        print("=" * 70)

        world = SimulationWorld()
        self._build_main_building(world)
        self._build_perimeter(world)
        self._create_parking_area(world)
        self._define_zones(world)
        self._deploy_sensors(world)
        self._add_security_personnel(world)
        self._add_traffic(world)

        print("\n✓ Building Security Ready")
        print(f"  Static: {len(world.static_objects)}")
        print(f"  Dynamic: {len(world.dynamic_objects)}")
        print(f"  Sensors: {len(world.sensors)}")
        print("=" * 70 + "\n")

        return world
    
    def _build_main_building(self, world: SimulationWorld):
        print("\n[1/7] Building main structure...")
        size = self.config.building_size
        
        building = Building(
            name="main_building",
            position=np.array([0, 0, 0]),
            size=size,
            color=(0.7, 0.7, 0.8, 1.0)
        )
        world.add_static_object(building)

        num_entrances = self.config.num_entrances
        for i in range(num_entrances):
            angle = (2 * np.pi * i) / num_entrances
            dist = (size[0] + size[1]) / 4
            x = np.cos(angle) * dist
            y = np.sin(angle) * dist

            entrance = Building(
                name=f"entrance_{i+1}",
                position=np.array([x, y, 0]),
                size=np.array([4, 4, 3]),
                color=(0.6, 0.6, 0.7, 1.0)
            )
            world.add_static_object(entrance)
        
        print(f"  ✓ Main building: {size[0]}x{size[1]}x{size[2]}m")
        print(f"  ✓ {num_entrances} entrances")
    
    def _build_perimeter(self, world: SimulationWorld):
        print("\n[2/7] Building perimeter...")
        size = 80
        
        fences = [
            ("north", [-size/2, size/2, 0], [size/2, size/2, 0]),
            ("east", [size/2, size/2, 0], [size/2, -size/2, 0]),
            ("south", [size/2, -size/2, 0], [-size/2, -size/2, 0]),
            ("west", [-size/2, -size/2, 0], [-size/2, size/2, 0])
        ]
        
        for name, start, end in fences:
            fence_wall = Wall(  # FIXED: renamed from 'wall'
                name=f"fence_{name}",
                start=np.array(start),
                end=np.array(end),
                height=2.5,  # FIXED: height (not heigth)
                thickness=0.2,  # FIXED: added explicit parameter (though 0.2 is default)
                color=(0.3, 0.3, 0.3, 1.0)
            )
            world.add_static_object(fence_wall)
        
        print(f"  ✓ Perimeter fence: {size}x{size}m")
    
    def _create_parking_area(self, world: SimulationWorld):
        print("\n[3/7] Creating parking area...")
        capacity = self.config.parking_capacity
        num_parked = min(5, capacity)
        
        for i in range(num_parked):
            x = -30 + i * 6
            y = -25
            
            vehicle = Vehicle(
                name=f"parked_vehicle_{i+1}",
                position=np.array([x, y, 0])
            )
            world.add_dynamic_object(vehicle)  # FIXED: was add_dynamics_objects
        
        print(f"  ✓ Parking: {num_parked}/{capacity} spaces occupied")
    
    def _define_zones(self, world: SimulationWorld):
        print("\n[4/7] Defining zones...")
        size = self.config.building_size
        
        world.add_zone(
            name="building_restricted",
            zone_type="restricted",
            polygon=[
                [-size[0]/2, -size[1]/2],
                [size[0]/2, -size[1]/2],
                [size[0]/2, size[1]/2],
                [-size[0]/2, size[1]/2]
            ]
        )
        
        world.add_zone(
            name="entrance_caution",
            zone_type="caution",
            polygon=[
                [-size[0]/2 - 10, -size[1]/2 - 10],
                [size[0]/2 + 10, -size[1]/2 - 10],
                [size[0]/2 + 10, size[1]/2 + 10],
                [-size[0]/2 - 10, size[1]/2 + 10]
            ]
        )
        
        world.add_zone(
            name="parking_safe",
            zone_type="safe",
            polygon=[
                [-35, -30],
                [35, -30],
                [35, -20],
                [-35, -20]
            ]
        )
        
        print("  ✓ 3 zones defined")
    
    def _deploy_sensors(self, world: SimulationWorld):
        print("\n[5/7] Deploying sensors...")
        size = self.config.building_size
        
        # Corner cameras
        corners = [
            [-size[0]/2, -size[1]/2],
            [size[0]/2, -size[1]/2],
            [size[0]/2, size[1]/2],
            [-size[0]/2, size[1]/2]
        ]
        
        for i, (x, y) in enumerate(corners):
            camera = VirtualCamera(
                name=f"corner_camera_{i+1}",
                position=np.array([x, y, size[2]]),
                rotation=np.arctan2(y, x) + np.pi,
                fov=110.0,
                max_range=50.0
            )
            world.add_sensor(camera)  # FIXED: was add_sensors
        
        # Entrance cameras
        for i in range(self.config.num_entrances):
            angle = (2 * np.pi * i) / self.config.num_entrances
            dist = (size[0] + size[1]) / 4
            x = np.cos(angle) * dist
            y = np.sin(angle) * dist
            
            camera = VirtualCamera(
                name=f"entrance_camera_{i+1}",
                position=np.array([x, y, 4]),
                rotation=angle,
                fov=90.0,
                max_range=30.0
            )
            world.add_sensor(camera)  # FIXED
        
        # Parking cameras
        for i in range(2):
            x = -20 + i * 40
            camera = VirtualCamera(
                name=f"parking_camera_{i+1}",
                position=np.array([x, -25, 8]),
                rotation=-np.pi / 2,  # Facing south
                fov=120.0,
                max_range=40.0
            )
            world.add_sensor(camera)  # FIXED
        
        # Roof LiDAR
        lidar = VirtualLiDAR(
            name="roof_lidar",
            position=np.array([0, 0, size[2] + 2]),
            max_range=60.0
        )
        world.add_sensor(lidar)  # FIXED
        
        total_cameras = 4 + self.config.num_entrances + 2
        print(f"  ✓ {total_cameras} cameras + 1 LiDAR")
    
    def _add_security_personnel(self, world: SimulationWorld):
        print("\n[6/7] Adding security personnel...")
        num_guards = self.config.num_guards
        
        patrol_areas = [
            ("front_entrance", [0, 30, 0], [0, 20, 0]),
            ("back_entrance", [0, -30, 0], [0, -20, 0]),
            ("east_perimeter", [35, 0, 0], [30, 0, 0]),
            ("west_perimeter", [-35, 0, 0], [-30, 0, 0]),
            ("parking_patrol", [-20, -25, 0], [20, -25, 0]),
            ("roof_patrol", [0, 0, 15], [0, 0, 15])  # FIXED: height for roof
        ]
        
        for i in range(num_guards):
            area = patrol_areas[i % len(patrol_areas)]
            name, start, end = area
            
            guard = Person(
                name=f"guard_{name}",
                position=np.array(start)
            )
            guard.walk_to(np.array(end))
            world.add_dynamic_object(guard)
        
        print(f"  ✓ {num_guards} guards")
    
    def _add_traffic(self, world: SimulationWorld):
        """Add traffic (employees, visitors, etc.)"""
        print("\n[7/7] Adding traffic...")
        
        if self.config.enable_employees:
            # Add some employees
            employee = Person(
                name="employee_1",
                position=np.array([40, 0, 0])
            )
            employee.walk_to(np.array([0, 20, 0]))  # Walking to entrance
            world.add_dynamic_object(employee)
        
        if self.config.enable_visitors:
            # Add a visitor vehicle
            visitor = Vehicle(
                name="visitor_car",
                position=np.array([40, 10, 0])
            )
            visitor.set_path(
                [[40, 10, 0], [0, 10, 0], [-20, -25, 0]],  # Drive to parking
                speed=8.0
            )
            world.add_dynamic_object(visitor)
        
        print("  ✓ Traffic added")


def create_building_security_scenario():
    """Factory function to create building security scenario"""
    config = BuildingSecurityConfig()
    env = BuildingSecurityEnvironment(config)
    return env.create_world()


# Test the module
if __name__ == "__main__":
    print("Testing Building Security Environment...")
    
    if IMPORTS_OK:
        try:
            world = create_building_security_scenario()
            
            # Run simulation for a few frames
            print("\nRunning simulation...")
            for i in range(30):
                world.update(0.033)
                
                if i % 10 == 0:
                    detections = world.get_detection()
                    violations = world.check_zone_violations()
                    context = world.get_astas_context()
                    print(f"Frame {i}: {len(detections)} detections, {len(violations)} violations")
            
            print("\n✓ Building Security Environment test complete")
        except Exception as e:
            print(f"\n✗ Test failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("✗ Cannot test - imports failed")