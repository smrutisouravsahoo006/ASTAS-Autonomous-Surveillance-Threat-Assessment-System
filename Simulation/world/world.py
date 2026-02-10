"""
Simulation World Module
Main world container for all simulation objects, sensors, and zones
"""

import numpy as np
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class Zone:
    """Defined zone with security classification"""
    name: str
    type: str  # 'safe', 'caution', 'restricted'
    polygon: np.ndarray
    color: tuple = (1.0, 0.0, 0.0, 0.3)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Set color based on zone type"""
        if self.type == 'restricted':
            self.color = (1.0, 0.0, 0.0, 0.3)  # Red
        elif self.type == 'caution':
            self.color = (1.0, 1.0, 0.0, 0.3)  # Yellow
        elif self.type == 'safe':
            self.color = (0.0, 1.0, 0.0, 0.3)  # Green
        else:
            self.color = (0.5, 0.5, 0.5, 0.3)  # Gray


class SimulationWorld:
    """
    Main simulation world container
    
    Manages:
    - Static objects (buildings, walls, terrain)
    - Dynamic objects (people, vehicles, drones)
    - Sensors (cameras, LiDAR, IMU)
    - Zones (restricted, caution, safe areas)
    - Events and detections
    """
    
    def __init__(self):
        """Initialize simulation world"""
        # Object containers
        self.static_objects = []
        self.dynamic_objects = []
        self.sensors = []
        self.zones = []
        
        # Simulation time
        self.time = 0.0
        self.dt = 0.033  # 30 FPS default
        
        # World bounds
        self.bounds = {
            'min': np.array([-100, -100, 0]),
            'max': np.array([100, 100, 50])
        }
        
        # Event log
        self.events = []
        self.max_events = 1000
        
        # Statistics
        self.stats = {
            'total_updates': 0,
            'total_detections': 0,
            'total_violations': 0
        }
    
    # =========================================================================
    # OBJECT MANAGEMENT
    # =========================================================================
    
    def add_static_object(self, obj):
        """
        Add static object to world
        
        Args:
            obj: Static object (Building, Wall, Ground, etc.)
            
        Returns:
            The added object
        """
        self.static_objects.append(obj)
        print(f"  Added static: {obj.name}")
        return obj
    
    def add_dynamic_object(self, obj):
        """
        Add dynamic object to world
        
        Args:
            obj: Dynamic object (Person, Vehicle, Drone, etc.)
            
        Returns:
            The added object
        """
        self.dynamic_objects.append(obj)
        print(f"  Added dynamic: {obj.name} ({obj.__class__.__name__})")
        return obj
    
    def add_sensor(self, sensor):  # FIXED: was add_sensors (plural)
        """
        Add sensor to world
        
        Args:
            sensor: Sensor object (VirtualCamera, VirtualLiDAR, etc.)
            
        Returns:
            The added sensor
        """
        self.sensors.append(sensor)
        print(f"  Added sensor: {sensor.name} ({sensor.__class__.__name__})")
        return sensor
    
    def add_zone(self, name: str, zone_type: str, polygon: List) -> Zone:
        """
        Add security zone to world
        
        Args:
            name: Zone name
            zone_type: 'restricted', 'caution', or 'safe'
            polygon: List of [x, y] points defining the zone boundary
            
        Returns:
            Created Zone object
        """
        zone = Zone(
            name=name,
            type=zone_type,
            polygon=np.array(polygon)
        )
        
        self.zones.append(zone)
        print(f"  Added zone: {name} ({zone_type})")
        return zone
    
    def remove_object(self, obj):
        """Remove object from world"""
        if obj in self.static_objects:
            self.static_objects.remove(obj)
        elif obj in self.dynamic_objects:
            self.dynamic_objects.remove(obj)
        elif obj in self.sensors:
            self.sensors.remove(obj)
    
    def get_object_by_name(self, name: str):
        """Find object by name"""
        # Check static objects
        for obj in self.static_objects:
            if obj.name == name:
                return obj
        
        # Check dynamic objects
        for obj in self.dynamic_objects:
            if obj.name == name:
                return obj
        
        # Check sensors
        for sensor in self.sensors:
            if sensor.name == name:
                return sensor
        
        return None
    
    # =========================================================================
    # SIMULATION UPDATE
    # =========================================================================
    
    def update(self, dt: Optional[float] = None):
        """
        Update world simulation
        
        Args:
            dt: Time step in seconds (uses default if None)
        """
        if dt is None:
            dt = self.dt
        
        # Update all dynamic objects
        for obj in self.dynamic_objects:
            if hasattr(obj, 'update'):
                obj.update(dt)
        
        # Update sensors (if they have update methods)
        for sensor in self.sensors:
            if hasattr(sensor, 'update'):
                sensor.update(dt)
        
        # Update time
        self.time += dt
        self.stats['total_updates'] += 1
    
    # =========================================================================
    # DETECTION & SENSING
    # =========================================================================
    
    def get_detections(self) -> List[Dict]:
        """
        Get all detections from all objects
        
        Returns:
            List of detection dictionaries
        """
        detections = []
        
        for obj in self.dynamic_objects:
            detection = {
                'name': obj.name,
                'type': obj.__class__.__name__.lower(),
                'position': obj.position.tolist() if hasattr(obj, 'position') else [0, 0, 0],
                'velocity': obj.velocity.tolist() if hasattr(obj, 'velocity') else [0, 0, 0],
                'speed': float(np.linalg.norm(obj.velocity)) if hasattr(obj, 'velocity') else 0.0,
                'timestamp': self.time
            }
            
            # Add state if available
            if hasattr(obj, 'state'):
                detection['state'] = str(obj.state)
            
            # Add rotation if available
            if hasattr(obj, 'rotation'):
                detection['rotation'] = float(obj.rotation)
            
            detections.append(detection)
        
        self.stats['total_detections'] = len(detections)
        return detections
    
    # Alias for backwards compatibility
    def get_detection(self) -> List[Dict]:
        """Alias for get_detections()"""
        return self.get_detections()
    
    # =========================================================================
    # ZONE MANAGEMENT
    # =========================================================================
    
    def check_zone_violations(self) -> List[Dict]:
        """
        Check for objects in restricted/caution zones
        
        Returns:
            List of violation dictionaries
        """
        violations = []
        
        for zone in self.zones:
            # Only check restricted and caution zones
            if zone.type not in ['restricted', 'caution']:
                continue
            
            for obj in self.dynamic_objects:
                if not hasattr(obj, 'position'):
                    continue
                
                # Check if object is in zone
                if self._point_in_polygon(obj.position[:2], zone.polygon):
                    violation = {
                        'object': obj.name,
                        'object_type': obj.__class__.__name__.lower(),
                        'zone': zone.name,
                        'zone_type': zone.type,
                        'position': obj.position.copy(),
                        'timestamp': self.time
                    }
                    violations.append(violation)
                    
                    # Log event
                    self._log_event('zone_violation', violation)
        
        self.stats['total_violations'] += len(violations)
        return violations
    
    def _point_in_polygon(self, point: np.ndarray, polygon: np.ndarray) -> bool:
        """
        Check if point is inside polygon (ray casting algorithm)
        
        Args:
            point: [x, y] coordinates
            polygon: Nx2 array of polygon vertices
            
        Returns:
            True if point is inside polygon
        """
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(n + 1):
            p2x, p2y = polygon[i % n]
            
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            
            p1x, p1y = p2x, p2y
        
        return inside
    
    def get_zone_at_position(self, position: np.ndarray) -> Optional[Zone]:
        """
        Get zone at given position
        
        Args:
            position: [x, y, z] or [x, y] coordinates
            
        Returns:
            Zone object or None
        """
        for zone in self.zones:
            if self._point_in_polygon(position[:2], zone.polygon):
                return zone
        return None
    
    # =========================================================================
    # SPATIAL QUERIES
    # =========================================================================
    
    def get_objects_near(self, position: np.ndarray, radius: float) -> List:
        """
        Get objects within radius of position
        
        Args:
            position: Center position [x, y, z]
            radius: Search radius in meters
            
        Returns:
            List of nearby objects
        """
        nearby = []
        
        for obj in self.dynamic_objects:
            if not hasattr(obj, 'position'):
                continue
            
            dist = np.linalg.norm(obj.position - position)
            if dist <= radius:
                nearby.append(obj)
        
        return nearby
    
    def get_objects_in_zone(self, zone_name: str) -> List:
        """
        Get all objects in a specific zone
        
        Args:
            zone_name: Name of the zone
            
        Returns:
            List of objects in zone
        """
        # Find zone
        zone = None
        for z in self.zones:
            if z.name == zone_name:
                zone = z
                break
        
        if zone is None:
            return []
        
        objects_in_zone = []
        for obj in self.dynamic_objects:
            if not hasattr(obj, 'position'):
                continue
            
            if self._point_in_polygon(obj.position[:2], zone.polygon):
                objects_in_zone.append(obj)
        
        return objects_in_zone
    
    # =========================================================================
    # ASTAS CONTEXT (for AI/ML models)
    # =========================================================================
    
    def get_astas_context(self) -> Dict:
        """
        Get comprehensive context for ASTAS (threat assessment)
        
        Returns:
            Dictionary with all relevant context information
        """
        detections = self.get_detections()
        violations = self.check_zone_violations()
        
        # Determine primary object type
        primary_object = 'unknown'
        if detections:
            primary_object = detections[0]['type']
        
        # Determine zone status
        zone_status = 'safe'
        if violations:
            # Prioritize restricted over caution
            for v in violations:
                if v['zone_type'] == 'restricted':
                    zone_status = 'restricted'
                    break
                elif v['zone_type'] == 'caution':
                    zone_status = 'caution'
        
        # Check for loitering
        loitering = False
        for obj in self.dynamic_objects:
            if hasattr(obj, 'state') and 'LOITERING' in str(obj.state):
                loitering = True
                break
        
        # Check for rapid movement
        rapid_movement = False
        max_speed = 0.0
        for obj in self.dynamic_objects:
            if hasattr(obj, 'velocity'):
                speed = np.linalg.norm(obj.velocity)
                max_speed = max(max_speed, speed)
                if speed > 4.0:  # > 4 m/s = rapid
                    rapid_movement = True
        
        # Determine time of day (simplified - could be enhanced)
        time_of_day = 'day' if 6 <= (self.time % 86400) / 3600 < 18 else 'night'
        
        return {
            # Basic info
            'detections': [d['type'] for d in detections],
            'num_detections': len(detections),
            'primary_object': primary_object,
            
            # Zone info
            'zone': zone_status,
            'restricted_area': zone_status == 'restricted',
            'num_violations': len(violations),
            
            # Motion analysis
            'motion_type': 'rapid' if rapid_movement else 'moderate',
            'speed': max_speed,
            'loitering': loitering,
            'rapid_movement': rapid_movement,
            
            # Environmental
            'time_of_day': time_of_day,
            'time_in_area': self.time,
            
            # Sensor data
            'lidar_objects': len(detections),
            'num_sensors': len(self.sensors),
            
            # Historical
            'previous_alerts': self.stats['total_violations'],
            'total_updates': self.stats['total_updates'],
            
            # Placeholder for advanced features
            'direction_changes': 0,
            'audio_events': [],
            'vibration': False,
            'unusual_pattern': False
        }
    
    # =========================================================================
    # EVENT LOGGING
    # =========================================================================
    
    def _log_event(self, event_type: str, data: Dict):
        """
        Log an event
        
        Args:
            event_type: Type of event
            data: Event data dictionary
        """
        event = {
            'type': event_type,
            'timestamp': self.time,
            'data': data
        }
        
        self.events.append(event)
        
        # Limit event history
        if len(self.events) > self.max_events:
            self.events.pop(0)
    
    def get_recent_events(self, event_type: Optional[str] = None, 
                         count: int = 10) -> List[Dict]:
        """
        Get recent events
        
        Args:
            event_type: Filter by event type (None = all types)
            count: Maximum number of events to return
            
        Returns:
            List of recent events
        """
        filtered_events = self.events
        
        if event_type is not None:
            filtered_events = [e for e in self.events if e['type'] == event_type]
        
        return filtered_events[-count:]
    
    # =========================================================================
    # STATISTICS & INFO
    # =========================================================================
    
    def get_statistics(self) -> Dict:
        """Get world statistics"""
        return {
            'static_objects': len(self.static_objects),
            'dynamic_objects': len(self.dynamic_objects),
            'sensors': len(self.sensors),
            'zones': len(self.zones),
            'simulation_time': self.time,
            'total_updates': self.stats['total_updates'],
            'total_detections': self.stats['total_detections'],
            'total_violations': self.stats['total_violations'],
            'total_events': len(self.events)
        }
    
    def print_summary(self):
        """Print world summary"""
        print("\n" + "=" * 70)
        print("SIMULATION WORLD SUMMARY")
        print("=" * 70)
        
        stats = self.get_statistics()
        
        print(f"Static Objects:  {stats['static_objects']}")
        print(f"Dynamic Objects: {stats['dynamic_objects']}")
        print(f"Sensors:         {stats['sensors']}")
        print(f"Zones:           {stats['zones']}")
        print(f"\nSimulation Time: {stats['simulation_time']:.2f}s")
        print(f"Total Updates:   {stats['total_updates']}")
        print(f"Total Violations: {stats['total_violations']}")
        print(f"Total Events:    {stats['total_events']}")
        print("=" * 70 + "\n")
    
    # =========================================================================
    # SERIALIZATION
    # =========================================================================
    
    def to_dict(self) -> Dict:
        """Serialize world to dictionary"""
        return {
            'time': self.time,
            'dt': self.dt,
            'bounds': {
                'min': self.bounds['min'].tolist(),
                'max': self.bounds['max'].tolist()
            },
            'stats': self.stats.copy(),
            'num_static_objects': len(self.static_objects),
            'num_dynamic_objects': len(self.dynamic_objects),
            'num_sensors': len(self.sensors),
            'num_zones': len(self.zones)
        }
    
    def __repr__(self):
        return (f"SimulationWorld(static={len(self.static_objects)}, "
                f"dynamic={len(self.dynamic_objects)}, "
                f"sensors={len(self.sensors)}, "
                f"zones={len(self.zones)}, "
                f"time={self.time:.2f}s)")


# =============================================================================
# TESTING
# =============================================================================
if __name__ == "__main__":
    print("Testing SimulationWorld...")
    
    # Create world
    world = SimulationWorld()
    print(f"Created: {world}")
    
    # Add zones
    world.add_zone(
        name="restricted_zone",
        zone_type="restricted",
        polygon=[[0, 0], [10, 0], [10, 10], [0, 10]]
    )
    
    world.add_zone(
        name="safe_zone",
        zone_type="safe",
        polygon=[[-10, -10], [0, -10], [0, 0], [-10, 0]]
    )
    
    # Create mock object for testing
    class MockObject:
        def __init__(self, name, pos):
            self.name = name
            self.position = np.array(pos)
            self.velocity = np.array([1.0, 0.0, 0.0])
        
        def update(self, dt):
            self.position += self.velocity * dt
    
    # Add objects
    obj1 = MockObject("obj1", [5, 5, 0])  # In restricted zone
    obj2 = MockObject("obj2", [-5, -5, 0])  # In safe zone
    
    world.add_dynamic_object(obj1)
    world.add_dynamic_object(obj2)
    
    # Update simulation
    for i in range(10):
        world.update(0.1)
    
    # Get detections
    detections = world.get_detections()
    print(f"\nDetections: {len(detections)}")
    
    # Check violations
    violations = world.check_zone_violations()
    print(f"Violations: {len(violations)}")
    
    # Get context
    context = world.get_astas_context()
    print(f"\nASTAS Context:")
    print(f"  Zone: {context['zone']}")
    print(f"  Detections: {context['num_detections']}")
    print(f"  Speed: {context['speed']:.2f} m/s")
    
    # Print summary
    world.print_summary()
    
    print("\n✓ SimulationWorld test complete")


