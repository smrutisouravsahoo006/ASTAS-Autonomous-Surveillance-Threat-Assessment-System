from .base_objects import SimulationObject, Transform, ObjectType

# Import objects based on what's available
try:
    from .human import Human, HumanState, HumanGroup
    HUMAN_AVAILABLE = True
except ImportError:
    HUMAN_AVAILABLE = False

try:
    from .vehicle import Vehicle, VehicleType
    VEHICLE_AVAILABLE = True
except ImportError:
    VEHICLE_AVAILABLE = False

try:
    from .drone import Drone
    DRONE_AVAILABLE = True
except ImportError:
    DRONE_AVAILABLE = False

try:
    from .Buildings import Building, Wall
    BUILDING_AVAILABLE = True
except ImportError:
    BUILDING_AVAILABLE = False

# Lightweight versions
try:
    from .dynamic_objects import Person, Vehicle as LightVehicle, Drone as LightDrone
    DYNAMIC_AVAILABLE = True
except ImportError:
    DYNAMIC_AVAILABLE = False

try:
    from .static_objects import Building as SimpleBuilding, Wall as SimpleWall, Ground
    STATIC_AVAILABLE = True
except ImportError:
    STATIC_AVAILABLE = False

__all__ = [
    # Base
    'SimulationObject',
    'Transform',
    'ObjectType',
]

# Add available objects to __all__
if HUMAN_AVAILABLE:
    __all__.extend(['Human', 'HumanState', 'HumanGroup'])

if VEHICLE_AVAILABLE:
    __all__.extend(['Vehicle', 'VehicleType'])

if DRONE_AVAILABLE:
    __all__.append('Drone')

if BUILDING_AVAILABLE:
    __all__.extend(['Building', 'Wall'])

if DYNAMIC_AVAILABLE:
    __all__.extend(['Person', 'LightVehicle', 'LightDrone'])

if STATIC_AVAILABLE:
    __all__.extend(['SimpleBuilding', 'SimpleWall', 'Ground'])

__version__ = '1.0.0'