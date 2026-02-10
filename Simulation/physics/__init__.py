# Physics engine
try:
    from .physics_engine import PhysicsEngine
    PHYSICS_ENGINE_AVAILABLE = True
except ImportError:
    PHYSICS_ENGINE_AVAILABLE = False

# Collision handling
try:
    from .collision_handler import (
        CollisionHandler,
        CollisionType,
        CollisionEvent,
        CollisionFilter,
        CollisionDebugger
    )
    COLLISION_HANDLER_AVAILABLE = True
except ImportError:
    COLLISION_HANDLER_AVAILABLE = False

__all__ = []

# Add available components to __all__
if PHYSICS_ENGINE_AVAILABLE:
    __all__.append('PhysicsEngine')

if COLLISION_HANDLER_AVAILABLE:
    __all__.extend([
        'CollisionHandler',
        'CollisionType',
        'CollisionEvent',
        'CollisionFilter',
        'CollisionDebugger'
    ])

__version__ = '1.0.0'