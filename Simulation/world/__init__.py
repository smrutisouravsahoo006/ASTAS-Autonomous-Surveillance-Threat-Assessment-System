"""
World Module
Simulation world management with objects, sensors, and zones

Available components:
- SimulationWorld: Complete world manager
- Zone: Security zone definition
"""

try:
    from .world import SimulationWorld, Zone
    WORLD_AVAILABLE = True
except ImportError:
    WORLD_AVAILABLE = False

__all__ = []

if WORLD_AVAILABLE:
    __all__.extend(['SimulationWorld', 'Zone'])

__version__ = '1.0.0'