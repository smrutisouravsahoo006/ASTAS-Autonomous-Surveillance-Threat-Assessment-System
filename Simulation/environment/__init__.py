"""
Environments Module
Pre-built simulation scenarios for testing and development

Available environments:
- Border Patrol: 200m fence line with guards and intruders
- Building Security: Office/warehouse complex with multiple zones
- Parking Lot: Parking facility with theft scenarios
- Custom Environment: Template for creating custom scenarios
"""

# Border patrol
try:
    from .border_patrol import (
        BorderPatrolEnvironment,
        BorderPatrolConfig,
        create_border_patrol_scenario
    )
    BORDER_PATROL_AVAILABLE = True
except ImportError:
    BORDER_PATROL_AVAILABLE = False

# Building security
try:
    from .Building_security import (
        BuildingSecurityEnvironment,
        BuildingSecurityConfig,
        create_building_security_scenario
    )
    BUILDING_SECURITY_AVAILABLE = True
except ImportError:
    BUILDING_SECURITY_AVAILABLE = False



__all__ = []

if BORDER_PATROL_AVAILABLE:
    __all__.extend([
        'BorderPatrolEnvironment',
        'BorderPatrolConfig',
        'create_border_patrol_scenario'
    ])

if BUILDING_SECURITY_AVAILABLE:
    __all__.extend([
        'BuildingSecurityEnvironment',
        'BuildingSecurityConfig',
        'create_building_security_scenario'
    ])


__version__ = '1.0.0'