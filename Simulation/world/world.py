import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Any


# =========================================================
# ZONE DEFINITIONS
# =========================================================

ZONE_COLORS = {
    "restricted": (1.0, 0.2, 0.2, 1.0),
    "caution":    (1.0, 0.8, 0.0, 1.0),
    "safe":       (0.2, 1.0, 0.2, 1.0),
}

class Zone:
    def __init__(self, name: str, zone_type: str, polygon: List[List[float]]):
        self.name      = name
        self.zone_type = zone_type
        self.polygon   = polygon
        self.color     = ZONE_COLORS.get(zone_type, (0.5, 0.5, 0.5, 1.0))


@dataclass
class ZoneViolation:
    object_name: str
    zone_name:   str
    zone_type:   str
    position:    np.ndarray
    timestamp:   float

    def __getitem__(self, key: str):
        """Dict-style access for backward compatibility with main.py."""
        _MAP = {
            'object':    self.object_name,
            'zone':      self.zone_name,
            'zone_type': self.zone_type,
            'position':  self.position,
            'timestamp': self.timestamp,
        }
        return _MAP[key]

    def get(self, key: str, default=None):
        try:
            return self[key]
        except KeyError:
            return default


# =========================================================
# SIMULATION WORLD
# =========================================================

class SimulationWorld:
    def __init__(self):
        self.dynamic_objects: List = []
        self.static_objects:  List = []
        self.sensors:         List = []
        self.zones:           List[Zone] = []
        self.simulation_time: float = 0.0
        self._alert_count:    int   = 0

    # ----------------------------------------------------------
    # ADD METHODS
    # ----------------------------------------------------------

    def add_dynamic_object(self, obj):
        self.dynamic_objects.append(obj)

    def add_static_object(self, obj):
        self.static_objects.append(obj)

    def add_sensor(self, sensor):
        self.sensors.append(sensor)

    def add_zone(self, name: str, zone_type: str, polygon):
        zone = Zone(name, zone_type, polygon)
        self.zones.append(zone)
        return zone

    # ----------------------------------------------------------
    # UPDATE LOOP
    # ----------------------------------------------------------

    def update(self, dt: float):
        self.simulation_time += dt
        for obj in self.dynamic_objects:
            obj.update(dt)

    # ----------------------------------------------------------
    # DETECTION
    # ----------------------------------------------------------

    def get_detections(self) -> List[Dict[str, Any]]:
        """
        Returns list of dicts with keys: name, position, state
        (format expected by main.py)
        """
        detections = []
        for obj in self.dynamic_objects:
            pos   = obj.get_position() if hasattr(obj, 'get_position') else getattr(obj, 'position', np.zeros(3))
            state = getattr(obj, 'state', None)
            detections.append({
                'name':     obj.name,
                'position': pos,
                'state':    state.name if hasattr(state, 'name') else str(state),
            })
        return detections

    # Legacy alias (kept for compatibility)
    def get_detection(self):
        return [obj.name for obj in self.dynamic_objects]

    # ----------------------------------------------------------
    # ZONE VIOLATIONS
    # ----------------------------------------------------------

    def check_zone_violations(self):
        """
        Returns list of ZoneViolation dataclass objects.
        Also supports dict-style access via main.py: v['object'], v['zone'], v['zone_type']
        Each ZoneViolation has: .object_name, .zone_name, .zone_type, .position, .timestamp
        """
        violations = []
        for obj in self.dynamic_objects:
            pos = (obj.get_position() if hasattr(obj, 'get_position')
                   else getattr(obj, 'position', np.zeros(3)))

            for zone in self.zones:
                if zone.zone_type in ("restricted", "caution"):
                    if self._point_in_polygon(pos[:2], zone.polygon):
                        v = ZoneViolation(
                            object_name = obj.name,
                            zone_name   = zone.name,
                            zone_type   = zone.zone_type,
                            position    = pos.copy(),
                            timestamp   = self.simulation_time,
                        )
                        violations.append(v)
                        self._alert_count += 1
        return violations

    # ----------------------------------------------------------
    # ASTAS CONTEXT SNAPSHOT
    # ----------------------------------------------------------

    def get_astas_context(self) -> Dict[str, Any]:
        """
        Returns rich context dict.  Keys accessed by main.py:
            zone, num_detections, loitering, rapid_movement, speed, time_of_day
        """
        detections     = self.get_detections()
        num_det        = len(detections)
        loitering      = False
        rapid_movement = False
        max_speed      = 0.0

        for obj in self.dynamic_objects:
            state = getattr(obj, 'state', None)
            if state is not None:
                sname = state.name if hasattr(state, 'name') else str(state)
                if sname.upper() in ('LOITERING',):
                    loitering = True
                if sname.upper() in ('RUNNING',):
                    rapid_movement = True
            vel = getattr(obj, 'velocity', None)
            if vel is not None:
                spd = float(np.linalg.norm(vel))
                if spd > max_speed:
                    max_speed = spd

        violations = self.check_zone_violations()
        zone = violations[0].zone_type if violations else 'safe'

        hour = int((self.simulation_time % 86400) / 3600)
        if 6 <= hour < 20:
            time_of_day = 'day'
        elif 20 <= hour < 22 or 4 <= hour < 6:
            time_of_day = 'dusk' if (20 <= hour < 22) else 'dawn'
        else:
            time_of_day = 'night'

        return {
            'simulation_time':  self.simulation_time,
            'zone':             zone,
            'num_detections':   num_det,
            'loitering':        loitering,
            'rapid_movement':   rapid_movement,
            'speed':            f"{max_speed:.1f} m/s",
            'time_of_day':      time_of_day,
            'detections':       [d['name'] for d in detections],
            'restricted_area':  any(v.zone_type == 'restricted' for v in violations),
            'dynamic_objects':  len(self.dynamic_objects),
            'static_objects':   len(self.static_objects),
            'zones':            len(self.zones),
            'sensors':          len(self.sensors),
            'previous_alerts':  self._alert_count,
        }

    # ----------------------------------------------------------
    # PRINT SUMMARY
    # ----------------------------------------------------------

    def summary(self) -> dict:
        """Return world stats as a dict (used by simulation_3d_launch)."""
        return {
            'simulation_time': self.simulation_time,
            'dynamic_objects': len(self.dynamic_objects),
            'static_objects':  len(self.static_objects),
            'zones':           len(self.zones),
            'sensors':         len(self.sensors),
            'total_alerts':    self._alert_count,
        }

    def print_summary(self):
        s = self.summary()
        print(f"\n  ── World Summary ──────────────────────────────────")
        print(f"  Simulation time : {s['simulation_time']:.2f}s")
        print(f"  Dynamic objects : {s['dynamic_objects']}")
        print(f"  Static objects  : {s['static_objects']}")
        print(f"  Zones           : {s['zones']}")
        print(f"  Sensors         : {s['sensors']}")
        print(f"  Total alerts    : {s['total_alerts']}")

    # ----------------------------------------------------------
    # INTERNAL GEOMETRY
    # ----------------------------------------------------------

    def _point_in_polygon(self, point, polygon) -> bool:
        """Ray-casting point-in-polygon."""
        x, y   = float(point[0]), float(point[1])
        inside = False
        j      = len(polygon) - 1
        for i in range(len(polygon)):
            xi, yi = float(polygon[i][0]), float(polygon[i][1])
            xj, yj = float(polygon[j][0]), float(polygon[j][1])
            if ((yi > y) != (yj > y)) and \
               (x < (xj - xi) * (y - yi) / (yj - yi + 1e-9) + xi):
                inside = not inside
            j = i
        return inside