import numpy as np

try:
    from .base_objects import SimulationObject, ObjectType, Transform
except ImportError:
    from base_objects import SimulationObject, ObjectType, Transform


# =========================================================
# BASE STATIC OBJECT
# =========================================================

class StaticObject(SimulationObject):
    def __init__(self, name, position, color=None, **kwargs):
        transform = Transform(
            position=np.array(position, dtype=float),
            rotation=np.zeros(3, dtype=float),
        )
        super().__init__(name, ObjectType.STATIC, transform)
        self.color             = color
        self.render_properties = kwargs

    @property
    def position(self) -> np.ndarray:
        return self.transform.position

    @position.setter
    def position(self, value):
        self.transform.position = np.array(value, dtype=float)


# =========================================================
# BUILDING
# =========================================================

class Building(StaticObject):
    def __init__(self, name, position, size, color=None, **kwargs):
        super().__init__(name, position, color=color, **kwargs)
        self.size = np.array(size, dtype=float)

    def get_corners(self):
        x, y, z  = self.transform.position
        dx, dy, dz = self.size / 2
        return [
            [x - dx, y - dy],
            [x + dx, y - dy],
            [x + dx, y + dy],
            [x - dx, y + dy],
        ]


# =========================================================
# WALL
# =========================================================

class Wall(StaticObject):
    def __init__(self, name, start, end, height=3.0, thickness=0.2, color=None, **kwargs):
        start    = np.array(start, dtype=float)
        end      = np.array(end,   dtype=float)
        position = (start + end) / 2
        if color is None:
            color = (0.65, 0.55, 0.40, 1.0)   # default tan/brown
        super().__init__(name, position, color=color, **kwargs)
        self.start     = start
        self.end       = end
        self.height    = float(height)
        self.thickness = float(thickness)

    def get_length(self):
        return float(np.linalg.norm(self.end - self.start))

    @property
    def length(self):
        """Direct attribute access: wall.length (used by simulation_3d_launch)."""
        return self.get_length()


# =========================================================
# OBSTACLE
# =========================================================

class Obstacle(StaticObject):
    def __init__(self, name, position, size, color=None, **kwargs):
        super().__init__(name, position, color=color, **kwargs)
        self.size = np.array(size, dtype=float)


# =========================================================
# TREE
# =========================================================

class Tree(StaticObject):
    def __init__(self, name, position, height=5.0, trunk_radius=0.3, color=None, **kwargs):
        if color is None:
            color = (0.20, 0.55, 0.20, 1.0)   # default green
        super().__init__(name, position, color=color, **kwargs)
        self.height        = float(height)
        self.trunk_radius  = float(trunk_radius)
        self.canopy_radius = float(trunk_radius) * 3
        # Alias used by simulation_3d_launch: obj.radius
        self.radius        = float(trunk_radius)


# =========================================================
# GROUND
# =========================================================

class Ground(StaticObject):
    def __init__(self):
        super().__init__("ground", [0.0, 0.0, 0.0])