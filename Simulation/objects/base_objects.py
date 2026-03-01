from enum import Enum
from dataclasses import dataclass
import numpy as np


class ObjectType(Enum):
    STATIC = "static"
    DYNAMIC = "dynamic"
    SENSOR = "sensor"


@dataclass
class Transform:
    position: np.ndarray
    rotation: np.ndarray


class SimulationObject:
    def __init__(self, name: str, object_type: ObjectType, transform: Transform):
        self.name = name
        self.object_type = object_type
        self.transform = transform
        self.size = np.array([1.0, 1.0, 1.0])

    def update(self, dt: float):
        pass

    def get_position(self):
        return self.transform.position

    def get_bounding_box(self):
        min_corner = self.transform.position - self.size / 2
        max_corner = self.transform.position + self.size / 2
        return min_corner, max_corner