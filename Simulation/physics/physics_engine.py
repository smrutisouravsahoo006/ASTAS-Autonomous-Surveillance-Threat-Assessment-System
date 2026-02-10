import numpy as np
from typing import Dict, List, Optional

try:
    import pybullet as p
    import pybullet_data
    PYBULLET_AVAILABLE = True
except ImportError:
    PYBULLET_AVAILABLE = False
    print("⚠ PyBullet not available - physics disabled")


class PhysicsEngine:
    def __init__(self, gui: bool = False, gravity: float = -9.81):
        if not PYBULLET_AVAILABLE:
            self.client_id = None
            return

        self.client_id = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, gravity, physicsClientId=self.client_id)
        p.setTimeStep(1 / 240, physicsClientId=self.client_id)

        # ✅ FIXED: loadURDF (not loadUDRF)
        self.ground_id = p.loadURDF("plane.urdf", physicsClientId=self.client_id)

        print(f"✓ Physics engine initialized (client_id={self.client_id})")

    def step(self):
        if self.client_id is not None:
            p.stepSimulation(physicsClientId=self.client_id)

    def create_box(
        self,
        position: np.ndarray,
        size: np.ndarray,
        mass: float = 1.0,
        color: tuple = (0.8, 0.8, 0.8, 1.0),
    ):
        if self.client_id is None:
            return None

        half_extents = (size / 2).tolist()

        collision_id = p.createCollisionShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            physicsClientId=self.client_id,
        )

        visual_id = p.createVisualShape(
            p.GEOM_BOX,
            halfExtents=half_extents,
            rgbaColor=color,
            physicsClientId=self.client_id,
        )

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position.tolist(),
            physicsClientId=self.client_id,
        )

        return body_id

    def create_sphere(
        self,
        position: np.ndarray,
        radius: float,
        mass: float = 1.0,
        color: tuple = (0.8, 0.2, 0.2, 1.0),
    ):
        if self.client_id is None:
            return None

        collision_id = p.createCollisionShape(
            p.GEOM_SPHERE,
            radius=radius,
            physicsClientId=self.client_id,
        )

        visual_id = p.createVisualShape(
            p.GEOM_SPHERE,
            radius=radius,
            rgbaColor=color,
            physicsClientId=self.client_id,
        )

        body_id = p.createMultiBody(
            baseMass=mass,
            baseCollisionShapeIndex=collision_id,
            baseVisualShapeIndex=visual_id,
            basePosition=position.tolist(),
            physicsClientId=self.client_id,
        )

        return body_id

    def get_position(self, body_id: int) -> np.ndarray:
        if self.client_id is None or body_id is None:
            return np.zeros(3)

        pos, _ = p.getBasePositionAndOrientation(
            body_id, physicsClientId=self.client_id
        )
        return np.array(pos)

    def get_velocity(self, body_id: int) -> np.ndarray:
        if self.client_id is None or body_id is None:
            return np.zeros(3)

        vel, _ = p.getBaseVelocity(body_id, physicsClientId=self.client_id)
        return np.array(vel)

    def set_velocity(self, body_id: int, velocity: np.ndarray):
        if self.client_id is not None and body_id is not None:
            p.resetBaseVelocity(
                body_id,
                linearVelocity=velocity.tolist(),
                physicsClientId=self.client_id,
            )

    def apply_force(self, body_id: int, force: np.ndarray):
        if self.client_id is not None and body_id is not None:
            p.applyExternalForce(
                body_id,
                -1,
                force.tolist(),
                [0, 0, 0],
                p.WORLD_FRAME,
                physicsClientId=self.client_id,
            )

    def disconnect(self):
        if self.client_id is not None:
            p.disconnect(physicsClientId=self.client_id)
            self.client_id = None


if __name__ == "__main__":
    print("Testing PhysicsEngine...")

    if PYBULLET_AVAILABLE:
        engine = PhysicsEngine(gui=False)

        box_id = engine.create_box(
            np.array([0, 0, 5]), np.array([1, 1, 1])
        )
        sphere_id = engine.create_sphere(
            np.array([2, 0, 5]), radius=0.5
        )

        for _ in range(240):
            engine.step()

        print("Box position:", engine.get_position(box_id))
        print("Sphere position:", engine.get_position(sphere_id))

        engine.disconnect()
        print("✓ PhysicsEngine OK")
    else:
        print("✓ PhysicsEngine module OK (PyBullet not installed)")
