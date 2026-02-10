"""
Camera Controller
Handles camera movement and controls for 3D scene
"""

import numpy as np
from typing import Tuple, Optional
from enum import Enum

try:
    from panda3d.core import Point3, Vec3
    PANDA3D_AVAILABLE = True
except:
    PANDA3D_AVAILABLE = False
    Point3 = None
    Vec3 = None


class CameraMode(Enum):
    """Camera control modes"""
    FREE = "free"           # Free flight
    ORBIT = "orbit"         # Orbit around target
    FOLLOW = "follow"       # Follow object
    FIXED = "fixed"         # Fixed position


class CameraController:
    """
    Camera controller for 3D scene
    
    Features:
    - Multiple camera modes
    - Smooth movement
    - Mouse/keyboard control
    - Zoom in/out
    - Pan
    - Look at target
    """
    
    def __init__(self, camera, initial_position: np.ndarray = None,
                 initial_target: np.ndarray = None):
        """
        Initialize camera controller
        
        Args:
            camera: Panda3D camera node
            initial_position: Starting camera position
            initial_target: Starting look-at point
        """
        self.camera = camera
        
        # Position and target - FIXED: explicit dtype
        if initial_position is None:
            initial_position = np.array([0.0, -50.0, 30.0], dtype=float)
        else:
            initial_position = np.array(initial_position, dtype=float)
            
        if initial_target is None:
            initial_target = np.array([0.0, 0.0, 0.0], dtype=float)
        else:
            initial_target = np.array(initial_target, dtype=float)
        
        self.position = initial_position
        self.target = initial_target
        
        # Orbit parameters
        self.distance = np.linalg.norm(self.position - self.target)
        
        direction = self.position - self.target
        self.angle_h = np.degrees(np.arctan2(direction[1], direction[0]))  # Horizontal angle
        self.angle_v = np.degrees(np.arcsin(direction[2] / self.distance))  # Vertical angle
        
        # Movement parameters
        self.move_speed = 0.5
        self.rotate_speed = 0.5
        self.zoom_speed = 2.0
        self.smooth_factor = 0.1
        
        # Target for smooth movement
        self.target_position = self.position.copy()
        self.target_look_at = self.target.copy()
        
        # Mode
        self.mode = CameraMode.ORBIT
        
        # Follow target
        self.follow_object = None
        self.follow_offset = np.array([0.0, -10.0, 5.0], dtype=float)  # FIXED
        
        # Constraints
        self.min_distance = 2.0
        self.max_distance = 200.0
        self.min_angle_v = -85
        self.max_angle_v = 85
        
        # Apply initial position
        self._update_camera()
    
    def set_mode(self, mode: CameraMode):
        """Set camera control mode"""
        self.mode = mode
        print(f"Camera mode: {mode.value}")
    
    def set_position(self, position: np.ndarray):
        """Set camera position directly"""
        self.position = np.array(position, dtype=float)  # FIXED
        self._update_camera()
    
    def set_target(self, target: np.ndarray):
        """Set look-at target"""
        self.target = np.array(target, dtype=float)  # FIXED
        self.distance = np.linalg.norm(self.position - self.target)
        self._update_camera()
    
    def look_at(self, target: np.ndarray):
        """Point camera at target"""
        self.target = np.array(target, dtype=float)  # FIXED
        if PANDA3D_AVAILABLE and self.camera is not None:
            self.camera.lookAt(target[0], target[1], target[2])
    
    def orbit(self, delta_h: float, delta_v: float):
        """
        Orbit camera around target
        
        Args:
            delta_h: Horizontal angle change (degrees)
            delta_v: Vertical angle change (degrees)
        """
        if self.mode != CameraMode.ORBIT:
            return
        
        self.angle_h += delta_h * self.rotate_speed
        self.angle_v += delta_v * self.rotate_speed
        
        # Clamp vertical angle
        self.angle_v = np.clip(self.angle_v, self.min_angle_v, self.max_angle_v)
        
        # Calculate new position
        self._calculate_orbit_position()
        self._update_camera()
    
    def zoom(self, delta: float):
        """
        Zoom camera in/out
        
        Args:
            delta: Distance change (positive = zoom out)
        """
        self.distance += delta * self.zoom_speed
        self.distance = np.clip(self.distance, self.min_distance, self.max_distance)
        
        if self.mode == CameraMode.ORBIT:
            self._calculate_orbit_position()
            self._update_camera()
    
    def pan(self, delta_x: float, delta_y: float):
        """
        Pan camera (move target and position together)
        
        Args:
            delta_x: Horizontal movement
            delta_y: Vertical movement
        """
        # Calculate camera right and up vectors
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=float))  # FIXED
        right = right / np.linalg.norm(right)
        
        up = np.cross(right, forward)
        
        # Move both position and target - FIXED: assignment instead of +=
        movement = right * delta_x * self.move_speed + up * delta_y * self.move_speed
        
        self.position = self.position + movement
        self.target = self.target + movement
        
        self._update_camera()
    
    def move_forward(self, amount: float):
        """Move camera forward"""
        forward = self.target - self.position
        forward = forward / np.linalg.norm(forward)
        
        # FIXED: assignment instead of +=
        self.position = self.position + forward * amount * self.move_speed
        self.target = self.target + forward * amount * self.move_speed
        
        self._update_camera()
    
    def follow_object_position(self, object_position: np.ndarray):
        """
        Follow an object
        
        Args:
            object_position: Position of object to follow
        """
        if self.mode != CameraMode.FOLLOW:
            return
        
        object_position = np.array(object_position, dtype=float)  # FIXED
        
        # Calculate desired position
        desired_position = object_position + self.follow_offset
        
        # Smooth follow - FIXED: assignment
        self.position = self.position + (desired_position - self.position) * self.smooth_factor
        self.target = object_position
        
        self._update_camera()
    
    def update(self, dt: float):
        """
        Update camera (smooth movement)
        
        Args:
            dt: Delta time
        """
        # Smooth position transition
        if not np.allclose(self.position, self.target_position):
            # FIXED: assignment
            self.position = self.position + (self.target_position - self.position) * self.smooth_factor
            self._update_camera()
    
    def _calculate_orbit_position(self):
        """Calculate position for orbit mode"""
        rad_h = np.radians(self.angle_h)
        rad_v = np.radians(self.angle_v)
        
        x = self.target[0] + self.distance * np.cos(rad_v) * np.cos(rad_h)
        y = self.target[1] + self.distance * np.cos(rad_v) * np.sin(rad_h)
        z = self.target[2] + self.distance * np.sin(rad_v)
        
        self.position = np.array([x, y, z], dtype=float)  # FIXED
    
    def _update_camera(self):
        """Apply position to camera"""
        if self.camera is None or not PANDA3D_AVAILABLE:
            return
        
        self.camera.setPos(self.position[0], self.position[1], self.position[2])
        self.camera.lookAt(self.target[0], self.target[1], self.target[2])
    
    def get_forward_vector(self) -> np.ndarray:
        """Get camera forward direction"""
        forward = self.target - self.position
        return forward / np.linalg.norm(forward)
    
    def get_right_vector(self) -> np.ndarray:
        """Get camera right direction"""
        forward = self.get_forward_vector()
        right = np.cross(forward, np.array([0.0, 0.0, 1.0], dtype=float))  # FIXED
        return right / np.linalg.norm(right)
    
    def get_up_vector(self) -> np.ndarray:
        """Get camera up direction"""
        forward = self.get_forward_vector()
        right = self.get_right_vector()
        return np.cross(right, forward)
    
    def screen_to_world(self, screen_x: float, screen_y: float,
                        screen_width: int, screen_height: int) -> np.ndarray:
        """
        Convert screen coordinates to world ray
        
        Args:
            screen_x, screen_y: Screen position (pixels)
            screen_width, screen_height: Screen dimensions
            
        Returns:
            Ray direction in world space
        """
        # Normalize screen coordinates to [-1, 1]
        ndc_x = (2.0 * screen_x) / screen_width - 1.0
        ndc_y = 1.0 - (2.0 * screen_y) / screen_height
        
        # Calculate ray direction
        forward = self.get_forward_vector()
        right = self.get_right_vector()
        up = self.get_up_vector()
        
        # Assume 60° FOV
        aspect = screen_width / screen_height
        fov_rad = np.radians(60)
        
        ray = forward + right * ndc_x * np.tan(fov_rad / 2) * aspect + up * ndc_y * np.tan(fov_rad / 2)
        return ray / np.linalg.norm(ray)
    
    def get_state(self) -> dict:
        """Get camera state"""
        return {
            'position': self.position.tolist(),
            'target': self.target.tolist(),
            'distance': self.distance,
            'angle_h': self.angle_h,
            'angle_v': self.angle_v,
            'mode': self.mode.value
        }
    
    def set_state(self, state: dict):
        """Restore camera state"""
        self.position = np.array(state['position'], dtype=float)  # FIXED
        self.target = np.array(state['target'], dtype=float)  # FIXED
        self.distance = state['distance']
        self.angle_h = state['angle_h']
        self.angle_v = state['angle_v']
        self.mode = CameraMode(state['mode'])
        self._update_camera()


class KeyboardMouseController:
    """
    Keyboard and mouse input handler for camera
    
    Usage with Panda3D:
        controller = KeyboardMouseController(camera_controller, base)
    """
    
    def __init__(self, camera_controller: CameraController, base):
        """
        Initialize input controller
        
        Args:
            camera_controller: CameraController instance
            base: Panda3D ShowBase instance
        """
        self.camera_controller = camera_controller
        self.base = base
        
        # Mouse state
        self.mouse_pressed = False
        self.last_mouse_x = 0
        self.last_mouse_y = 0
        
        # Setup input
        self._setup_keyboard()
        self._setup_mouse()
    
    def _setup_keyboard(self):
        """Setup keyboard controls"""
        # WASD movement
        self.base.accept("w", self._on_key_w)
        self.base.accept("s", self._on_key_s)
        self.base.accept("a", self._on_key_a)
        self.base.accept("d", self._on_key_d)
        
        # QE for up/down
        self.base.accept("q", self._on_key_q)
        self.base.accept("e", self._on_key_e)
        
        # Arrow keys for orbit
        self.base.accept("arrow_left", lambda: self.camera_controller.orbit(-5, 0))
        self.base.accept("arrow_right", lambda: self.camera_controller.orbit(5, 0))
        self.base.accept("arrow_up", lambda: self.camera_controller.orbit(0, 5))
        self.base.accept("arrow_down", lambda: self.camera_controller.orbit(0, -5))
        
        # Number keys for camera modes
        self.base.accept("1", lambda: self.camera_controller.set_mode(CameraMode.FREE))
        self.base.accept("2", lambda: self.camera_controller.set_mode(CameraMode.ORBIT))
        self.base.accept("3", lambda: self.camera_controller.set_mode(CameraMode.FOLLOW))
    
    def _setup_mouse(self):
        """Setup mouse controls"""
        # Mouse buttons
        self.base.accept("mouse1", self._on_mouse_down)
        self.base.accept("mouse1-up", self._on_mouse_up)
        
        # Mouse wheel for zoom
        self.base.accept("wheel_up", lambda: self.camera_controller.zoom(-2))
        self.base.accept("wheel_down", lambda: self.camera_controller.zoom(2))
    
    def _on_key_w(self):
        self.camera_controller.move_forward(1.0)
    
    def _on_key_s(self):
        self.camera_controller.move_forward(-1.0)
    
    def _on_key_a(self):
        self.camera_controller.pan(-1.0, 0)
    
    def _on_key_d(self):
        self.camera_controller.pan(1.0, 0)
    
    def _on_key_q(self):
        self.camera_controller.pan(0, -1.0)
    
    def _on_key_e(self):
        self.camera_controller.pan(0, 1.0)
    
    def _on_mouse_down(self):
        self.mouse_pressed = True
        if self.base.mouseWatcherNode.hasMouse():
            self.last_mouse_x = self.base.mouseWatcherNode.getMouseX()
            self.last_mouse_y = self.base.mouseWatcherNode.getMouseY()
    
    def _on_mouse_up(self):
        self.mouse_pressed = False
    
    def update(self):
        """Update mouse drag (call every frame)"""
        if self.mouse_pressed and self.base.mouseWatcherNode.hasMouse():
            mouse_x = self.base.mouseWatcherNode.getMouseX()
            mouse_y = self.base.mouseWatcherNode.getMouseY()
            
            delta_x = (mouse_x - self.last_mouse_x) * 100
            delta_y = (mouse_y - self.last_mouse_y) * 100
            
            self.camera_controller.orbit(delta_x, delta_y)
            
            self.last_mouse_x = mouse_x
            self.last_mouse_y = mouse_y


if __name__ == "__main__":
    print("Testing CameraController...")
    
    # Create mock camera
    class MockCamera:
        def setPos(self, x, y, z):
            print(f"Camera pos: ({x:.1f}, {y:.1f}, {z:.1f})")
        
        def lookAt(self, x, y, z):
            print(f"Looking at: ({x:.1f}, {y:.1f}, {z:.1f})")
    
    camera = MockCamera()
    controller = CameraController(camera)
    
    print(f"Position dtype: {controller.position.dtype}")
    print(f"Target dtype: {controller.target.dtype}")
    
    # Test orbit
    print("\nTest orbit:")
    controller.orbit(45, 10)
    
    # Test zoom
    print("\nTest zoom:")
    controller.zoom(10)
    
    # Test pan
    print("\nTest pan:")
    controller.pan(5, 0)
    
    # Test state save/load
    print("\nTest state:")
    state = controller.get_state()
    print(f"State: {state}")
    
    print("\n✓ CameraController test complete")