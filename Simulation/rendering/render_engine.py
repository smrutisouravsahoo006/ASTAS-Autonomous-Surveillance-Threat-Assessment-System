"""
Render Engine - Panda3D Wrapper
Handles all 3D visualization and rendering
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from enum import Enum

try:
    from direct.showbase.ShowBase import ShowBase
    from direct.task import Task
    from panda3d.core import *
    PANDA3D_AVAILABLE = True
except ImportError:
    PANDA3D_AVAILABLE = False
    print("⚠ Panda3D not available - rendering disabled")
    ShowBase = object          # harmless placeholder so class body parses


class RenderMode(Enum):
    """Rendering modes"""
    SOLID = "solid"
    WIREFRAME = "wireframe"
    TEXTURED = "textured"
    DEBUG = "debug"


class RenderEngine(ShowBase):
    """
    3D Rendering Engine using Panda3D
    
    Features:
    - Real-time 3D rendering
    - Camera control
    - Object management
    - Lighting
    - Materials
    - Grid/axes display
    """
    
    def __init__(self, window_title: str = "ASTAS 3D Simulation", 
                 window_size: Tuple[int, int] = (1280, 720),
                 show_fps: bool = True):
        """
        Initialize render engine
        
        Args:
            window_title: Window title
            window_size: (width, height)
            show_fps: Show FPS counter
        """
        if not PANDA3D_AVAILABLE:
            self.available = False
            return
        
        # Initialize Panda3D
        super().__init__()
        
        self.available = True
        
        # Window setup
        props = WindowProperties() # type: ignore
        props.setTitle(window_title)
        props.setSize(window_size[0], window_size[1])
        self.win.requestProperties(props)
        
        # FPS
        if show_fps:
            self.setFrameRateMeter(True)
        
        # Disable default mouse control
        self.disableMouse()
        
        # Node tracking
        self.render_nodes = {}
        self.object_count = 0
        
        # Setup scene
        self._setup_camera()
        self._setup_lighting()
        self._setup_ground()
        
        # Render mode
        self.render_mode = RenderMode.SOLID
        
        print(f"✓ Render engine initialized: {window_size[0]}x{window_size[1]}")
    
    def _setup_camera(self):
        """Setup default camera"""
        # Camera position
        self.camera.setPos(0, -50, 30)
        self.camera.lookAt(0, 0, 0)
        
        # Camera parameters
        self.camera_distance = 50
        self.camera_angle_h = 0
        self.camera_angle_v = 30
        self.camera_target = Point3(0, 0, 0) # type: ignore
    
    def _setup_lighting(self):
        """Setup scene lighting"""
        # Ambient light
        ambient = AmbientLight("ambient") # type: ignore
        ambient.setColor((0.3, 0.3, 0.3, 1))
        self.ambient_light = self.render.attachNewNode(ambient)
        self.render.setLight(self.ambient_light)
        
        # Directional light (sun)
        sun = DirectionalLight("sun") # type: ignore
        sun.setColor((0.8, 0.8, 0.7, 1))
        sun_node = self.render.attachNewNode(sun)
        sun_node.setHpr(45, -60, 0)
        self.render.setLight(sun_node)
        
        # Point light (overhead)
        point = PointLight("point") # type: ignore
        point.setColor((0.5, 0.5, 0.5, 1))
        point_node = self.render.attachNewNode(point)
        point_node.setPos(0, 0, 50)
        self.render.setLight(point_node)
    
    def _setup_ground(self):
        """Setup ground plane with grid"""
        # Ground plane
        cm = CardMaker("ground") # type: ignore
        cm.setFrame(-100, 100, -100, 100)
        ground = self.render.attachNewNode(cm.generate())
        ground.setP(-90)  # Rotate to horizontal
        ground.setPos(0, 0, 0)
        ground.setColor(0.3, 0.5, 0.3, 1)  # Green
        
        # Grid lines
        self._create_grid(size=100, spacing=10)
    
    def _create_grid(self, size: float = 100, spacing: float = 10):
        """Create grid lines"""
        lines = LineSegs() # type: ignore
        lines.setColor(0.4, 0.4, 0.4, 0.5)
        lines.setThickness(1)
        
        # Grid lines
        for i in range(int(-size), int(size) + 1, int(spacing)):
            # Lines parallel to X axis
            lines.moveTo(i, -size, 0.01)
            lines.drawTo(i, size, 0.01)
            
            # Lines parallel to Y axis
            lines.moveTo(-size, i, 0.01)
            lines.drawTo(size, i, 0.01)
        
        grid_node = self.render.attachNewNode(lines.create())
        grid_node.setName("grid")
    
    def create_box(self, name: str, position: np.ndarray, size: np.ndarray,
                   color: Tuple[float, float, float, float] = (0.8, 0.8, 0.8, 1.0)) -> object:
        """
        Create box mesh
        
        Args:
            name: Object name
            position: [x, y, z]
            size: [width, depth, height]
            color: RGBA
            
        Returns:
            NodePath for the object
        """
        if not self.available:
            return None
        
        # Create box geometry
        box = self.loader.loadModel("models/box")
        box.reparentTo(self.render)
        box.setPos(position[0], position[1], position[2] + size[2]/2)
        box.setScale(size[0]/2, size[1]/2, size[2]/2)
        box.setColor(color[0], color[1], color[2], color[3])
        box.setName(name)
        
        self.render_nodes[name] = box
        self.object_count += 1
        
        return box
    
    def create_sphere(self, name: str, position: np.ndarray, radius: float,
                     color: Tuple[float, float, float, float] = (0.8, 0.2, 0.2, 1.0)) -> object:
        """Create sphere mesh"""
        if not self.available:
            return None
        
        sphere = self.loader.loadModel("models/sphere")
        sphere.reparentTo(self.render)
        sphere.setPos(position[0], position[1], position[2])
        sphere.setScale(radius)
        sphere.setColor(color[0], color[1], color[2], color[3])
        sphere.setName(name)
        
        self.render_nodes[name] = sphere
        self.object_count += 1
        
        return sphere
    
    def create_capsule(self, name: str, position: np.ndarray, radius: float, height: float,
                      color: Tuple[float, float, float, float] = (0.2, 0.8, 0.2, 1.0)) -> object:
        """Create capsule mesh (for humans)"""
        if not self.available:
            return None
        
        # Capsule as cylinder + 2 spheres
        capsule = self.render.attachNewNode(name)
        
        # Cylinder body
        cylinder = self.loader.loadModel("models/cylinder")
        cylinder.reparentTo(capsule)
        cylinder.setScale(radius, radius, height/2)
        cylinder.setPos(0, 0, height/2)
        
        # Top sphere
        top = self.loader.loadModel("models/sphere")
        top.reparentTo(capsule)
        top.setScale(radius)
        top.setPos(0, 0, height)
        
        # Bottom sphere
        bottom = self.loader.loadModel("models/sphere")
        bottom.reparentTo(capsule)
        bottom.setScale(radius)
        bottom.setPos(0, 0, 0)
        
        capsule.setPos(position[0], position[1], position[2])
        capsule.setColor(color[0], color[1], color[2], color[3])
        
        self.render_nodes[name] = capsule
        self.object_count += 1
        
        return capsule
    
    def update_position(self, name: str, position: np.ndarray):
        """Update object position"""
        if not self.available or name not in self.render_nodes:
            return
        
        node = self.render_nodes[name]
        node.setPos(position[0], position[1], position[2])
    
    def update_rotation(self, name: str, rotation: np.ndarray):
        """Update object rotation (roll, pitch, yaw in radians)"""
        if not self.available or name not in self.render_nodes:
            return
        
        node = self.render_nodes[name]
        # Convert radians to degrees
        node.setHpr(
            np.degrees(rotation[2]),  # Heading (yaw)
            np.degrees(rotation[1]),  # Pitch
            np.degrees(rotation[0])   # Roll
        )
    
    def remove_object(self, name: str):
        """Remove object from scene"""
        if not self.available or name not in self.render_nodes:
            return
        
        self.render_nodes[name].removeNode()
        del self.render_nodes[name]
        self.object_count -= 1
    
    def set_camera_position(self, position: np.ndarray, target: np.ndarray):
        """Set camera position and look-at target"""
        if not self.available:
            return
        
        self.camera.setPos(position[0], position[1], position[2])
        self.camera.lookAt(target[0], target[1], target[2])
        self.camera_target = Point3(target[0], target[1], target[2]) # type: ignore
    
    def orbit_camera(self, delta_h: float, delta_v: float):
        """Orbit camera around target"""
        if not self.available:
            return
        
        self.camera_angle_h += delta_h
        self.camera_angle_v += delta_v
        
        # Clamp vertical angle
        self.camera_angle_v = np.clip(self.camera_angle_v, 5, 85)
        
        # Calculate new position
        rad_h = np.radians(self.camera_angle_h)
        rad_v = np.radians(self.camera_angle_v)
        
        x = self.camera_target.x + self.camera_distance * np.cos(rad_v) * np.sin(rad_h)
        y = self.camera_target.y - self.camera_distance * np.cos(rad_v) * np.cos(rad_h)
        z = self.camera_target.z + self.camera_distance * np.sin(rad_v)
        
        self.camera.setPos(x, y, z)
        self.camera.lookAt(self.camera_target)
    
    def zoom_camera(self, delta: float):
        """Zoom camera in/out"""
        if not self.available:
            return
        
        self.camera_distance += delta
        self.camera_distance = np.clip(self.camera_distance, 5, 200)
        
        # Update position
        self.orbit_camera(0, 0)
    
    def set_render_mode(self, mode: RenderMode):
        """Set rendering mode"""
        if not self.available:
            return
        
        self.render_mode = mode
        
        if mode == RenderMode.WIREFRAME:
            self.render.setRenderModeWireframe()
        elif mode == RenderMode.SOLID:
            self.render.setRenderModeFilled()
        elif mode == RenderMode.DEBUG:
            self.render.setRenderModeWireframe()
            self.render.showBounds()
    
    def get_screenshot(self, filename: str = "screenshot.png"):
        """Save screenshot"""
        if not self.available:
            return
        
        self.screenshot(namePrefix=filename.replace('.png', ''))
        print(f"Screenshot saved: {filename}")
    
    def get_stats(self) -> Dict:
        """Get rendering statistics"""
        if not self.available:
            return {}
        
        return {
            'objects': self.object_count,
            'nodes': len(self.render_nodes),
            'fps': globalClock.getAverageFrameRate() if self.available else 0 # type: ignore
        }


class SimpleRenderEngine:
    """
    Fallback renderer without Panda3D
    Just tracks objects for headless mode
    """
    
    def __init__(self, **kwargs):
        self.available = False
        self.render_nodes = {}
        self.object_count = 0
        print("⚠ Simple render engine (no visualization)")
    
    def create_box(self, name, position, size, color=(0.8, 0.8, 0.8, 1.0)):
        self.render_nodes[name] = {'type': 'box', 'position': position, 'size': size}
        self.object_count += 1
        return None
    
    def create_sphere(self, name, position, radius, color=(0.8, 0.2, 0.2, 1.0)):
        self.render_nodes[name] = {'type': 'sphere', 'position': position, 'radius': radius}
        self.object_count += 1
        return None
    
    def create_capsule(self, name, position, radius, height, color=(0.2, 0.8, 0.2, 1.0)):
        self.render_nodes[name] = {'type': 'capsule', 'position': position}
        self.object_count += 1
        return None
    
    def update_position(self, name, position):
        if name in self.render_nodes:
            self.render_nodes[name]['position'] = position
    
    def update_rotation(self, name, rotation):
        if name in self.render_nodes:
            self.render_nodes[name]['rotation'] = rotation
    
    def remove_object(self, name):
        if name in self.render_nodes:
            del self.render_nodes[name]
            self.object_count -= 1
    
    def get_stats(self):
        return {'objects': self.object_count, 'fps': 0}


def create_render_engine(**kwargs):
    """Factory function to create render engine"""
    if PANDA3D_AVAILABLE:
        return RenderEngine(**kwargs)
    else:
        return SimpleRenderEngine(**kwargs)


if __name__ == "__main__":
    print("Testing RenderEngine...")
    
    if PANDA3D_AVAILABLE:
        # Create engine
        engine = RenderEngine(window_title="Test", window_size=(800, 600))
        
        # Create test objects
        engine.create_box("box1", np.array([0, 0, 1]), np.array([2, 2, 2]))
        engine.create_sphere("sphere1", np.array([5, 0, 1]), radius=1.0)
        engine.create_capsule("capsule1", np.array([-5, 0, 0]), radius=0.3, height=1.75)
        
        print(f"Stats: {engine.get_stats()}")
        print("✓ RenderEngine test complete")
        print("Close the window to exit...")
        
        # Run
        engine.run()
    else:
        engine = SimpleRenderEngine()
        engine.create_box("box1", np.array([0, 0, 1]), np.array([2, 2, 2]))
        print(f"Stats: {engine.get_stats()}")
        print("✓ SimpleRenderEngine test complete")

