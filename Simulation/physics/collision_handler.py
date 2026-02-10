"""
Collision Handler
Detects and responds to collisions between objects
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from enum import Enum

try:
    import pybullet as p
    PYBULLET_AVAILABLE = True
except:
    PYBULLET_AVAILABLE = False


class CollisionType(Enum):
    """Types of collision responses"""
    NONE = "none"
    BOUNCE = "bounce"
    STOP = "stop"
    TRIGGER = "trigger"
    DESTROY = "destroy"


@dataclass
class CollisionEvent:
    """Represents a collision between two objects"""
    body_a: int
    body_b: int
    position: np.ndarray
    normal: np.ndarray
    impulse: float
    timestamp: float
    
    def __repr__(self):
        return f"Collision({self.body_a} <-> {self.body_b}, impulse={self.impulse:.2f})"


class CollisionHandler:
    """
    Handles collision detection and response
    
    Features:
    - Collision detection (PyBullet or manual)
    - Collision callbacks
    - Collision filtering
    - Event logging
    """
    
    def __init__(self, physics_engine=None):
        """
        Initialize collision handler
        
        Args:
            physics_engine: PhysicsEngine instance (optional)
        """
        self.physics_engine = physics_engine
        self.collision_callbacks = {}
        self.collision_history = []
        self.max_history = 1000
        
        # Collision pairs to ignore
        self.ignore_pairs = set()
        
        # Collision response types
        self.collision_types = {}
        
        print("✓ CollisionHandler initialized")
    
    def register_callback(self, body_id: int, callback: Callable):
        """
        Register callback for when body collides
        
        Args:
            body_id: Body ID to monitor
            callback: Function(collision_event) to call
        """
        self.collision_callbacks[body_id] = callback
    
    def ignore_collision(self, body_a: int, body_b: int):
        """Ignore collisions between two bodies"""
        pair = tuple(sorted([body_a, body_b]))
        self.ignore_pairs.add(pair)
    
    def set_collision_type(self, body_id: int, collision_type: CollisionType):
        """Set collision response type for body"""
        self.collision_types[body_id] = collision_type
    
    def detect_collisions_pybullet(self) -> List[CollisionEvent]:
        """
        Detect collisions using PyBullet
        
        Returns:
            List of collision events
        """
        if not PYBULLET_AVAILABLE or self.physics_engine is None:
            return []
        
        if self.physics_engine.client_id is None:
            return []
        
        collisions = []
        
        # Get all contact points
        contact_points = p.getContactPoints(physicsClientId=self.physics_engine.client_id)
        
        for contact in contact_points:
            body_a = contact[1]
            body_b = contact[2]
            
            # Check if this pair should be ignored
            pair = tuple(sorted([body_a, body_b]))
            if pair in self.ignore_pairs:
                continue
            
            # Extract collision data
            position = np.array(contact[5])  # Contact position on A
            normal = np.array(contact[7])    # Contact normal
            impulse = contact[9]              # Normal impulse
            
            collision = CollisionEvent(
                body_a=body_a,
                body_b=body_b,
                position=position,
                normal=normal,
                impulse=impulse,
                timestamp=0.0  # Would need to track time
            )
            
            collisions.append(collision)
            
            # Add to history
            self._add_to_history(collision)
            
            # Trigger callbacks
            self._trigger_callbacks(collision)
        
        return collisions
    
    def detect_collisions_manual(self, objects: List) -> List[CollisionEvent]:
        """
        Detect collisions using simple bounding box checks
        
        Args:
            objects: List of objects with position and size
            
        Returns:
            List of collision events
        """
        collisions = []
        
        for i, obj_a in enumerate(objects):
            for obj_b in objects[i+1:]:
                if self._check_collision(obj_a, obj_b):
                    # Calculate collision data
                    position = (obj_a.position + obj_b.position) / 2
                    normal = obj_b.position - obj_a.position
                    if np.linalg.norm(normal) > 0:
                        normal = normal / np.linalg.norm(normal)
                    else:
                        normal = np.array([1, 0, 0])
                    
                    collision = CollisionEvent(
                        body_a=id(obj_a),
                        body_b=id(obj_b),
                        position=position,
                        normal=normal,
                        impulse=0.0,
                        timestamp=0.0
                    )
                    
                    collisions.append(collision)
                    self._add_to_history(collision)
        
        return collisions
    
    def _check_collision(self, obj_a, obj_b) -> bool:
        """
        Check if two objects are colliding (AABB)
        
        Args:
            obj_a, obj_b: Objects with position and size/radius
            
        Returns:
            True if colliding
        """
        # Simple sphere collision
        distance = np.linalg.norm(obj_a.position - obj_b.position)
        
        # Get radii (assume objects have size or radius attribute)
        radius_a = getattr(obj_a, 'radius', None) or np.linalg.norm(getattr(obj_a, 'size', np.array([1,1,1]))) / 2
        radius_b = getattr(obj_b, 'radius', None) or np.linalg.norm(getattr(obj_b, 'size', np.array([1,1,1]))) / 2
        
        return distance < (radius_a + radius_b)
    
    def respond_to_collision(self, collision: CollisionEvent):
        """
        Apply collision response
        
        Args:
            collision: Collision event to respond to
        """
        # Get collision types
        type_a = self.collision_types.get(collision.body_a, CollisionType.BOUNCE)
        type_b = self.collision_types.get(collision.body_b, CollisionType.BOUNCE)
        
        if type_a == CollisionType.BOUNCE or type_b == CollisionType.BOUNCE:
            self._bounce_response(collision)
        elif type_a == CollisionType.STOP or type_b == CollisionType.STOP:
            self._stop_response(collision)
        elif type_a == CollisionType.TRIGGER or type_b == CollisionType.TRIGGER:
            self._trigger_response(collision)
    
    def _bounce_response(self, collision: CollisionEvent):
        """Make objects bounce apart"""
        if not PYBULLET_AVAILABLE or self.physics_engine is None:
            return
        
        # Apply impulse in normal direction
        impulse_force = collision.normal * collision.impulse * 10
        
        self.physics_engine.apply_force(collision.body_a, -impulse_force)
        self.physics_engine.apply_force(collision.body_b, impulse_force)
    
    def _stop_response(self, collision: CollisionEvent):
        """Stop objects from moving"""
        if not PYBULLET_AVAILABLE or self.physics_engine is None:
            return
        
        self.physics_engine.set_velocity(collision.body_a, np.zeros(3))
        self.physics_engine.set_velocity(collision.body_b, np.zeros(3))
    
    def _trigger_response(self, collision: CollisionEvent):
        """Trigger event without physical response"""
        # Just log, callbacks already triggered
        pass
    
    def _trigger_callbacks(self, collision: CollisionEvent):
        """Call registered callbacks"""
        if collision.body_a in self.collision_callbacks:
            self.collision_callbacks[collision.body_a](collision)
        
        if collision.body_b in self.collision_callbacks:
            self.collision_callbacks[collision.body_b](collision)
    
    def _add_to_history(self, collision: CollisionEvent):
        """Add collision to history"""
        self.collision_history.append(collision)
        
        # Limit history size
        if len(self.collision_history) > self.max_history:
            self.collision_history.pop(0)
    
    def get_collisions_for_body(self, body_id: int) -> List[CollisionEvent]:
        """Get all collisions involving a specific body"""
        return [c for c in self.collision_history 
                if c.body_a == body_id or c.body_b == body_id]
    
    def clear_history(self):
        """Clear collision history"""
        self.collision_history.clear()
    
    def get_collision_statistics(self) -> Dict:
        """Get statistics about collisions"""
        if not self.collision_history:
            return {
                'total': 0,
                'average_impulse': 0.0,
                'max_impulse': 0.0,
                'unique_bodies': 0
            }
        
        impulses = [c.impulse for c in self.collision_history]
        bodies = set()
        for c in self.collision_history:
            bodies.add(c.body_a)
            bodies.add(c.body_b)
        
        return {
            'total': len(self.collision_history),
            'average_impulse': np.mean(impulses),
            'max_impulse': np.max(impulses),
            'unique_bodies': len(bodies)
        }


class CollisionFilter:
    """
    Filters collisions based on rules
    
    Example:
        filter = CollisionFilter()
        filter.add_rule(lambda c: c.impulse > 10.0)  # Only strong collisions
    """
    
    def __init__(self):
        self.rules = []
    
    def add_rule(self, rule: Callable[[CollisionEvent], bool]):
        """Add filtering rule"""
        self.rules.append(rule)
    
    def filter(self, collisions: List[CollisionEvent]) -> List[CollisionEvent]:
        """Apply all rules to filter collisions"""
        filtered = collisions
        
        for rule in self.rules:
            filtered = [c for c in filtered if rule(c)]
        
        return filtered


class CollisionDebugger:
    """
    Debug collision detection
    
    Features:
    - Visualize collision points
    - Print collision logs
    - Export collision data
    """
    
    def __init__(self, physics_engine=None):
        self.physics_engine = physics_engine
        self.debug_lines = []
    
    def visualize_collision(self, collision: CollisionEvent, duration: float = 1.0):
        """
        Draw collision point and normal in PyBullet
        
        Args:
            collision: Collision to visualize
            duration: How long to show (seconds)
        """
        if not PYBULLET_AVAILABLE or self.physics_engine is None:
            return
        
        if self.physics_engine.client_id is None:
            return
        
        # Draw point
        end_point = collision.position + collision.normal * 0.5
        
        line_id = p.addUserDebugLine(
            collision.position.tolist(),
            end_point.tolist(),
            [1, 0, 0],  # Red
            lineWidth=3,
            lifeTime=duration,
            physicsClientId=self.physics_engine.client_id
        )
        
        self.debug_lines.append(line_id)
    
    def print_collision(self, collision: CollisionEvent):
        """Print collision details"""
        print(f"Collision: {collision.body_a} <-> {collision.body_b}")
        print(f"  Position: {collision.position}")
        print(f"  Normal: {collision.normal}")
        print(f"  Impulse: {collision.impulse:.2f}")
    
    def clear_debug_lines(self):
        """Remove all debug visualizations"""
        if not PYBULLET_AVAILABLE or self.physics_engine is None:
            return
        
        for line_id in self.debug_lines:
            p.removeUserDebugItem(line_id, physicsClientId=self.physics_engine.client_id)
        
        self.debug_lines.clear()


if __name__ == "__main__":
    print("Testing CollisionHandler...")
    
    # Test without PyBullet
    handler = CollisionHandler()
    
    # Test collision callback
    def on_collision(event):
        print(f"Collision detected: {event}")
    
    handler.register_callback(1, on_collision)
    
    # Test collision ignoring
    handler.ignore_collision(1, 2)
    print(f"Ignoring collision pair: (1, 2)")
    
    # Test collision types
    handler.set_collision_type(1, CollisionType.BOUNCE)
    handler.set_collision_type(2, CollisionType.STOP)
    
    print(f"Collision types set")
    
    # Test statistics
    stats = handler.get_collision_statistics()
    print(f"Statistics: {stats}")
    
    # Test filter
    filter = CollisionFilter()
    filter.add_rule(lambda c: c.impulse > 5.0)
    print("Filter created with impulse > 5.0 rule")
    
    print("\n✓ CollisionHandler test complete")
    print("\nTo test with physics:")
    print("  from physics.physics_engine import PhysicsEngine")
    print("  engine = PhysicsEngine()")
    print("  handler = CollisionHandler(engine)")
    print("  collisions = handler.detect_collisions_pybullet()")