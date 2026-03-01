import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
import time


@dataclass
class ControlOutput:
    value: float
    timestamp: float
    setpoint: float
    error: float


class PIDcontroller:
    def __init__(self, kp: float = 1.0, ki: float = 0.1, kd: float = 0.05,
                 output_limits: Tuple[float, float] = (-10.0, 10.0),
                 sample_time: float = 0.01):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.output_limits = output_limits
        self.sample_time = sample_time
        self.setpoint = 0.0
        self.last_error = 0.0
        self.integral = 0.0
        self.last_time = None
        self.integral_limit = 100.0

    def compute(self, measurement: float, setpoint: Optional[float] = None) -> ControlOutput:
        current_time = time.time()
        if setpoint is not None:
            self.setpoint = setpoint
        error = self.setpoint - measurement
        if self.last_time is None:
            dt = self.sample_time
        else:
            dt = current_time - self.last_time
            if dt < 1e-6:
                dt = self.sample_time
        p_term = self.kp * error
        self.integral += error * dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        i_term = self.ki * self.integral
        if dt > 0:
            derivative = (error - self.last_error) / dt
        else:
            derivative = 0.0
        d_term = self.kd * derivative
        output = p_term + i_term + d_term
        output = np.clip(output, self.output_limits[0], self.output_limits[1])
        self.last_error = error
        self.last_time = current_time
        return ControlOutput(
            value=float(output),
            timestamp=current_time,
            setpoint=self.setpoint,
            error=error
        )

    def reset(self):
        self.integral = 0.0
        self.last_error = 0.0
        self.last_time = None

    def set_gains(self, kp: Optional[float] = None, ki: Optional[float] = None,
                  kd: Optional[float] = None):
        if kp is not None:
            self.kp = kp
        if ki is not None:
            self.ki = ki
        if kd is not None:
            self.kd = kd


class CameraController:
    def __init__(self, max_rotation_speed: float = 30.0, smooth_factor: float = 0.8):
        self.pan_controller = PIDcontroller(
            kp=1.5, ki=0.1, kd=0.3,
            output_limits=(-max_rotation_speed, max_rotation_speed)
        )
        self.tilt_controller = PIDcontroller(
            kp=1.5, ki=0.1, kd=0.3,
            output_limits=(-max_rotation_speed, max_rotation_speed)
        )
        self.smooth_factor = smooth_factor
        self.current_pan = 0.0
        self.current_tilt = 0.0
        self.smoothed_pan = 0.0
        self.smoothed_tilt = 0.0

    def track_target(self, target_position: Tuple[float, float],
                     frame_center: Tuple[float, float],
                     frame_size: Tuple[int, int]) -> Tuple[float, float]:
        error_x = target_position[0] - frame_center[0]
        error_y = target_position[1] - frame_center[1]
        normalized_error_x = error_x / (frame_size[0] / 2)
        normalized_error_y = error_y / (frame_size[1] / 2)
        self.smoothed_pan = (self.smooth_factor * self.smoothed_pan +
                             (1 - self.smooth_factor) * normalized_error_x)
        self.smoothed_tilt = (self.smooth_factor * self.smoothed_tilt +
                              (1 - self.smooth_factor) * normalized_error_y)
        pan_output = self.pan_controller.compute(0, self.smoothed_pan)
        tilt_output = self.tilt_controller.compute(0, self.smoothed_tilt)
        return pan_output.value, tilt_output.value

    def update_position(self, pan_speed: float, tilt_speed: float, dt: float):
        self.current_pan += pan_speed * dt
        self.current_tilt += tilt_speed * dt
        # FIX: was clipping tilt into pan, and clamping pan not tilt
        self.current_pan = np.clip(self.current_pan, -180, 180)
        self.current_tilt = np.clip(self.current_tilt, -90, 90)


@dataclass
class Waypoint:
    x: float
    y: float
    heading: float = 0.0
    speed: float = 1.0


class TrajectoryPlanner:
    def __init__(self, max_speed: float = 2.0, max_acceleration: float = 1.0):
        # FIX: __init__ was empty (pass), missing required state attributes
        self.max_speed = max_speed
        self.max_acceleration = max_acceleration
        self.current_position = np.array([0.0, 0.0])
        self.current_velocity = np.array([0.0, 0.0])
        self.current_heading = 0.0

    def plan_path(self, waypoints: List[Waypoint]) -> List[np.ndarray]:
        if len(waypoints) < 2:
            return []
        path = []
        for i in range(len(waypoints) - 1):
            start = np.array([waypoints[i].x, waypoints[i].y])
            end = np.array([waypoints[i + 1].x, waypoints[i + 1].y])
            num_points = int(np.linalg.norm(end - start) * 10)
            num_points = max(num_points, 2)
            for t in np.linspace(0, 1, num_points):
                point = start * (1 - t) + end * t
                path.append(point)
        return path

    def compute_velocity_profile(self, path: List[np.ndarray]) -> List[float]:
        # FIX: was named follow_path but computed velocities; signature/body corrected
        if len(path) < 2:
            return [0.0]
        velocities = [0.0]
        for i in range(1, len(path)):
            distance = np.linalg.norm(path[i] - path[i - 1])
            prev_velocity = velocities[-1]
            achievable_velocity = np.sqrt(prev_velocity ** 2 + 2 * self.max_acceleration * distance)
            velocity = min(achievable_velocity, self.max_speed)
            velocities.append(velocity)
        # Backward pass to respect deceleration
        for i in range(len(velocities) - 2, -1, -1):
            distance = np.linalg.norm(path[i + 1] - path[i])
            next_velocity = velocities[i + 1]
            max_velocity = np.sqrt(next_velocity ** 2 + 2 * self.max_acceleration * distance)
            velocities[i] = min(velocities[i], max_velocity)
        return velocities

    def follow_path(self, path: List[np.ndarray], velocities: List[float],
                    current_index: int, dt: float) -> Tuple[np.ndarray, int]:
        # FIX: was named compute_velocity_profile but did path-following; corrected
        if current_index >= len(path) - 1:
            return np.array([0.0, 0.0]), current_index
        target = path[current_index]
        direction = target - self.current_position
        distance = np.linalg.norm(direction)
        if distance < 0.1:
            current_index = min(current_index + 1, len(path) - 1)
            target = path[current_index]
            direction = target - self.current_position
            distance = np.linalg.norm(direction)
        if distance > 0:
            direction_normalized = direction / distance
            desired_speed = velocities[min(current_index, len(velocities) - 1)]
            desired_velocity = direction_normalized * desired_speed
        else:
            desired_velocity = np.array([0.0, 0.0])
        velocity_error = desired_velocity - self.current_velocity
        max_velocity_change = self.max_acceleration * dt
        if np.linalg.norm(velocity_error) > max_velocity_change:
            velocity_error = velocity_error / np.linalg.norm(velocity_error) * max_velocity_change
        velocity_command = self.current_velocity + velocity_error
        return velocity_command, current_index

    def update_state(self, velocity: np.ndarray, dt: float):
        self.current_velocity = velocity
        self.current_position += velocity * dt
        if np.linalg.norm(velocity) > 0:
            self.current_heading = np.arctan2(velocity[1], velocity[0])


if __name__ == "__main__":
    print("Testing PID Controller...")
    pid = PIDcontroller(kp=1.0, ki=0.1, kd=0.05)
    pid.setpoint = 100.0
    measurement = 0.0
    for i in range(100):
        output = pid.compute(measurement)
        measurement += output.value * 0.1
        if i % 20 == 0:
            print(f"Step {i}: Setpoint={output.setpoint:.1f}, "
                  f"Measurement={measurement:.1f}, "
                  f"Error={output.error:.1f}, "
                  f"Output={output.value:.2f}")

    print("\nTesting Camera Controller...")
    camera = CameraController(max_rotation_speed=30.0)
    target_pos = (800, 400)
    frame_center = (640, 360)
    frame_size = (1280, 720)
    for i in range(50):
        target_pos = (target_pos[0] + 10, target_pos[1] + 5)
        pan_speed, tilt_speed = camera.track_target(target_pos, frame_center, frame_size)
        camera.update_position(pan_speed, tilt_speed, 0.033)
        if i % 10 == 0:
            print(f"Frame {i}: Pan={camera.current_pan:.1f}°, "
                  f"Tilt={camera.current_tilt:.1f}°, "
                  f"Speeds=({pan_speed:.1f}, {tilt_speed:.1f}) °/s")

    print("\nTesting Trajectory Planner...")
    planner = TrajectoryPlanner(max_speed=2.0, max_acceleration=1.0)
    waypoints = [
        Waypoint(0, 0), Waypoint(5, 5), Waypoint(10, 5), Waypoint(10, 10)
    ]
    path = planner.plan_path(waypoints)
    velocities = planner.compute_velocity_profile(path)
    print(f"Generated path with {len(path)} points")
    print(f"Max velocity in profile: {max(velocities):.2f} m/s")
    print("\nSimulating path following:")
    current_index = 0
    for step in range(100):
        velocity_cmd, current_index = planner.follow_path(path, velocities, current_index, 0.1)
        planner.update_state(velocity_cmd, 0.1)
        if step % 20 == 0:
            print(f"Step {step}: Position=({planner.current_position[0]:.1f}, "
                  f"{planner.current_position[1]:.1f}), "
                  f"Velocity={np.linalg.norm(planner.current_velocity):.2f} m/s")
        if current_index >= len(path) - 1:
            print(f"Reached goal at step {step}")
            break