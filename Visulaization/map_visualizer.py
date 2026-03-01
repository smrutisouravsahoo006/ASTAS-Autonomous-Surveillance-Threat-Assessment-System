"""
Map Visualizer Module
Real-time visualization of surveillance data, tracking, and threat assessments.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon, Arrow
from matplotlib.animation import FuncAnimation
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime
import json


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    width: int = 1920
    height: int = 1080
    fps: int = 30
    show_zones: bool = True
    show_trajectories: bool = True
    show_sensor_data: bool = True
    show_heatmap: bool = False
    trajectory_length: int = 30
    colormap: str = 'viridis'


class MapVisualizer:
    """Real-time map visualization for surveillance system"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize map visualizer
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        
        # Color scheme
        self.colors = {
            'background': (30, 30, 30),
            'grid': (50, 50, 50),
            'zones': {
                'restricted': (100, 100, 200, 100),  # Red zone (RGBA)
                'caution': (200, 200, 100, 100),      # Yellow zone
                'safe': (100, 200, 100, 50)           # Green zone
            },
            'threat_levels': {
                'low': (0, 255, 0),        # Green
                'medium': (255, 255, 0),   # Yellow
                'high': (255, 165, 0),     # Orange
                'critical': (255, 0, 0)    # Red
            },
            'objects': {
                'person': (0, 255, 0),
                'vehicle': (255, 0, 0),
                'animal': (0, 255, 255),
                'unknown': (128, 128, 128)
            }
        }
        
        # Font settings
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.font_scale = 0.6
        self.font_thickness = 2
        
        # Heatmap accumulator
        self.heatmap = np.zeros((self.config.height, self.config.width), dtype=np.float32)
        self.heatmap_alpha = 0.95  # Decay factor
        
    def create_base_frame(self) -> np.ndarray:
        """Create base visualization frame with grid"""
        frame = np.zeros((self.config.height, self.config.width, 3), dtype=np.uint8)
        frame[:] = self.colors['background']
        
        # Draw grid
        grid_spacing = 50
        for x in range(0, self.config.width, grid_spacing):
            cv2.line(frame, (x, 0), (x, self.config.height), 
                    self.colors['grid'], 1)
        for y in range(0, self.config.height, grid_spacing):
            cv2.line(frame, (0, y), (self.config.width, y), 
                    self.colors['grid'], 1)
        
        return frame
    
    def draw_zones(self, frame: np.ndarray, zones: List[Dict]) -> np.ndarray:
        """
        Draw restricted/caution zones
        
        Args:
            frame: Input frame
            zones: List of zone dictionaries with 'type', 'polygon'
        """
        overlay = frame.copy()
        
        for zone in zones:
            zone_type = zone.get('type', 'safe')
            polygon = zone.get('polygon')
            
            if polygon is not None and len(polygon) > 0:
                # Get color
                color = self.colors['zones'].get(zone_type, (100, 100, 100, 100))
                
                # Draw filled polygon
                pts = np.array(polygon, dtype=np.int32)
                cv2.fillPoly(overlay, [pts], color[:3])
                
                # Draw border
                cv2.polylines(frame, [pts], True, color[:3], 3)
                
                # Draw label
                center = pts.mean(axis=0).astype(int)
                label = zone_type.upper()
                cv2.putText(frame, label, tuple(center), 
                           self.font, 1.0, (255, 255, 255), 2)
        
        # Blend overlay
        cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
        
        return frame
    
    def draw_detections(self, frame: np.ndarray, detections: List) -> np.ndarray:
        """
        Draw object detections
        
        Args:
            frame: Input frame
            detections: List of Detection objects
        """
        for det in detections:
            x1, y1, x2, y2 = det.bbox
            
            # Get color based on class
            color = self.colors['objects'].get(det.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw center point
            cv2.circle(frame, det.center, 5, color, -1)
            
            # Draw label with confidence
            label = f"{det.class_name} {det.confidence:.2f}"
            (text_w, text_h), baseline = cv2.getTextSize(
                label, self.font, self.font_scale, self.font_thickness
            )
            
            # Background for text
            cv2.rectangle(frame, (x1, y1 - text_h - 10), 
                         (x1 + text_w, y1), color, -1)
            
            # Text
            cv2.putText(frame, label, (x1, y1 - 5), 
                       self.font, self.font_scale, (255, 255, 255), 
                       self.font_thickness)
        
        return frame
    
    def draw_tracks(self, frame: np.ndarray, tracks: List) -> np.ndarray:
        """
        Draw tracking information with trajectories
        
        Args:
            frame: Input frame
            tracks: List of Track objects
        """
        for track in tracks:
            x1, y1, x2, y2 = track.bbox
            
            # Color based on track age
            age_normalized = min(track.age / 100, 1.0)
            color = (
                int(255 * (1 - age_normalized)),  # Red decreases
                int(255 * age_normalized),         # Green increases
                100
            )
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw trajectory
            if self.config.show_trajectories and len(track.trajectory) > 1:
                points = np.array(list(track.trajectory), dtype=np.int32)
                
                # Draw fading trajectory
                for i in range(len(points) - 1):
                    alpha = (i + 1) / len(points)  # Fade in
                    thickness = max(1, int(3 * alpha))
                    
                    pt1 = tuple(points[i])
                    pt2 = tuple(points[i + 1])
                    cv2.line(frame, pt1, pt2, color, thickness)
                
                # Draw velocity arrow
                if len(points) > 1:
                    start = tuple(points[-1])
                    vel = track.velocity
                    speed = np.sqrt(vel[0]**2 + vel[1]**2)
                    if speed > 1:
                        scale = 20  # Arrow length
                        end = (
                            int(start[0] + vel[0] * scale),
                            int(start[1] + vel[1] * scale)
                        )
                        cv2.arrowedLine(frame, start, end, color, 2, tipLength=0.3)
            
            # Draw track info
            label = f"ID:{track.track_id} {track.class_name}"
            speed = np.sqrt(track.velocity[0]**2 + track.velocity[1]**2)
            label += f" {speed:.1f}px/s"
            
            cv2.putText(frame, label, (x1, y1 - 10), 
                       self.font, self.font_scale, color, self.font_thickness)
            
            # Draw track ID on center
            cv2.putText(frame, str(track.track_id), 
                       (track.center[0] - 10, track.center[1] + 5),
                       self.font, 0.8, (255, 255, 255), 2)
        
        return frame
    
    def draw_sensor_data(self, frame: np.ndarray, sensor_data: Dict) -> np.ndarray:
        """
        Draw sensor data overlays
        
        Args:
            frame: Input frame
            sensor_data: Dictionary of sensor readings
        """
        if not self.config.show_sensor_data:
            return frame
        
        y_offset = 120
        x_pos = 10
        line_height = 30
        
        # IMU data
        if 'imu' in sensor_data:
            imu = sensor_data['imu']
            
            cv2.putText(frame, "IMU Data:", (x_pos, y_offset), 
                       self.font, 0.7, (100, 200, 255), 2)
            y_offset += line_height
            
            if 'orientation' in imu:
                orient = imu['orientation']
                text = f"Roll: {np.degrees(orient['roll']):.1f}°"
                cv2.putText(frame, text, (x_pos + 20, y_offset), 
                           self.font, 0.5, (255, 255, 255), 1)
                y_offset += line_height - 5
                
                text = f"Pitch: {np.degrees(orient['pitch']):.1f}°"
                cv2.putText(frame, text, (x_pos + 20, y_offset), 
                           self.font, 0.5, (255, 255, 255), 1)
                y_offset += line_height - 5
            
            if imu.get('vibration_detected', False):
                cv2.putText(frame, "VIBRATION!", (x_pos + 20, y_offset), 
                           self.font, 0.6, (0, 0, 255), 2)
                y_offset += line_height
        
        # LiDAR data
        if 'lidar_objects' in sensor_data:
            lidar_objs = sensor_data['lidar_objects']
            
            cv2.putText(frame, f"LiDAR: {len(lidar_objs)} objects", 
                       (x_pos, y_offset), self.font, 0.7, 
                       (100, 255, 200), 2)
            y_offset += line_height
        
        # Audio events
        if 'audio_events' in sensor_data:
            audio = sensor_data['audio_events']
            
            cv2.putText(frame, "Audio Events:", (x_pos, y_offset), 
                       self.font, 0.7, (255, 200, 100), 2)
            y_offset += line_height
            
            for event in audio[:3]:  # Show first 3
                text = f"• {event.event_type} ({event.confidence:.2f})"
                cv2.putText(frame, text, (x_pos + 20, y_offset), 
                           self.font, 0.5, (255, 255, 255), 1)
                y_offset += line_height - 5
        
        return frame
    
    def draw_threat_assessment(self, frame: np.ndarray, 
                               assessment: Dict) -> np.ndarray:
        """
        Draw threat assessment overlay
        
        Args:
            frame: Input frame
            assessment: Threat assessment dictionary
        """
        threat_level = assessment.get('threat_level', 'low')
        threat_score = assessment.get('threat_score', 0.0)
        reasoning = assessment.get('reasoning', 'No assessment')
        actions = assessment.get('recommended_actions', [])
        
        # Get color for threat level
        color = self.colors['threat_levels'].get(threat_level, (255, 255, 255))
        
        # Draw threat indicator box
        box_height = 150 + len(actions) * 25
        box_width = 450
        x_pos = self.config.width - box_width - 20
        y_pos = 20
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x_pos, y_pos), 
                     (x_pos + box_width, y_pos + box_height), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
        
        # Border with threat color
        cv2.rectangle(frame, (x_pos, y_pos), 
                     (x_pos + box_width, y_pos + box_height), 
                     color, 3)
        
        # Title
        cv2.putText(frame, "THREAT ASSESSMENT", (x_pos + 10, y_pos + 30), 
                   self.font, 0.8, (255, 255, 255), 2)
        
        # Threat level
        level_text = f"Level: {threat_level.upper()}"
        cv2.putText(frame, level_text, (x_pos + 10, y_pos + 60), 
                   self.font, 0.9, color, 2)
        
        # Score bar
        bar_width = 350
        bar_x = x_pos + 50
        bar_y = y_pos + 75
        
        # Background bar
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + bar_width, bar_y + 20), 
                     (50, 50, 50), -1)
        
        # Filled bar
        fill_width = int(bar_width * threat_score)
        cv2.rectangle(frame, (bar_x, bar_y), 
                     (bar_x + fill_width, bar_y + 20), 
                     color, -1)
        
        # Score text
        score_text = f"{threat_score:.2f}"
        cv2.putText(frame, score_text, (bar_x + bar_width + 10, bar_y + 15), 
                   self.font, 0.6, (255, 255, 255), 2)
        
        # Reasoning (wrapped)
        y_text = y_pos + 110
        max_chars = 50
        if len(reasoning) > max_chars:
            reasoning = reasoning[:max_chars] + "..."
        cv2.putText(frame, reasoning, (x_pos + 10, y_text), 
                   self.font, 0.5, (200, 200, 200), 1)
        
        # Actions
        y_text += 30
        cv2.putText(frame, "Actions:", (x_pos + 10, y_text), 
                   self.font, 0.6, (255, 255, 255), 2)
        
        for i, action in enumerate(actions[:3]):  # Show first 3
            y_text += 25
            action_text = f"• {action}"
            if len(action_text) > 45:
                action_text = action_text[:45] + "..."
            cv2.putText(frame, action_text, (x_pos + 20, y_text), 
                       self.font, 0.5, (200, 200, 200), 1)
        
        return frame
    
    def draw_heatmap(self, frame: np.ndarray, positions: List[Tuple[int, int]]) -> np.ndarray:
        """
        Draw movement heatmap
        
        Args:
            frame: Input frame
            positions: List of (x, y) positions to add to heatmap
        """
        if not self.config.show_heatmap:
            return frame
        
        # Decay existing heatmap
        self.heatmap *= self.heatmap_alpha
        
        # Add new positions
        for pos in positions:
            x, y = pos
            if 0 <= x < self.config.width and 0 <= y < self.config.height:
                # Gaussian blob
                cv2.circle(self.heatmap, (x, y), 20, 1.0, -1)
        
        # Normalize and colorize
        if self.heatmap.max() > 0:
            heatmap_normalized = (self.heatmap / self.heatmap.max() * 255).astype(np.uint8)
            heatmap_colored = cv2.applyColorMap(heatmap_normalized, cv2.COLORMAP_JET)
            
            # Blend with frame
            frame = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        
        return frame
    
    def draw_statistics(self, frame: np.ndarray, stats: Dict) -> np.ndarray:
        """
        Draw system statistics
        
        Args:
            frame: Input frame
            stats: Statistics dictionary
        """
        x_pos = 10
        y_pos = 30
        line_height = 30
        
        # Timestamp
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(frame, timestamp, (x_pos, y_pos), 
                   self.font, 0.7, (255, 255, 255), 2)
        y_pos += line_height
        
        # Frame count
        if 'frame_count' in stats:
            text = f"Frame: {stats['frame_count']}"
            cv2.putText(frame, text, (x_pos, y_pos), 
                       self.font, 0.6, (255, 255, 255), 2)
            y_pos += line_height
        
        # FPS
        if 'fps' in stats:
            text = f"FPS: {stats['fps']:.1f}"
            color = (0, 255, 0) if stats['fps'] >= 25 else (255, 165, 0)
            cv2.putText(frame, text, (x_pos, y_pos), 
                       self.font, 0.6, color, 2)
            y_pos += line_height
        
        # Detection/Track counts
        if 'num_detections' in stats:
            text = f"Detections: {stats['num_detections']}"
            cv2.putText(frame, text, (x_pos, y_pos), 
                       self.font, 0.6, (100, 200, 255), 2)
        
        if 'num_tracks' in stats:
            text = f" | Tracks: {stats['num_tracks']}"
            cv2.putText(frame, text, (x_pos + 200, y_pos), 
                       self.font, 0.6, (100, 255, 200), 2)
        
        return frame
    
    def visualize_complete(self, detections: List, tracks: List, 
                          zones: List[Dict], sensor_data: Dict,
                          threat_assessment: Dict, stats: Dict) -> np.ndarray:
        """
        Create complete visualization with all elements
        
        Args:
            detections: List of detections
            tracks: List of tracks
            zones: List of zone definitions
            sensor_data: Sensor data dictionary
            threat_assessment: Threat assessment results
            stats: System statistics
            
        Returns:
            Visualized frame
        """
        # Create base
        frame = self.create_base_frame()
        
        # Draw zones
        if self.config.show_zones:
            frame = self.draw_zones(frame, zones)
        
        # Draw heatmap
        positions = [track.center for track in tracks]
        frame = self.draw_heatmap(frame, positions)
        
        # Draw detections
        frame = self.draw_detections(frame, detections)
        
        # Draw tracks
        frame = self.draw_tracks(frame, tracks)
        
        # Draw sensor data
        frame = self.draw_sensor_data(frame, sensor_data)
        
        # Draw threat assessment
        frame = self.draw_threat_assessment(frame, threat_assessment)
        
        # Draw statistics
        frame = self.draw_statistics(frame, stats)
        
        return frame


def create_matplotlib_dashboard(data_history: List[Dict]) -> plt.Figure:
    """
    Create matplotlib dashboard with plots
    
    Args:
        data_history: List of historical data points
        
    Returns:
        Matplotlib figure
    """
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('ASTAS Surveillance Dashboard', fontsize=16, fontweight='bold')
    
    # Extract data
    timestamps = [d['timestamp'] for d in data_history]
    threat_scores = [d['threat_score'] for d in data_history]
    num_detections = [d['num_detections'] for d in data_history]
    num_tracks = [d['num_tracks'] for d in data_history]
    
    # Plot 1: Threat Score Over Time
    axes[0, 0].plot(timestamps, threat_scores, 'r-', linewidth=2)
    axes[0, 0].axhline(y=0.85, color='r', linestyle='--', label='Critical')
    axes[0, 0].axhline(y=0.65, color='orange', linestyle='--', label='High')
    axes[0, 0].axhline(y=0.40, color='y', linestyle='--', label='Medium')
    axes[0, 0].set_title('Threat Score Timeline')
    axes[0, 0].set_xlabel('Time')
    axes[0, 0].set_ylabel('Threat Score')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Detection/Track Counts
    axes[0, 1].plot(timestamps, num_detections, 'b-', label='Detections', linewidth=2)
    axes[0, 1].plot(timestamps, num_tracks, 'g-', label='Tracks', linewidth=2)
    axes[0, 1].set_title('Object Counts')
    axes[0, 1].set_xlabel('Time')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Threat Level Distribution
    threat_levels = [d['threat_level'] for d in data_history]
    level_counts = {}
    for level in ['low', 'medium', 'high', 'critical']:
        level_counts[level] = threat_levels.count(level)
    
    axes[1, 0].bar(level_counts.keys(), level_counts.values(), 
                   color=['green', 'yellow', 'orange', 'red'])
    axes[1, 0].set_title('Threat Level Distribution')
    axes[1, 0].set_ylabel('Count')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: System Statistics
    stats_text = f"""
    Total Frames: {len(data_history)}
    Avg Detections: {np.mean(num_detections):.1f}
    Avg Tracks: {np.mean(num_tracks):.1f}
    Avg Threat Score: {np.mean(threat_scores):.2f}
    Max Threat Score: {max(threat_scores):.2f}
    
    Critical Events: {level_counts.get('critical', 0)}
    High Threats: {level_counts.get('high', 0)}
    """
    
    axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, 
                    verticalalignment='center', family='monospace')
    axes[1, 1].axis('off')
    axes[1, 1].set_title('Summary Statistics')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    print("Testing Map Visualizer...")
    
    # Create visualizer
    config = VisualizationConfig(width=1280, height=720)
    visualizer = MapVisualizer(config)
    
    # Mock data
    from Object_Detection.object_detection import Detection
    from Object_Detection.motion_tracking import Track
    from collections import deque
    
    detections = [
        Detection(0, "person", 0.92, (100, 150, 180, 350), (140, 250), 8000, 0.0),
        Detection(2, "car", 0.87, (400, 200, 600, 400), (500, 300), 40000, 0.0)
    ]
    
    tracks = [
        Track(1, "person", (100, 150, 180, 350), (140, 250), 
              velocity=(2.0, 1.0), trajectory=deque([(120, 240), (130, 245), (140, 250)])),
        Track(2, "car", (400, 200, 600, 400), (500, 300),
              velocity=(5.0, 0.5), trajectory=deque([(480, 295), (490, 297), (500, 300)]))
    ]
    
    zones = [
        {'type': 'restricted', 'polygon': np.array([[0, 0], [640, 0], [640, 240], [0, 240]])}
    ]
    
    sensor_data = {
        'imu': {
            'orientation': {'roll': 0.1, 'pitch': -0.05, 'yaw': 0.0},
            'vibration_detected': False
        },
        'lidar_objects': [{'type': 'person', 'distance': 15.2}],
        'audio_events': []
    }
    
    assessment = {
        'threat_level': 'medium',
        'threat_score': 0.55,
        'reasoning': 'Person detected in restricted zone',
        'recommended_actions': ['Monitor closely', 'Track continuously']
    }
    
    stats = {
        'frame_count': 150,
        'fps': 29.5,
        'num_detections': 2,
        'num_tracks': 2
    }
    
    # Create visualization
    frame = visualizer.visualize_complete(
        detections, tracks, zones, sensor_data, assessment, stats
    )
    
    print(f"Generated visualization: {frame.shape}")
    print("Visualization test complete!")







