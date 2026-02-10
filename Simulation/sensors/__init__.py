"""
Sensors Module
Virtual sensor suite for object detection and environmental monitoring

Available sensors:
- VirtualCamera: FOV-based object detection
- VirtualLiDAR: 360° point cloud scanning
- VirtualIMU: Acceleration and orientation
- AudioSensor: Event detection
- Advanced cameras with environmental effects
- Advanced LiDAR with 3D scanning and clustering
"""

# Basic sensors
try:
    from .virtual_sensors import (
        VirtualCamera as BasicCamera,
        VirtualLiDAR as BasicLiDAR,
        VirtualIMU,
        AudioSensor
    )
    BASIC_SENSORS_AVAILABLE = True
except ImportError:
    BASIC_SENSORS_AVAILABLE = False

# Advanced camera
try:
    from .camera_sensor import (
        VirtualCamera,
        CameraType,
        CameraIntrinsics,
        Detection,
        DetectionConfidence
    )
    ADVANCED_CAMERA_AVAILABLE = True
except ImportError:
    ADVANCED_CAMERA_AVAILABLE = False

# Advanced LiDAR
try:
    from .lidar_sensor import (
        VirtualLiDAR,
        LiDARType,
        ScanPattern,
        PointCloudData
    )
    ADVANCED_LIDAR_AVAILABLE = True
except ImportError:
    ADVANCED_LIDAR_AVAILABLE = False

__all__ = []

# Add available sensors to __all__
if BASIC_SENSORS_AVAILABLE:
    __all__.extend(['BasicCamera', 'BasicLiDAR', 'VirtualIMU', 'AudioSensor'])

if ADVANCED_CAMERA_AVAILABLE:
    __all__.extend([
        'VirtualCamera',
        'CameraType',
        'CameraIntrinsics',
        'Detection',
        'DetectionConfidence'
    ])

if ADVANCED_LIDAR_AVAILABLE:
    __all__.extend([
        'VirtualLiDAR',
        'LiDARType',
        'ScanPattern',
        'PointCloudData'
    ])

__version__ = '1.0.0'