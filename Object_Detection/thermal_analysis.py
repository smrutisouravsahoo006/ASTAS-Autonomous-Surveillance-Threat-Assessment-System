"""
Thermal Analysis Module
Processes thermal/infrared imagery for enhanced night vision and heat signature detection.
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class HeatSignature:
    """Container for detected heat signature"""
    bbox: Tuple[int, int, int, int]
    center: Tuple[int, int]
    temperature: float  # Estimated temperature in Celsius
    area: int
    intensity: float
    signature_type: str  # person, vehicle, animal, unknown
    confidence: float


class ThermalAnalyzer:
    """Analyze thermal/infrared imagery"""
    
    def __init__(self, temp_min: float = -20.0, temp_max: float = 50.0):
        """
        Initialize thermal analyzer
        
        Args:
            temp_min: Minimum temperature range
            temp_max: Maximum temperature range
        """
        self.temp_min = temp_min
        self.temp_max = temp_max
        
        # Temperature thresholds for classification
        self.person_temp_range = (35.0, 37.5)  # Human body temp
        self.vehicle_temp_range = (40.0, 100.0)  # Engine heat
        self.animal_temp_range = (36.0, 40.0)
        
        # Size thresholds (in pixels)
        self.person_size_range = (1000, 10000)
        self.vehicle_size_range = (5000, 50000)
        
    def process_thermal_frame(self, thermal_frame: np.ndarray) -> List[HeatSignature]:
        """
        Process thermal frame to detect heat signatures
        
        Args:
            thermal_frame: Thermal image (single channel, 8-bit or 16-bit)
            
        Returns:
            List of detected heat signatures
        """
        # Normalize to 8-bit if necessary
        if thermal_frame.dtype == np.uint16:
            thermal_frame = cv2.normalize(thermal_frame, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        
        # Apply threshold to detect hot objects
        _, binary = cv2.threshold(thermal_frame, 150, 255, cv2.THRESH_BINARY)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        signatures = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            
            # Filter small detections
            if area < 500:
                continue
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            bbox = (x, y, x + w, y + h)
            center = (x + w // 2, y + h // 2)
            
            # Calculate average intensity in region
            mask = np.zeros(thermal_frame.shape, dtype=np.uint8)
            cv2.drawContours(mask, [contour], -1, 255, -1)
            intensity = cv2.mean(thermal_frame, mask=mask)[0]
            
            # Estimate temperature (linear mapping from intensity)
            temperature = self._intensity_to_temperature(intensity)
            
            # Classify signature
            signature_type, confidence = self._classify_signature(area, temperature, w, h)
            
            signatures.append(HeatSignature(
                bbox=bbox,
                center=center,
                temperature=temperature,
                area=area,
                intensity=intensity,
                signature_type=signature_type,
                confidence=confidence
            ))
        
        return signatures
    
    def _intensity_to_temperature(self, intensity: float) -> float:
        """Convert pixel intensity to estimated temperature"""
        # Linear mapping from intensity range to temperature range
        temp = self.temp_min + (intensity / 255.0) * (self.temp_max - self.temp_min)
        return temp
    
    def _classify_signature(self, area: int, temperature: float, 
                           width: int, height: int) -> Tuple[str, float]:
        """Classify heat signature based on characteristics"""
        confidence = 0.0
        signature_type = "unknown"
        
        # Check temperature ranges
        if self.person_temp_range[0] <= temperature <= self.person_temp_range[1]:
            # Check size for person
            if self.person_size_range[0] <= area <= self.person_size_range[1]:
                # Check aspect ratio (standing person)
                aspect_ratio = height / width if width > 0 else 0
                if 1.5 <= aspect_ratio <= 3.0:
                    signature_type = "person"
                    confidence = 0.8
                else:
                    signature_type = "person"
                    confidence = 0.6
        
        elif self.vehicle_temp_range[0] <= temperature <= self.vehicle_temp_range[1]:
            # Check size for vehicle
            if self.vehicle_size_range[0] <= area <= self.vehicle_size_range[1]:
                signature_type = "vehicle"
                confidence = 0.7
        
        elif self.animal_temp_range[0] <= temperature <= self.animal_temp_range[1]:
            if 500 <= area <= 5000:
                signature_type = "animal"
                confidence = 0.6
        
        # Default to unknown with low confidence
        if signature_type == "unknown":
            confidence = 0.3
        
        return signature_type, confidence
    
    def enhance_thermal_image(self, thermal_frame: np.ndarray) -> np.ndarray:
        """
        Enhance thermal image for better visualization
        
        Args:
            thermal_frame: Input thermal image
            
        Returns:
            Enhanced thermal image
        """
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(thermal_frame)
        
        # Apply colormap for visualization
        colored = cv2.applyColorMap(enhanced, cv2.COLORMAP_JET)
        
        return colored
    
    def fusion_with_rgb(self, rgb_frame: np.ndarray, thermal_frame: np.ndarray,
                       alpha: float = 0.5) -> np.ndarray:
        """
        Fuse RGB and thermal images
        
        Args:
            rgb_frame: RGB image
            thermal_frame: Thermal image
            alpha: Blending factor (0-1)
            
        Returns:
            Fused image
        """
        # Ensure same size
        if rgb_frame.shape[:2] != thermal_frame.shape[:2]:
            thermal_frame = cv2.resize(thermal_frame, 
                                      (rgb_frame.shape[1], rgb_frame.shape[0]))
        
        # Convert thermal to color if needed
        if len(thermal_frame.shape) == 2:
            thermal_colored = cv2.applyColorMap(thermal_frame, cv2.COLORMAP_JET)
        else:
            thermal_colored = thermal_frame
        
        # Blend images
        fused = cv2.addWeighted(rgb_frame, alpha, thermal_colored, 1 - alpha, 0)
        
        return fused
    
    def draw_signatures(self, frame: np.ndarray, signatures: List[HeatSignature],
                       show_temperature: bool = True) -> np.ndarray:
        """Draw heat signatures on frame"""
        output = frame.copy()
        
        for sig in signatures:
            x1, y1, x2, y2 = sig.bbox
            
            # Color based on type
            colors = {
                'person': (0, 255, 0),
                'vehicle': (255, 0, 0),
                'animal': (0, 255, 255),
                'unknown': (128, 128, 128)
            }
            color = colors.get(sig.signature_type, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{sig.signature_type}"
            if show_temperature:
                label += f" {sig.temperature:.1f}°C"
            
            cv2.putText(output, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            # Draw center
            cv2.circle(output, sig.center, 5, color, -1)
        
        return output


def generate_synthetic_thermal_frame(width: int = 640, height: int = 480) -> np.ndarray:
    """Generate synthetic thermal frame for testing"""
    # Base temperature (ambient)
    frame = np.random.randint(80, 120, (height, width), dtype=np.uint8)
    
    # Add hot signature (person)
    person_x, person_y = np.random.randint(100, width-100), np.random.randint(100, height-100)
    cv2.ellipse(frame, (person_x, person_y), (30, 60), 0, 0, 360, 200, -1)
    
    # Add vehicle signature
    vehicle_x, vehicle_y = np.random.randint(100, width-100), np.random.randint(100, height-100)
    cv2.rectangle(frame, (vehicle_x-40, vehicle_y-30), (vehicle_x+40, vehicle_y+30), 220, -1)
    
    # Add noise
    noise = np.random.randint(-10, 10, (height, width), dtype=np.int16)
    frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Blur for realistic thermal diffusion
    frame = cv2.GaussianBlur(frame, (15, 15), 0)
    
    return frame


if __name__ == "__main__":
    print("Testing Thermal Analysis Module...")
    
    analyzer = ThermalAnalyzer(temp_min=0, temp_max=50)
    
    # Generate synthetic thermal frame
    thermal_frame = generate_synthetic_thermal_frame()
    
    print(f"Generated thermal frame: {thermal_frame.shape}")
    
    # Detect heat signatures
    signatures = analyzer.process_thermal_frame(thermal_frame)
    
    print(f"\nDetected {len(signatures)} heat signatures:")
    for i, sig in enumerate(signatures):
        print(f"\nSignature {i+1}:")
        print(f"  Type: {sig.signature_type} (confidence: {sig.confidence:.2f})")
        print(f"  Temperature: {sig.temperature:.1f}°C")
        print(f"  Area: {sig.area} pixels")
        print(f"  Location: {sig.center}")
    
    # Enhance and visualize
    enhanced = analyzer.enhance_thermal_image(thermal_frame)
    print(f"\nEnhanced thermal frame: {enhanced.shape}")
    
    print("\nThermal analysis test complete!")