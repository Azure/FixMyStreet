#!/usr/bin/env python3
"""
Pothole detection algorithms - Unified detection system
"""

import cv2
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class Detection:
    """Data class for pothole detections"""
    
    def __init__(self, bbox: Dict[str, int], confidence: float, 
                 area_pixels: float, metrics: Dict[str, float]):
        self.bbox = bbox  # {'x': int, 'y': int, 'width': int, 'height': int}
        self.confidence = confidence
        self.area_pixels = area_pixels
        self.metrics = metrics  # Additional metrics like aspect_ratio, circularity, etc.


class PotholeDetector(ABC):
    """Abstract base class for pothole detection algorithms"""
    
    def __init__(self, sensitivity: float = 0.7):
        self.sensitivity = sensitivity
        
    @abstractmethod
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect potholes in a frame"""
        pass
    
    def _get_road_area(self, frame: np.ndarray) -> tuple:
        """Get the road area coordinates for focused detection"""
        height, width = frame.shape[:2]
        road_top = int(height * 0.30)
        road_bottom = int(height * 0.95)
        road_left = int(width * 0.05)
        road_right = int(width * 0.95)
        return road_top, road_bottom, road_left, road_right
    
    def _calculate_metrics(self, contour: np.ndarray, bbox: Dict[str, int]) -> Dict[str, float]:
        """Calculate detection metrics"""
        area = cv2.contourArea(contour)
        x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
        
        aspect_ratio = w / h if h > 0 else 0
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        perimeter = cv2.arcLength(contour, True)
        circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
        
        # Calculate convex hull for additional metrics
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)
        convexity = area / hull_area if hull_area > 0 else 0
        solidity = convexity  # Alias for backward compatibility
        
        return {
            'aspect_ratio': aspect_ratio,
            'extent': extent,
            'circularity': circularity,
            'convexity': convexity,
            'solidity': solidity
        }


class EdgeBasedDetector(PotholeDetector):
    """Edge-based pothole detection using Canny edge detection"""
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect potholes using edge detection"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        road_top, road_bottom, road_left, road_right = self._get_road_area(frame)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Edge detection with sensitivity-based thresholds
        low_threshold = max(10, int(50 - (1 - self.sensitivity) * 30))
        high_threshold = max(30, int(100 - (1 - self.sensitivity) * 50))
        edges = cv2.Canny(blurred, low_threshold, high_threshold)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        dilated = cv2.dilate(closed, kernel, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        return self._process_contours(contours, road_top, road_bottom, road_left, road_right)
    
    def _process_contours(self, contours, road_top, road_bottom, road_left, road_right) -> List[Detection]:
        """Process contours and create Detection objects"""
        detections = []
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:
                continue
                
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if in road area
            if y < road_top or y + h > road_bottom or x < road_left or x + w > road_right:
                continue
            
            bbox = {'x': x, 'y': y, 'width': w, 'height': h}
            metrics = self._calculate_metrics(contour, bbox)
            
            # Filter based on shape characteristics
            if self._is_valid_pothole_shape(metrics, area, w, h):
                confidence = self._calculate_confidence(metrics, area)
                if confidence > self.sensitivity:
                    detections.append(Detection(bbox, confidence, area, metrics))
        
        return detections
    
    def _is_valid_pothole_shape(self, metrics: Dict[str, float], area: float, w: int, h: int) -> bool:
        """Check if shape characteristics match pothole criteria"""
        return (0.2 <= metrics['aspect_ratio'] <= 5.0 and 
                0.1 <= metrics['extent'] <= 0.9 and 
                w > 15 and h > 15 and area > 200)
    
    def _calculate_confidence(self, metrics: Dict[str, float], area: float) -> float:
        """Calculate detection confidence"""
        size_factor = min(area / 1000, 1.0)
        extent_factor = metrics['extent']
        circularity_factor = min(metrics['circularity'] * 2, 1.0)
        
        confidence = (size_factor * 0.3 + extent_factor * 0.5 + circularity_factor * 0.2)
        
        # Boost for pothole-like shapes
        if 0.5 <= metrics['aspect_ratio'] <= 2.0 and area > 300:
            confidence *= 1.2
            
        return min(confidence, 1.0)


class DarkRegionDetector(PotholeDetector):
    """Dark region detection for identifying shadow areas (potholes)"""
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect potholes by identifying dark regions"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        road_top, road_bottom, road_left, road_right = self._get_road_area(frame)
        
        # Extract road area
        road_gray = gray[road_top:road_bottom, road_left:road_right]
        
        # Adaptive threshold based on local statistics
        mean_intensity = np.mean(road_gray)
        std_intensity = np.std(road_gray)
        threshold = max(20, mean_intensity - (2 - self.sensitivity) * std_intensity)
        
        # Create binary mask for dark regions
        _, binary = cv2.threshold(road_gray, threshold, 255, cv2.THRESH_BINARY_INV)
        
        # Morphological operations to clean up
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
        cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 300:
                continue
            
            x, y, w, h = cv2.boundingRect(contour)
            # Adjust coordinates back to full frame
            x += road_left
            y += road_top
            
            bbox = {'x': x, 'y': y, 'width': w, 'height': h}
            metrics = self._calculate_metrics(contour, bbox)
            
            if self._is_valid_dark_region(metrics, area, w, h):
                confidence = self._calculate_dark_confidence(metrics, area)
                if confidence > self.sensitivity:
                    detections.append(Detection(bbox, confidence, area, metrics))
        
        return detections
    
    def _is_valid_dark_region(self, metrics: Dict[str, float], area: float, w: int, h: int) -> bool:
        """Check if dark region matches pothole criteria"""
        return (0.3 <= metrics['aspect_ratio'] <= 3.0 and 
                metrics['extent'] > 0.3 and 
                w > 20 and h > 20 and area > 300)
    
    def _calculate_dark_confidence(self, metrics: Dict[str, float], area: float) -> float:
        """Calculate confidence for dark region detection"""
        size_factor = min(area / 2000, 1.0)
        shape_factor = 1.0 - abs(metrics['aspect_ratio'] - 1.0) * 0.3  # Prefer square-ish shapes
        convexity_factor = metrics['convexity']
        
        confidence = (size_factor * 0.4 + shape_factor * 0.3 + convexity_factor * 0.3)
        return min(confidence, 1.0)


class EnhancedDetector(PotholeDetector):
    """Enhanced detector that combines multiple detection methods"""
    
    def __init__(self, sensitivity: float = 0.7):
        super().__init__(sensitivity)
        self.edge_detector = EdgeBasedDetector(sensitivity)
        self.dark_detector = DarkRegionDetector(sensitivity)
    
    def detect(self, frame: np.ndarray) -> List[Detection]:
        """Detect potholes using multiple algorithms and merge results"""
        edge_detections = self.edge_detector.detect(frame)
        dark_detections = self.dark_detector.detect(frame)
        
        # Combine and merge overlapping detections
        all_detections = edge_detections + dark_detections
        merged_detections = self._merge_overlapping_detections(all_detections)
        
        return merged_detections
    
    def _merge_overlapping_detections(self, detections: List[Detection], 
                                    overlap_threshold: float = 0.3) -> List[Detection]:
        """Merge overlapping detections to avoid duplicates"""
        if not detections:
            return []
        
        # Sort by confidence (highest first)
        detections.sort(key=lambda d: d.confidence, reverse=True)
        
        merged = []
        for detection in detections:
            is_duplicate = False
            
            for existing in merged:
                if self._calculate_overlap(detection.bbox, existing.bbox) > overlap_threshold:
                    # If overlap is significant, keep the one with higher confidence
                    if detection.confidence > existing.confidence:
                        merged.remove(existing)
                        merged.append(detection)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                merged.append(detection)
        
        return merged
    
    def _calculate_overlap(self, bbox1: Dict[str, int], bbox2: Dict[str, int]) -> float:
        """Calculate overlap ratio between two bounding boxes"""
        x1, y1, w1, h1 = bbox1['x'], bbox1['y'], bbox1['width'], bbox1['height']
        x2, y2, w2, h2 = bbox2['x'], bbox2['y'], bbox2['width'], bbox2['height']
        
        # Calculate intersection
        ix1 = max(x1, x2)
        iy1 = max(y1, y2)
        ix2 = min(x1 + w1, x2 + w2)
        iy2 = min(y1 + h1, y2 + h2)
        
        if ix2 <= ix1 or iy2 <= iy1:
            return 0.0
        
        intersection = (ix2 - ix1) * (iy2 - iy1)
        area1 = w1 * h1
        area2 = w2 * h2
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


class DetectorFactory:
    """Factory class for creating detector instances"""
    
    @staticmethod
    def create_detector(detector_type: str = "enhanced", sensitivity: float = 0.7) -> PotholeDetector:
        """Create a detector instance"""
        if detector_type == "edge":
            return EdgeBasedDetector(sensitivity)
        elif detector_type == "dark":
            return DarkRegionDetector(sensitivity)
        elif detector_type == "enhanced":
            return EnhancedDetector(sensitivity)
        else:
            raise ValueError(f"Unknown detector type: {detector_type}")
