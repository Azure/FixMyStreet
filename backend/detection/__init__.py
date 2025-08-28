"""
Pothole detection module

This module provides unified pothole detection capabilities with multiple algorithms
and video processing features.

Usage:
    from backend.detection import PotholeVideoProcessor, DetectorFactory
    
    # Create a processor
    processor = PotholeVideoProcessor("enhanced", sensitivity=0.7)
    
    # Process a video
    result = processor.process_video("video.mp4")
    
    # Or create a custom detector
    detector = DetectorFactory.create_detector("edge", sensitivity=0.8)
"""

from .detection_algorithms import (
    PotholeDetector,
    EdgeBasedDetector, 
    DarkRegionDetector,
    EnhancedDetector,
    DetectorFactory,
    Detection
)

from .video_processor import (
    PotholeVideoProcessor,
    VideoDetectionResult,
    detect_potholes_in_frame,  # Legacy compatibility
    detect_potholes_enhanced_from_path  # Legacy compatibility
)

__all__ = [
    # Core classes
    'PotholeDetector',
    'Detection',
    'DetectorFactory',
    
    # Detection algorithms
    'EdgeBasedDetector',
    'DarkRegionDetector', 
    'EnhancedDetector',
    
    # Video processing
    'PotholeVideoProcessor',
    'VideoDetectionResult',
    
    # Legacy functions for backward compatibility
    'detect_potholes_in_frame',
    'detect_potholes_enhanced_from_path'
]
