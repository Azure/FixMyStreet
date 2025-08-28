#!/usr/bin/env python3
"""
Video processing module for pothole detection
"""

import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from .detection_algorithms import DetectorFactory, Detection
from backend.utils.extract_location import extract_gps_from_video_overlay


class VideoDetectionResult:
    """Result container for video detection"""
    
    def __init__(self):
        self.detections: List[Dict[str, Any]] = []
        self.gps_data: List[Dict[str, Any]] = []
        self.video_info: Dict[str, Any] = {}
        self.summary: Dict[str, Any] = {}


class PotholeVideoProcessor:
    """Process videos for pothole detection with GPS extraction"""
    
    def __init__(self, detector_type: str = "enhanced", sensitivity: float = 0.7):
        self.detector = DetectorFactory.create_detector(detector_type, sensitivity)
        self.sensitivity = sensitivity
    
    def process_video(self, video_path: str, sample_interval: int = 30, 
                     extract_gps: bool = True) -> VideoDetectionResult:
        """
        Process a video file for pothole detection
        
        Args:
            video_path: Path to video file
            sample_interval: Frame sampling interval
            extract_gps: Whether to extract GPS data
            
        Returns:
            VideoDetectionResult containing all detection data
        """
        result = VideoDetectionResult()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        result.video_info = {
            'fps': fps,
            'total_frames': total_frames,
            'duration_seconds': duration,
            'processed_frames': 0
        }
        
        # Extract GPS data if requested
        if extract_gps:
            try:
                result.gps_data = extract_gps_from_video_overlay(video_path, sample_interval)
            except Exception as e:
                print(f"GPS extraction failed: {e}")
                result.gps_data = []
        
        # Process frames
        frame_number = 0
        processed_frames = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Sample frames based on interval
                if frame_number % sample_interval == 0:
                    detections = self._process_frame(frame, frame_number, fps)
                    
                    # Add GPS location to detections
                    gps_location = self._get_gps_for_frame(frame_number, fps, result.gps_data)
                    
                    for detection in detections:
                        detection_dict = self._detection_to_dict(detection, frame_number, fps)
                        detection_dict['location'] = gps_location
                        result.detections.append(detection_dict)
                    
                    processed_frames += 1
                
                frame_number += 1
                
        finally:
            cap.release()
        
        result.video_info['processed_frames'] = processed_frames
        result.summary = self._calculate_summary(result)
        
        return result
    
    def process_frame(self, frame: np.ndarray, frame_number: int = 0, fps: float = 30.0) -> List[Dict[str, Any]]:
        """
        Process a single frame for pothole detection
        
        Args:
            frame: OpenCV frame
            frame_number: Frame number
            fps: Video FPS
            
        Returns:
            List of detection dictionaries
        """
        detections = self._process_frame(frame, frame_number, fps)
        return [self._detection_to_dict(d, frame_number, fps) for d in detections]
    
    def _process_frame(self, frame: np.ndarray, frame_number: int, fps: float) -> List[Detection]:
        """Internal frame processing"""
        return self.detector.detect(frame)
    
    def _detection_to_dict(self, detection: Detection, frame_number: int, fps: float) -> Dict[str, Any]:
        """Convert Detection object to dictionary"""
        timestamp_seconds = frame_number / fps if fps > 0 else 0
        timestamp_str = f"{int(timestamp_seconds//60):02d}:{int(timestamp_seconds%60):02d}"
        
        return {
            'frame': frame_number,
            'timestamp': timestamp_str,
            'video_time_seconds': timestamp_seconds,
            'bbox': detection.bbox,
            'confidence': detection.confidence,
            'area_pixels': detection.area_pixels,
            'metrics': detection.metrics,
            'location': None  # Will be filled by caller
        }
    
    def _get_gps_for_frame(self, frame_number: int, fps: float, gps_data: List[Dict[str, Any]]) -> Optional[Dict[str, float]]:
        """Get GPS location for a specific frame"""
        if not gps_data:
            return None
        
        video_time = frame_number / fps if fps > 0 else 0
        
        # Find closest GPS point by time
        closest_gps = None
        min_time_diff = float('inf')
        
        for gps_point in gps_data:
            # Parse timestamp (format: "MM:SS")
            try:
                time_parts = gps_point['timestamp'].split(':')
                gps_time = int(time_parts[0]) * 60 + int(time_parts[1])
                time_diff = abs(video_time - gps_time)
                
                if time_diff < min_time_diff:
                    min_time_diff = time_diff
                    closest_gps = gps_point
            except (ValueError, KeyError, IndexError):
                continue
        
        if closest_gps and min_time_diff <= 5:  # Within 5 seconds
            return {
                'lat': closest_gps.get('lat'),
                'lon': closest_gps.get('lon')
            }
        
        return None
    
    def _calculate_summary(self, result: VideoDetectionResult) -> Dict[str, Any]:
        """Calculate detection summary statistics"""
        total_detections = len(result.detections)
        frames_with_detections = len(set(d['frame'] for d in result.detections))
        
        if total_detections > 0:
            avg_confidence = sum(d['confidence'] for d in result.detections) / total_detections
            gps_coverage = len([d for d in result.detections if d.get('location')]) / total_detections * 100
        else:
            avg_confidence = 0.0
            gps_coverage = 0.0
        
        return {
            'total_potholes': total_detections,
            'frames_with_potholes': frames_with_detections,
            'avg_confidence': avg_confidence,
            'gps_coverage': gps_coverage
        }


def detect_potholes_in_frame(frame: np.ndarray, frame_number: int, fps: float, 
                           sensitivity: float = 0.7) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility
    """
    processor = PotholeVideoProcessor("enhanced", sensitivity)
    return processor.process_frame(frame, frame_number, fps)


def detect_potholes_enhanced_from_path(image_path: str, sensitivity: float = 0.7) -> List[Dict[str, Any]]:
    """
    Legacy function for backward compatibility - detect from image path
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return []
    
    processor = PotholeVideoProcessor("enhanced", sensitivity)
    detections = processor._process_frame(frame, 0, 30.0)
    return [processor._detection_to_dict(d, 0, 30.0) for d in detections]
