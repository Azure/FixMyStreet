#!/usr/bin/env python3
"""
Enhanced pothole detection algorithm optimized for real-world pothole images
"""

import cv2
import numpy as np

def detect_potholes_enhanced_from_path(image_path, sensitivity=0.7):
    """
    Enhanced pothole detection from image path
    
    Args:
        image_path: Path to image file
        sensitivity: Detection sensitivity (0.5-0.9)
    
    Returns:
        List of pothole detections with bounding boxes and confidence
    """
    frame = cv2.imread(image_path)
    if frame is None:
        return []
    return detect_potholes_enhanced(frame, sensitivity)

def detect_potholes_enhanced(frame, sensitivity=0.7):
    """
    Enhanced pothole detection that uses multiple approaches for better detection
    
    Args:
        frame: OpenCV frame
        sensitivity: Detection sensitivity (0.5-0.9)
    
    Returns:
        List of pothole detections with bounding boxes and confidence
    """
    height, width = frame.shape[:2]
    
    # Less restrictive road area - expand detection zone
    road_top = int(height * 0.30)    # Start from 30% down (was 60%)
    road_bottom = int(height * 0.95) # Go to 95% (was 80%)
    road_left = int(width * 0.05)    # 5% from left edge (was 20%)
    road_right = int(width * 0.95)   # 95% from left edge (was 80%)
    
    # Extract road area
    road_area = frame[road_top:road_bottom, road_left:road_right]
    
    if road_area.size == 0:
        return []
    
    # Convert to different color spaces for better detection
    gray = cv2.cvtColor(road_area, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(road_area, cv2.COLOR_BGR2LAB)
    hsv = cv2.cvtColor(road_area, cv2.COLOR_BGR2HSV)
    
    # Method 1: Traditional edge-based detection
    potholes_edges = detect_by_edges(gray, road_left, road_top, sensitivity)
    
    # Method 2: Dark region detection
    potholes_dark = detect_dark_regions(gray, road_left, road_top, sensitivity)
    
    # Method 3: Texture analysis
    potholes_texture = detect_by_texture(gray, road_left, road_top, sensitivity)
    
    # Method 4: Color-based detection (darker patches)
    potholes_color = detect_by_color(lab, road_left, road_top, sensitivity)
    
    # Combine all detections
    all_potholes = potholes_edges + potholes_dark + potholes_texture + potholes_color
    
    # Remove duplicates and merge nearby detections
    merged_potholes = merge_detections(all_potholes)
    
    # Sort by confidence
    merged_potholes.sort(key=lambda x: x['confidence'], reverse=True)
    
    return merged_potholes[:10]  # Return top 10 detections

def detect_by_edges(gray, offset_x, offset_y, sensitivity):
    """Detect potholes using edge detection"""
    potholes = []
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection with Canny
    edges = cv2.Canny(blurred, 50, 150)
    
    # Morphological operations to close gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # More flexible size constraints
        if area < 50 or area > 5000:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        
        # Adjust coordinates
        x += offset_x
        y += offset_y
        
        # Less restrictive shape constraints
        aspect_ratio = w / h if h > 0 else 0
        if 0.3 <= aspect_ratio <= 3.0 and w >= 8 and h >= 8:
            confidence = min(0.3 + (1000 - area) / 2000, 0.8)  # Size-based confidence
            confidence = max(confidence * (1.0 - (1.0 - sensitivity)), 0.1)
            
            potholes.append({
                'bbox': (x, y, w, h),
                'area': area,
                'confidence': confidence,
                'method': 'edges',
                'contour': contour.tolist()
            })
    
    return potholes

def detect_dark_regions(gray, offset_x, offset_y, sensitivity):
    """Detect potholes as dark regions"""
    potholes = []
    
    # Create a local adaptive threshold
    adaptive_thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                          cv2.THRESH_BINARY_INV, 15, 5)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(adaptive_thresh, cv2.MORPH_OPEN, kernel)
    
    # Find contours
    contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < 30 or area > 8000:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        x += offset_x
        y += offset_y
        
        # Check if region is actually dark
        roi = gray[max(0, y-offset_y):min(gray.shape[0], y-offset_y+h), 
                  max(0, x-offset_x):min(gray.shape[1], x-offset_x+w)]
        
        if roi.size > 0:
            mean_intensity = np.mean(roi)
            overall_mean = np.mean(gray)
            
            # Dark regions should be significantly darker than average
            if mean_intensity < overall_mean * 0.8:
                confidence = min(0.4 + (overall_mean - mean_intensity) / overall_mean, 0.9)
                confidence = max(confidence * (1.0 - (1.0 - sensitivity)), 0.1)
                
                potholes.append({
                    'bbox': (x, y, w, h),
                    'area': area,
                    'confidence': confidence,
                    'method': 'dark_regions',
                    'mean_intensity': mean_intensity,
                    'contour': contour.tolist()
                })
    
    return potholes

def detect_by_texture(gray, offset_x, offset_y, sensitivity):
    """Detect potholes using texture analysis"""
    potholes = []
    
    # Use Laplacian for texture detection
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    
    # Threshold to find areas with low texture (smooth areas like water in potholes)
    _, thresh = cv2.threshold(laplacian, 30, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < 100 or area > 10000:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        x += offset_x
        y += offset_y
        
        confidence = min(0.5 + area / 5000, 0.8)
        confidence = max(confidence * (1.0 - (1.0 - sensitivity)), 0.1)
        
        potholes.append({
            'bbox': (x, y, w, h),
            'area': area,
            'confidence': confidence,
            'method': 'texture',
            'contour': contour.tolist()
        })
    
    return potholes

def detect_by_color(lab, offset_x, offset_y, sensitivity):
    """Detect potholes using color analysis in LAB space"""
    potholes = []
    
    l_channel = lab[:, :, 0]  # Lightness channel
    
    # Find very dark areas
    _, dark_mask = cv2.threshold(l_channel, 80, 255, cv2.THRESH_BINARY_INV)
    
    # Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cleaned = cv2.morphologyEx(dark_mask, cv2.MORPH_OPEN, kernel)
    
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        if area < 80 or area > 15000:
            continue
            
        x, y, w, h = cv2.boundingRect(contour)
        x += offset_x
        y += offset_y
        
        confidence = min(0.4 + area / 8000, 0.7)
        confidence = max(confidence * (1.0 - (1.0 - sensitivity)), 0.1)
        
        potholes.append({
            'bbox': (x, y, w, h),
            'area': area,
            'confidence': confidence,
            'method': 'color',
            'contour': contour.tolist()
        })
    
    return potholes

def merge_detections(detections, overlap_threshold=0.3):
    """Merge overlapping detections"""
    if not detections:
        return []
    
    # Sort by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)
    
    merged = []
    used = set()
    
    for i, det in enumerate(detections):
        if i in used:
            continue
            
        # Find overlapping detections
        overlapping = [det]
        used.add(i)
        
        for j, other_det in enumerate(detections[i+1:], i+1):
            if j in used:
                continue
                
            if calculate_overlap(det['bbox'], other_det['bbox']) > overlap_threshold:
                overlapping.append(other_det)
                used.add(j)
        
        # Merge overlapping detections
        if len(overlapping) == 1:
            merged.append(overlapping[0])
        else:
            merged_det = merge_multiple_detections(overlapping)
            merged.append(merged_det)
    
    return merged

def calculate_overlap(bbox1, bbox2):
    """Calculate overlap ratio between two bounding boxes"""
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    
    # Calculate intersection
    x_left = max(x1, x2)
    y_top = max(y1, y2)
    x_right = min(x1 + w1, x2 + w2)
    y_bottom = min(y1 + h1, y2 + h2)
    
    if x_right < x_left or y_bottom < y_top:
        return 0.0
    
    intersection = (x_right - x_left) * (y_bottom - y_top)
    area1 = w1 * h1
    area2 = w2 * h2
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0

def merge_multiple_detections(detections):
    """Merge multiple overlapping detections into one"""
    # Find bounding box that encompasses all
    min_x = min(det['bbox'][0] for det in detections)
    min_y = min(det['bbox'][1] for det in detections)
    max_x = max(det['bbox'][0] + det['bbox'][2] for det in detections)
    max_y = max(det['bbox'][1] + det['bbox'][3] for det in detections)
    
    merged_bbox = (min_x, min_y, max_x - min_x, max_y - min_y)
    merged_area = sum(det['area'] for det in detections) / len(detections)
    merged_confidence = max(det['confidence'] for det in detections)
    
    methods = list(set(det['method'] for det in detections))
    
    return {
        'bbox': merged_bbox,
        'area': merged_area,
        'confidence': min(merged_confidence * 1.2, 1.0),  # Boost confidence for multiple detections
        'method': '+'.join(methods),
        'merged_count': len(detections)
    }
