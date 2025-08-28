#!/usr/bin/env python3
"""
Improved pothole detection system with enhanced detection algorithms
"""

import cv2
import numpy as np
import argparse
import json
import csv
import os
from pathlib import Path
import subprocess
import sys
from backend.utils.extract_location import extract_gps_from_video_overlay

def detect_potholes_in_frame(frame, frame_number, fps, sensitivity=0.7):
    """
    Enhanced pothole detection using multiple detection methods
    
    Args:
        frame: OpenCV frame
        frame_number: Current frame number
        fps: Video FPS
        sensitivity: Detection sensitivity (0.5-0.9, higher = more strict)
    
    Returns:
        List of pothole detections with bounding boxes and confidence
    """
    # Use the enhanced detection algorithm
    from backend.detection.enhanced_pothole_detection import detect_potholes_enhanced
    
    try:
        detections = detect_potholes_enhanced(frame, sensitivity)
        
        # Add frame information to each detection
        for detection in detections:
            detection['frame_number'] = frame_number
            detection['timestamp'] = frame_number / fps if fps > 0 else 0
            
        return detections
    
    except Exception as e:
        print(f"Enhanced detection failed, falling back to original: {e}")
        # Fallback to original detection if enhanced fails
        return detect_potholes_in_frame_original(frame, frame_number, fps, sensitivity)

def detect_potholes_in_frame_original(frame, frame_number, fps, sensitivity=0.7):
    """
    Original pothole detection method (kept as fallback)
    
    Args:
        frame: OpenCV frame
        frame_number: Current frame number
        fps: Video FPS
        sensitivity: Detection sensitivity (0.5-0.9, higher = more strict)
    
    Returns:
        List of pothole detections with bounding boxes and confidence
    """
    height, width = frame.shape[:2]
    
    # Much more restrictive road area to exclude dashboard/UI
    road_top = int(height * 0.60)    # Start from 60% down (was 40%)
    road_bottom = int(height * 0.80) # Stop at 80% (exclude bottom dashboard)
    road_left = int(width * 0.20)    # 20% from left edge
    road_right = int(width * 0.80)   # 80% from left edge
    
    # Extract only the actual road surface
    road_area = frame[road_top:road_bottom, road_left:road_right]
    
    if road_area.size == 0:
        return []
    
    # Convert to grayscale
    gray = cv2.cvtColor(road_area, cv2.COLOR_BGR2GRAY)
    
    # Use a more sophisticated approach for road surface analysis
    # 1. Apply bilateral filter to preserve edges while reducing noise
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Calculate local mean to detect depressions (darker areas)
    kernel_size = 15
    kernel = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size * kernel_size)
    local_mean = cv2.filter2D(filtered, -1, kernel)
    
    # 3. Find areas that are darker than local average (potential potholes)
    diff = local_mean.astype(np.float32) - filtered.astype(np.float32)
    
    # 4. Threshold to find significant depressions
    _, thresh = cv2.threshold(diff, 15, 255, cv2.THRESH_BINARY)
    thresh = thresh.astype(np.uint8)
    
    # 5. Apply morphological operations to clean up
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    opened = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel_small)
    
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel_large)
    
    # Find contours
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potholes = []
    
    for contour in contours:
        area = cv2.contourArea(contour)
        
        # Realistic pothole size constraints
        if area < 200 or area > 2000:
            continue
        
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Adjust coordinates back to full frame
        x += road_left
        y += road_top
        
        # Realistic pothole shape constraints
        aspect_ratio = w / h
        rect_area = w * h
        extent = area / rect_area if rect_area > 0 else 0
        
        # Potholes should be reasonably compact and not too elongated
        if (0.5 <= aspect_ratio <= 2.0 and    # Reasonable aspect ratio
            0.4 <= extent <= 0.8 and          # Well-filled shape
            w >= 15 and h >= 15 and           # Minimum meaningful size
            w <= 80 and h <= 80):             # Maximum reasonable size
            
            # Calculate shape quality metrics
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter) if perimeter > 0 else 0
            
            # Calculate convexity
            hull = cv2.convexHull(contour)
            hull_area = cv2.contourArea(hull)
            convexity = area / hull_area if hull_area > 0 else 0
            
            # Calculate solidity (how solid the shape is)
            solidity = area / hull_area if hull_area > 0 else 0
            
            # Real potholes have specific characteristics:
            # - Reasonably circular (circularity > 0.4)
            # - Fairly convex (convexity > 0.75)
            # - Solid shape (solidity > 0.75)
            if (circularity > 0.4 and 
                convexity > 0.75 and 
                solidity > 0.75):
                
                # Calculate confidence based on how "pothole-like" the shape is
                circularity_score = min(circularity * 2, 1.0)
                convexity_score = convexity
                solidity_score = solidity
                extent_score = extent
                
                confidence = (circularity_score * 0.4 + 
                            convexity_score * 0.3 + 
                            solidity_score * 0.2 + 
                            extent_score * 0.1)
                
                # Apply sensitivity threshold
                if confidence >= sensitivity:
                    potholes.append({
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': min(confidence, 1.0),
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'circularity': circularity,
                        'convexity': convexity,
                        'solidity': solidity,
                        'contour': contour.tolist()
                    })
    
    # Sort by confidence (highest first)
    potholes.sort(key=lambda x: x['confidence'], reverse=True)
    
    # Limit to top 3 detections per frame to avoid false positives
    return potholes[:3]

def run_gps_extraction(video_path):
    """Run GPS extraction using the existing extract_location.py script"""
    try:
        result = subprocess.run([
            sys.executable, 'extract_location.py', video_path
        ], capture_output=True, text=True, cwd=Path(__file__).parent)
        
        if result.returncode == 0:
            print("GPS extraction completed successfully")
        else:
            print(f"GPS extraction warning: {result.stderr}")
        
        # Load GPS data
        csv_path = Path(video_path).stem + "_route.csv"
        if Path(csv_path).exists():
            gps_data = []
            with open(csv_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    gps_data.append({
                        'timestamp': float(row['timestamp']),
                        'latitude': float(row['latitude']),
                        'longitude': float(row['longitude'])
                    })
            return gps_data
        
    except Exception as e:
        print(f"GPS extraction failed: {e}")
    
    return []

def get_gps_for_frame(frame_number, fps, gps_data):
    """Get GPS coordinates for a specific frame"""
    if not gps_data:
        return None, None
    
    video_time = frame_number / fps
    
    # Find the closest GPS point
    closest_point = min(gps_data, key=lambda x: abs(x['timestamp'] - video_time))
    
    # Only use GPS if within 5 seconds
    if abs(closest_point['timestamp'] - video_time) <= 5.0:
        return closest_point['latitude'], closest_point['longitude']
    
    return None, None

def main():
    parser = argparse.ArgumentParser(description="Detect potholes in dashcam video")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--sensitivity", type=float, default=0.8, 
                       help="Detection sensitivity (0.5-0.9, higher = more strict)")
    parser.add_argument("--interval", type=int, default=10, 
                       help="Frame sampling interval (process every N frames)")
    parser.add_argument("--output-dir", default="potholes_output", 
                       help="Output directory for results")
    
    args = parser.parse_args()
    
    # Validate sensitivity
    if not 0.5 <= args.sensitivity <= 0.9:
        print("Error: Sensitivity must be between 0.5 and 0.9")
        return
    
    video_path = args.video
    if not Path(video_path).exists():
        print(f"Error: Video file not found: {video_path}")
        return
    
    print(f"Starting improved pothole detection on: {video_path}")
    print(f"Sensitivity: {args.sensitivity}, Sample interval: {args.interval} frames")
    
    # Step 1: Extract GPS coordinates
    print("1. Extracting GPS coordinates...")
    gps_data = run_gps_extraction(video_path)
    if gps_data:
        print(f"Loaded {len(gps_data)} GPS points from {Path(video_path).stem}_route.csv")
    else:
        print("No GPS data available")
    
    # Step 2: Open video
    print("2. Opening video for pothole detection...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    
    print(f"Video: {fps:.1f} fps, {total_frames} frames, {duration:.1f}s duration")
    print(f"Processing every {args.interval} frames (~{args.interval/fps:.1f}s intervals)")
    
    # Step 3: Process frames
    print("3. Detecting potholes...")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    all_detections = []
    frame_number = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every N frames
        if frame_number % args.interval == 0:
            potholes = detect_potholes_in_frame(frame, frame_number, fps, args.sensitivity)
            
            for pothole in potholes:
                # Get GPS coordinates for this frame
                lat, lon = get_gps_for_frame(frame_number, fps, gps_data)
                
                # Calculate timestamp
                timestamp_seconds = frame_number / fps
                timestamp_str = f"{int(timestamp_seconds//60):02d}:{int(timestamp_seconds%60):02d}"
                
                detection = {
                    'frame': frame_number,
                    'timestamp': timestamp_str,
                    'video_time_seconds': timestamp_seconds,
                    'confidence': pothole['confidence'],
                    'bbox_x': pothole['bbox'][0],
                    'bbox_y': pothole['bbox'][1], 
                    'bbox_width': pothole['bbox'][2],
                    'bbox_height': pothole['bbox'][3],
                    'area_pixels': pothole['area'],
                    'latitude': lat,
                    'longitude': lon
                }
                
                all_detections.append(detection)
                
                print(f"Frame {frame_number}: Found pothole at confidence {pothole['confidence']:.2f}")
                if lat and lon:
                    print(f"  GPS: {lat:.6f}, {lon:.6f}")
                else:
                    print(f"  GPS: Not available")
                
                # Save annotated frame
                annotated_frame = frame.copy()
                x, y, w, h = pothole['bbox']
                cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(annotated_frame, f"Pothole {pothole['confidence']:.2f}", 
                           (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                frame_filename = f"{args.output_dir}/pothole_frame_{frame_number:06d}.jpg"
                cv2.imwrite(frame_filename, annotated_frame)
            
            processed_frames += 1
            if processed_frames % 50 == 0:
                progress = frame_number / total_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")
        
        frame_number += 1
    
    cap.release()
    
    # Step 4: Save results
    print("4. Saving results...")
    
    # Save CSV
    csv_file = f"{args.output_dir}/potholes_detected.csv"
    if all_detections:
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=all_detections[0].keys())
            writer.writeheader()
            writer.writerows(all_detections)
    
    # Save JSON with metadata
    json_data = {
        'video_file': video_path,
        'detection_params': {
            'sensitivity': args.sensitivity,
            'sample_interval': args.interval,
            'fps': fps
        },
        'total_detections': len(all_detections),
        'detections': all_detections
    }
    
    json_file = f"{args.output_dir}/potholes_detected.json"
    with open(json_file, 'w') as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Total potholes detected: {len(all_detections)}")
    print(f"Results saved to:")
    print(f"  CSV: {csv_file}")
    print(f"  JSON: {json_file}")
    if all_detections:
        print(f"  Annotated frames: {args.output_dir}/pothole_frame_*.jpg")
    
    # Summary stats
    with_gps = [d for d in all_detections if d['latitude'] is not None]
    print(f"Potholes with GPS coordinates: {len(with_gps)}")
    
    if with_gps:
        print("Sample locations:")
        for i, detection in enumerate(with_gps[:5]):
            print(f"  {i+1}. Frame {detection['frame']}: {detection['latitude']:.6f}, {detection['longitude']:.6f}")

if __name__ == "__main__":
    main()
