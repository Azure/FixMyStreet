import argparse
import csv
import os
import sys
import subprocess
import cv2
import numpy as np
from datetime import datetime
import json

def detect_potholes_in_frame(frame, frame_number, fps, sensitivity=0.5):
    """
    Detect potholes in a single frame using computer vision techniques.
    
    Args:
        frame: OpenCV frame
        frame_number: Current frame number
        fps: Video FPS
        sensitivity: Detection sensitivity (0.1-1.0, lower = more sensitive)
    
    Returns:
        List of pothole detections with bounding boxes and confidence
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection using Canny with lower thresholds for more sensitivity
    edges = cv2.Canny(blurred, 30, 100)
    
    # Morphological operations to close gaps in edges
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
    
    # Additional dilation to connect nearby edges
    dilated = cv2.dilate(closed, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    potholes = []
    height, width = frame.shape[:2]
    
    # Focus on road area (bottom 60% of frame, excluding edges)
    road_top = int(height * 0.4)
    road_left = int(width * 0.1)
    road_right = int(width * 0.9)
    
    for contour in contours:
        # Calculate contour properties
        area = cv2.contourArea(contour)
        if area < 200:  # Lower threshold for more sensitivity
            continue
            
        # Get bounding rectangle
        x, y, w, h = cv2.boundingRect(contour)
        
        # Check if contour is in road area
        if y < road_top or x < road_left or x + w > road_right:
            continue
            
        # Calculate aspect ratio and extent
        aspect_ratio = w / h
        rect_area = w * h
        extent = area / rect_area
        
        # More lenient criteria for real road conditions
        # Potholes can be irregular shapes, not just circular
        if (0.2 <= aspect_ratio <= 5.0 and 
            0.1 <= extent <= 0.9 and 
            w > 15 and h > 15 and
            area > 200):
            
            # Calculate confidence based on multiple factors
            perimeter = cv2.arcLength(contour, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter * perimeter)
                
                # Weighted confidence calculation
                # Give more weight to size and extent, less to perfect circularity
                size_factor = min(area / 1000, 1.0)  # Normalize by 1000 pixels
                extent_factor = extent
                circularity_factor = min(circularity * 2, 1.0)  # Double circularity weight
                
                # Combined confidence with weights
                confidence = (size_factor * 0.3 + extent_factor * 0.5 + circularity_factor * 0.2)
                
                # Boost confidence for shapes that look more like potholes
                if 0.5 <= aspect_ratio <= 2.0 and area > 300:
                    confidence *= 1.2
                
                confidence = min(confidence, 1.0)
                
                if confidence > sensitivity:
                    timestamp = frame_number / fps
                    potholes.append({
                        'frame': frame_number,
                        'timestamp': f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                        'bbox': (x, y, w, h),
                        'area': area,
                        'confidence': confidence,
                        'aspect_ratio': aspect_ratio,
                        'extent': extent,
                        'circularity': circularity,
                        'contour': contour.tolist()  # For drawing later if needed
                    })
    
    return potholes

def extract_gps_for_frame(gps_data, frame_number, fps):
    """
    Find the closest GPS coordinate for a given frame.
    
    Args:
        gps_data: List of GPS points from extract_location.py
        frame_number: Current frame number
        fps: Video FPS
    
    Returns:
        GPS coordinates (lat, lon) or (None, None) if not found
    """
    if not gps_data:
        return None, None
    
    frame_time_seconds = frame_number / fps
    
    # Find closest GPS point by time
    best_match = None
    min_time_diff = float('inf')
    
    for gps_point in gps_data:
        # Parse timestamp from GPS data (format: "MM:SS")
        try:
            if ':' in gps_point['timestamp']:
                time_parts = gps_point['timestamp'].split(':')
                gps_time_seconds = int(time_parts[0]) * 60 + int(time_parts[1])
            else:
                continue
                
            time_diff = abs(frame_time_seconds - gps_time_seconds)
            if time_diff < min_time_diff:
                min_time_diff = time_diff
                best_match = gps_point
        except:
            continue
    
    if best_match and min_time_diff < 5:  # Within 5 seconds
        return best_match['lat'], best_match['lon']
    
    return None, None

def run_gps_extraction(video_path):
    """
    Run the extract_location.py script to get GPS data.
    
    Args:
        video_path: Path to the video file
    
    Returns:
        List of GPS points or empty list if extraction fails
    """
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        extract_script = os.path.join(script_dir, "extract_location.py")
        
        if not os.path.exists(extract_script):
            print(f"Error: extract_location.py not found at {extract_script}")
            return []
        
        # Run the GPS extraction script
        result = subprocess.run([
            sys.executable, extract_script, video_path
        ], capture_output=True, text=True, cwd=script_dir)
        
        if result.returncode != 0:
            print(f"GPS extraction failed: {result.stderr}")
            return []
        
        # Read the generated CSV file
        base_name = os.path.splitext(video_path)[0]
        csv_path = f"{base_name}_route.csv"
        
        if not os.path.exists(csv_path):
            print(f"GPS CSV file not found: {csv_path}")
            return []
        
        gps_data = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row['latitude'] and row['longitude']:
                    gps_data.append({
                        'timestamp': row['timestamp'],
                        'lat': float(row['latitude']),
                        'lon': float(row['longitude']),
                        'speed': float(row['speed_mps']) if row['speed_mps'] else None
                    })
        
        print(f"Loaded {len(gps_data)} GPS points from {csv_path}")
        return gps_data
        
    except Exception as e:
        print(f"Error extracting GPS data: {e}")
        return []

def detect_potholes_in_video(video_path, output_dir="potholes_output", sensitivity=0.5, sample_interval=15):
    """
    Detect potholes in video and correlate with GPS coordinates.
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save results
        sensitivity: Pothole detection sensitivity (0.1-1.0)
        sample_interval: Process every N frames (default: 15 = ~0.5s at 30fps)
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Starting pothole detection on: {video_path}")
    print(f"Sensitivity: {sensitivity}, Sample interval: {sample_interval} frames")
    
    # Extract GPS data first
    print("\n1. Extracting GPS coordinates...")
    gps_data = run_gps_extraction(video_path)
    
    # Open video
    print("\n2. Opening video for pothole detection...")
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file: {video_path}")
        return
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {fps:.1f} fps, {total_frames} frames, {duration:.1f}s duration")
    print(f"Processing every {sample_interval} frames (~{sample_interval/fps:.1f}s intervals)")
    
    # Process video frames
    print("\n3. Detecting potholes...")
    all_detections = []
    frame_number = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_number % sample_interval == 0:
            # Detect potholes in current frame
            potholes = detect_potholes_in_frame(frame, frame_number, fps, sensitivity)
            
            if potholes:
                # Get GPS coordinates for this frame
                lat, lon = extract_gps_for_frame(gps_data, frame_number, fps)
                
                for pothole in potholes:
                    pothole['latitude'] = lat
                    pothole['longitude'] = lon
                    pothole['video_time'] = frame_number / fps
                    all_detections.append(pothole)
                    
                    print(f"Frame {frame_number}: Found pothole at confidence {pothole['confidence']:.2f}")
                    if lat and lon:
                        print(f"  GPS: {lat:.6f}, {lon:.6f}")
                    else:
                        print(f"  GPS: Not available")
                
                # Save frame with detections
                annotated_frame = frame.copy()
                for pothole in potholes:
                    x, y, w, h = pothole['bbox']
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, f"Pothole {pothole['confidence']:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                
                frame_filename = os.path.join(output_dir, f"pothole_frame_{frame_number:06d}.jpg")
                cv2.imwrite(frame_filename, annotated_frame)
            
            processed_frames += 1
            
            # Progress indicator
            if processed_frames % 50 == 0:
                progress = frame_number / total_frames * 100
                print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")
        
        frame_number += 1
    
    cap.release()
    
    # Save results
    print(f"\n4. Saving results...")
    print(f"Total potholes detected: {len(all_detections)}")
    
    if all_detections:
        # Save as CSV
        csv_filename = os.path.join(output_dir, "potholes_detected.csv")
        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['frame', 'timestamp', 'video_time_seconds', 'confidence', 
                           'bbox_x', 'bbox_y', 'bbox_width', 'bbox_height', 
                           'area_pixels', 'latitude', 'longitude'])
            
            for detection in all_detections:
                x, y, w, h = detection['bbox']
                writer.writerow([
                    detection['frame'],
                    detection['timestamp'],
                    detection['video_time'],
                    detection['confidence'],
                    x, y, w, h,
                    detection['area'],
                    detection['latitude'] or '',
                    detection['longitude'] or ''
                ])
        
        # Save as JSON for more detailed data
        json_filename = os.path.join(output_dir, "potholes_detected.json")
        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump({
                'video_file': video_path,
                'detection_params': {
                    'sensitivity': sensitivity,
                    'sample_interval': sample_interval,
                    'fps': fps
                },
                'total_detections': len(all_detections),
                'detections': all_detections
            }, f, indent=2)
        
        print(f"Results saved to:")
        print(f"  CSV: {csv_filename}")
        print(f"  JSON: {json_filename}")
        print(f"  Annotated frames: {output_dir}/pothole_frame_*.jpg")
        
        # Summary by GPS location
        gps_potholes = [d for d in all_detections if d['latitude'] and d['longitude']]
        if gps_potholes:
            print(f"\nPotholes with GPS coordinates: {len(gps_potholes)}")
            print("Sample locations:")
            for i, detection in enumerate(gps_potholes[:5]):
                print(f"  {i+1}. Frame {detection['frame']}: {detection['latitude']:.6f}, {detection['longitude']:.6f}")
        
    else:
        print("No potholes detected in the video.")

def main():
    parser = argparse.ArgumentParser(description="Detect potholes in dashcam video with GPS correlation")
    parser.add_argument("video", help="Path to dashcam video file")
    parser.add_argument("-o", "--output", default="potholes_output", 
                       help="Output directory for results (default: potholes_output)")
    parser.add_argument("-s", "--sensitivity", type=float, default=0.5, 
                       help="Detection sensitivity 0.1-1.0 (default: 0.5, lower = more sensitive)")
    parser.add_argument("-i", "--interval", type=int, default=15,
                       help="Process every N frames (default: 15)")
    
    args = parser.parse_args()
    
    # Validate sensitivity
    if not 0.1 <= args.sensitivity <= 1.0:
        print("Error: Sensitivity must be between 0.1 and 1.0")
        sys.exit(1)
    
    detect_potholes_in_video(args.video, args.output, args.sensitivity, args.interval)

if __name__ == "__main__":
    main()
