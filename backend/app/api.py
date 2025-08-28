#!/usr/bin/env python3
"""
Pothole Detection Microservice
REST API for detecting potholes in videos and images
"""

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os
import tempfile
import shutil
import uuid
import json
import cv2
import numpy as np
from pathlib import Path
import subprocess
import sys
from datetime import datetime
import zipfile
import io

# Import our refactored detection modules
from backend.detection import PotholeVideoProcessor, detect_potholes_in_frame, detect_potholes_enhanced_from_path
from backend.utils.extract_location import extract_gps_from_video_overlay, extract_gps_from_image, extract_with_exiftool

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max file size

# Configure upload folders
UPLOAD_FOLDER = '/tmp/uploads'
RESULTS_FOLDER = '/tmp/results'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Allowed file extensions
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv', 'flv', 'webm'}
ALLOWED_IMAGE_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'webp'}

def allowed_file(filename, allowed_extensions):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions

def create_error_response(message, status_code=400):
    """Create standardized error response"""
    return jsonify({
        'success': False,
        'error': message,
        'timestamp': datetime.utcnow().isoformat()
    }), status_code

def create_success_response(data):
    """Create standardized success response"""
    return jsonify({
        'success': True,
        'data': data,
        'timestamp': datetime.utcnow().isoformat()
    })

def detect_potholes_in_image(image_path, sensitivity=0.7):
    """
    Detect potholes in a single image using enhanced detection algorithm
    
    Args:
        image_path: Path to image file
        sensitivity: Detection sensitivity (0.5-0.9)
    
    Returns:
        List of detected potholes with confidence scores and shape metrics
    """
    try:
        # Use the enhanced detection algorithm
        detections = detect_potholes_enhanced_from_path(image_path, sensitivity=sensitivity)
        print(f"Raw detections: {len(detections)}")
        print(f"First detection sample: {detections[0] if detections else 'None'}")
        
        # Convert to the format expected by the API
        potholes = []
        for i, detection in enumerate(detections):
            print(f"Processing detection {i+1}: {type(detection)} - keys: {detection.keys()}")
            # Handle bbox tuple format (x, y, w, h)
            bbox = detection['bbox']
            print(f"bbox type: {type(bbox)}, value: {bbox}")
            
            x, y, w, h = bbox
            area = detection['area']
            
            # Calculate shape metrics that are expected by the API
            aspect_ratio = w / h if h > 0 else 0
            extent = area / (w * h) if (w * h) > 0 else 0
            
            # For missing metrics, provide reasonable defaults
            # These would normally be calculated from the contour
            circularity = 0.5  # Default value
            convexity = 0.8    # Default value
            solidity = 0.7     # Default value
            
            pothole = {
                'id': i + 1,
                'bbox': bbox,  # Keep bbox format for annotation
                'x': int(x),
                'y': int(y),
                'width': int(w),
                'height': int(h),
                'confidence': round(detection['confidence'], 3),
                'area': int(area),
                'method': detection['method'],
                'timestamp': 0.0,  # For image, timestamp is 0
                'aspect_ratio': round(aspect_ratio, 3),
                'extent': round(extent, 3),
                'circularity': round(circularity, 3),
                'convexity': round(convexity, 3),
                'solidity': round(solidity, 3)
            }
            potholes.append(pothole)
        
        print(f"Enhanced detection found {len(potholes)} potholes in image")
        return potholes
        
    except Exception as e:
        print(f"Error in enhanced detection: {e}")
        import traceback
        traceback.print_exc()
        return []

def process_video(video_path, sensitivity=0.7, interval=30):
    """
    Process video for pothole detection using the refactored video processor
    
    Args:
        video_path: Path to video file
        sensitivity: Detection sensitivity
        interval: Frame sampling interval
    
    Returns:
        Dict with detection results
    """
    # Generate unique output directory
    result_id = str(uuid.uuid4())
    output_dir = os.path.join(RESULTS_FOLDER, result_id)
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # Use the new refactored video processor
        processor = PotholeVideoProcessor("enhanced", sensitivity)
        detection_result = processor.process_video(video_path, interval, extract_gps=True)
        
        # Save annotated frames for the first 10 detections
        annotated_frames = []
        if detection_result.detections:
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                saved_frames = set()
                for detection in detection_result.detections[:10]:  # Limit to 10 frames
                    frame_num = detection['frame']
                    if frame_num not in saved_frames:
                        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
                        ret, frame = cap.read()
                        if ret:
                            # Draw bounding box
                            bbox = detection['bbox']
                            x, y, w, h = bbox['x'], bbox['y'], bbox['width'], bbox['height']
                            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                            cv2.putText(frame, f"Pothole {detection['confidence']:.2f}", 
                                       (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            
                            frame_filename = os.path.join(output_dir, f"pothole_frame_{frame_num:06d}.jpg")
                            cv2.imwrite(frame_filename, frame)
                            annotated_frames.append(f"pothole_frame_{frame_num:06d}.jpg")
                            saved_frames.add(frame_num)
                cap.release()
        
        # Format result for backward compatibility
        result = {
            'result_id': result_id,
            'video_info': detection_result.video_info,
            'detection_params': {
                'sensitivity': sensitivity,
                'sample_interval': interval
            },
            'summary': detection_result.summary,
            'detections': detection_result.detections,
            'annotated_frames': annotated_frames
        }
        
        # Save detailed results
        results_file = os.path.join(output_dir, 'results.json')
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        return result
        
    except Exception as e:
        # Clean up on error
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)
        raise e

@app.route('/', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return create_success_response({
        'service': 'Pothole Detection API',
        'version': '1.0.0',
        'status': 'healthy',
        'endpoints': {
            'video': '/api/detect/video',
            'image_detect': '/api/detect/image',
            'results': '/api/results/<result_id>',
            'image_view': '/api/image/<result_id>',
            'download': '/api/download/<result_id>'
        }
    })

@app.route('/api/detect/video', methods=['POST'])
def detect_potholes_video():
    """
    Detect potholes in uploaded video
    
    Body:
    - file: Video file (mp4, avi, mov, etc.)
    - sensitivity: Detection sensitivity (0.5-0.9, default: 0.7)
    - interval: Frame sampling interval (default: 30)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return create_error_response('No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return create_error_response('No file selected')
        
        if not allowed_file(file.filename, ALLOWED_VIDEO_EXTENSIONS):
            return create_error_response(f'Invalid file type. Allowed: {", ".join(ALLOWED_VIDEO_EXTENSIONS)}')
        
        # Get parameters
        sensitivity = float(request.form.get('sensitivity', 0.7))
        interval = int(request.form.get('interval', 30))
        
        # Validate parameters
        if not 0.5 <= sensitivity <= 0.9:
            return create_error_response('Sensitivity must be between 0.5 and 0.9')
        
        if not 1 <= interval <= 300:
            return create_error_response('Interval must be between 1 and 300 frames')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        file.save(upload_path)
        
        try:
            # Process video
            result = process_video(upload_path, sensitivity, interval)
            
            # Clean up uploaded file
            os.remove(upload_path)
            
            return create_success_response(result)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(upload_path):
                os.remove(upload_path)
            raise e
            
    except Exception as e:
        return create_error_response(f'Processing failed: {str(e)}', 500)

@app.route('/api/detect/image', methods=['POST'])
def detect_potholes_image():
    """
    Detect potholes in uploaded image
    
    Body:
    - file: Image file (jpg, png, etc.)
    - sensitivity: Detection sensitivity (0.5-0.9, default: 0.7)
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return create_error_response('No file uploaded')
        
        file = request.files['file']
        if file.filename == '':
            return create_error_response('No file selected')
        
        if not allowed_file(file.filename, ALLOWED_IMAGE_EXTENSIONS):
            return create_error_response(f'Invalid file type. Allowed: {", ".join(ALLOWED_IMAGE_EXTENSIONS)}')
        
        # Get parameters
        sensitivity = float(request.form.get('sensitivity', 0.7))
        
        # Validate parameters
        if not 0.5 <= sensitivity <= 0.9:
            return create_error_response('Sensitivity must be between 0.5 and 0.9')
        
        # Save uploaded file
        filename = secure_filename(file.filename)
        file_id = str(uuid.uuid4())
        upload_path = os.path.join(UPLOAD_FOLDER, f"{file_id}_{filename}")
        file.save(upload_path)
        
        try:
            # Extract GPS coordinates from image EXIF data
            print("Extracting GPS coordinates from image...")
            image_gps = extract_gps_from_image(upload_path)
            
            # Temporary fallback for testing if no GPS found
            if not image_gps:
                print("No GPS data found in image")
                image_gps = None
            
            # Process image
            potholes = detect_potholes_in_image(upload_path, sensitivity)
            
            # Create annotated image if potholes found
            annotated_image_path = None
            if potholes:
                frame = cv2.imread(upload_path)
                for pothole in potholes:
                    # Use bbox format for consistency
                    if 'bbox' in pothole:
                        x, y, w, h = pothole['bbox']
                    else:
                        x, y, w, h = pothole['x'], pothole['y'], pothole['width'], pothole['height']
                    
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(frame, f"Pothole {pothole['confidence']:.2f}", 
                               (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                
                # Save annotated image
                result_id = str(uuid.uuid4())
                output_dir = os.path.join(RESULTS_FOLDER, result_id)
                os.makedirs(output_dir, exist_ok=True)
                annotated_image_path = os.path.join(output_dir, 'annotated_image.jpg')
                cv2.imwrite(annotated_image_path, frame)
            
            # Prepare response
            detections = []
            for pothole in potholes:
                detection = {
                    'confidence': pothole['confidence'],
                    'bbox': {
                        'x': pothole['x'],
                        'y': pothole['y'],
                        'width': pothole['width'],
                        'height': pothole['height']
                    },
                    'area_pixels': pothole['area'],
                    'metrics': {
                        'aspect_ratio': pothole['aspect_ratio'],
                        'extent': pothole['extent'],
                        'circularity': pothole['circularity'],
                        'convexity': pothole['convexity'],
                        'solidity': pothole['solidity']
                    }
                }
                
                # Add GPS location if available
                if image_gps:
                    detection['location'] = {
                        'lat': image_gps['lat'],
                        'lon': image_gps['lon']
                    }
                else:
                    detection['location'] = None
                    
                detections.append(detection)
            
            result = {
                'result_id': result_id if potholes else None,
                'image_info': {
                    'filename': filename,
                    'size': os.path.getsize(upload_path)
                },
                'detection_params': {
                    'sensitivity': sensitivity
                },
                'summary': {
                    'total_potholes': len(potholes),
                    'avg_confidence': sum(p['confidence'] for p in potholes) / len(potholes) if potholes else 0,
                    'gps_coverage': 1 if image_gps and potholes else 0
                },
                'detections': detections,
                'annotated_image_available': annotated_image_path is not None
            }
            
            # Save results.json for image detection if potholes were found
            if potholes and result_id:
                results_file = os.path.join(output_dir, 'results.json')
                with open(results_file, 'w') as f:
                    json.dump(result, f, indent=2)
            
            # Clean up uploaded file
            os.remove(upload_path)
            
            return create_success_response(result)
            
        except Exception as e:
            # Clean up on error
            if os.path.exists(upload_path):
                os.remove(upload_path)
            raise e
            
    except Exception as e:
        return create_error_response(f'Processing failed: {str(e)}', 500)

@app.route('/api/results/<result_id>', methods=['GET'])
def get_results(result_id):
    """Get detailed results for a specific result ID"""
    try:
        results_file = os.path.join(RESULTS_FOLDER, result_id, 'results.json')
        if not os.path.exists(results_file):
            return create_error_response('Result not found', 404)
        
        with open(results_file, 'r') as f:
            result = json.load(f)
        
        return create_success_response(result)
        
    except Exception as e:
        return create_error_response(f'Failed to retrieve results: {str(e)}', 500)

@app.route('/api/download/<result_id>', methods=['GET'])
def download_results(result_id):
    """Download all results as a ZIP file"""
    try:
        result_dir = os.path.join(RESULTS_FOLDER, result_id)
        if not os.path.exists(result_dir):
            return create_error_response('Result not found', 404)
        
        # Create ZIP file in memory
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for root, dirs, files in os.walk(result_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    arc_name = os.path.relpath(file_path, result_dir)
                    zip_file.write(file_path, arc_name)
        
        zip_buffer.seek(0)
        
        return send_file(
            io.BytesIO(zip_buffer.read()),
            mimetype='application/zip',
            as_attachment=True,
            download_name=f'pothole_results_{result_id}.zip'
        )
        
    except Exception as e:
        return create_error_response(f'Failed to download results: {str(e)}', 500)

@app.route('/api/image/<result_id>', methods=['GET'])
def get_annotated_image(result_id):
    """Get the annotated image for a specific result ID"""
    try:
        # Check for annotated_image.jpg first (image detection)
        image_path = os.path.join(RESULTS_FOLDER, result_id, 'annotated_image.jpg')
        if os.path.exists(image_path):
            return send_file(image_path, mimetype='image/jpeg')
        
        # Check for pothole frame files (video detection)
        result_dir = os.path.join(RESULTS_FOLDER, result_id)
        if os.path.exists(result_dir):
            # Find the first pothole frame image
            for file in os.listdir(result_dir):
                if file.startswith('pothole_frame_') and file.endswith('.jpg'):
                    image_path = os.path.join(result_dir, file)
                    return send_file(image_path, mimetype='image/jpeg')
        
        return create_error_response('Annotated image not found', 404)
        
    except Exception as e:
        return create_error_response(f'Failed to retrieve image: {str(e)}', 500)

@app.errorhandler(413)
def file_too_large(e):
    return create_error_response('File too large. Maximum size: 100MB', 413)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
