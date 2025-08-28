# Pothole Detection Module - Refactored

## Overview

The pothole detection system has been refactored for better maintainability, code reuse, and extensibility. The new architecture follows object-oriented principles and provides clear separation of concerns.

## Architecture

### Core Components

1. **Detection Algorithms** (`detection_algorithms.py`)
   - `PotholeDetector` - Abstract base class for all detectors
   - `EdgeBasedDetector` - Edge detection using Canny algorithm
   - `DarkRegionDetector` - Dark region analysis for shadow detection
   - `EnhancedDetector` - Combines multiple detection methods
   - `DetectorFactory` - Factory pattern for creating detector instances

2. **Video Processing** (`video_processor.py`)
   - `PotholeVideoProcessor` - Main class for video processing
   - `VideoDetectionResult` - Result container with GPS integration
   - GPS extraction and frame-to-location mapping

3. **Data Models**
   - `Detection` - Data class for individual pothole detections
   - Standardized bbox format: `{'x': int, 'y': int, 'width': int, 'height': int}`
   - Comprehensive metrics: aspect_ratio, circularity, convexity, etc.

## Usage Examples

### Basic Video Processing

```python
from backend.detection import PotholeVideoProcessor

# Create processor with enhanced detection
processor = PotholeVideoProcessor("enhanced", sensitivity=0.7)

# Process video with GPS extraction
result = processor.process_video("dashcam_video.mp4", sample_interval=30)

# Access results
print(f"Found {result.summary['total_potholes']} potholes")
print(f"GPS coverage: {result.summary['gps_coverage']}%")

for detection in result.detections:
    print(f"Frame {detection['frame']}: {detection['confidence']:.2f} confidence")
    if detection['location']:
        print(f"  Location: {detection['location']['lat']}, {detection['location']['lon']}")
```

### Custom Detection Algorithm

```python
from backend.detection import DetectorFactory

# Create specific detector types
edge_detector = DetectorFactory.create_detector("edge", sensitivity=0.8)
dark_detector = DetectorFactory.create_detector("dark", sensitivity=0.6)
enhanced_detector = DetectorFactory.create_detector("enhanced", sensitivity=0.7)

# Process single frame
import cv2
frame = cv2.imread("pothole_image.jpg")
detections = enhanced_detector.detect(frame)

for detection in detections:
    print(f"Pothole at {detection.bbox} with confidence {detection.confidence}")
```

### Legacy Compatibility

```python
# The old function signatures are maintained for backward compatibility
from backend.detection import detect_potholes_in_frame, detect_potholes_enhanced_from_path

# These work exactly as before
detections = detect_potholes_in_frame(frame, frame_number, fps, sensitivity)
image_detections = detect_potholes_enhanced_from_path("image.jpg", sensitivity)
```

## Algorithm Details

### EdgeBasedDetector
- Uses Canny edge detection with adaptive thresholds
- Morphological operations for noise reduction
- Shape-based filtering for pothole characteristics
- Confidence calculation based on size, extent, and circularity

### DarkRegionDetector
- Identifies dark regions that may indicate potholes
- Adaptive thresholding based on local image statistics
- Focuses on road surface shadows and depressions
- Optimized for varying lighting conditions

### EnhancedDetector
- Combines results from multiple detection algorithms
- Merges overlapping detections to reduce duplicates
- Weighted confidence scoring
- Best overall performance for diverse road conditions

## Configuration

### Sensitivity Settings
- **0.5-0.6**: High sensitivity, more detections (may include false positives)
- **0.7**: Balanced setting (recommended for most use cases)
- **0.8-0.9**: Low sensitivity, fewer but more confident detections

### Sample Intervals
- **15-30 frames**: Detailed analysis, slower processing
- **30-60 frames**: Balanced performance (recommended)
- **60+ frames**: Fast processing, may miss smaller potholes

## Performance Optimizations

1. **Road Area Focusing**: Detection limited to road surface (30%-95% of frame height)
2. **Efficient Contour Processing**: Pre-filtering by area and aspect ratio
3. **GPU Acceleration**: OpenCV operations utilize available GPU acceleration
4. **Memory Management**: Proper cleanup and resource management

## Integration with GPS

The video processor automatically:
1. Extracts GPS data using ExifTool (embedded metadata)
2. Falls back to OCR-based overlay extraction
3. Maps GPS coordinates to video timestamps
4. Associates location data with each detection

## Migration from Old System

### Files Renamed/Replaced
- `detect_potholes.py` → `detect_potholes.py.backup`
- `detect_potholes_improved.py` → `detect_potholes_improved.py.backup`
- `enhanced_pothole_detection.py` → `enhanced_pothole_detection.py.backup`

### New Structure
```
backend/detection/
├── __init__.py                 # Clean module interface
├── detection_algorithms.py    # Core detection algorithms
├── video_processor.py         # Video processing logic
└── *.py.backup                # Original files (preserved)
```

### API Changes
The main API (`api.py`) now uses:
- `PotholeVideoProcessor` for video processing
- Cleaner, more maintainable code
- Better error handling and resource management
- Preserved backward compatibility

## Benefits of Refactoring

1. **Code Reusability**: Modular design allows easy algorithm swapping
2. **Maintainability**: Clear separation of concerns and standardized interfaces
3. **Extensibility**: Easy to add new detection algorithms
4. **Testing**: Better unit testing capabilities with isolated components
5. **Performance**: Optimized processing pipeline
6. **Documentation**: Comprehensive code documentation and type hints

## Future Enhancements

The refactored architecture supports:
- Machine learning-based detection algorithms
- Real-time video processing
- Advanced GPS tracking features
- Custom detection parameter tuning
- Multi-threaded processing
- Cloud-based detection services
