import argparse
import csv
import os
import re
import subprocess
import sys
from datetime import datetime
import shutil

try:
    import cv2
    import pytesseract
    
    # Set Tesseract path for Windows
    tesseract_paths = [
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe", 
        r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.environ.get('USERNAME', '')),
        r"C:\tesseract\tesseract.exe"
    ]
    
    for path in tesseract_paths:
        if os.path.exists(path):
            pytesseract.pytesseract.tesseract_cmd = path
            break
    
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False

# Optional parsers for sidecar files
def parse_gpx(gpx_path):
    import gpxpy
    points = []
    with open(gpx_path, "r", encoding="utf-8") as f:
        gpx = gpxpy.parse(f)
    for trk in gpx.tracks:
        for seg in trk.segments:
            for p in seg.points:
                points.append({
                    "timestamp": p.time.isoformat() if p.time else "",
                    "lat": p.latitude,
                    "lon": p.longitude,
                    "speed": getattr(p, "speed", None)
                })
    return points

def parse_nmea(nmea_path):
    import pynmea2
    points = []
    with open(nmea_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line or not line.startswith("$"):
                continue
            try:
                msg = pynmea2.parse(line)
            except Exception:
                continue
            # Prefer RMC for lat/lon/time/speed
            if msg.sentence_type == "RMC" and hasattr(msg, "latitude") and hasattr(msg, "longitude"):
                ts = ""
                try:
                    # Combine date+time from RMC into ISO if available
                    if getattr(msg, "datestamp", None) and getattr(msg, "timestamp", None):
                        dt = datetime.combine(msg.datestamp, msg.timestamp)
                        ts = dt.isoformat()
                except Exception:
                    pass
                spd = None
                try:
                    # speed over ground in knots -> m/s or km/h if you prefer
                    if getattr(msg, "spd_over_grnd", None) is not None:
                        spd = float(msg.spd_over_grnd) * 0.514444  # knots to m/s
                except Exception:
                    pass
                points.append({
                    "timestamp": ts,
                    "lat": float(msg.latitude),
                    "lon": float(msg.longitude),
                    "speed": spd
                })
    return points

def find_exiftool():
    """Locate ExifTool executable.
    Order: env vars -> PATH -> known installation locations.
    """
    # 1. Environment variables
    for key in ("EXIFTOOL", "EXIFTOOL_PATH"):
        p = os.environ.get(key)
        if p and os.path.exists(p):
            return p

    # 2. On PATH
    p = shutil.which("exiftool")
    if p:
        return p

    # 3. Common installation locations
    script_dir = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        # User's main installation first
        r"F:\exiftool-13.33_64\exiftool.exe",
        r"F:\exiftool-13.33_64\exiftool(-k).exe",
        # Common locations
        r"C:\exiftool\exiftool.exe",
        r"C:\exiftool\exiftool(-k).exe",
        r"C:\Program Files\exiftool\exiftool.exe",
        r"C:\Program Files (x86)\exiftool\exiftool.exe",
        # Local to this script (last resort)
        os.path.join(script_dir, "exiftool-13.33_64", "exiftool.exe"),
        os.path.join(script_dir, "exiftool-13.33_64", "exiftool(-k).exe"),
        os.path.join(script_dir, "exiftool.exe"),
        os.path.join(script_dir, "exiftool(-k).exe"),
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None

def exiftool_available():
    try:
        exe = find_exiftool()
        if not exe:
            return False
        subprocess.run([exe, "-ver"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except Exception:
        return False

def extract_with_exiftool(video_path):
    """
    Use ExifTool to extract GPS track samples (-ee).
    Outputs a list of dicts with timestamp, lat, lon, speed.
    """
    exe = find_exiftool()
    if not exe:
        print("ExifTool not found. Install exiftool or set EXIFTOOL/EXIFTOOL_PATH environment variable.")
        return []

    print(f"Using ExifTool: {exe}")

    # Ask for numeric values (-n) and extended embedded metadata (-ee), CSV for easy parsing.
    # Common columns: "GPSLatitude","GPSLongitude","GPSDateTime","GPSSpeed" (names can vary).
    cmd = [
        exe,
        "-ee",
        "-n",
        "-csv",
        "-api", "largefilesupport=1",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSDateTime",
        "-GPSSpeed",
        video_path
    ]
    try:
        print(f"Running command: {' '.join(cmd)}")
        print("Processing video... This may take a while for large files.")
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=5)  # 5 minute timeout
        print(f"ExifTool completed. Output length: {len(proc.stdout)} chars")
    except subprocess.TimeoutExpired:
        print("ExifTool timed out after 5 seconds. Try with a smaller video file.")
        return []
    except subprocess.CalledProcessError as e:
        print(f"ExifTool failed with return code {e.returncode}")
        print(f"Error output: {e.stderr.strip() or e.stdout.strip()}")
        return []

    lines = proc.stdout.splitlines()
    print(f"Got {len(lines)} lines of output")
    if not lines:
        print("No output from ExifTool")
        return []

    print(f"First few lines of output:")
    for i, line in enumerate(lines[:3]):
        print(f"  {i}: {line}")

    # Parse CSV header
    header = [h.strip() for h in lines[0].split(",")]
    col_idx = {name: i for i, name in enumerate(header)}
    
    print(f"CSV columns found: {header}")
    
    # Check if we have GPS columns
    gps_cols = ["GPSLatitude", "GPSLongitude", "GPSDateTime", "GPSSpeed"]
    found_gps_cols = [col for col in gps_cols if col in header]
    
    if not found_gps_cols:
        print("No GPS columns found in ExifTool output. This video doesn't contain embedded GPS data.")
        return []
    
    print(f"Found GPS columns: {found_gps_cols}")

    def get(row, key):
        i = col_idx.get(key)
        return row[i].strip() if (i is not None and i < len(row)) else ""

    points = []
    for line in lines[1:]:
        row = [c.strip() for c in split_csv_line(line)]
        lat = get(row, "GPSLatitude")
        lon = get(row, "GPSLongitude")
        dt = get(row, "GPSDateTime")
        sp = get(row, "GPSSpeed")

        if not lat or not lon:
            continue
        try:
            latf = float(lat)
            lonf = float(lon)
        except ValueError:
            continue

        # Normalize timestamp to ISO if present (ExifTool often outputs "YYYY:MM:DD HH:MM:SSZ")
        ts = normalize_exiftool_dt(dt)
        spf = None
        try:
            spf = float(sp) if sp else None
        except ValueError:
            spf = None

        points.append({
            "timestamp": ts,
            "lat": latf,
            "lon": lonf,
            "speed": spf
        })
    return points

def split_csv_line(line):
    # Minimal CSV splitter for simple ExifTool output (no embedded commas expected in our fields)
    # If needed, use Python csv module, but we’re avoiding locale quirks.
    return [p for p in re.split(r',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)]

def normalize_exiftool_dt(s):
    if not s:
        return ""
    s = s.strip().replace("/", "-")
    # Convert "YYYY:MM:DD HH:MM:SSZ" -> "YYYY-MM-DDTHH:MM:SSZ"
    s = re.sub(r"^(\d{4}):(\d{2}):(\d{2})\s+(\d{2}:\d{2}:\d{2})(Z?)$", r"\1-\2-\3T\4\5", s)
    return s

def find_sidecar(video_path):
    base, _ = os.path.splitext(video_path)
    candidates = [
        base + ".gpx",
        base + ".GPX",
        base + ".nmea",
        base + ".NMEA",
        base + ".gps",
        base + ".GPS",
        base + ".txt",   # some dashcams emit NMEA in .txt
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    return None

def write_csv(points, out_csv):
    if not points:
        print("No points to write.")
        return
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "latitude", "longitude", "speed_mps"])
        for p in points:
            w.writerow([p.get("timestamp", ""), p["lat"], p["lon"], p.get("speed")])

def extract_gps_from_image(image_path):
    """
    Extract GPS coordinates from image EXIF data.
    Args:
        image_path: Path to the image file
    Returns:
        dict with lat, lon if found, None otherwise
    """
    exe = find_exiftool()
    if not exe:
        print("ExifTool not found. Install exiftool or set EXIFTOOL/EXIFTOOL_PATH environment variable.")
        return None

    cmd = [
        exe,
        "-n",  # Numeric GPS coordinates
        "-csv",
        "-GPSLatitude",
        "-GPSLongitude",
        "-GPSDateTime",
        image_path
    ]
    
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, check=True, timeout=30)
        lines = proc.stdout.splitlines()
        
        if len(lines) < 2:  # Need header + data
            return None
            
        # Parse CSV header to find column indices
        header = [c.strip() for c in lines[0].split(',')]
        data = [c.strip() for c in lines[1].split(',')]
        
        lat_idx = next((i for i, h in enumerate(header) if 'GPSLatitude' in h), None)
        lon_idx = next((i for i, h in enumerate(header) if 'GPSLongitude' in h), None)
        
        if lat_idx is None or lon_idx is None or lat_idx >= len(data) or lon_idx >= len(data):
            return None
            
        lat = data[lat_idx]
        lon = data[lon_idx]
        
        if not lat or not lon or lat == '-' or lon == '-':
            return None
            
        try:
            lat_f = float(lat)
            lon_f = float(lon)
            return {
                "lat": lat_f,
                "lon": lon_f
            }
        except ValueError:
            return None
            
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired, Exception) as e:
        print(f"GPS extraction from image failed: {e}")
        return None

def extract_gps_from_video_overlay(video_path, sample_interval=30):
    """
    Extract GPS coordinates from video overlay text using OCR.
    Args:
        video_path: Path to the video file
        sample_interval: Extract GPS data every N frames (default: 30 = ~1 second at 30fps)
    """
    if not OPENCV_AVAILABLE:
        print("OpenCV and pytesseract not available. Install with: pip install opencv-python pytesseract")
        print("Also install Tesseract OCR: https://github.com/UB-Mannheim/tesseract/wiki")
        return []
    
    print(f"Extracting GPS from video overlay using OCR...")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}")
        return []
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video: {fps:.1f} fps, {total_frames} frames, {duration:.1f}s duration")
    print(f"Sampling every {sample_interval} frames (~{sample_interval/fps:.1f}s intervals)")
    
    points = []
    frame_number = 0
    
    # Pattern to match GPS coordinates in various formats
    # Examples: "N12.345678 E123.456789", "12.345678N 123.456789E", "12°34'56.78"N 123°45'67.89"E"
    gps_patterns = [
        # OCR-corrupted format: N12.9227-£77-6874 or N12.8227-E7 7 (more flexible)
        r'[NnSs](\d{1,3})[.\-_](\d+)[\-_][£EeWw](\d{1,3})[\-_\s]?(\d+)',
        # Standard decimal degrees: N12.345678 E123.456789 or 12.345678N 123.456789E
        r'([NS]?\s*\d{1,3}\.\d+)\s*([NS]?)\s+([EW]?\s*\d{1,3}\.\d+)\s*([EW]?)',
        # Degrees minutes seconds: 12°34'56.78"N 123°45'67.89"E
        r'(\d{1,3})°(\d{1,2})\'([\d.]+)"([NS])\s+(\d{1,3})°(\d{1,2})\'([\d.]+)"([EW])',
        # Degrees minutes: 12°34.5678'N 123°45.6789'E  
        r'(\d{1,3})°([\d.]+)\'([NS])\s+(\d{1,3})°([\d.]+)\'([EW])',
    ]
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame_number % sample_interval == 0:
            # Focus on bottom portion of frame where GPS overlay is typically located
            height, width = frame.shape[:2]
            bottom_region = frame[int(height * 0.7):, :]  # Bottom 30% of frame
            
            # Convert to grayscale and enhance contrast for better OCR
            gray = cv2.cvtColor(bottom_region, cv2.COLOR_BGR2GRAY)
            # Threshold to make text more readable
            _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            try:
                # Extract text from the frame
                text = pytesseract.image_to_string(thresh, config='--psm 6')
                
                # Debug: Print OCR text for first few frames
                if frame_number <= 90:  # First 3 samples
                    print(f"Frame {frame_number} OCR text: '{text.strip()}'")
                
                # Try to find GPS coordinates in the extracted text
                for pattern in gps_patterns:
                    matches = re.findall(pattern, text, re.IGNORECASE)
                    for match in matches:
                        try:
                            lat, lon = parse_gps_match(match, pattern)
                            if lat and lon:
                                timestamp = frame_number / fps  # Time in seconds
                                points.append({
                                    "timestamp": f"{int(timestamp//60):02d}:{int(timestamp%60):02d}",
                                    "lat": lat,
                                    "lon": lon,
                                    "speed": None
                                })
                                print(f"Frame {frame_number}: Found GPS {lat:.6f}, {lon:.6f}")
                                break
                        except Exception as e:
                            continue
                            
            except Exception as e:
                print(f"OCR error at frame {frame_number}: {e}")
                
        frame_number += 1
        
        # Progress indicator
        if frame_number % (sample_interval * 30) == 0:  # Every ~30 samples
            progress = frame_number / total_frames * 100
            print(f"Progress: {progress:.1f}% ({frame_number}/{total_frames} frames)")
    
    cap.release()
    return points

def parse_gps_match(match, pattern_used):
    """Parse GPS coordinates from regex match based on the pattern used."""
    try:
        if len(match) == 4 and all(match[i].isdigit() for i in [0, 1, 2, 3]):  # OCR corrupted format
            lat_deg, lat_dec, lon_deg, lon_dec = match
            
            # Reconstruct decimal coordinates
            lat_val = float(f"{lat_deg}.{lat_dec}")
            lon_val = float(f"{lon_deg}.{lon_dec}")
            
            # Validate coordinates are reasonable (for India region)
            if lat_val < 8 or lat_val > 37 or lon_val < 68 or lon_val > 97:
                # If longitude seems truncated (like 7.7 instead of 77.x), try to fix it
                if lon_val < 10 and lat_val > 10:
                    lon_val = float(f"7{lon_deg}.{lon_dec}")  # Assume 77.x for Bangalore area
            
            return lat_val, lon_val
            
        elif len(match) == 4:  # Standard decimal degrees pattern
            lat_str, lat_dir, lon_str, lon_dir = match
            
            # Clean up the strings
            lat_val = float(re.sub(r'[^\d.]', '', lat_str))
            lon_val = float(re.sub(r'[^\d.]', '', lon_str))
            
            # Determine directions
            lat_dir = lat_dir or ('S' if 'S' in lat_str.upper() else 'N')
            lon_dir = lon_dir or ('W' if 'W' in lon_str.upper() else 'E')
            
            # Apply direction
            if lat_dir.upper() == 'S':
                lat_val = -lat_val
            if lon_dir.upper() == 'W':
                lon_val = -lon_val
                
            return lat_val, lon_val
            
        elif len(match) == 8:  # Degrees minutes seconds pattern
            lat_deg, lat_min, lat_sec, lat_dir, lon_deg, lon_min, lon_sec, lon_dir = match
            
            lat_val = float(lat_deg) + float(lat_min)/60 + float(lat_sec)/3600
            lon_val = float(lon_deg) + float(lon_min)/60 + float(lon_sec)/3600
            
            if lat_dir.upper() == 'S':
                lat_val = -lat_val
            if lon_dir.upper() == 'W':
                lon_val = -lon_val
                
            return lat_val, lon_val
            
        elif len(match) == 6:  # Degrees minutes pattern
            lat_deg, lat_min, lat_dir, lon_deg, lon_min, lon_dir = match
            
            lat_val = float(lat_deg) + float(lat_min)/60
            lon_val = float(lon_deg) + float(lon_min)/60
            
            if lat_dir.upper() == 'S':
                lat_val = -lat_val
            if lon_dir.upper() == 'W':
                lon_val = -lon_val
                
            return lat_val, lon_val
            
    except Exception as e:
        pass
        
    return None, None

def main():
    ap = argparse.ArgumentParser(description="Extract GPS (lat,lon) from dashcam video or sidecar.")
    ap.add_argument(
        "input",
        nargs="?",
        default="video1.mp4",
        help="Path to dashcam video (default: video1.mp4 in current directory). Also supports sidecar files (.gpx/.nmea/.gps)",
    )
    ap.add_argument("-o", "--output", help="Output CSV path (default: <input_basename>_route.csv)")
    args = ap.parse_args()

    src = args.input
    if not os.path.exists(src):
        print(f"Input not found: {src}")
        sys.exit(1)

    out_csv = args.output or (os.path.splitext(src)[0] + "_route.csv")

    points = []
    # If input itself is a sidecar, parse directly
    ext = os.path.splitext(src)[1].lower()
    try:
        if ext in [".gpx"]:
            points = parse_gpx(src)
        elif ext in [".nmea", ".gps", ".txt"]:
            points = parse_nmea(src)
        else:
            # Try sidecar next to the video
            sidecar = find_sidecar(src)
            if sidecar:
                if sidecar.lower().endswith(".gpx"):
                    points = parse_gpx(sidecar)
                else:
                    points = parse_nmea(sidecar)
            if not points:
                # Try OCR extraction from video overlay first
                if OPENCV_AVAILABLE:
                    points = extract_gps_from_video_overlay(src)
                else:
                    print("OpenCV not available for OCR. Install with: pip install opencv-python pytesseract")
                
                # Fallback to ExifTool from the video metadata if OCR didn't work
                if not points:
                    print("Trying ExifTool as fallback...")
                    points = extract_with_exiftool(src)
    except ImportError as e:
        print(f"Missing package: {e}. Install gpxpy and pynmea2 if using sidecar files.")
    except Exception as e:
        print(f"Failed to parse: {e}")

    if not points:
        print("No GPS data found.")
        sys.exit(2)

    write_csv(points, out_csv)
    print(f"Wrote {len(points)} points to: {out_csv}")

    # Print a preview
    preview = points[:5]
    print("Sample points:")
    for p in preview:
        print(p)

if __name__ == "__main__":
    main()