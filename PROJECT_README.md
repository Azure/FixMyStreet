# Pothole Detection System

A comprehensive pothole detection system with REST API backend and web frontend.

## 🏗️ Project Structure

```
FixMyStreet/
├── backend/                       # Flask API backend
│   ├── app/
│   │   ├── __init__.py
│   │   └── api.py                # Main Flask application
│   ├── detection/                # Detection algorithms
│   │   ├── __init__.py
│   │   ├── detect_potholes.py
│   │   ├── detect_potholes_improved.py
│   │   └── enhanced_pothole_detection.py
│   ├── utils/                    # Utility functions
│   │   ├── __init__.py
│   │   └── extract_location.py
│   └── requirements.txt          # Python dependencies
├── frontend/                     # Web UI
│   ├── index.html               # Main web interface
│   ├── serve.py                 # Development server
│   ├── README.md               # Original frontend docs
│   └── README_NEW.md           # Updated frontend documentation
├── docker/                      # Docker configurations
│   ├── Dockerfile              # Basic Dockerfile
│   ├── Dockerfile.alpine       # Alpine-based image
│   ├── Dockerfile.minimal      # Minimal configuration
│   ├── Dockerfile.opencv       # Main production Dockerfile with OpenCV
│   ├── Dockerfile.simple       # Simple configuration
│   └── Dockerfile.working      # Working development version
├── datasets/                    # Sample data for testing
│   ├── 1.jpg, 2.jpg           # Test images
│   └── video1.mp4, video2.mp4  # Test videos
├── results/                     # API output results (gitignored)
├── uploads/                     # Temporary upload storage (gitignored)
├── logs/                        # Application logs (gitignored)
├── copilot-instructions.md      # AI development guidelines
├── deploy.bat                   # Windows deployment script
├── docker-compose.yml          # Production configuration
├── docker-compose.dev.yml      # Development configuration
├── PROJECT_README.md           # This file - detailed project docs
├── README.md                   # Main project documentation
├── SECURITY.md                 # Security guidelines
└── run.py                      # Local development entry point
```

## 🚀 Quick Start

### Development Mode (Recommended)

1. **Start development environment:**
   ```bash
   docker-compose -f docker-compose.yml -f docker-compose.dev.yml up --build
   ```
   
   This enables:
   - Live code reloading
   - Volume mounts for instant changes
   - Debug mode enabled

2. **Make changes to backend code:**
   - Edit files in `backend/` directory
   - Changes are immediately reflected in the running container
   - No need to rebuild for code changes

### Production Mode

1. **Build and run:**
   ```bash
   docker-compose up --build -d
   ```

### Local Development (without Docker)

1. **Install dependencies:**
   ```bash
   cd backend
   pip install -r requirements.txt
   ```

2. **Run locally:**
   ```bash
   python ../run.py
   ```

## 📡 API Endpoints

### Health Check
```bash
GET /
```

### Image Detection
```bash
POST /api/detect/image
```
**Parameters:**
- `file`: Image file (jpg, png, etc.)
- `sensitivity`: Detection sensitivity (0.5-0.9, default: 0.7)

**Example:**
```bash
curl -X POST -F "file=@road_image.jpg" -F "sensitivity=0.8" http://localhost:5000/api/detect/image
```

### Video Detection
```bash
POST /api/detect/video
```
**Parameters:**
- `file`: Video file (mp4, avi, etc.)
- `sensitivity`: Detection sensitivity (0.5-0.9, default: 0.7)
- `interval`: Frame sampling interval (default: 30)

### Get Results
```bash
GET /api/results/{result_id}
```

### Download Results
```bash
GET /api/download/{result_id}
```

## 🔧 Development Workflow

### Making Backend Changes

1. **Edit code in `backend/` directory**
2. **Changes are automatically reloaded** (in development mode)
3. **Test the API** using curl or your preferred tool

### Adding New Features

1. **Detection algorithms**: Add to `backend/detection/`
2. **Utility functions**: Add to `backend/utils/`
3. **API endpoints**: Modify `backend/app/api.py`

### Building for Production

1. **Update requirements.txt** if you added new dependencies
2. **Test with production config:**
   ```bash
   docker-compose down
   docker-compose up --build
   ```

## 🎨 Frontend Development (Future)

The `frontend/` directory is prepared for UI development with options for:

- **Simple Web UI**: HTML + CSS + JavaScript
- **Flask Templates**: Server-side rendered with Jinja2
- **React/Vue SPA**: Modern single-page application
- **Streamlit**: Python-based rapid prototyping

See `frontend/README.md` for detailed plans.

## 🐳 Docker Configuration

### Files
- `docker-compose.yml`: Production configuration
- `docker-compose.dev.yml`: Development overrides
- `docker/Dockerfile.opencv`: Main Dockerfile with OpenCV

### Environment Variables
- `FLASK_ENV`: `development` or `production`
- `FLASK_DEBUG`: Enable Flask debug mode
- `PYTHONPATH`: Python module search path

## 📝 API Response Format

### Success Response
```json
{
  "success": true,
  "data": {
    "result_id": "uuid-string",
    "detections": [...],
    "summary": {...}
  },
  "timestamp": "2025-08-27T10:30:00.000Z"
}
```

### Error Response
```json
{
  "success": false,
  "error": "Error message",
  "timestamp": "2025-08-27T10:30:00.000Z"
}
```

## 🤝 Contributing

1. Make changes in the appropriate directory (`backend/` for API, `frontend/` for UI)
2. Test in development mode first
3. Ensure production build works
4. Update documentation as needed

## 📊 Sample Usage

```bash
# Test with sample image
curl -X POST \
  -F "file=@datasets/1.jpg" \
  -F "sensitivity=0.7" \
  http://localhost:5000/api/detect/image

# Check health
curl http://localhost:5000/

# Get results
curl http://localhost:5000/api/results/{result_id}
```
