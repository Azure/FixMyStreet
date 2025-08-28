# Pothole Detection Frontend

A modern, responsive web interface for the Pothole Detection API.

## Features

- **File Upload**: Support for images (JPG, PNG) and videos (MP4, AVI, MOV, etc.)
- **Real-time Processing**: Live feedback during detection processing
- **Results Visualization**: 
  - Summary statistics with confidence scores
  - Annotated image preview for image uploads
  - Download links for processed results
- **Responsive Design**: Works on desktop and mobile devices
- **Error Handling**: User-friendly error messages and status updates

## Quick Start

### Option 1: Python Server (Recommended)
```bash
# Start the frontend server
cd frontend
python serve.py
```

The frontend will be available at `http://localhost:3000` and will automatically open in your browser.

### Option 2: Direct File Access
Simply open `index.html` in your web browser. Note that some features may be limited due to CORS restrictions.

## Usage

1. **Start Backend**: Ensure the backend API is running on `http://localhost:5000`
2. **Start Frontend**: Run the frontend server as described above
3. **Upload File**: Click "Choose File" and select an image or video
4. **Detect Potholes**: Click "Detect Potholes" to process the file
5. **View Results**: See detection statistics and download annotated results

## API Integration

The frontend communicates with the backend API at `http://localhost:5000`:

- `GET /` - API health check
- `POST /api/detect/image` - Image pothole detection
- `POST /api/detect/video` - Video pothole detection  
- `GET /api/results/<id>` - Get annotated image
- `GET /api/download/<id>` - Download result file

## File Support

### Images
- JPG, JPEG, PNG formats
- Any resolution (automatically processed)

### Videos  
- MP4, AVI, MOV, MKV, WMV, FLV, WebM formats
- Processes every 30th frame by default
- GPS extraction support if available in video metadata

## Development

The frontend is built with vanilla HTML, CSS, and JavaScript for simplicity and performance.

### Key Components

- **index.html**: Main application interface
- **serve.py**: Development server with CORS support
- **CSS**: Modern responsive design with gradients and animations
- **JavaScript**: Async API communication and dynamic UI updates

### Customization

- Modify colors in the CSS gradient definitions
- Adjust upload file types in the `accept` attribute
- Change API endpoints in the JavaScript fetch calls
- Update styling for different branding requirements

## Troubleshooting

### Common Issues

1. **API Connection Error**: Ensure backend is running on localhost:5000
2. **File Upload Fails**: Check file format and size limits
3. **CORS Issues**: Use the Python server instead of direct file access
4. **Slow Processing**: Large videos may take several minutes to process

### Browser Compatibility

- Chrome 60+
- Firefox 55+  
- Safari 12+
- Edge 79+

## Future Enhancements

- Real-time video streaming detection
- Batch processing for multiple files
- Interactive map for GPS-enabled results
- User authentication and result history
- Advanced filtering and search capabilities
