@echo off
REM Pothole Detection API Deployment Script for Windows

echo Pothole Detection API Deployment
echo ====================================

REM Check if Docker is installed
docker --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker is not installed. Please install Docker Desktop first.
    echo    Visit: https://docs.docker.com/desktop/windows/
    pause
    exit /b 1
)

REM Check if Docker Compose is installed
docker-compose --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Docker Compose is not installed. Please install Docker Compose.
    echo    Usually comes with Docker Desktop.
    pause
    exit /b 1
)

echo Docker and Docker Compose are available

REM Create necessary directories
echo Creating directories...
if not exist "results" mkdir results
if not exist "uploads" mkdir uploads
if not exist "logs" mkdir logs

echo Building and starting the service...
docker-compose up --build -d

REM Wait a moment for the service to start
echo Waiting for service to start...
timeout /t 10 >nul

REM Check if service is running
curl -f http://localhost:5000/ >nul 2>&1
if %errorlevel% equ 0 (
    echo Service is running successfully!
    echo.
    echo API Endpoints:
    echo    Health Check: http://localhost:5000/
    echo    Image API:    http://localhost:5000/api/detect/image
    echo    Video API:    http://localhost:5000/api/detect/video
    echo.
    echo View logs: docker-compose logs -f
    echo Stop service: docker-compose down
    echo Run tests: python test_api.py
    echo.
    echo Quick Test Commands:
    echo    curl http://localhost:5000/
    echo    curl -X POST -F "file=@image.jpg" http://localhost:5000/api/detect/image
    echo    curl -X POST -F "file=@video.mp4" http://localhost:5000/api/detect/video
) else (
    echo Service failed to start. Check logs:
    echo    docker-compose logs
    pause
    exit /b 1
)

pause
