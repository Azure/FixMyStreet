# Copilot Instructions – Python API in Docker

## Context
This repo is a Python project exposing REST APIs for detecting potholes in videos and images using computer vision and deployed in Docker. 

## Features

- **Video Processing**: Detect potholes in dashcam videos with GPS correlation
- **Image Processing**: Detect potholes in single images
- **GPS Integration**: Extract GPS coordinates from video overlays
- **Confidence Scoring**: Provides confidence levels for each detection
- **Annotated Results**: Generate annotated images showing detected potholes
- **Containerized**: Easy deployment using Docker


## Rules

### Code & Structure
- Follow PEP 8 and use type hints.
- Structure: app/api/, app/services/, app/models/, tests/.
- Separate API routes from business logic.
- Use FastAPI with Pydantic for request/response validation.

### Config & Security
- Never hardcode secrets; load from .env or Docker secrets.
- Use separate configs for dev/staging/prod.
- Follow 12-factor app principles.

### Docker
- Multi-stage builds with python:3.x-slim.
- Use a non-root user in containers.
- Pin dependency versions in requirements.txt.
- Use `.dockerignore` to exclude caches, __pycache__, and dev files.
- Keep logs in stdout/stderr for Docker logging.

### API & Deployment
- RESTful endpoints with proper status codes.
- Add OpenAPI docs via FastAPI auto-gen.
- Use gunicorn with uvicorn workers for prod, not `flask run`.
- Healthcheck in Dockerfile or compose.
- Keep containers stateless (DB/cache external).

### Testing & CI
- Write pytest unit and integration tests.
- Lint with black/ruff and check types with mypy.
- Run tests in CI before deployment.

### Observability
- Add structured JSON logging.
- Include metrics and error monitoring.

## Output Expectation
When generating code, Copilot should follow these rules by default.
