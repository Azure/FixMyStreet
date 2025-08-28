#!/usr/bin/env python3
"""
Main entry point for the Pothole Detection API
"""

import sys
import os

# Add the project root to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

if __name__ == '__main__':
    from backend.app.api import app
    app.run(host='0.0.0.0', port=5000, debug=True)
