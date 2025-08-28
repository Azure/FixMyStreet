#!/usr/bin/env python3
"""
Simple HTTP server to serve the frontend
"""

import http.server
import socketserver
import os
import webbrowser
from pathlib import Path

PORT = 3000
FRONTEND_DIR = Path(__file__).parent

class CORSHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

def serve_frontend():
    """Start the frontend server"""
    os.chdir(FRONTEND_DIR)
    
    with socketserver.TCPServer(("", PORT), CORSHTTPRequestHandler) as httpd:
        print(f"Frontend server running at http://localhost:{PORT}")
        print(f"Serving from: {FRONTEND_DIR}")
        print("Press Ctrl+C to stop the server")
        
        # Open browser
        webbrowser.open(f"http://localhost:{PORT}")
        
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nShutting down frontend server...")
            httpd.shutdown()

if __name__ == "__main__":
    serve_frontend()
