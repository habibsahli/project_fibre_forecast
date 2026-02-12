#!/usr/bin/env python3
"""
Run the Fibre Forecasting Web Application
Starts the simple HTTP server and opens web interface
"""

import sys
import os
import site

# Add user site-packages to path
user_site = site.getusersitepackages()
if user_site not in sys.path:
    sys.path.insert(0, user_site)

import subprocess
import time
import webbrowser
from pathlib import Path

def start_api():
    """Start the simple HTTP API server"""
    print("🚀 Starting HTTP API server...")
    api_process = subprocess.Popen([
        sys.executable, "simple_api.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait a bit for server to start
    time.sleep(2)

    # Check if server started successfully
    if api_process.poll() is None:
        print("✅ HTTP API server started successfully on http://localhost:5000")
        return api_process
    else:
        stdout, stderr = api_process.communicate()
        print("❌ Failed to start HTTP API server:")
        print(stderr.decode())
        return None

def open_web_interface():
    """Open the web interface in browser"""
    print("🌐 Opening web interface in browser...")
    webbrowser.open("http://localhost:5000")

def main():
    print("🔮 Fibre Subscription Forecasting - Web Application")
    print("="*55)

    # Start API server
    api_process = start_api()
    if not api_process:
        print("❌ Cannot start web application without API server")
        return 1

    # Open web interface
    open_web_interface()

    print("\n" + "="*55)
    print("🎉 Web application is running!")
    print("🌐 Web Interface: http://localhost:5000")
    print("🔗 API Endpoints: http://localhost:5000")
    print("🛑 Press Ctrl+C to stop the application")
    print("="*55)

    try:
        # Keep running until user stops
        api_process.wait()
    except KeyboardInterrupt:
        print("\n🛑 Stopping application...")
        api_process.terminate()
        api_process.wait()
        print("✅ Application stopped")

    return 0

if __name__ == "__main__":
    sys.exit(main())