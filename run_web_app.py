#!/usr/bin/env python3
"""
Run the Fibre Forecasting Web Application
Starts both Flask API and web interface
"""

import subprocess
import sys
import time
import webbrowser
from pathlib import Path

def start_api():
    """Start the Flask API in background"""
    print("🚀 Starting Flask API...")
    api_process = subprocess.Popen([
        sys.executable, "api.py"
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    # Wait a bit for API to start
    time.sleep(3)

    # Check if API started successfully
    if api_process.poll() is None:
        print("✅ Flask API started successfully on http://localhost:5000")
        return api_process
    else:
        stdout, stderr = api_process.communicate()
        print("❌ Failed to start Flask API:")
        print(stderr.decode())
        return None

def open_web_interface():
    """Open the web interface in browser"""
    web_path = Path("web_interface.html").absolute()
    print(f"🌐 Opening web interface: file://{web_path}")
    webbrowser.open(f"file://{web_path}")

def main():
    print("🔮 Fibre Subscription Forecasting - Web Application")
    print("="*55)

    # Start API
    api_process = start_api()
    if not api_process:
        print("❌ Cannot start web application without API")
        return 1

    # Open web interface
    open_web_interface()

    print("\n" + "="*55)
    print("🎉 Web application is running!")
    print("📱 Web Interface: Open web_interface.html in your browser")
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