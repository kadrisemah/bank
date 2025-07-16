"""
Simple script to run the Banking ML project
Usage: python run_project.py
"""

import subprocess
import sys
import time
import webbrowser
from threading import Thread

def run_backend():
    """Run FastAPI backend"""
    print("🚀 Starting Backend API...")
    subprocess.run([sys.executable, "-m", "uvicorn", "src.api.app:app", "--reload", "--host", "0.0.0.0", "--port", "8000"])

def run_frontend():
    """Run Dash frontend"""
    print("🚀 Starting Dashboard...")
    time.sleep(5)  # Wait for backend to start
    subprocess.run([sys.executable, "src/dashboard/app.py"])

def main():
    print("🏦 Starting Banking ML Project")
    print("=" * 50)
    
    # Start backend in a thread
    backend_thread = Thread(target=run_backend, daemon=True)
    backend_thread.start()
    
    # Give backend time to start
    print("⏳ Waiting for backend to start...")
    time.sleep(5)
    
    # Open browser
    print("🌐 Opening dashboard in browser...")
    webbrowser.open("http://localhost:8050")
    
    # Run frontend in main thread
    try:
        run_frontend()
    except KeyboardInterrupt:
        print("\n👋 Shutting down...")

if __name__ == "__main__":
    main()